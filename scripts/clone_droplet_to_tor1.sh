#!/usr/bin/env bash
# =============================================================================
# DigitalOcean Droplet Clone: NYC2 source -> TOR1 destination (Best Practices)
# =============================================================================
# Clones a droplet by:
#   1) Powering off the source droplet for a crash-consistent snapshot
#   2) Creating a snapshot image
#   3) Creating a new droplet in TOR1 from that snapshot
#   4) Applying the same tags (so firewalls that target tags apply automatically)
#
# Best practices (2025):
#   - Use doctl + jq; authenticate with a PAT via: doctl auth init
#   - Power off before snapshot for FS consistency
#   - Verify size availability in the target region; fallback if not offered there
#   - Use the target region's default VPC (or allow override)
#   - Use --enable-monitoring to install the agent; optionally --droplet-agent=true to make install errors fatal
#   - Reuse an existing snapshot if present to avoid re-snapshotting (optional)
#   - Avoid hardcoding SSH keys; allow explicit --ssh-keys
#   - Preserve source tags so firewalls bound to tags continue to apply
#
# Requirements:
#   - doctl (auth set up) and jq
#
# Usage:
#   ./scripts/clone_droplet_to_tor1.sh \
#       --src-name dev-station-nyc2 \
#       [--dst-name dev-station-tor1] \
#       [--dst-region tor1] \
#       [--dst-vpc-uuid <uuid-of-tor1-vpc>] \
#       [--ssh-keys keyID1,keyID2] \
#       [--enable-ipv6] \
#       [--enable-monitoring] \
#       [--enable-backups] \
#       [--with-droplet-agent] \
#       [--reuse-snapshot] \
#       [--auto-size-fallback] \
#       [--yes]
#
# Notes:
#   - This clones only the droplet root disk. Block storage volumes must be
#     snapshot/migrated separately.
#   - Snapshots are global; once "available", they can be used in any region.
#   - Firewalls: most setups target tags. Copying tags is usually enough.
# =============================================================================
set -euo pipefail

# ---------- defaults ----------
DST_NAME=""
DST_REGION="tor1"
DST_VPC_UUID=""
SRC_NAME=""
SSH_KEYS=""
ENABLE_IPV6=false
ENABLE_MONITORING=false
ENABLE_BACKUPS=false
DROPLET_AGENT_FATAL=false
ASSUME_YES=false
REUSE_SNAPSHOT=false
AUTO_SIZE_FALLBACK=false
PROJECT_ID=""
SRC_ID=""

# ---------- helpers ----------
die() { echo "ERROR: $*" >&2; exit 1; }
need() { command -v "$1" &>/dev/null || die "Missing required binary: $1"; }

confirm() {
  if $ASSUME_YES; then return 0; fi
  printf "%s [y/N]: " "$1"
  read -r ans
  [[ "${ans:-}" =~ ^[Yy]$ ]]
}

wait_for_status() {
  local droplet_id="$1"
  local desired="$2"
  local timeout="${3:-900}"
  local start ts status
  start=$(date +%s)
  while true; do
    status="$(doctl compute droplet get "$droplet_id" -o json | jq -r '.[0].status')"
    [[ -n "$status" ]] || die "Failed to fetch droplet status (id=$droplet_id)"
    if [[ "$status" == "$desired" ]]; then
      break
    fi
    ts=$(($(date +%s) - start))
    if (( ts > timeout )); then
      die "Timed out waiting for droplet $droplet_id to be '$desired' (last=$status)"
    fi
    sleep 5
  done
}

# Attempt graceful shutdown, then hard power-off as fallback
shutdown_or_poweroff() {
  local droplet_id="$1"
  local soft_timeout="${2:-180}"
  doctl compute droplet-action shutdown "$droplet_id" >/dev/null 2>&1 || true
  if ! wait_for_status "$droplet_id" "off" "$soft_timeout" 2>/dev/null; then
    echo "    Graceful shutdown timed out; forcing power-off..."
    doctl compute droplet-action power-off "$droplet_id" >/dev/null
    wait_for_status "$droplet_id" "off" 900
  fi
}

# Return image id when a user snapshot by name is "available"
wait_for_image_available() {
  local snap_name="$1"
  local timeout="${2:-3600}"
  local start ts state
  start=$(date +%s)
  while true; do
    # Might not exist immediately
    state="$(doctl compute image list-user -o json | jq -r --arg n "$snap_name" '.[] | select(.name==$n) | .status' | head -n1 || true)"
    if [[ "$state" == "available" ]]; then
      break
    fi
    ts=$(($(date +%s) - start))
    if (( ts > timeout )); then
      die "Timed out waiting for snapshot '$snap_name' to become available (last='$state')"
    fi
    sleep 10
  done
  doctl compute image list-user -o json | jq -r --arg n "$snap_name" '.[] | select(.name==$n) | .id' | head -n1
}

# Check if a size slug is offered in a region; echo "yes"/"no"
size_offered_in_region() {
  local slug="$1" region="$2"
  # sizes include list of regions they are available in
  if doctl compute size list -o json | jq -e --arg s "$slug" --arg r "$region" '.[] | select(.slug==$s) | select(.regions[]?==$r)' >/dev/null; then
    echo "yes"
  else
    echo "no"
  fi
}

# Pick a fallback size in a region roughly equivalent to a "2 vCPU / 4GB" class
pick_fallback_size() {
  local region="$1"
  # Try common general-purpose sizes in order
  local candidates=("s-2vcpu-4gb" "s-2vcpu-2gb" "s-4vcpu-8gb")
  for c in "${candidates[@]}"; do
    if [[ "$(size_offered_in_region "$c" "$region")" == "yes" ]]; then
      echo "$c"
      return 0
    fi
  done
  # As a last resort, pick the first size offered in region
  doctl compute size list -o json | jq -r --arg r "$region" \
    '[ .[] | select(.regions[]?==$r) | .slug ] | first' | sed -n '1p'
}

# Pick the smallest size in region whose disk >= min_disk (GB)
pick_size_for_mindisk() {
  local region="$1" min_disk="$2"
  doctl compute size list -o json | jq -r --arg r "$region" --argjson m "$min_disk" \
    '[ .[]
      | select(.regions[]?==$r)
      | select((.disk // 0) >= $m)
    ] | sort_by(.disk) | first | .slug // empty'
}

# Validate SSH key identifiers exist (IDs or fingerprints)
validate_ssh_keys() {
  local csv="$1"
  [[ -z "$csv" ]] && return 0
  local keys_json
  keys_json="$(doctl compute ssh-key list -o json)"
  IFS=',' read -r -a arr <<< "$csv"
  for k in "${arr[@]}"; do
    k="${k// /}"
    if ! echo "$keys_json" | jq -e --arg k "$k" 'map(.id|tostring) + map(.fingerprint) | index($k) != null' >/dev/null; then
      die "SSH key not found (id or fingerprint): $k"
    fi
  done
}

# Get default VPC UUID for a region
default_vpc_uuid_for_region() {
  local region="$1"
  # vpcs list output contains region or region_slug, prefer "region" if present
  local vpc
  vpc="$(doctl vpcs list -o json | jq -r --arg r "$region" \
    '.[] | select((.region==$r) or (.region_slug==$r)) | select(.default==true) | .id' | head -n1)"
  if [[ -z "$vpc" ]]; then
    # Fallback to first VPC in region
    vpc="$(doctl vpcs list -o json | jq -r --arg r "$region" \
      '.[] | select((.region==$r) or (.region_slug==$r)) | .id' | head -n1)"
  fi
  echo "$vpc"
}

# ---------- args ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --src-name) SRC_NAME="${2:-}"; shift 2 ;;
    --src-id) SRC_ID="${2:-}"; shift 2 ;;
    --dst-name) DST_NAME="${2:-}"; shift 2 ;;
    --dst-region) DST_REGION="${2:-}"; shift 2 ;;
    --dst-vpc-uuid) DST_VPC_UUID="${2:-}"; shift 2 ;;
    --ssh-keys) SSH_KEYS="${2:-}"; shift 2 ;;
    --enable-ipv6) ENABLE_IPV6=true; shift ;;
    --enable-monitoring) ENABLE_MONITORING=true; shift ;;
    --enable-backups) ENABLE_BACKUPS=true; shift ;;
    --with-droplet-agent|--droplet-agent-fatal) DROPLET_AGENT_FATAL=true; shift ;;
    --reuse-snapshot) REUSE_SNAPSHOT=true; shift ;;
    --auto-size-fallback) AUTO_SIZE_FALLBACK=true; shift ;;
    --project-id) PROJECT_ID="${2:-}"; shift 2 ;;
    --yes|-y) ASSUME_YES=true; shift ;;
    -h|--help)
      sed -n '1,200p' "$0"; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

# ---------- preflight ----------
need doctl
need jq
if [[ -z "$SRC_ID" ]]; then
  [[ -n "$SRC_NAME" ]] || die "--src-name or --src-id is required"
fi
doctl account get >/dev/null 2>&1 || die "doctl not authenticated. Run: doctl auth init"

# Lookup source droplet
if [[ -n "$SRC_ID" ]]; then
  SRC_JSON="$(doctl compute droplet get "$SRC_ID" -o json | jq -c '.[0]')"
  [[ -n "$SRC_JSON" ]] || die "Source droplet not found by id: $SRC_ID"
else
  # Ensure unique match by name
  local_matches="$(doctl compute droplet list -o json | jq -c --arg n "$SRC_NAME" '[ .[] | select(.name==$n) ]')"
  count="$(echo "$local_matches" | jq 'length')"
  (( count > 0 )) || die "Source droplet not found by name: $SRC_NAME"
  (( count == 1 )) || die "Multiple droplets match --src-name=$SRC_NAME; use --src-id to disambiguate"
  SRC_JSON="$(echo "$local_matches" | jq -c '.[0]')"
fi

SRC_ID="$(echo "$SRC_JSON" | jq -r '.id')"
SRC_REGION="$(echo "$SRC_JSON" | jq -r '.region.slug')"
SIZE_SLUG="$(echo "$SRC_JSON" | jq -r '.size.slug')"
SRC_TAGS_CSV="$(echo "$SRC_JSON" | jq -r '.tags | join(",")')"
SRC_TAGS="${SRC_TAGS_CSV:-}"

if [[ -z "$DST_NAME" ]]; then
  DST_NAME="${SRC_NAME}-tor1"
fi

# Validate size availability in destination region
CHOSEN_SIZE="$SIZE_SLUG"
if [[ "$(size_offered_in_region "$SIZE_SLUG" "$DST_REGION")" != "yes" ]]; then
  if $AUTO_SIZE_FALLBACK; then
    echo "Source size '$SIZE_SLUG' not offered in $DST_REGION; selecting fallback..."
    CHOSEN_SIZE="$(pick_fallback_size "$DST_REGION")"
    [[ -n "$CHOSEN_SIZE" ]] || die "Could not select a fallback size for region $DST_REGION"
    echo "Using fallback size: $CHOSEN_SIZE"
  else
    die "Size '$SIZE_SLUG' not offered in $DST_REGION. Re-run with --auto-size-fallback or provide a region-supported size."
  fi
fi

# Determine destination VPC
if [[ -z "$DST_VPC_UUID" ]]; then
  DST_VPC_UUID="$(default_vpc_uuid_for_region "$DST_REGION" || true)"
  if [[ -z "$DST_VPC_UUID" ]]; then
    echo "WARNING: No VPC found in region $DST_REGION; droplet will use the public network only."
  else
    echo "Destination VPC: $DST_VPC_UUID"
  fi
fi

echo "=============================================="
echo "DigitalOcean Droplet Clone"
echo "Source droplet:      $SRC_NAME (id=$SRC_ID, region=$SRC_REGION, size=$SIZE_SLUG)"
echo "Destination name:    $DST_NAME"
echo "Destination region:  $DST_REGION"
echo "Destination size:    $CHOSEN_SIZE"
echo "Destination VPC:     ${DST_VPC_UUID:-<none>}"
echo "Tags to carry:       ${SRC_TAGS:-<none>}"
echo "SSH Keys:            ${SSH_KEYS:-<none specified>}"
echo "IPv6:                $ENABLE_IPV6"
echo "Monitoring:          $ENABLE_MONITORING"
echo "Backups:             $ENABLE_BACKUPS"
echo "Droplet Agent fatal: $DROPLET_AGENT_FATAL"
echo "Reuse snapshot:      $REUSE_SNAPSHOT"
echo "Auto size fallback:  $AUTO_SIZE_FALLBACK"
echo "Project ID:          ${PROJECT_ID:-<none>}"
echo "=============================================="

confirm "Proceed with snapshot and clone?" || die "Aborted by user"

# ---------- snapshot (optionally reuse) ----------
SNAP_NAME="${SRC_NAME:-src}-${SRC_ID}-snapshot-$(date -u +%Y%m%dT%H%M%SZ)"
IMAGE_ID=""
SOURCE_WAS_POWERED_OFF=false

# Ensure the source gets powered back on if we abort mid-flight
cleanup() {
  if $SOURCE_WAS_POWERED_OFF; then
    echo "Cleaning up: re-powering source droplet $SRC_ID ..."
    doctl compute droplet-action power-on "$SRC_ID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if $REUSE_SNAPSHOT; then
  # Try to find the newest available snapshot matching prefix "${SRC_NAME}-snapshot-"
  EXISTING_IMAGE_ID="$(doctl compute image list-user -o json | jq -r --arg p "${SRC_NAME}-snapshot-" \
    '[ .[] | select(.name | startswith($p)) | select(.status=="available") ] | sort_by(.created_at) | last | .id // empty')"
  if [[ -n "$EXISTING_IMAGE_ID" ]]; then
    echo "Reusing existing snapshot image id: $EXISTING_IMAGE_ID"
    IMAGE_ID="$EXISTING_IMAGE_ID"
  fi
fi

if [[ -z "$IMAGE_ID" ]]; then
  echo "[1/5] Powering off source droplet for a consistent snapshot..."
  shutdown_or_poweroff "$SRC_ID" 180
  SOURCE_WAS_POWERED_OFF=true
  echo "    Source is off."

  echo "[2/5] Creating snapshot: $SNAP_NAME"
  doctl compute droplet-action snapshot "$SRC_ID" --snapshot-name "$SNAP_NAME" >/dev/null

  echo "    Waiting for snapshot to be available (this can take time)..."
  IMAGE_ID="$(wait_for_image_available "$SNAP_NAME" 3600)"
  [[ -n "$IMAGE_ID" ]] || die "Snapshot did not yield an image id"

  echo "[3/5] Powering on source droplet..."
  doctl compute droplet-action power-on "$SRC_ID" >/dev/null
  wait_for_status "$SRC_ID" "active" 900
  SOURCE_WAS_POWERED_OFF=false
  echo "    Source is active."
fi

echo "Snapshot image id: $IMAGE_ID"

# Validate that chosen size's disk >= image MinDisk; select fallback if needed
IMAGE_MIN_DISK="$(doctl compute image list-user -o json | jq -r --arg id "$IMAGE_ID" '[ .[] | select((.id|tostring)==$id) ] | first | .min_disk // .MinDisk // 0')"
SIZE_DISK_GB="$(doctl compute size list -o json | jq -r --arg s "$CHOSEN_SIZE" '[ .[] | select(.slug==$s) ] | first | .disk // 0')"
if [[ -n "$IMAGE_MIN_DISK" && -n "$SIZE_DISK_GB" ]] && (( SIZE_DISK_GB < IMAGE_MIN_DISK )); then
  echo "Chosen size '$CHOSEN_SIZE' has disk ${SIZE_DISK_GB}GB, less than image minimum ${IMAGE_MIN_DISK}GB."
  if $AUTO_SIZE_FALLBACK; then
    new_size="$(pick_size_for_mindisk "$DST_REGION" "$IMAGE_MIN_DISK")"
    [[ -n "$new_size" ]] || die "Could not find a size in $DST_REGION meeting min disk ${IMAGE_MIN_DISK}GB"
    echo "Using fallback size meeting MinDisk: $new_size"
    CHOSEN_SIZE="$new_size"
  else
    die "Size '$CHOSEN_SIZE' disk ${SIZE_DISK_GB}GB < image MinDisk ${IMAGE_MIN_DISK}GB. Use --auto-size-fallback or specify a larger size."
  fi
fi

# ---------- create destination droplet ----------
echo "[4/5] Creating destination droplet in $DST_REGION ..."
CREATE_ARGS=(compute droplet create "$DST_NAME" --region "$DST_REGION" --size "$CHOSEN_SIZE" --image "$IMAGE_ID" --wait)

if [[ -n "$DST_VPC_UUID" ]]; then
  CREATE_ARGS+=(--vpc-uuid "$DST_VPC_UUID")
fi
if [[ -n "$SRC_TAGS" ]]; then
  CREATE_ARGS+=(--tag-names "$SRC_TAGS")
fi
if [[ -n "$SSH_KEYS" ]]; then
  validate_ssh_keys "$SSH_KEYS"
  CREATE_ARGS+=(--ssh-keys "$SSH_KEYS")
fi
if $ENABLE_IPV6; then
  CREATE_ARGS+=(--enable-ipv6)
fi
if $ENABLE_MONITORING; then
  CREATE_ARGS+=(--enable-monitoring)
fi
if $ENABLE_BACKUPS; then
  CREATE_ARGS+=(--enable-backups)
fi
if $DROPLET_AGENT_FATAL; then
  CREATE_ARGS+=(--droplet-agent=true)
fi
if [[ -n "$PROJECT_ID" ]]; then
  CREATE_ARGS+=(--project-id "$PROJECT_ID")
fi
# Capture the created droplet ID directly
DST_ID="$(doctl "${CREATE_ARGS[@]}" --format ID --no-header)"
[[ -n "$DST_ID" ]] || die "Failed to capture created droplet ID"

# ---------- fetch destination info ----------
echo "[5/5] Fetching destination droplet details..."
DST_JSON="$(doctl compute droplet get "$DST_ID" -o json | jq -c '.[0]')"
[[ -n "$DST_JSON" ]] || die "Destination droplet not found after creation"

DST_IP4="$(echo "$DST_JSON" | jq -r '.networks.v4[] | select(.type=="public") | .ip_address' | head -n1)"
DST_PRIV="$(echo "$DST_JSON" | jq -r '.networks.v4[] | select(.type=="private") | .ip_address' | head -n1)"
DST_STATUS="$(echo "$DST_JSON" | jq -r '.status')"

echo "=============================================="
echo "Clone complete."
echo "Destination id:   $DST_ID"
echo "Public IPv4:      ${DST_IP4:-<none yet>}"
echo "Private IPv4:     ${DST_PRIV:-<none>}"
echo "Status:           $DST_STATUS"
echo "=============================================="
echo
echo "Post-steps:"
echo " - If you use Cloud Firewalls bound to tags, they should already apply (copied tags: ${SRC_TAGS:-<none>})."
echo " - Verify SSH access (use --ssh-keys to attach at creation if needed)."
echo " - If the source used block storage volumes, snapshot and attach those separately in TOR1."
echo " - Consider reserving a floating IP in TOR1 and assigning it to this droplet if you need a stable address."