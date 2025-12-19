#!/usr/bin/env bash
# =============================================================================
# Qdrant PVC Migration: NYC2 (source) -> TOR1 (destination)
# =============================================================================
# This script clones Qdrant data from a source DOKS cluster to a destination
# DOKS cluster (different region). It streams a tarball of the Qdrant storage
# from the source cluster and restores it into the destination cluster's PVC.
#
# Prereqs:
#   - bash, tar
#   - kubectl available locally
#   - Source kubeconfig (NYC2) and Destination kubeconfig (TOR1)
#   - Repo manifests present: k8s/namespace.yaml and k8s/qdrant.yaml
#
# Usage:
#   ./scripts/migrate_qdrant_to_tor1.sh \
#       --src-kubeconfig k8s/k8s-nyc2.yaml \
#       --dst-kubeconfig /path/to/tor1-kubeconfig.yaml \
#       [--namespace emailops]
#
# Notes:
#   - The script scales down Qdrant in the destination while restoring.
#   - The destination Qdrant PVC name must be 'qdrant-storage' per k8s/qdrant.yaml.
#   - No external object storage required; data is streamed locally.
#
# Files referenced:
#   - Namespace: k8s/namespace.yaml
#   - Qdrant:    k8s/qdrant.yaml
# =============================================================================
set -euo pipefail

NS="emailops"
SRC_KCFG=""
DST_KCFG=""
BACKUPS_DIR="./backups"
RESTORER_POD="qdrant-restorer"
QDRANT_DEPLOYMENT="qdrant"
PVC_NAME="qdrant-storage"

die() { echo "ERROR: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src-kubeconfig) SRC_KCFG="${2:-}"; shift 2 ;;
    --dst-kubeconfig) DST_KCFG="${2:-}"; shift 2 ;;
    --namespace) NS="${2:-emailops}"; shift 2 ;;
    -h|--help)
      sed -n '1,80p' "$0"
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

[[ -n "${SRC_KCFG}" ]] || die "--src-kubeconfig is required"
[[ -n "${DST_KCFG}" ]] || die "--dst-kubeconfig is required"
[[ -f "${SRC_KCFG}" ]] || die "Source kubeconfig not found: ${SRC_KCFG}"
[[ -f "${DST_KCFG}" ]] || die "Destination kubeconfig not found: ${DST_KCFG}"

echo "=============================================="
echo "Qdrant Migration NYC2 -> TOR1"
echo "Namespace:      ${NS}"
echo "Source kubeconfig:      ${SRC_KCFG}"
echo "Destination kubeconfig: ${DST_KCFG}"
echo "=============================================="
echo

mkdir -p "${BACKUPS_DIR}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ARCHIVE="${BACKUPS_DIR}/qdrant-storage-${STAMP}.tgz"

# -----------------------------------------------------------------------------
# 1) Verify cluster connectivity
# -----------------------------------------------------------------------------
echo "[1/9] Verifying cluster connectivity"
kubectl --kubeconfig "${SRC_KCFG}" cluster-info >/dev/null || die "Cannot reach SOURCE cluster"
kubectl --kubeconfig "${DST_KCFG}" cluster-info >/dev/null || die "Cannot reach DESTINATION cluster"

# -----------------------------------------------------------------------------
# 2) Ensure namespace and Qdrant exist on destination (PVC + Deployment)
# -----------------------------------------------------------------------------
echo "[2/9] Applying namespace and Qdrant manifests on DESTINATION"
kubectl --kubeconfig "${DST_KCFG}" apply -f k8s/namespace.yaml
kubectl --kubeconfig "${DST_KCFG}" -n "${NS}" apply -f k8s/qdrant.yaml

# Immediately scale down Qdrant to avoid file locks during restore
echo "[3/9] Scaling down DESTINATION Qdrant before restore"
kubectl --kubeconfig "${DST_KCFG}" -n "${NS}" scale deploy/${QDRANT_DEPLOYMENT} --replicas=0 || true

# Wait for pods to terminate
kubectl --kubeconfig "${DST_KCFG}" -n "${NS}" wait --for=delete pod -l app=emailops,component=qdrant --timeout=120s || true

# -----------------------------------------------------------------------------
# 3) Export (stream) Qdrant data from SOURCE into a local tarball
# -----------------------------------------------------------------------------
echo "[4/9] Exporting SOURCE Qdrant storage -> ${ARCHIVE}"
# Ensure Qdrant is running on source
kubectl --kubeconfig "${SRC_KCFG}" -n "${NS}" get deploy/${QDRANT_DEPLOYMENT} >/dev/null || die "Qdrant deployment not found on SOURCE"

# Stream tarball to local file
# We run tar from the Qdrant pod to package /qdrant/storage contents
SRC_POD="$(kubectl --kubeconfig "${SRC_KCFG}" -n "${NS}" get pods -l app=emailops,component=qdrant -o jsonpath='{.items[0].metadata.name}')"
[[ -n "${SRC_POD}" ]] || die "No Qdrant pod found on SOURCE"

echo "    Source pod: ${SRC_POD}"
kubectl --kubeconfig "${SRC_KCFG}" -n "${NS}" exec "${SRC_POD}" -- sh -lc 'tar -C /qdrant/storage -czf - .' > "${ARCHIVE}"
ls -lh "${ARCHIVE}"

# -----------------------------------------------------------------------------
# 4) Create a temporary restorer pod on DESTINATION with PVC mounted
# -----------------------------------------------------------------------------
echo "[5/9] Creating DESTINATION restorer pod (${RESTORER_POD}) with PVC '${PVC_NAME}'"
# Clean up any previous restorer
kubectl --kubeconfig "${DST_KCFG}" -n "${NS}" delete pod "${RESTORER_POD}" --ignore-not-found=true

cat <<EOF | kubectl --kubeconfig "${DST_KCFG}" -n "${NS}" apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: ${RESTORER_POD}
  labels:
    app: emailops
    component: qdrant-restorer
spec:
  restartPolicy: Never
  containers:
    - name: restorer
      image: alpine:3.20
      command: ["sh","-lc","sleep 36000"]
      volumeMounts:
        - name: qdrant-storage
          mountPath: /qdrant/storage
        - name: work
          mountPath: /work
  volumes:
    - name: qdrant-storage
      persistentVolumeClaim:
        claimName: ${PVC_NAME}
    - name: work
      emptyDir: {}
EOF

kubectl --kubeconfig "${DST_KCFG}" -n "${NS}" wait --for=condition=Ready pod/${RESTORER_POD} --timeout=120s

# -----------------------------------------------------------------------------
# 5) Copy archive into restorer pod and unpack into PVC
# -----------------------------------------------------------------------------
echo "[6/9] Copying archive into DESTINATION restorer pod"
kubectl --kubeconfig "${DST_KCFG}" -n "${NS}" cp "${ARCHIVE}" "${RESTORER_POD}:/work/upload.tgz"
echo "[7/9] Restoring into PVC (this may take a while)"
kubectl --kubeconfig "${DST_KCFG}" -n "${NS}" exec "${RESTORER_POD}" -- sh -lc '
  set -e
  apk add --no-cache tar >/dev/null
  rm -rf /qdrant/storage/*
  tar -xzf /work/upload.tgz -C /qdrant/storage
  sync
  echo "Restore finished"
'

# -----------------------------------------------------------------------------
# 6) Cleanup restorer and scale Qdrant back up
# -----------------------------------------------------------------------------
echo "[8/9] Deleting restorer pod"
kubectl --kubeconfig "${DST_KCFG}" -n "${NS}" delete pod "${RESTORER_POD}" --wait=true

echo "[9/9] Scaling up DESTINATION Qdrant and waiting for readiness"
kubectl --kubeconfig "${DST_KCFG}" -n "${NS}" scale deploy/${QDRANT_DEPLOYMENT} --replicas=1
kubectl --kubeconfig "${DST_KCFG}" -n "${NS}" rollout status deploy/${QDRANT_DEPLOYMENT} --timeout=180s

echo
echo "=============================================="
echo "Migration complete."
echo "Destination Qdrant should now serve the restored data."
echo "Health check from a cluster pod:"
echo "  kubectl --kubeconfig ${DST_KCFG} -n ${NS} run curl --rm -it --image=curlimages/curl --restart=Never -- \\"
echo "      sh -lc \"curl -s http://qdrant.${NS}.svc.cluster.local:6333/healthz\""
echo "=============================================="