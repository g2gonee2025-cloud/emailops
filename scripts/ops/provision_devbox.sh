#!/bin/bash
# =============================================================================
# Provision Development Droplet ("DevStation")
# =============================================================================
# Creates a dedicated Droplet in NYC2 (same VPC as DB/Cluster) for:
# - VS Code Remote - SSH access
# - Direct DB access (Private Network)
# - S3 / Spaces access (High speed)
# =============================================================================

set -euo pipefail

# Pre-flight checks
if ! command -v doctl >/dev/null 2>&1; then
    echo "Error: 'doctl' command not found."
    echo "Please install the DigitalOcean CLI (doctl) and authenticate."
    exit 1
fi

if ! doctl account get >/dev/null 2>&1; then
    echo "Error: Not authenticated with DigitalOcean."
    echo "Please run 'doctl auth init' to configure your account."
    exit 1
fi

REGION="nyc2"
SIZE="s-2vcpu-4gb" # Cost-effective dev machine
IMAGE="ubuntu-22-04-x64"
NAME="dev-station-nyc2"
SSH_KEY_ID="52624582" # From 'doctl compute ssh-key list'
TAG_NAMES="dev,emailops" # Comma-separated list of tags
VPC_UUID="YOUR_VPC_UUID_HERE" # Replace with your actual VPC UUID from 'doctl compute vpc list'

echo "Creating Development Droplet: $NAME ($REGION, $SIZE)..."

# Create Droplet and capture its ID
echo "Creating Droplet and capturing its ID..."
DROPLET_ID=$(doctl compute droplet create "$NAME" \
    --region "$REGION" \
    --size "$SIZE" \
    --image "$IMAGE" \
    --ssh-keys "$SSH_KEY_ID" \
    --tag-names "$TAG_NAMES" \
    --enable-monitoring \
    --vpc-uuid "$VPC_UUID" \
    --user-data-file "scripts/ops/cloud-init.yaml" \
    --format "ID" --no-header)

echo "Droplet creation initiated with ID: $DROPLET_ID. Waiting for it to become active..."

# Wait for the Droplet to be active
while true; do
    STATUS=$(doctl compute droplet get "$DROPLET_ID" --format "Status" --no-header)
    if [ "$STATUS" == "active" ]; then
        echo "Droplet is active."
        break
    fi
    echo "Current status: $STATUS. Waiting..."
    sleep 5
done

# Retrieve the IP address using the Droplet ID
IP=$(doctl compute droplet get "$DROPLET_ID" --format "PublicIPv4" --no-header)
if [ -z "$IP" ]; then
    echo "Error: Could not retrieve IP address for Droplet ID $DROPLET_ID."
    exit 1
fi

echo ""
echo "=============================================="
echo "DevStation Created!"
echo "=============================================="
echo "IP Address:   $IP"
echo "Region:       $REGION"
echo "SSH User:     root"
echo "=============================================="
echo ""
echo "To connect via VS Code:"
echo "1. Open VS Code Remote - SSH."
echo "2. Connect to: root@$IP"
echo "3. Accept the host key."
echo ""
echo "Security Recommendations:"
echo "-------------------------"
echo "1. SSH into the droplet and set a strong password for the root user."
echo "2. Create a non-root user for daily operations."
echo "3. Configure the UFW firewall to allow only necessary traffic (e.g., SSH):"
echo "   ufw allow OpenSSH"
echo "   ufw enable"
echo "4. Consider setting up SSH key-based authentication for your non-root user."
echo ""
