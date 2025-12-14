#!/bin/bash
# =============================================================================
# Provision Development Droplet ("DevStation")
# =============================================================================
# Creates a dedicated Droplet in NYC2 (same VPC as DB/Cluster) for:
# - VS Code Remote - SSH access
# - Direct DB access (Private Network)
# - S3 / Spaces access (High speed)
# =============================================================================

set -e

REGION="nyc2"
SIZE="s-2vcpu-4gb" # Cost-effective dev machine
IMAGE="ubuntu-22-04-x64"
NAME="dev-station-nyc2"
SSH_KEY_ID="52624582" # From 'doctl compute ssh-key list'
TAGS="dev,emailops"

echo "Creating Development Droplet: $NAME ($REGION, $SIZE)..."

# Create Droplet with Cloud-Init user-data to set up environment
doctl compute droplet create "$NAME" \
    --region "$REGION" \
    --size "$SIZE" \
    --image "$IMAGE" \
    --ssh-keys "$SSH_KEY_ID" \
    --tag-name "$TAGS" \
    --enable-monitoring \
    --enable-private-networking \
    --user-data-file - <<EOF
#cloud-config
packages:
  - git
  - python3-pip
  - docker.io
  - postgresql-client
  - unzip

runcmd:
  # Install Docker Compose
  - curl -L "https://github.com/docker/compose/releases/download/v2.24.5/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
  - chmod +x /usr/local/bin/docker-compose

  # Setup User
  - usermod -aG docker root

  # Clone Repository (Public/Deployment) or Setup Directory
  - mkdir -p /root/workspace
  - cd /root/workspace

  # Note: Since the repo is private on user's local, we just prep the folder.
  # User will 'rsync' or 'git clone' via VS Code later.

EOF

echo ""
echo "Droplet creation initiated. Waiting for IP address..."
sleep 15
IP=$(doctl compute droplet list "$NAME" --format PublicIPv4 --no-header)

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
echo "1. Open VS Code Remote - SSH"
echo "2. Connect to: root@$IP"
echo "3. Accept the host key."
echo ""
