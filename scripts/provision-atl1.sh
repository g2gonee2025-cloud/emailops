#!/bin/bash
# provision-atl1.sh - Provision an Atlanta (ATL1) Kubernetes cluster with H200
#
# Usage: ./provision-atl1.sh

set -euo pipefail

# --- Configuration ---
CLUSTER_NAME="k8s-atl1-retrieval"
REGION="atl1"
K8S_VERSION="1.34.1-do.1"
GENERAL_NODE_POOL_NAME="pool-general"
GENERAL_NODE_POOL_SIZE="s-2vcpu-8gb-amd"
GENERAL_NODE_POOL_COUNT=1
GPU_NODE_POOL_NAME="pool-gpu-h200"
GPU_NODE_POOL_SIZE="gpu-h200x1-141gb"
GPU_NODE_POOL_COUNT=1

# --- Cleanup ---
CLUSTER_CREATED_BY_SCRIPT=false
cleanup() {
  if [ "$CLUSTER_CREATED_BY_SCRIPT" = true ]; then
    echo ""
    echo "An error occurred during provisioning." >&2
    read -p "Do you want to delete the partially created cluster '$CLUSTER_NAME'? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      echo "Deleting cluster '$CLUSTER_NAME'..."
      doctl kubernetes cluster delete "$CLUSTER_NAME" --force
      echo "Cluster '$CLUSTER_NAME' deleted."
    fi
  fi
}
trap cleanup ERR

# --- Preflight Checks ---
echo "=== Running Preflight Checks ==="

# Check for required tools
for tool in doctl kubectl; do
  if ! command -v "$tool" &> /dev/null; then
    echo "Error: Required tool '$tool' is not installed." >&2
    exit 1
  fi
done
echo "✔ Required tools (doctl, kubectl) are installed."

# Check for doctl authentication
if ! doctl account get > /dev/null 2>&1; then
  echo "Error: Not authenticated with doctl. Please run 'doctl auth init'." >&2
  exit 1
fi
echo "✔ Authenticated with doctl."
echo ""


echo "=== Provisioning ATL1 Cluster ==="
echo "Cluster Name: $CLUSTER_NAME"
echo "Region: $REGION"
echo "K8s Version: $K8S_VERSION"
echo ""

# Check if cluster already exists
echo "Step 1: Checking for existing cluster..."
if CLUSTER_ID=$(doctl kubernetes cluster get "$CLUSTER_NAME" --format ID --no-header 2>/dev/null); then
  echo "Cluster '$CLUSTER_NAME' already exists with ID: $CLUSTER_ID"
else
  echo "Cluster '$CLUSTER_NAME' not found. Creating new cluster..."
  CLUSTER_CREATED_BY_SCRIPT=true
  # Create cluster with general node pool
  doctl kubernetes cluster create "$CLUSTER_NAME" \
    --region "$REGION" \
    --version "$K8S_VERSION" \
    --node-pool "name=$GENERAL_NODE_POOL_NAME;size=$GENERAL_NODE_POOL_SIZE;count=$GENERAL_NODE_POOL_COUNT" \
    --wait

  # Get cluster ID
  CLUSTER_ID=$(doctl kubernetes cluster get "$CLUSTER_NAME" --format ID --no-header)
  echo "Cluster created with ID: $CLUSTER_ID"
fi

if [ -z "$CLUSTER_ID" ]; then
  echo "Error: Failed to retrieve Cluster ID. Aborting." >&2
  exit 1
fi


# Add GPU node pool (H200 - The only one available)
echo ""
echo "Step 2: Adding H200 GPU node pool..."
doctl kubernetes cluster node-pool create "$CLUSTER_ID" \
  --name "$GPU_NODE_POOL_NAME" \
  --size "$GPU_NODE_POOL_SIZE" \
  --count $GPU_NODE_POOL_COUNT \
  --wait

echo ""
echo "Step 3: Saving kubeconfig..."
doctl kubernetes cluster kubeconfig save "$CLUSTER_ID"

# --- Success ---
# Disable cleanup trap on successful exit
CLUSTER_CREATED_BY_SCRIPT=false
trap - ERR

echo ""
echo "=== ATL1 Cluster Ready ==="
echo "Cluster ID: $CLUSTER_ID"
echo ""
echo "Next steps:"
echo "1. Create the emailops namespace: kubectl create namespace emailops"
echo "2. Apply the retrieval manifest: kubectl apply -f k8s/retrieval-api.yaml"
