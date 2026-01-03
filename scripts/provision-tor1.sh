#!/bin/bash
# provision-tor1.sh - Provision a Toronto (TOR1) Kubernetes cluster for retrieval services
#
# This script creates a DOKS cluster in Toronto with:
# - A general-purpose node pool for essential Kubernetes workloads
# - A GPU node pool using L40S for cost-effective AI inference
#
# Usage: ./provision-tor1.sh

set -euo pipefail

# --- Configuration ---
CLUSTER_NAME="k8s-tor1-retrieval"
REGION="tor1"
K8S_VERSION="1.30.5-do.0"
GENERAL_NODE_SIZE="s-2vcpu-8gb-amd"
GENERAL_NODE_COUNT=1
GPU_NODE_SIZE="gpu-l40sx1-48gb"
GPU_NODE_COUNT=1

# --- Cleanup ---
cleanup() {
  echo ""
  echo "--- Running cleanup ---"
  # Ask for confirmation before deleting the cluster if it exists
  if doctl kubernetes cluster get "$CLUSTER_NAME" --format Name --no-header &>/dev/null; then
    read -p "Do you want to delete the cluster '$CLUSTER_NAME'? (y/n): " confirm
    if [[ "$confirm" == "y" ]]; then
      echo "Deleting cluster '$CLUSTER_NAME'..."
      doctl kubernetes cluster delete "$CLUSTER_NAME" --force
    else
      echo "Skipping cluster deletion."
    fi
  else
    echo "Cluster '$CLUSTER_NAME' does not exist, no cleanup needed."
  fi
}

# --- Preflight Checks ---
preflight_checks() {
  echo "--- Running preflight checks ---"
  command -v doctl >/dev/null 2>&1 || { echo >&2 "doctl is not installed. Aborting."; exit 1; }
  doctl auth init --check >/dev/null 2>&1 || { echo >&2 "doctl is not authenticated. Aborting."; exit 1; }
  echo "Checks passed."
  echo ""
}

# --- Main Script ---
trap 'echo "An error occurred. Exiting..."; cleanup' ERR

preflight_checks

echo "=== Provisioning TOR1 Cluster ==="
echo "Cluster Name: $CLUSTER_NAME"
echo "Region: $REGION"
echo "K8s Version: $K8S_VERSION"
echo ""

# Check if cluster already exists
if doctl kubernetes cluster get "$CLUSTER_NAME" --format Name --no-header &>/dev/null; then
  echo "Cluster '$CLUSTER_NAME' already exists. Skipping creation."
else
  # Create cluster with general node pool
  echo "Step 1: Creating cluster with general node pool..."
  doctl kubernetes cluster create "$CLUSTER_NAME" \
    --region "$REGION" \
    --version "$K8S_VERSION" \
    --node-pool "name=pool-general;size=$GENERAL_NODE_SIZE;count=$GENERAL_NODE_COUNT" \
    --wait
fi

# Get cluster ID
echo ""
echo "Fetching cluster ID..."
CLUSTER_ID=$(doctl kubernetes cluster get "$CLUSTER_NAME" --format ID --no-header)

if [ -z "$CLUSTER_ID" ]; then
  echo "Error: Failed to get cluster ID for '$CLUSTER_NAME'."
  exit 1
fi
echo "Cluster ID: $CLUSTER_ID"

# Add GPU node pool (L40S) if it doesn't exist
if doctl kubernetes cluster node-pool get "$CLUSTER_ID" pool-gpu-l40s --format Name --no-header &>/dev/null; then
  echo "GPU node pool 'pool-gpu-l40s' already exists. Skipping creation."
else
  echo ""
  echo "Step 2: Adding L40S GPU node pool..."
  doctl kubernetes cluster node-pool create "$CLUSTER_ID" \
    --name "pool-gpu-l40s" \
    --size "$GPU_NODE_SIZE" \
    --count "$GPU_NODE_COUNT" \
    --wait
fi

echo ""
echo "Step 3: Saving kubeconfig..."
doctl kubernetes cluster kubeconfig save "$CLUSTER_ID"

echo ""
echo "=== TOR1 Cluster Ready ==="
echo "Cluster ID: $CLUSTER_ID"
echo ""
echo "Next steps:"
echo "1. Create the emailops namespace: kubectl create namespace emailops"
echo "2. Apply the retrieval manifest: kubectl apply -f k8s/retrieval-api.yaml"
echo "3. Set up ArgoCD for automatic sync"
