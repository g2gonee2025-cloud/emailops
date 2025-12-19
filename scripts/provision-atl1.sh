#!/bin/bash
# provision-atl1.sh - Provision an Atlanta (ATL1) Kubernetes cluster with H200
#
# Usage: ./provision-atl1.sh

set -e

CLUSTER_NAME="k8s-atl1-retrieval"
REGION="atl1"
K8S_VERSION="1.34.1-do.1"

echo "=== Provisioning ATL1 Cluster ==="
echo "Cluster Name: $CLUSTER_NAME"
echo "Region: $REGION"
echo "K8s Version: $K8S_VERSION"
echo ""

# Create cluster with general node pool
echo "Step 1: Creating cluster with general node pool..."
doctl kubernetes cluster create "$CLUSTER_NAME" \
  --region "$REGION" \
  --version "$K8S_VERSION" \
  --node-pool "name=pool-general;size=s-2vcpu-8gb-amd;count=1" \
  --wait

# Get cluster ID
CLUSTER_ID=$(doctl kubernetes cluster list --format ID,Name --no-header | grep "$CLUSTER_NAME" | awk '{print $1}')
echo "Cluster created with ID: $CLUSTER_ID"

# Add GPU node pool (H200 - The only one available)
echo ""
echo "Step 2: Adding H200 GPU node pool..."
doctl kubernetes cluster node-pool create "$CLUSTER_ID" \
  --name "pool-gpu-h200" \
  --size "gpu-h200x1-141gb" \
  --count 1 \
  --wait

echo ""
echo "Step 3: Saving kubeconfig..."
doctl kubernetes cluster kubeconfig save "$CLUSTER_ID"

echo ""
echo "=== ATL1 Cluster Ready ==="
echo "Cluster ID: $CLUSTER_ID"
echo ""
echo "Next steps:"
echo "1. Create the emailops namespace: kubectl create namespace emailops"
echo "2. Apply the retrieval manifest: kubectl apply -f k8s/retrieval-api.yaml"
