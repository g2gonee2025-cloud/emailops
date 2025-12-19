#!/bin/bash
# provision-tor1.sh - Provision a Toronto (TOR1) Kubernetes cluster for retrieval services
#
# This script creates a DOKS cluster in Toronto with:
# - A general-purpose node pool for control plane workloads
# - A GPU node pool using L40S for cost-effective AI inference
#
# Usage: ./provision-tor1.sh

set -e

CLUSTER_NAME="k8s-tor1-retrieval"
REGION="tor1"
K8S_VERSION="1.34.1-do.1"

echo "=== Provisioning TOR1 Cluster ==="
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

# Add GPU node pool (L40S)
echo ""
echo "Step 2: Adding L40S GPU node pool..."
doctl kubernetes cluster node-pool create "$CLUSTER_ID" \
  --name "pool-gpu-l40s" \
  --size "gpu-l40sx1-48gb" \
  --count 1 \
  --wait

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
