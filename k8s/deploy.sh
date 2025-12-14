#!/bin/bash
# =============================================================================
# EmailOps Kubernetes Deployment Script (v3.3 DOKS Edition)
# =============================================================================
# Deploys the full stack to DigitalOcean Kubernetes (DOKS)
# using Managed PostgreSQL and Spaces.
# =============================================================================

set -e

NAMESPACE="emailops"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# DOKS Cluster Info (Verified from User Input)
CLUSTER_ID="23c013d9-4d8d-4d3d-a813-7e5cbc3d0af1"
CLUSTER_NAME="k8s-1-34-1-do-1-nyc2-1765360390845"
GPU_POOL_ID="e359afb3-1891-4bba-94b4-c7ab1d2e1736" # pool-f4x80anpj (H200)
CPU_POOL_ID="c949fea3-5d57-4c30-9fd4-35acdf6973af" # pool-ewvdhlkec

echo "=============================================="
echo "EmailOps Kubernetes Deployment (NYC2)"
echo "Cluster: $CLUSTER_NAME"
echo "=============================================="
echo ""

# Check kubectl connection
echo "[1/8] Checking cluster connection..."
kubectl cluster-info || { echo "ERROR: Cannot connect to cluster. Check your kubeconfig."; exit 1; }

# Create namespace
echo "[2/8] Creating namespace '$NAMESPACE'..."
kubectl apply -f "$SCRIPT_DIR/namespace.yaml"

# Apply secrets
echo "[3/8] Applying secrets..."
if [ -f "$SCRIPT_DIR/secrets_live.yaml" ]; then
    # Auto-rename secrets_live to secrets if needed or just apply it
    echo "    Using secrets_live.yaml..."
    kubectl apply -f "$SCRIPT_DIR/secrets_live.yaml"
elif [ -f "$SCRIPT_DIR/secrets.yaml" ]; then
    kubectl apply -f "$SCRIPT_DIR/secrets.yaml"
else
    echo "ERROR: No secrets file found!"
    echo "Please ensure k8s/secrets_live.yaml or k8s/secrets.yaml exists."
    exit 1
fi

# Apply ConfigMap
echo "[4/8] Applying ConfigMap..."
kubectl apply -f "$SCRIPT_DIR/configmap.yaml"

# Deploy Database Strategy: Managed
echo "[5/8] Configuring Database..."
echo "    Using DigitalOcean Managed PostgreSQL (emailops-db-nyc2)."
echo "    Skipping in-cluster Postgres deployment."
# We do NOT run kubectl apply -f postgres.yaml

# Redis/Valkey: Using DO Managed (skip in-cluster deployment)
echo "[6/8] Redis/Valkey..."
echo "    Using DigitalOcean Managed Valkey (emailops-redis)."
echo "    Skipping in-cluster Redis deployment."
# We do NOT run kubectl apply -f redis.yaml

# Deploy Embeddings API (vLLM on GPU)
echo "[7/8] Deploying Embeddings API (vLLM)..."
# Ensure GPU pool is scaled up? (Optional check)
echo "    Applying PVC and Service..."
kubectl apply -f "$SCRIPT_DIR/embeddings-pvc.yaml" || true
kubectl apply -f "$SCRIPT_DIR/embeddings-vllm.yaml"
echo "    Note: vLLM pod will stay Pending until H200 GPU node is available."

# Deploy Backend
echo "[8/8] Deploying Backend API..."
kubectl apply -f "$SCRIPT_DIR/backend-deployment.yaml"

# Optional: Deploy MiniMax M2 LLM
if [ "$DEPLOY_LLM" = "true" ]; then
    echo "[OPTIONAL] Deploying MiniMax M2 LLM..."
    kubectl apply -f "$SCRIPT_DIR/minimax-m2-llm.yaml"
fi

# Apply Ingress
echo "[9/8] Applying Ingress..."
kubectl apply -f "$SCRIPT_DIR/ingress.yaml" || true

# Apply HPA
echo "[10/8] Applying HPA..."
kubectl apply -f "$SCRIPT_DIR/hpa.yaml" || true

echo ""
echo "=============================================="
echo "Deployment Submitted!"
echo "=============================================="
echo ""
echo "Next Steps:"
echo "1. Scale up the GPU node pool for embeddings:"
echo "   doctl kubernetes cluster node-pool update $CLUSTER_ID $GPU_POOL_ID --count 1"
echo ""
echo "2. Monitor pod status:"
echo "   kubectl get pods -n $NAMESPACE -w"
echo ""
