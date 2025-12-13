#!/bin/bash
# =============================================================================
# EmailOps Kubernetes Deployment Script
# =============================================================================
# Deploys the full stack to DigitalOcean Kubernetes (DOKS)
# =============================================================================

set -e

NAMESPACE="emailops"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# DOKS Cluster Info (NYC2)
CLUSTER_ID="23c013d9-4d8d-4d3d-a813-7e5cbc3d0af1"
CLUSTER_NAME="k8s-1-34-1-do-1-nyc2-1765360390845"
GPU_POOL_ID="e359afb3-1891-4bba-94b4-c7ab1d2e1736"
CPU_POOL_ID="c949fea3-5d57-4c30-9fd4-35acdf6973af"

echo "=============================================="
echo "EmailOps Kubernetes Deployment (NYC2)"
echo "Cluster: $CLUSTER_NAME"
echo "=============================================="

# Check kubectl connection
echo "[1/8] Checking cluster connection..."
kubectl cluster-info || { echo "ERROR: Cannot connect to cluster"; exit 1; }

# Create namespace
echo "[2/8] Creating namespace..."
kubectl apply -f "$SCRIPT_DIR/namespace.yaml"

# Apply secrets (must exist!)
echo "[3/8] Applying secrets..."
if [ ! -f "$SCRIPT_DIR/secrets.yaml" ]; then
    echo "ERROR: secrets.yaml not found!"
    echo "Copy secrets-template.yaml to secrets.yaml and fill in your values."
    exit 1
fi
kubectl apply -f "$SCRIPT_DIR/secrets.yaml"

# Apply ConfigMap
echo "[4/8] Applying ConfigMap..."
kubectl apply -f "$SCRIPT_DIR/configmap.yaml"

# Deploy PostgreSQL (StatefulSet with PVC)
echo "[5/8] Deploying PostgreSQL..."
kubectl apply -f "$SCRIPT_DIR/postgres.yaml"
echo "    Waiting for PostgreSQL to be ready..."
kubectl rollout status statefulset/postgres -n $NAMESPACE --timeout=120s || true

# Deploy Redis
echo "[6/8] Deploying Redis..."
kubectl apply -f "$SCRIPT_DIR/redis.yaml"
kubectl rollout status deployment/redis -n $NAMESPACE --timeout=60s || true

# Deploy Embeddings API (vLLM on GPU)
echo "[7/8] Deploying Embeddings API..."
kubectl apply -f "$SCRIPT_DIR/embeddings-pvc.yaml" || true
kubectl apply -f "$SCRIPT_DIR/embeddings-vllm.yaml"
echo "    Note: Embeddings requires GPU node. Waiting for pod..."

# Deploy Backend
echo "[8/8] Deploying Backend..."
kubectl apply -f "$SCRIPT_DIR/backend-deployment.yaml"

# Optional: Deploy MiniMax M2 LLM (requires H200 GPU)
if [ "$DEPLOY_LLM" = "true" ]; then
    echo "[OPTIONAL] Deploying MiniMax M2 LLM..."
    kubectl apply -f "$SCRIPT_DIR/minimax-m2-llm.yaml"
    echo "    Note: LLM requires H200 GPU node pool to be scaled up."
fi

# Apply Ingress
echo "[9/8] Applying Ingress..."
kubectl apply -f "$SCRIPT_DIR/ingress.yaml" || true

# Apply HPA
echo "[10/8] Applying HPA..."
kubectl apply -f "$SCRIPT_DIR/hpa.yaml" || true

echo ""
echo "=============================================="
echo "Deployment Complete!"
echo "=============================================="
echo ""
echo "Check status with:"
echo "  kubectl get pods -n $NAMESPACE"
echo ""
echo "To scale up GPU pool for embeddings/LLM:"
echo "  doctl kubernetes cluster node-pool update $CLUSTER_ID $GPU_POOL_ID --count 1"
echo ""
echo "To scale down GPU pool (save costs):"
echo "  doctl kubernetes cluster node-pool update $CLUSTER_ID $GPU_POOL_ID --count 0"
echo ""
echo "To deploy MiniMax M2 LLM:"
echo "  DEPLOY_LLM=true ./deploy.sh"
echo ""
