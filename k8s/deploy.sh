#!/bin/bash
# =============================================================================
# EmailOps Kubernetes Deployment Script (v4.0 Modular GPU Edition)
# =============================================================================
# Usage: ./deploy.sh [--gpu h100|h200]
# =============================================================================

set -e

NAMESPACE="emailops"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default GPU Profile
GPU_TYPE="h100"

# Parse Arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu) GPU_TYPE="$2"; shift ;;
        --help) echo "Usage: ./deploy.sh [--gpu h100|h200]"; exit 0 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# GPU Profiles (Node Pool IDs & Configs)
# H200 Pool ID: pool-f4x80anpj (e359afb3-1891-4bba-94b4-c7ab1d2e1736)
# H100 Pool ID: pool-h100 (6bd0fb19-90a0-42eb-bc6b-8beccc704d89) - Assuming based on creation
# NOTE: Using Pool Names for `doks.digitalocean.com/node-pool` selector is simpler and strictly correct.

if [ "$GPU_TYPE" == "h100" ]; then
    echo "Using Profile: H100 (80GB VRAM)"
    export NODE_POOL="pool-h100"
    export MAX_BATCH_SIZE="256"
elif [ "$GPU_TYPE" == "h200" ]; then
    echo "Using Profile: H200 (141GB VRAM)"
    export NODE_POOL="pool-f4x80anpj"
    export MAX_BATCH_SIZE="256" # Conservative start, can go up to 512
else
    echo "Error: Unsupported GPU type '$GPU_TYPE'. Use 'h100' or 'h200'."
    exit 1
fi

echo "=============================================="
echo "EmailOps Kubernetes Deployment (NYC2)"
echo "Target GPU: $GPU_TYPE"
echo "Node Pool:  $NODE_POOL"
echo "Batch Size: $MAX_BATCH_SIZE"
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
    echo "    Using secrets_live.yaml..."
    kubectl apply -f "$SCRIPT_DIR/secrets_live.yaml"
elif [ -f "$SCRIPT_DIR/secrets.yaml" ]; then
    kubectl apply -f "$SCRIPT_DIR/secrets.yaml"
else
    echo "ERROR: No secrets file found!"
    exit 1
fi

# Apply ConfigMap (Update BATCH_SIZE dynamically if needed, but we rely on CLI args for vLLM)
echo "[4/8] Applying ConfigMap..."
kubectl apply -f "$SCRIPT_DIR/configmap.yaml"

# Apply Priority Classes
echo "[5/8] Applying Priority Classes..."
kubectl apply -f "$SCRIPT_DIR/priority-classes.yaml"

# Databases (Managed)
echo "[6/8] Configuring Database..."
echo "    Using DigitalOcean Managed PostgreSQL."
echo "[7/8] Redis/Valkey..."
echo "    Using DigitalOcean Managed Valkey."

# Deploy AI Bundle (Embeddings + Reranker)
echo "[8/8] Deploying AI Bundle (Embeddings + Reranker) on $GPU_TYPE..."
kubectl apply -f "$SCRIPT_DIR/ai-bundle.yaml"

# Deploy Qdrant (Vector DB)
echo "[9/8] Deploying Qdrant vector DB..."
kubectl apply -f "$SCRIPT_DIR/qdrant.yaml"

# Deploy SonarQube
echo "[9/8] Deploying SonarQube & Postgres..."
kubectl apply -f "$SCRIPT_DIR/sonarqube-postgres.yaml"
kubectl apply -f "$SCRIPT_DIR/sonarqube.yaml"

# Deploy Backend
echo "[10/8] Deploying Backend API..."
kubectl apply -f "$SCRIPT_DIR/backend-deployment.yaml"

# Apply Ingress
echo "[11/8] Applying Ingress..."
kubectl apply -f "$SCRIPT_DIR/ingress.yaml" || true

# Scaling & Optimization
echo "[12/8] Applying Scaling & Optimization Layer..."

# HPA
echo "    Applying HPA..."
kubectl apply -f "$SCRIPT_DIR/hpa.yaml" || true

# GPU Buffer Pods
echo "    Applying GPU Buffer Pods ($NODE_POOL)..."
envsubst < "$SCRIPT_DIR/gpu-buffer.yaml.template" > "$SCRIPT_DIR/gpu-buffer.yaml"
kubectl apply -f "$SCRIPT_DIR/gpu-buffer.yaml" || true

echo ""
echo "=============================================="
echo "Deployment Submitted!"
echo "=============================================="
echo ""
echo "Monitor pod status:"
echo "   kubectl get pods -n $NAMESPACE -w"
echo ""
