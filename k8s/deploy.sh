#!/bin/bash
# =============================================================================
# EmailOps Kubernetes Deployment Script (v4.0 Modular GPU Edition)
# =============================================================================
# Usage: ./deploy.sh [--gpu h100|h200] [--env dev|prod]
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
GPU_TYPE="h100"
ENV_MODE="prod"

# Parse Arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu) GPU_TYPE="$2"; shift ;;
        --env) ENV_MODE="$2"; shift ;;
        --help) echo "Usage: ./deploy.sh [--gpu h100|h200|l40s] [--env dev|prod]"; exit 0 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ "$ENV_MODE" == "dev" ]; then
    NAMESPACE="emailops-dev"
    CONFIG_FILE="configmap-dev.yaml"
    NS_FILE="namespace-dev.yaml"
    INGRESS_FILE="ingress-dev.yaml"
    echo "Deploying to DEVELOPMENT environment ($NAMESPACE)..."
else
    NAMESPACE="emailops"
    CONFIG_FILE="configmap.yaml"
    NS_FILE="namespace.yaml"
    INGRESS_FILE="ingress.yaml"
    echo "Deploying to PRODUCTION environment ($NAMESPACE)..."
fi

# GPU Profiles (Node Pool IDs & Configs)
# H200 Pool ID: pool-f4x80anpj (e359afb3-1891-4bba-94b4-c7ab1d2e1736)
# H100 Pool ID: pool-h100 (6bd0fb19-90a0-42eb-bc6b-8beccc704d89)
# L40S Pool ID (TOR1): pool-bxlxnlbu1 (8527e1b6-2c51-4d54-9a1d-de27bb3aea14)

if [ "$GPU_TYPE" == "h100" ]; then
    echo "Using Profile: H100 (80GB VRAM) [NYC2]"
    export NODE_POOL="pool-h100"
    export MAX_BATCH_SIZE="256"
elif [ "$GPU_TYPE" == "h200" ]; then
    echo "Using Profile: H200 (141GB VRAM) [NYC2]"
    export NODE_POOL="pool-f4x80anpj"
    export MAX_BATCH_SIZE="256"
elif [ "$GPU_TYPE" == "l40s" ]; then
    echo "Using Profile: L40S (48GB VRAM) [TOR1]"
    export NODE_POOL="pool-bxlxnlbu1"
    export MAX_BATCH_SIZE="128" # Reduced for 48GB VRAM
else
    echo "Error: Unsupported GPU type '$GPU_TYPE'. Use 'h100', 'h200', or 'l40s'."
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
kubectl apply -f "$SCRIPT_DIR/$NS_FILE"

# Apply secrets
echo "[3/8] Applying secrets..."
# For dev, we might want a different secrets file, but falling back to live secrets
# (and relying on the Namespace to isolate) is a common pattern if they share the same DO account resources.
# Ideally, create secrets-dev.yaml.
SECRETS_FILE="$SCRIPT_DIR/secrets.yaml"
if [ "$ENV_MODE" == "dev" ] && [ -f "$SCRIPT_DIR/secrets-dev.yaml" ]; then
    SECRETS_FILE="$SCRIPT_DIR/secrets-dev.yaml"
elif [ -f "$SCRIPT_DIR/secrets_live.yaml" ]; then
    SECRETS_FILE="$SCRIPT_DIR/secrets_live.yaml"
fi

echo "    Using secrets file: $SECRETS_FILE"
# Ensure we apply secrets to the correct namespace
kubectl apply -f "$SECRETS_FILE" -n $NAMESPACE

# Apply ConfigMap
echo "[4/8] Applying ConfigMap ($CONFIG_FILE)..."
kubectl apply -f "$SCRIPT_DIR/$CONFIG_FILE"

# Apply Priority Classes
echo "[5/8] Applying Priority Classes..."
kubectl apply -f "$SCRIPT_DIR/priority-classes.yaml"

# Databases (Managed)
echo "[6/8] Configuring Database..."
echo "    Using DigitalOcean Managed PostgreSQL."
echo "[7/8] Redis/Valkey..."
echo "    Using DigitalOcean Managed Valkey."

# Deploy Embeddings API (vLLM)
echo "[8/8] Deploying Embeddings API (vLLM) on $GPU_TYPE..."
# Generate manifest from template
envsubst < "$SCRIPT_DIR/embeddings-vllm.yaml.template" > "$SCRIPT_DIR/embeddings-vllm.yaml"
kubectl apply -f "$SCRIPT_DIR/embeddings-vllm.yaml" -n $NAMESPACE

# Deploy Qdrant (Vector DB)
echo "[9/8] Deploying Qdrant vector DB..."
kubectl apply -f "$SCRIPT_DIR/qdrant.yaml" -n $NAMESPACE

# Deploy Backend
echo "[10/8] Deploying Backend API..."
kubectl apply -f "$SCRIPT_DIR/backend-deployment.yaml" -n $NAMESPACE

# Apply Ingress
echo "[11/8] Applying Ingress ($INGRESS_FILE)..."
kubectl apply -f "$SCRIPT_DIR/$INGRESS_FILE" -n $NAMESPACE || true

# Scaling & Optimization
echo "[12/8] Applying Scaling & Optimization Layer..."

# HPA
echo "    Applying HPA..."
kubectl apply -f "$SCRIPT_DIR/hpa.yaml" -n $NAMESPACE || true

# GPU Buffer Pods
echo "    Applying GPU Buffer Pods ($NODE_POOL)..."
envsubst < "$SCRIPT_DIR/gpu-buffer.yaml.template" > "$SCRIPT_DIR/gpu-buffer.yaml"
kubectl apply -f "$SCRIPT_DIR/gpu-buffer.yaml" -n $NAMESPACE || true

echo ""
echo "=============================================="
echo "Deployment Submitted!"
echo "=============================================="
echo ""
echo "Monitor pod status:"
echo "   kubectl get pods -n $NAMESPACE -w"
echo ""
