#!/bin/bash
# =============================================================================
# One-Click Dev Deployment Script (TOR1 / L40S)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER_CONTEXT="do-tor1-k8s-tor1-retrieval"
CLUSTER_ID="56eec486-1e93-4dbd-ad9b-a17643a9df31"
POOL_NAME="pool-bxlxnlbu1"

echo "=============================================="
echo "EmailOps Dev Deployment (TOR1)"
echo "=============================================="

# 1. Setup Secrets
echo "[1/4] Checking secrets..."
if [ ! -f "$SCRIPT_DIR/secrets-dev.yaml" ]; then
    if [ -f "$SCRIPT_DIR/secrets_live.yaml" ]; then
        echo "    Creating secrets-dev.yaml from secrets_live.yaml..."
        cp "$SCRIPT_DIR/secrets_live.yaml" "$SCRIPT_DIR/secrets-dev.yaml"
        # Using sed to patch the namespace in place (compatible with Linux/Mac)
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' 's/namespace: emailops/namespace: emailops-dev/g' "$SCRIPT_DIR/secrets-dev.yaml"
        else
            sed -i 's/namespace: emailops/namespace: emailops-dev/g' "$SCRIPT_DIR/secrets-dev.yaml"
        fi
    else
        echo "ERROR: No secrets_live.yaml found to clone. Please create k8s/secrets-dev.yaml manually."
        exit 1
    fi
else
    echo "    secrets-dev.yaml exists."
fi

# 2. Switch Context
echo "[2/4] Switching Kubernetes context..."
current_ctx=$(kubectl config current-context)
if [ "$current_ctx" != "$CLUSTER_CONTEXT" ]; then
    kubectl config use-context "$CLUSTER_CONTEXT"
else
    echo "    Already on context $CLUSTER_CONTEXT"
fi

# 3. Configure Autoscaling
echo "[3/4] Ensuring GPU autoscaling is enabled..."
# Check quickly to avoid slow API call if possible, or just run update idempotently
CURRENT_CONFIG=$(doctl kubernetes cluster node-pool get $CLUSTER_ID $POOL_NAME --format AutoScale,MinNodes,MaxNodes --no-header 2>/dev/null || true)

if [ -z "$CURRENT_CONFIG" ]; then
    echo "    WARNING: Could not fetch node pool config via doctl. Skipping autoscaling check."
else
    read AUTO_SCALE MIN MAX <<< "$CURRENT_CONFIG"
    if [ "$AUTO_SCALE" == "true" ] && [ "$MIN" == "0" ] && [ "$MAX" == "3" ]; then
        echo "    Autoscaling already configured (0-3 nodes)."
    else
        echo "    Updating node pool to AutoScale=true (Min=0, Max=3)..."
        doctl kubernetes cluster node-pool update $CLUSTER_ID $POOL_NAME \
            --auto-scale \
            --min-nodes 0 \
            --max-nodes 3
    fi
fi

# 4. Deploy
echo "[4/4] Deploying stack..."
"$SCRIPT_DIR/deploy.sh" --env dev --gpu l40s

echo ""
echo "=============================================="
echo "Dev Environment Deployed Successfully!"
echo "URL: https://dev-api.emailops.linkos.me"
echo "Monitor pods: kubectl get pods -n emailops-dev -w"
echo "=============================================="