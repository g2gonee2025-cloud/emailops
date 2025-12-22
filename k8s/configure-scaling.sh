#!/bin/bash
# =============================================================================
# Configure GPU Autoscaling for Dev (L40S)
# =============================================================================

CLUSTER_ID="56eec486-1e93-4dbd-ad9b-a17643a9df31"
POOL_NAME="pool-bxlxnlbu1"
MIN_NODES=0
MAX_NODES=3

echo "Configuring autoscaling for node pool '$POOL_NAME' on cluster '$CLUSTER_ID'..."

# Check current status
CURRENT_CONFIG=$(doctl kubernetes cluster node-pool get $CLUSTER_ID $POOL_NAME --format AutoScale,MinNodes,MaxNodes --no-header)
read AUTO_SCALE MIN MAX <<< "$CURRENT_CONFIG"

echo "Current config: AutoScale=$AUTO_SCALE, Min=$MIN, Max=$MAX"

if [ "$AUTO_SCALE" == "true" ] && [ "$MIN" == "$MIN_NODES" ] && [ "$MAX" == "$MAX_NODES" ]; then
    echo "Autoscaling is already correctly configured."
    exit 0
fi

# Enable/Update autoscaling
echo "Updating node pool to AutoScale=true (Min=$MIN_NODES, Max=$MAX_NODES)..."
doctl kubernetes cluster node-pool update $CLUSTER_ID $POOL_NAME \
    --auto-scale \
    --min-nodes $MIN_NODES \
    --max-nodes $MAX_NODES

if [ $? -eq 0 ]; then
    echo "Successfully updated autoscaling configuration."
else
    echo "Error updating node pool."
    exit 1
fi