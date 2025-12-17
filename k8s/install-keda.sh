#!/bin/bash
# =============================================================================
# KEDA Installation Script for EmailOps
# =============================================================================
# Reference: https://keda.sh/docs/2.16/deploy/#helm
# =============================================================================

set -e

echo "=============================================="
echo "Installing KEDA (Kubernetes Event-driven Autoscaler)"
echo "=============================================="

# Step 1: Add KEDA Helm repo
echo "[1/3] Adding KEDA Helm repository..."
helm repo add kedacore https://kedacore.github.io/charts
helm repo update

# Step 2: Install KEDA
echo "[2/3] Installing KEDA in 'keda' namespace..."
helm install keda kedacore/keda \
  --namespace keda \
  --create-namespace \
  --set watchNamespace="emailops"  # Only watch emailops namespace

# Step 3: Verify installation
echo "[3/3] Verifying KEDA installation..."
kubectl get pods -n keda

echo ""
echo "=============================================="
echo "KEDA Installation Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Deploy Prometheus (if not already running)"
echo "  2. Apply ScaledObject: kubectl apply -f keda-scaledobjects.yaml"
echo ""
