#!/bin/bash
# =============================================================================
# Prometheus Stack Installation Script
# =============================================================================
# Installs kube-prometheus-stack (Prometheus + Grafana + AlertManager)
# Needed for KEDA to scrape metrics.
# =============================================================================

set -e

echo "=============================================="
echo "Installing Prometheus Stack"
echo "=============================================="

# Step 1: Add Prometheus Community Helm repo
echo "[1/3] Adding Prometheus Community Helm repository..."
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Step 2: Install kube-prometheus-stack
# We install in 'monitoring' namespace
echo "[2/3] Installing kube-prometheus-stack in 'monitoring' namespace..."
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace kube-prometheus-stack \
  --create-namespace \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
  --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false

# Step 3: Verify
echo "[3/3] Verifying installation..."
kubectl get pods -n kube-prometheus-stack -l release=prometheus

echo ""
echo "=============================================="
echo "Prometheus Stack Installation Complete!"
echo "Prometheus Server: http://prometheus-kube-prometheus-prometheus.kube-prometheus-stack.svc.cluster.local:9090"
echo "Grafana: http://prometheus-grafana.kube-prometheus-stack.svc.cluster.local"
echo "=============================================="
