# ArgoCD Setup for Multi-Region Deployment

This directory contains ArgoCD resources for syncing applications across NYC2 and TOR1 clusters.

## Prerequisites

1. Install ArgoCD in each cluster:
   ```bash
   kubectl create namespace argocd
   kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
   ```

2. Get the initial admin password:
   ```bash
   kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d
   ```

3. Port-forward to access ArgoCD UI:
   ```bash
   kubectl port-forward svc/argocd-server -n argocd 8080:443
   ```

## Registering Clusters

To enable hub-spoke sync (one ArgoCD managing both clusters):

```bash
# From the hub cluster (e.g., NYC2), register TOR1 as a remote cluster
argocd cluster add <tor1-context-name>
```

## Application Sync

Apply the `emailops-app.yaml` to automatically sync the `emailops` namespace across both clusters.
