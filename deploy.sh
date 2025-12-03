#!/bin/bash
# EmailOps Kubernetes Deployment Script
# Usage: ./deploy.sh [build|deploy|all]

set -e

# Configuration
REGISTRY="registry.digitalocean.com/sf-registry"
IMAGE_NAME="emailops-backend"
TAG="${IMAGE_TAG:-latest}"
CLUSTER_NAME="emailops-k8s"
NAMESPACE="emailops"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check prerequisites
check_prereqs() {
    log_info "Checking prerequisites..."
    
    if ! command -v doctl &> /dev/null; then
        log_error "doctl is not installed. Install from https://docs.digitalocean.com/reference/doctl/how-to/install/"
        exit 1
    fi
    
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed."
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed."
        exit 1
    fi
    
    log_info "All prerequisites met."
}

# Connect to DOKS cluster
connect_cluster() {
    log_info "Connecting to DOKS cluster: $CLUSTER_NAME..."
    doctl kubernetes cluster kubeconfig save $CLUSTER_NAME
    kubectl config current-context
}

# Create container registry if it doesn't exist
setup_registry() {
    log_info "Setting up container registry..."
    
    # Check if registry exists
    if ! doctl registry get 2>/dev/null; then
        log_info "Creating container registry..."
        doctl registry create emailops-registry --region sgp1
    fi
    
    # Login to registry
    doctl registry login
    
    # Configure Kubernetes to pull from registry
    doctl registry kubernetes-manifest | kubectl apply -f -
}

# Build and push Docker image
build_and_push() {
    log_info "Building Docker image..."
    
    cd "$(dirname "$0")"
    docker build -t ${REGISTRY}/${IMAGE_NAME}:${TAG} -f backend/Dockerfile .
    
    log_info "Pushing to registry..."
    docker push ${REGISTRY}/${IMAGE_NAME}:${TAG}
    
    log_info "Image pushed: ${REGISTRY}/${IMAGE_NAME}:${TAG}"
}

# Deploy to Kubernetes
deploy() {
    log_info "Deploying to Kubernetes..."
    
    # Apply namespace first
    kubectl apply -f k8s/namespace.yaml
    
    # Apply secrets and configmap
    log_warn "Make sure to update k8s/secrets.yaml with your actual credentials before deploying!"
    kubectl apply -f k8s/configmap.yaml
    kubectl apply -f k8s/secrets.yaml
    
    # Apply deployment
    kubectl apply -f k8s/backend-deployment.yaml
    
    # Apply HPA
    kubectl apply -f k8s/hpa.yaml
    
    # Apply ingress (includes LoadBalancer service)
    kubectl apply -f k8s/ingress.yaml
    
    log_info "Deployment complete. Checking status..."
    
    # Wait for rollout
    kubectl -n $NAMESPACE rollout status deployment/emailops-backend --timeout=300s
    
    # Get service info
    log_info "Service endpoints:"
    kubectl -n $NAMESPACE get svc
    
    # Get external IP
    log_info "Waiting for LoadBalancer IP..."
    sleep 10
    kubectl -n $NAMESPACE get svc emailops-backend-lb -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
    echo ""
}

# Show status
status() {
    log_info "Cluster status:"
    kubectl -n $NAMESPACE get all
    echo ""
    log_info "Pod logs (last 20 lines):"
    kubectl -n $NAMESPACE logs -l app=emailops --tail=20 || true
}

# Main
case "${1:-all}" in
    build)
        check_prereqs
        setup_registry
        build_and_push
        ;;
    deploy)
        check_prereqs
        connect_cluster
        deploy
        ;;
    status)
        connect_cluster
        status
        ;;
    all)
        check_prereqs
        connect_cluster
        setup_registry
        build_and_push
        deploy
        ;;
    *)
        echo "Usage: $0 [build|deploy|status|all]"
        exit 1
        ;;
esac

log_info "Done!"
