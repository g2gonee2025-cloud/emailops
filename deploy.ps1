# EmailOps Kubernetes Deployment Script for Windows
# Usage: .\deploy.ps1 -Action [build|deploy|status|all]

param(
    [Parameter(Position=0)]
    [ValidateSet("build", "deploy", "status", "all")]
    [string]$Action = "all",
    
    [string]$ImageTag = "latest"
)

$ErrorActionPreference = "Stop"

# Configuration
$REGISTRY = "registry.digitalocean.com/sf-registry"
$IMAGE_NAME = "emailops-backend"
$TAG = $ImageTag
$CLUSTER_NAME = "emailops-k8s"
$NAMESPACE = "emailops"

function Write-Info { param($Message) Write-Host "[INFO] $Message" -ForegroundColor Green }
function Write-Warn { param($Message) Write-Host "[WARN] $Message" -ForegroundColor Yellow }
function Write-Err { param($Message) Write-Host "[ERROR] $Message" -ForegroundColor Red }

# Check prerequisites
function Test-Prerequisites {
    Write-Info "Checking prerequisites..."
    
    if (-not (Get-Command doctl -ErrorAction SilentlyContinue)) {
        Write-Err "doctl is not installed. Install from https://docs.digitalocean.com/reference/doctl/how-to/install/"
        exit 1
    }
    
    if (-not (Get-Command kubectl -ErrorAction SilentlyContinue)) {
        Write-Err "kubectl is not installed."
        exit 1
    }
    
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Err "docker is not installed."
        exit 1
    }
    
    Write-Info "All prerequisites met."
}

# Connect to DOKS cluster
function Connect-Cluster {
    Write-Info "Connecting to DOKS cluster: $CLUSTER_NAME..."
    doctl kubernetes cluster kubeconfig save $CLUSTER_NAME
    kubectl config current-context
}

# Setup container registry
function Setup-Registry {
    Write-Info "Setting up container registry..."
    
    $registryExists = doctl registry get 2>$null
    if (-not $registryExists) {
        Write-Info "Creating container registry..."
        doctl registry create emailops-registry --region sgp1
    }
    
    # Login to registry
    doctl registry login
    
    # Configure Kubernetes to pull from registry
    doctl registry kubernetes-manifest | kubectl apply -f -
}

# Build and push Docker image
function Build-AndPush {
    Write-Info "Building Docker image..."
    
    Push-Location $PSScriptRoot
    try {
        docker build -t "${REGISTRY}/${IMAGE_NAME}:${TAG}" -f backend/Dockerfile .
        
        Write-Info "Pushing to registry..."
        docker push "${REGISTRY}/${IMAGE_NAME}:${TAG}"
        
        Write-Info "Image pushed: ${REGISTRY}/${IMAGE_NAME}:${TAG}"
    }
    finally {
        Pop-Location
    }
}

# Deploy to Kubernetes
function Deploy-ToK8s {
    Write-Info "Deploying to Kubernetes..."
    
    # Apply namespace first
    kubectl apply -f k8s/namespace.yaml
    
    # Apply configmap
    kubectl apply -f k8s/configmap.yaml

    # Apply secrets with env substitution
    Write-Info "Applying secrets from .env..."
    if (Test-Path .env) {
        $envVars = @{}
        Get-Content .env | ForEach-Object {
            if ($_ -match "^([^#=]+)=(.*)$") {
                $name = $matches[1].Trim()
                $val = $matches[2].Trim()
                if ($val -match "^['`"](.*)['`"]$") { $val = $matches[1] }
                $envVars[$name] = $val
            }
        }
        
        $secretsYaml = Get-Content k8s/secrets.yaml -Raw
        foreach ($key in $envVars.Keys) {
            $placeholder = "`$`{$key`}"
            if ($secretsYaml.Contains($placeholder)) {
                $secretsYaml = $secretsYaml.Replace($placeholder, $envVars[$key])
            }
        }
        
        $secretsYaml | kubectl apply -f -
    } else {
        Write-Warn ".env file not found! Applying secrets.yaml directly (placeholders may fail)..."
        kubectl apply -f k8s/secrets.yaml
    }
    
    # Apply deployment
    kubectl apply -f k8s/backend-deployment.yaml
    
    # Apply HPA
    kubectl apply -f k8s/hpa.yaml
    
    # Apply ingress (includes LoadBalancer service)
    kubectl apply -f k8s/ingress.yaml
    
    Write-Info "Deployment complete. Checking status..."
    
    # Wait for rollout
    kubectl -n $NAMESPACE rollout status deployment/emailops-backend --timeout=300s
    
    # Get service info
    Write-Info "Service endpoints:"
    kubectl -n $NAMESPACE get svc
    
    # Get external IP
    Write-Info "Waiting for LoadBalancer IP..."
    Start-Sleep -Seconds 10
    $externalIP = kubectl -n $NAMESPACE get svc emailops-backend-lb -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
    Write-Info "External IP: $externalIP"
}

# Show status
function Get-Status {
    Write-Info "Cluster status:"
    kubectl -n $NAMESPACE get all
    
    Write-Info "`nPod logs (last 20 lines):"
    kubectl -n $NAMESPACE logs -l app=emailops --tail=20 2>$null
}

# Main
switch ($Action) {
    "build" {
        Test-Prerequisites
        Setup-Registry
        Build-AndPush
    }
    "deploy" {
        Test-Prerequisites
        Connect-Cluster
        Deploy-ToK8s
    }
    "status" {
        Connect-Cluster
        Get-Status
    }
    "all" {
        Test-Prerequisites
        Connect-Cluster
        Setup-Registry
        Build-AndPush
        Deploy-ToK8s
    }
}

Write-Info "Done!"
