<#
DigitalOcean deploy helper (PowerShell)
Actions: provision | build | deploy | status | gpu-up | gpu-down | all

provision: ensure DOKS cluster + managed Postgres, apply manifests, optional migrations
build:     build + push backend image to DO registry
deploy:    apply manifests (configmap, secrets with .env templating, backend, embeddings, hpa, ingress)
status:    show cluster status / rollout
gpu-up:    scale GPU node pool to 1 (for embeddings/LLM)
gpu-down:  scale GPU node pool to 0 (save costs)
all:       provision -> build -> deploy
#>

param(
    [Parameter(Position=0)]
    [ValidateSet("provision", "build", "deploy", "status", "gpu-up", "gpu-down", "all")]
    [string]$Action = "all",

    # Infra - NYC2 cluster (existing)
    [string]$ClusterId = "23c013d9-4d8d-4d3d-a813-7e5cbc3d0af1",
    [string]$ClusterName = "k8s-1-34-1-do-1-nyc2-1765360390845",
    [string]$GpuPoolId = "e359afb3-1891-4bba-94b4-c7ab1d2e1736",
    [string]$CpuPoolId = "c949fea3-5d57-4c30-9fd4-35acdf6973af",
    [string]$Region = "nyc2",
    [string]$NodeSize = "s-2vcpu-8gb-amd",
    [int]$NodeCount = 1,
    [switch]$KubeSetCurrent,
    [switch]$SkipMigrations,
    [string]$Namespace = "emailops",

    # Database (in-cluster PostgreSQL, not managed DB)
    [switch]$UseInClusterDb,

    # Image / registry
    [string]$Registry = "registry.digitalocean.com/sf-registry",
    [string]$ImageName = "emailops-backend",
    [string]$ImageTag = "latest"
)

$ErrorActionPreference = "Stop"

function Test-CommandRequired($name) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        throw "Missing required command: $name"
    }
}

# Basic prereqs
Test-CommandRequired doctl
Test-CommandRequired kubectl
if ($Action -in @("build", "deploy", "all")) {
    Test-CommandRequired docker
}

if (-not $env:DIGITALOCEAN_ACCESS_TOKEN) {
    throw "DIGITALOCEAN_ACCESS_TOKEN is not set"
}

# Paths
$RepoRoot = (Resolve-Path "$PSScriptRoot/..")
$K8sDir = Join-Path $RepoRoot "k8s"
$MigrationsDir = Join-Path $RepoRoot "backend/migrations"

function Get-DotenvVars {
    param([string]$Path = "$RepoRoot/.env")
    $vars = @{}
    if (-not (Test-Path $Path)) { return $vars }
    Get-Content $Path | ForEach-Object {
        if ($_ -match "^\s*#" -or $_ -notmatch "=") { continue }
        $parts = $_.Split("=", 2)
        $name = $parts[0].Trim()
        $val = $parts[1].Trim()
        if ($val -match '^[''"](.*)[''"]$') { $val = $matches[1] }
        if ($name) { $vars[$name] = $val }
    }
    return $vars
}

function Update-Registry {
    Write-Host "Ensuring registry exists..."
    $regExists = doctl registry get 2>$null
    if (-not $regExists) {
        Write-Host "Creating registry..."
        doctl registry create (Split-Path $Registry -Leaf) --region $Region
    }
    doctl registry login
    doctl registry kubernetes-manifest | kubectl apply -f -
}

function Build-And-Push {
    Write-Host "Building Docker image..."
    Push-Location $RepoRoot
    try {
        docker build -t "$($Registry)/$($ImageName):$($ImageTag)" -f "backend/Dockerfile" .
        Write-Host "Pushing image..."
        docker push "$($Registry)/$($ImageName):$($ImageTag)"
    }
    finally {
        Pop-Location
    }
}

function Update-Cluster {
    Write-Host "Checking cluster $ClusterName (NYC2)..."
    
    # Check if cluster exists
    $clusterInfo = doctl kubernetes cluster get $ClusterId --output json 2>$null | ConvertFrom-Json
    if (-not $clusterInfo) {
        throw "Cluster $ClusterId not found. Please create it first or update ClusterId."
    }
    
    Write-Host "Cluster found: $($clusterInfo.name) in $($clusterInfo.region)"
    Write-Host "  Status: $($clusterInfo.status.state)"
    Write-Host "  Node Pools: $($clusterInfo.node_pools.Count)"
    
    Write-Host "Saving kubeconfig..."
    $kubeArgs = @("kubernetes", "cluster", "kubeconfig", "save", $ClusterId)
    if ($KubeSetCurrent) { $kubeArgs += "--set-current-context" }
    doctl @kubeArgs | Out-Null
}

function Scale-GpuPool {
    param([int]$Count)
    Write-Host "Scaling GPU pool to $Count nodes..."
    doctl kubernetes cluster node-pool update $ClusterId $GpuPoolId --count $Count
    
    if ($Count -gt 0) {
        Write-Host "GPU pool scaling up. This may take 2-5 minutes."
        Write-Host "Cost: ~`$3.81/hour for H200 GPU"
    } else {
        Write-Host "GPU pool scaled to 0. No GPU costs will be incurred."
    }
}

function Set-Secrets {
    $dotenv = Get-DotenvVars
    $secretsPath = Join-Path $K8sDir "secrets.yaml"
    if (-not (Test-Path $secretsPath)) { 
        Write-Warning "secrets.yaml not found. Copy secrets-template.yaml and fill in values."
        throw "secrets.yaml not found" 
    }
    $content = Get-Content $secretsPath -Raw
    foreach ($k in $dotenv.Keys) {
        $placeholder = "$" + "{" + $k + "}"
        $content = $content.Replace($placeholder, $dotenv[$k])
    }
    $content | kubectl apply -f -
}

function Install-Manifests {
    Write-Host "Applying manifests..."
    kubectl apply -f (Join-Path $K8sDir "namespace.yaml")
    kubectl apply -f (Join-Path $K8sDir "configmap.yaml")
    Set-Secrets
    
    # Deploy in-cluster PostgreSQL and Redis
    if ($UseInClusterDb -or (Test-Path (Join-Path $K8sDir "postgres.yaml"))) {
        Write-Host "Deploying in-cluster PostgreSQL..."
        kubectl apply -f (Join-Path $K8sDir "postgres.yaml")
    }
    kubectl apply -f (Join-Path $K8sDir "redis.yaml")
    
    # Deploy backend
    kubectl apply -f (Join-Path $K8sDir "backend-deployment.yaml")
    
    # Deploy embeddings (requires GPU)
    kubectl apply -f (Join-Path $K8sDir "embeddings-pvc.yaml") 2>$null
    kubectl apply -f (Join-Path $K8sDir "embeddings-vllm.yaml")
    
    # Optional resources
    kubectl apply -f (Join-Path $K8sDir "ingress.yaml") 2>$null
    kubectl apply -f (Join-Path $K8sDir "hpa.yaml") 2>$null
}

function Invoke-Migrations {
    if ($SkipMigrations) { 
        Write-Host "Skipping migrations (use -SkipMigrations:$false to run)"
        return 
    }
    if (-not (Get-Command alembic -ErrorAction SilentlyContinue)) {
        Write-Warning "alembic not found; skipping migrations"
        return
    }
    Write-Host "Running migrations..."
    Push-Location $MigrationsDir
    try {
        alembic -c alembic.ini upgrade head
    }
    finally {
        Pop-Location
    }
}

function Show-Status {
    Write-Host "`n=== Cluster Info ===" -ForegroundColor Cyan
    kubectl cluster-info
    
    Write-Host "`n=== Node Pools ===" -ForegroundColor Cyan
    doctl kubernetes cluster node-pool list $ClusterId
    
    Write-Host "`n=== Pods ===" -ForegroundColor Cyan
    kubectl -n $Namespace get pods -o wide
    
    Write-Host "`n=== Services ===" -ForegroundColor Cyan
    kubectl -n $Namespace get svc
    
    Write-Host "`n=== Rollout Status ===" -ForegroundColor Cyan
    kubectl -n $Namespace rollout status deployment/emailops-backend --timeout=60s 2>$null
}

switch ($Action) {
    "provision" {
        Update-Cluster
        Install-Manifests
        Invoke-Migrations
    }
    "build" {
        Update-Registry
        Build-And-Push
    }
    "deploy" {
        Install-Manifests
    }
    "status" {
        Show-Status
    }
    "gpu-up" {
        Scale-GpuPool -Count 1
    }
    "gpu-down" {
        Scale-GpuPool -Count 0
    }
    "all" {
        Update-Cluster
        Update-Registry
        Build-And-Push
        Install-Manifests
        Invoke-Migrations
        Show-Status
    }
}

Write-Host "`nDone." -ForegroundColor Green