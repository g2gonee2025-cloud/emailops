<#
DigitalOcean deploy helper (PowerShell)
Actions: provision | build | deploy | status | all

provision: ensure DOKS cluster + managed Postgres, apply manifests, optional migrations
build:     build + push backend image to DO registry
deploy:    apply manifests (configmap, secrets with .env templating, backend, embeddings, hpa, ingress)
status:    show cluster status / rollout
all:       provision -> build -> deploy
#>

param(
    [Parameter(Position=0)]
    [ValidateSet("provision", "build", "deploy", "status", "all")]
    [string]$Action = "all",

    # Infra
    [string]$ClusterName = "do-tor1-emailops-k8s",
    [string]$DbName = "emailops-db-tor1",
    [string]$Region = "tor1",
    [string]$NodeSize = "s-1vcpu-2gb",
    [int]$NodeCount = 1,
    [string]$DbSize = "db-s-1vcpu-1gb",
    [switch]$KubeSetCurrent,
    [switch]$SkipMigrations,
    [string]$Namespace = "emailops",

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
Test-CommandRequired jq
Test-CommandRequired alembic
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

function Update-ClusterAndDb {
    Write-Host "Ensuring cluster $ClusterName exists..."
    $clusterId = $(doctl kubernetes cluster list --output json | jq -r --arg name "$ClusterName" '.[] | select(.name == $name) | .id')
    if (-not $clusterId) {
        Write-Host "Creating cluster $ClusterName ..."
        doctl kubernetes cluster create $ClusterName --region $Region --node-pool "name=pool1;size=$NodeSize;count=$NodeCount"
    } else {
        Write-Host "Cluster found with ID: $clusterId"
    }

    Write-Host "Ensuring database $DbName exists..."
    $dbId = $(doctl databases list --output json | jq -r --arg name "$DbName" '.[] | select(.name == $name) | .id')
    if (-not $dbId) {
        Write-Host "Creating database..."
        $createJson = doctl databases create $DbName --engine pg --size $DbSize --region $Region --num-nodes 1 --output json | Out-String
        $dbId = ($createJson | jq -r '.[0].id')
    } else {
        Write-Host "Database found with ID: $dbId"
    }

    Write-Host "Fetching DB connection URL..."
    $connJson = doctl databases connection $dbId --output json | Out-String
    $script:OUTLOOKCORTEX_DB_URL = ($connJson | jq -r '.uri')
    if (-not $script:OUTLOOKCORTEX_DB_URL) { throw "Failed to get DB URL" }

    Write-Host "Saving kubeconfig..."
    $kubeArgs = @("kubernetes", "cluster", "kubeconfig", "save", $ClusterName)
    if ($KubeSetCurrent) { $kubeArgs += "--set-current" }
    doctl @kubeArgs | Out-Null
}

function Set-Secrets {
    $dotenv = Get-DotenvVars
    $secretsPath = Join-Path $K8sDir "secrets.yaml"
    if (-not (Test-Path $secretsPath)) { throw "secrets.yaml not found" }
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
    kubectl apply -f (Join-Path $K8sDir "backend-deployment.yaml")
    kubectl apply -f (Join-Path $K8sDir "embeddings.yaml")
    kubectl apply -f (Join-Path $K8sDir "ingress.yaml")
    kubectl apply -f (Join-Path $K8sDir "hpa.yaml")
}

function Invoke-Migrations {
    if ($SkipMigrations) { return }
    Write-Host "Running migrations..."
    Push-Location $MigrationsDir
    try {
        $env:OUTLOOKCORTEX_DB_URL = $script:OUTLOOKCORTEX_DB_URL
        alembic -c alembic.ini upgrade head
    }
    finally {
        Pop-Location
    }
}

function Show-Status {
    kubectl -n $Namespace get all
    Write-Host "\nRollout status:"; kubectl -n $Namespace rollout status deployment/emailops-backend --timeout=120s
}

switch ($Action) {
    "provision" {
        Update-ClusterAndDb
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
    "all" {
        Update-ClusterAndDb
        Update-Registry
        Build-And-Push
        Install-Manifests
        Invoke-Migrations
        Show-Status
    }
}

Write-Host "Done."