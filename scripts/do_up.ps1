<#
Purpose: Bring up DigitalOcean infra for emailops on-demand (cluster + managed PG), run migrations, and apply k8s manifests.
Prereqs: doctl, kubectl, alembic, jq (for JSON parsing), psql/pg_dump in PATH; DIGITALOCEAN_ACCESS_TOKEN set.
Safety: Creates resources if missing; does not delete anything. Idempotent-ish (skips existing cluster/db by name).
#>
param(
    [string]$ClusterName = "do-sgp1-emailops-k8s",
    [string]$DbName = "emailops-db",
    [string]$Region = "sgp1",
    [string]$NodeSize = "s-1vcpu-2gb",
    [int]$NodeCount = 1,
    [string]$DbSize = "db-s-1vcpu-1gb",
    [switch]$KubeSetCurrent,
    [switch]$SkipMigrations,
    [string]$Namespace = "emailops"
)

$ErrorActionPreference = "Stop"
function Require-Cmd($name) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        throw "Missing required command: $name"
    }
}

Require-Cmd doctl
Require-Cmd kubectl
Require-Cmd jq
Require-Cmd alembic

if (-not $env:DIGITALOCEAN_ACCESS_TOKEN) {
    throw "DIGITALOCEAN_ACCESS_TOKEN is not set"
}

# Paths
$RepoRoot = (Resolve-Path "$PSScriptRoot/..")
$K8sDir = Join-Path $RepoRoot "k8s"
$MigrationsDir = Join-Path $RepoRoot "backend/migrations"

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
$DB_URL = ($connJson | jq -r '.uri')
if (-not $DB_URL) { throw "Failed to get DB URL" }

Write-Host "Saving kubeconfig..."
$kubeArgs = @("kubernetes","cluster","kubeconfig","save",$ClusterName)
if ($KubeSetCurrent) { $kubeArgs += "--set-current" }
doctl @kubeArgs | Out-Null

Write-Host "Applying namespace and config..."
kubectl apply -f (Join-Path $K8sDir "namespace.yaml")
kubectl apply -f (Join-Path $K8sDir "configmap.yaml")
kubectl apply -f (Join-Path $K8sDir "secrets.yaml")
kubectl apply -f (Join-Path $K8sDir "backend-deployment.yaml")
kubectl apply -f (Join-Path $K8sDir "embeddings.yaml")
kubectl apply -f (Join-Path $K8sDir "ingress.yaml")
kubectl apply -f (Join-Path $K8sDir "hpa.yaml")

if (-not $SkipMigrations) {
    Write-Host "Running migrations..."
    pushd $MigrationsDir
    $env:DB_URL = $DB_URL
    alembic -c alembic.ini upgrade head
    popd
}

Write-Host "Done. Cluster and DB are up."
Write-Host "DB_URL (export for app/workers): $DB_URL"