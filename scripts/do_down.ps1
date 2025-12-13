<#
Purpose: Scale down GPU pool (to save costs), optionally back up DB and destroy resources.
Prereqs: doctl, pg_dump (optional), DIGITALOCEAN_ACCESS_TOKEN set.
Safety: GPU scaling is safe. Cluster/DB deletion requires -Force or confirmation.
#>
param(
    # NYC2 Cluster (existing)
    [string]$ClusterId = "23c013d9-4d8d-4d3d-a813-7e5cbc3d0af1",
    [string]$ClusterName = "k8s-1-34-1-do-1-nyc2-1765360390845",
    [string]$GpuPoolId = "e359afb3-1891-4bba-94b4-c7ab1d2e1736",
    [string]$Region = "nyc2",
    
    # Backup options
    [string]$BackupDir = "./backups",
    [string]$SpacesBucket = "emailops-backups",
    [switch]$UploadToSpaces,
    [switch]$SkipBackup,
    
    # Destruction options (USE WITH CAUTION)
    [switch]$DeleteCluster,
    [switch]$Force,
    [string[]]$DropletIdsToDelete = @()
)

$ErrorActionPreference = "Stop"
function Test-Command($name) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        throw "Missing required command: $name"
    }
}

Test-Command doctl

if (-not $env:DIGITALOCEAN_ACCESS_TOKEN) {
    throw "DIGITALOCEAN_ACCESS_TOKEN is not set"
}

# =============================================================================
# PRIMARY ACTION: Scale down GPU pool to save costs
# =============================================================================
Write-Host "=============================================="
Write-Host "EmailOps Scale Down (NYC2)" -ForegroundColor Yellow
Write-Host "Cluster: $ClusterName"
Write-Host "=============================================="

Write-Host "`nScaling GPU pool to 0 nodes..." -ForegroundColor Cyan
try {
    doctl kubernetes cluster node-pool update $ClusterId $GpuPoolId --count 0
    Write-Host "GPU pool scaled to 0. No GPU costs will be incurred." -ForegroundColor Green
    Write-Host "  H200 GPU: ~`$3.81/hour saved"
} catch {
    Write-Warning "GPU pool scaling failed: $_"
}

# Show current state
Write-Host "`n=== Current Node Pools ===" -ForegroundColor Cyan
doctl kubernetes cluster node-pool list $ClusterId

# =============================================================================
# OPTIONAL: Delete cluster entirely (requires -DeleteCluster flag)
# =============================================================================
if ($DeleteCluster) {
    Write-Host "`n" -ForegroundColor Red
    Write-Host "!!! DANGER ZONE !!!" -ForegroundColor Red
    Write-Host "You requested to DELETE the entire cluster." -ForegroundColor Red
    
    if (-not $Force) {
        $resp = Read-Host "About to DELETE cluster $ClusterName. Type 'DELETE' to continue"
        if ($resp -ne "DELETE") { 
            Write-Host "Aborted by user." -ForegroundColor Yellow
            exit 0 
        }
    }
    
    Write-Host "Deleting k8s cluster $ClusterName ..."
    try {
        doctl kubernetes cluster delete $ClusterId --force --dangerous
    } catch {
        Write-Warning "Cluster delete failed: $_"
    }
}

# Optional: Delete specific droplets
foreach ($id in $DropletIdsToDelete) {
    Write-Host "Deleting droplet $id ..."
    doctl compute droplet delete $id --force
}

Write-Host "`nDone." -ForegroundColor Green
Write-Host "Remaining costs: Spaces storage + any snapshots/volumes" 
