<#
Purpose: Back up DB to a local dump (optional upload), then destroy DOKS cluster, managed PG, and any listed droplets.
Prereqs: doctl, pg_dump, optional aws/s3cmd for upload; DIGITALOCEAN_ACCESS_TOKEN set.
Safety: Will delete resources by name; confirm via -Force to skip prompts.
#>
param(
    [string]$ClusterName = "do-sgp1-emailops-k8s",
    [string]$DbName = "emailops-db",
    [string]$BackupDir = "./backups",
    [string]$SpacesBucket = "emailops-backups",
    [string]$Region = "sgp1",
    [switch]$UploadToSpaces,
    [switch]$Force,
    [switch]$SkipBackup,
    [string[]]$DropletIdsToDelete = @()
)

$ErrorActionPreference = "Stop"
function Require-Cmd($name) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        throw "Missing required command: $name"
    }
}

Require-Cmd doctl

$pgDumpAvailable = $true
if (-not (Get-Command pg_dump -ErrorAction SilentlyContinue)) {
    $pgDumpAvailable = $false
}

if (-not $env:DIGITALOCEAN_ACCESS_TOKEN) {
    throw "DIGITALOCEAN_ACCESS_TOKEN is not set"
}

# Ensure backup dir
$BackupPath = Resolve-Path (New-Item -ItemType Directory -Force -Path $BackupDir)

Write-Host "Fetching DB connection URL..."
$DB_URL = $null
try {
    $connJson = doctl databases connection $DbName --output json | Out-String
    $connObj = $connJson | ConvertFrom-Json
    if ($connObj -and $connObj[0].uri) {
        $DB_URL = $connObj[0].uri
    }
} catch {
    Write-Warning "Could not fetch DB URL (db may already be deleted): $_"
}

if (-not $DB_URL) {
    Write-Warning "DB URL unavailable; skipping backup"
    $SkipBackup = $true
}

if ($SkipBackup) {
    Write-Warning "SkipBackup set; skipping pg_dump"
} elseif (-not $pgDumpAvailable) {
    Write-Warning "pg_dump not found; skipping backup. Install PostgreSQL client tools to enable backups."
} else {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $dumpFile = Join-Path $BackupPath "${DbName}_${timestamp}.dump"
    Write-Host "Dumping DB to $dumpFile ..."
    pg_dump --format=custom --file="$dumpFile" "$DB_URL"

    if ($UploadToSpaces) {
        if (Get-Command aws -ErrorAction SilentlyContinue) {
            aws s3 cp "$dumpFile" "s3://$SpacesBucket/"
        } elseif (Get-Command s3cmd -ErrorAction SilentlyContinue) {
            s3cmd put "$dumpFile" "s3://$SpacesBucket/"
        } else {
            Write-Warning "Upload requested but aws/s3cmd not found; skipping upload."
        }
    }
}

if (-not $Force) {
    $resp = Read-Host "About to DELETE cluster $ClusterName and DB $DbName. Type 'yes' to continue"
    if ($resp -ne "yes") { throw "Aborted by user" }
}

Write-Host "Deleting k8s cluster $ClusterName ..."
try {
    doctl kubernetes cluster delete $ClusterName --force
} catch {
    Write-Warning "Cluster delete failed or not found: $_"
}

Write-Host "Deleting database $DbName ..."
try {
    doctl databases delete $DbName --force
} catch {
    Write-Warning "Database delete failed or not found: $_"
}

foreach ($id in $DropletIdsToDelete) {
    Write-Host "Deleting droplet $id ..."
    doctl compute droplet delete $id --force
}

Write-Host "Done. Remaining cost should be Spaces + any snapshots." 
