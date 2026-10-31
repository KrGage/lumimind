param([string]$ProjectDirectory)
$ErrorActionPreference = 'Stop'

function Get-Sha256([string]$FilePath) {
    $algorithm = [Security.Cryptography.SHA256]::Create()
    $stream = [IO.File]::OpenRead($FilePath)
    try { return [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() }
    finally { $stream.Dispose(); $algorithm.Dispose() }
}

function Resolve-ChildPath([string]$BaseDirectory, [string]$RelativePath) {
    $basePath = [IO.Path]::GetFullPath($BaseDirectory).TrimEnd('\', '/')
    $childPath = [IO.Path]::GetFullPath((Join-Path $basePath $RelativePath))
    if (!$childPath.StartsWith($basePath + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Invalid package path: $RelativePath"
    }
    return $childPath
}

if (!$ProjectDirectory) {
    Write-Host 'Close the agent backend before installing. Enter the project root containing backend and fontend.'
    $ProjectDirectory = Read-Host 'Project folder'
}
$project = (Resolve-Path -LiteralPath $ProjectDirectory.Trim('"')).Path
foreach ($relative in @('backend\app\main.py', 'fontend\index-V2.html')) {
    if (!(Test-Path -LiteralPath (Resolve-ChildPath $project $relative) -PathType Leaf)) {
        throw "Not a supported LumiMind project folder: missing $relative"
    }
}
$payload = Join-Path $PSScriptRoot 'payload'
$manifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$conflicts = @()
foreach ($file in $manifest.files) {
    $source = Resolve-ChildPath $payload $file.path
    $destination = Resolve-ChildPath $project $file.path
    if (!(Test-Path -LiteralPath $source -PathType Leaf)) { throw "Package file missing: $($file.path)" }
    $sourceHash = Get-Sha256 $source
    if ($sourceHash -ne $file.sha256) { throw "Package integrity check failed: $($file.path)" }
    if ($file.mode -eq 'merge_config') {
        if (Test-Path -LiteralPath $destination -PathType Leaf) {
            $existing = Get-Content -LiteralPath $destination -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($existing -isnot [PSCustomObject]) { throw 'Existing light_config.json must be an object.' }
        }
    } elseif ($file.mode -ne 'preserve' -and (Test-Path -LiteralPath $destination -PathType Leaf)) {
        $targetHash = Get-Sha256 $destination
        if ($targetHash -ne $file.sha256 -and $targetHash -ne $file.base_sha256) { $conflicts += $file.path }
    } elseif ($file.required_existing -and !(Test-Path -LiteralPath $destination -PathType Leaf)) {
        $conflicts += $file.path
    }
}
if ($conflicts.Count -gt 0) {
    throw ("Different agent code version detected. No project files were changed. Send these files for adaptation; do not force overwrite:`n" + ($conflicts -join "`n"))
}

$backup = Join-Path $project ('.lumimind-backups\emotion-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N').Substring(0,8))
New-Item -ItemType Directory -Path $backup -Force | Out-Null
$changed = @()
foreach ($file in $manifest.files) {
    $source = Resolve-ChildPath $payload $file.path
    $destination = Resolve-ChildPath $project $file.path
    $exists = Test-Path -LiteralPath $destination -PathType Leaf
    if ($exists -and $file.mode -eq 'preserve') { continue }
    if ($exists) {
        $backupFile = Resolve-ChildPath $backup $file.path
        New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($backupFile)) -Force | Out-Null
        Copy-Item -LiteralPath $destination -Destination $backupFile
    }
    New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($destination)) -Force | Out-Null
    if ($file.mode -eq 'merge_config' -and $exists) {
        $merged = Get-Content -LiteralPath $source -Raw -Encoding UTF8 | ConvertFrom-Json
        $existing = Get-Content -LiteralPath $destination -Raw -Encoding UTF8 | ConvertFrom-Json
        foreach ($property in $existing.PSObject.Properties) {
            $merged | Add-Member -MemberType NoteProperty -Name $property.Name -Value $property.Value -Force
        }
        $merged | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $destination -Encoding UTF8
    } else {
        Copy-Item -LiteralPath $source -Destination $destination -Force
    }
    $changed += $file.path
}
@{ changed = $changed; installed_at = (Get-Date -Format o) } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $backup 'installation.json') -Encoding UTF8
Write-Host "Installed. Backup: $backup"
Write-Host "Set the single lamp IP in: $(Join-Path $project 'light_control\light_config.json')"
Write-Host 'Restart the original backend with ONE worker, then open /page/monitor and hard refresh (Ctrl+F5).'
Write-Host 'Accept recommendation now calculates and sends automatically. No manual Python call or compilation is needed.'
