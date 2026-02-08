# WatchPull Daemon - Standalone background process
param(
    [string]$LocalPath,
    [string]$RemotePath,
    [string]$ServerName,
    [string]$ServerHost,
    [string]$ServerUser,
    [string]$ServerPass,
    [string]$PlinkPath,
    [string]$PscpPath,
    [string]$LogPath,
    [int]$Interval = 30
)

# Helper function to append log with proper encoding (UTF-8 without BOM)
function Write-Log {
    param([string]$Message)
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::AppendAllText($LogPath, "$Message`r`n", $utf8NoBom)
}

# Write startup message
Write-Log "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $ServerName DAEMON STARTED (Interval: ${Interval}s)"

while ($true) {
    try {
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        
        # Find all .dat, log*, .plt files in main directory (no marker logic - pure hash comparison)
        $cmd = "find $RemotePath -maxdepth 1 -type f \( -name '*.dat' -o -name 'log*' -o -name '*.plt' \) 2>/dev/null"
        $result = & $PlinkPath -ssh -pw $ServerPass -batch "$ServerUser@$ServerHost" $cmd 2>$null
        
        # Also get files in result/statistics directories (include .bin and .vtk)
        $resultCmd = "find $RemotePath/result $RemotePath/statistics -type f \( -name '*.dat' -o -name '*.plt' -o -name '*.bin' -o -name '*.vtk' -o -name 'log*' \) 2>/dev/null"
        $resultFiles = & $PlinkPath -ssh -pw $ServerPass -batch "$ServerUser@$ServerHost" $resultCmd 2>$null
        
        $allFiles = @()
        if ($result) { $allFiles += $result }
        if ($resultFiles) { $allFiles += $resultFiles }
        $allFiles = $allFiles | Sort-Object -Unique
        
        foreach ($remotefile in $allFiles) {
            if (-not $remotefile -or $remotefile -match '\.git/' -or $remotefile -match '\.vscode/') { continue }
            
            $relativePath = $remotefile.Replace("$RemotePath/", "")
            $localFile = Join-Path $LocalPath $relativePath.Replace("/", "\")
            
            # Get remote file hash
            $hashCmd = "md5sum '$remotefile' 2>/dev/null | cut -d' ' -f1"
            $remoteHash = & $PlinkPath -ssh -pw $ServerPass -batch "$ServerUser@$ServerHost" $hashCmd 2>$null
            
            if (-not $remoteHash) { continue }
            
            # Compare with local
            $needDownload = $false
            if (-not (Test-Path $localFile)) {
                $needDownload = $true
            }
            else {
                try {
                    $localHash = (Get-FileHash $localFile -Algorithm MD5).Hash.ToLower()
                    if ($localHash -ne $remoteHash.Trim().ToLower()) {
                        $needDownload = $true
                    }
                }
                catch { $needDownload = $true }
            }
            
            if ($needDownload) {
                # Create directory if needed
                $localDir = Split-Path $localFile -Parent
                if (-not (Test-Path $localDir)) {
                    New-Item -ItemType Directory -Path $localDir -Force | Out-Null
                }
                
                # Download file (suppress all output)
                $remoteUri = "$ServerUser@${ServerHost}:$remotefile"
                $null = & $PscpPath -pw $ServerPass -batch $remoteUri $localFile 2>&1
                
                Write-Log "[$timestamp] $ServerName DOWNLOADED: $relativePath"
            }
        }
    }
    catch {
        Write-Log "[$timestamp] $ServerName ERROR: $_"
    }
    
    Start-Sleep -Seconds $Interval
}
