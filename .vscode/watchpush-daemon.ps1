# WatchPush Daemon - Standalone background process
param(
    [string]$LocalPath,
    [string]$RemotePath,
    [string]$ServersJson,
    [string]$PlinkPath,
    [string]$PscpPath,
    [string]$LogPath,
    [int]$Interval = 10
)

$servers = $ServersJson | ConvertFrom-Json
# 排除規則：程式碼輸出檔案不上傳（避免覆蓋遠端正在寫入的檔案）
$excludePatterns = @(
    ".git/*", ".vscode/*",           # 系統資料夾
    "a.out", "*.o", "*.exe",         # 編譯產物
    "*.dat", "*.DAT",                # 輸出資料檔
    "log*",                          # log 檔案
    "*.plt",                         # 繪圖檔案
    "result/*", "backup/*", "statistics/*"  # 輸出資料夾
)
$lastHashes = @{}

# Helper function to append log with proper encoding (UTF-8 without BOM)
function Write-Log {
    param([string]$Message)
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::AppendAllText($LogPath, "$Message`r`n", $utf8NoBom)
}

# Write startup message
Write-Log "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] DAEMON STARTED (Interval: ${Interval}s)"

while ($true) {
    try {
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        
        # Get all local files
        $changedFiles = @()
        Get-ChildItem -Path $LocalPath -Recurse -File -ErrorAction SilentlyContinue | ForEach-Object {
            $relativePath = $_.FullName.Substring($LocalPath.Length + 1).Replace("\", "/")
            
            # Check exclusions
            $exclude = $false
            foreach ($pattern in $excludePatterns) {
                if ($relativePath -like $pattern) { $exclude = $true; break }
            }
            
            if (-not $exclude) {
                try {
                    $currentHash = (Get-FileHash $_.FullName -Algorithm MD5 -ErrorAction Stop).Hash
                    if (-not $lastHashes.ContainsKey($relativePath) -or $lastHashes[$relativePath] -ne $currentHash) {
                        $changedFiles += @{ Path = $relativePath; FullPath = $_.FullName }
                        $lastHashes[$relativePath] = $currentHash
                    }
                }
                catch { }
            }
        }
        
        # Upload changed files
        if ($changedFiles.Count -gt 0) {
            foreach ($file in $changedFiles) {
                foreach ($server in $servers) {
                    $remoteFile = "$RemotePath/$($file.Path)"
                    $remoteUri = "$($server.User)@$($server.Host):$remoteFile"
                    
                    # Create remote directory if needed
                    $remoteDir = [System.IO.Path]::GetDirectoryName($remoteFile).Replace("\", "/")
                    $mkdirCmd = "mkdir -p '$remoteDir' 2>/dev/null"
                    $null = & $PlinkPath -ssh -pw $server.Password -batch "$($server.User)@$($server.Host)" $mkdirCmd 2>&1
                    
                    # Upload file
                    $null = & $PscpPath -pw $server.Password -batch $file.FullPath $remoteUri 2>&1
                }
                
                Write-Log "[$timestamp] UPLOADED: $($file.Path)"
            }
        }
    }
    catch {
        Write-Log "[$timestamp] ERROR: $_"
    }
    
    Start-Sleep -Seconds $Interval
}
