<# 
.SYNOPSIS
    MobaXterm-style Sync Commands for D3Q27_PeriodicHill
.DESCRIPTION
    Git-like sync commands: check, add, push, pull, status
.EXAMPLE
    mobaxterm check   - Compare local vs remote
    mobaxterm add .   - Show changed files
    mobaxterm push    - Push to remote servers
#>

param(
    [Parameter(Position=0)]
    [string]$Command,
    [Parameter(Position=1, ValueFromRemainingArguments=$true)]
    [string[]]$Arguments
)

# Configuration
$script:Config = @{
    LocalPath = "c:\Users\88697.CHENPENGCHUNG12\Desktop\GitHub-PeriodicHill\D3Q27_PeriodicHill"
    RemotePath = "/home/chenpengchung/D3Q27_PeriodicHill"
    Servers = @(
        @{ Name = ".87"; Host = "140.114.58.87"; User = "chenpengchung"; Password = "1256" },
        @{ Name = ".154"; Host = "140.114.58.154"; User = "chenpengchung"; Password = "1256" }
    )
    PscpPath = "C:\Program Files\PuTTY\pscp.exe"
    PlinkPath = "C:\Program Files\PuTTY\plink.exe"
    # 遠端同步：只排除 .git 和編譯產物，其他全部同步
    ExcludePatterns = @(".git/*", "a.out", "*.o", "*.exe")
    # 同步所有檔案類型
    SyncExtensions = @("*")
    SyncAll = $true  # 同步所有檔案到遠端工作區
}

function Write-Color {
    param([string]$Text, [string]$Color = "White")
    Write-Host $Text -ForegroundColor $Color
}

function Get-LocalFiles {
    $files = @()
    Get-ChildItem -Path $Config.LocalPath -Recurse -File -ErrorAction SilentlyContinue | ForEach-Object {
        $relativePath = $_.FullName.Substring($Config.LocalPath.Length + 1).Replace("\", "/")
        $exclude = $false
        foreach ($pattern in $Config.ExcludePatterns) {
            if ($relativePath -like $pattern) { $exclude = $true; break }
        }
        if (-not $exclude) {
            $hash = ""
            try { $hash = (Get-FileHash $_.FullName -Algorithm MD5 -ErrorAction Stop).Hash }
            catch { $hash = "ERROR" }
            $files += @{
                Path = $relativePath
                FullPath = $_.FullName
                Size = $_.Length
                Modified = $_.LastWriteTime
                Hash = $hash
            }
        }
    }
    return $files
}

function Get-RemoteFiles {
    param([hashtable]$Server)
    
    # 只排除 .git 目錄，其他全部同步
    $excludeGrep = "grep -v '/.git/'"
    $cmd = "find $($Config.RemotePath) -type f -exec md5sum {} \; 2>/dev/null | $excludeGrep"
    $result = & $Config.PlinkPath -ssh -pw $Server.Password -batch "$($Server.User)@$($Server.Host)" $cmd 2>$null
    
    $files = @()
    if ($result) {
        foreach ($line in $result) {
            if ($line -match "^([a-f0-9]+)\s+(.+)$") {
                $hash = $Matches[1].ToUpper()
                $path = $Matches[2].Replace($Config.RemotePath + "/", "")
                $files += @{ Path = $path; Hash = $hash }
            }
        }
    }
    return $files
}

function Compare-Files {
    param(
        [hashtable]$Server,
        [switch]$Silent
    )
    
    if (-not $Silent) {
        Write-Color "`nConnecting to $($Server.Name) ($($Server.Host))..." "Cyan"
    }
    
    $localFiles = Get-LocalFiles
    $remoteFiles = Get-RemoteFiles -Server $Server
    
    $remoteHash = @{}
    foreach ($f in $remoteFiles) { $remoteHash[$f.Path] = $f.Hash }
    
    $localHash = @{}
    foreach ($f in $localFiles) { $localHash[$f.Path] = $f.Hash }
    
    $results = @{
        New = @()
        Modified = @()
        Deleted = @()
        Same = @()
    }
    
    foreach ($f in $localFiles) {
        if (-not $remoteHash.ContainsKey($f.Path)) {
            $results.New += $f.Path
        }
        elseif ($remoteHash[$f.Path] -ne $f.Hash) {
            $results.Modified += $f.Path
        }
        else {
            $results.Same += $f.Path
        }
    }
    
    foreach ($f in $remoteFiles) {
        if (-not $localHash.ContainsKey($f.Path)) {
            $results.Deleted += $f.Path
        }
    }
    
    return $results
}

function Show-CompareResults {
    param([hashtable]$Results, [string]$ServerName)
    
    Write-Color "`n=== $ServerName Sync Status ===" "Yellow"
    
    if ($Results.New.Count -gt 0) {
        Write-Color "`n[NEW] Local only (not on remote):" "Green"
        foreach ($f in $Results.New) { Write-Color "  + $f" "Green" }
    }
    
    if ($Results.Modified.Count -gt 0) {
        Write-Color "`n[MODIFIED] Content differs:" "Yellow"
        foreach ($f in $Results.Modified) { Write-Color "  ~ $f" "Yellow" }
    }
    
    if ($Results.Deleted.Count -gt 0) {
        Write-Color "`n[REMOTE ONLY] Not in local:" "Red"
        foreach ($f in $Results.Deleted) { Write-Color "  - $f" "Red" }
    }
    
    if ($Results.New.Count -eq 0 -and $Results.Modified.Count -eq 0 -and $Results.Deleted.Count -eq 0) {
        Write-Color "`n[OK] Fully synchronized!" "Green"
    }
    
    Write-Color "`nStats: Same=$($Results.Same.Count) | New=$($Results.New.Count) | Modified=$($Results.Modified.Count) | RemoteOnly=$($Results.Deleted.Count)" "Cyan"
}

# Command handlers
switch ($Command) {
    # ===== Git-like Commands =====
    
    # git diff - 比較本地與遠端差異
    { $_ -in "diff", "check" } {
        Write-Color "[DIFF] Comparing local vs remote..." "Magenta"
        foreach ($server in $Config.Servers) {
            $results = Compare-Files -Server $server
            Show-CompareResults -Results $results -ServerName $server.Name
        }
    }
    
    # git status - 顯示同步狀態
    "status" {
        Write-Color "[STATUS] Sync overview" "Magenta"
        
        $localFiles = Get-LocalFiles
        Write-Color "`nLocal files: $($localFiles.Count)" "White"
        
        foreach ($server in $Config.Servers) {
            $results = Compare-Files -Server $server
            $needsPush = $results.New.Count + $results.Modified.Count
            $needsDelete = $results.Deleted.Count
            if ($needsPush -eq 0 -and $needsDelete -eq 0) {
                $status = "[OK] Synced"
            }
            elseif ($needsPush -gt 0) {
                $status = "[!] Needs push"
            }
            else {
                $status = "[!] Remote has extra files"
            }
            Write-Color "$($server.Name): $status (push: $needsPush, remote-only: $needsDelete)" "White"
        }
    }
    
    # git add - 顯示待推送的變更
    "add" {
        Write-Color "[ADD] Showing pending changes..." "Magenta"
        $allChanges = @()
        foreach ($server in $Config.Servers) {
            $results = Compare-Files -Server $server
            if ($results.New.Count -gt 0 -or $results.Modified.Count -gt 0) {
                $allChanges += $results.New
                $allChanges += $results.Modified
            }
        }
        $allChanges = $allChanges | Select-Object -Unique
        
        if ($allChanges.Count -gt 0) {
            Write-Color "`nFiles to be pushed:" "Green"
            foreach ($f in $allChanges) {
                Write-Color "  -> $f" "White"
            }
            Write-Color "`nUse 'mobaxterm push' to sync these files" "Cyan"
        }
        else {
            Write-Color "`nNo pending changes" "Green"
        }
    }
    
    "push" {
        Write-Color "[PUSH] Uploading ALL files to remote servers..." "Magenta"
        
        $localFiles = Get-LocalFiles
        # 同步所有檔案（已經過 Get-LocalFiles 過濾排除項目）
        $syncFiles = $localFiles
        
        foreach ($server in $Config.Servers) {
            Write-Color "`nPushing to $($server.Name) ($($server.Host))..." "Cyan"
            
            $successCount = 0
            $failCount = 0
            
            foreach ($file in $syncFiles) {
                $localPath = $file.FullPath
                $remoteDest = "$($server.User)@$($server.Host):$($Config.RemotePath)/$($file.Path)"
                
                $remoteDir = [System.IO.Path]::GetDirectoryName("$($Config.RemotePath)/$($file.Path)").Replace("\", "/")
                & $Config.PlinkPath -ssh -pw $server.Password -batch "$($server.User)@$($server.Host)" "mkdir -p '$remoteDir'" 2>$null
                
                $null = & $Config.PscpPath -pw $server.Password -q $localPath $remoteDest 2>&1
                if ($LASTEXITCODE -eq 0) {
                    Write-Color "  [OK] $($file.Path)" "Green"
                    $successCount++
                }
                else {
                    Write-Color "  [FAIL] $($file.Path)" "Red"
                    $failCount++
                }
            }
            
            Write-Color "`n$($server.Name): Success=$successCount | Failed=$failCount" "Cyan"
        }
        
        Write-Color "`nPush completed!" "Green"
    }
    
    "pull" {
        Write-Color "[PULL] Downloading from remote..." "Magenta"
        
        # Allow selecting server: mobaxterm pull .154 or mobaxterm pull .87
        $targetServer = $null
        if ($Arguments -and $Arguments.Count -gt 0) {
            $serverArg = $Arguments[0]
            foreach ($s in $Config.Servers) {
                if ($s.Name -eq $serverArg -or $s.Name -eq ".$serverArg" -or $serverArg -like "*$($s.Name)*") {
                    $targetServer = $s
                    break
                }
            }
        }
        if (-not $targetServer) {
            # Default to .87
            $targetServer = $Config.Servers[0]
        }
        
        Write-Color "`nPulling from $($targetServer.Name) ($($targetServer.Host))..." "Cyan"
        
        $extPattern = "-name '*.cu' -o -name '*.h' -o -name '*.c' -o -name '*.json' -o -name '*.md'"
        $cmd = "find $($Config.RemotePath) $extPattern 2>/dev/null | grep -v result | grep -v backup | grep -v initial_D3Q19"
        $remoteList = & $Config.PlinkPath -ssh -pw $targetServer.Password -batch "$($targetServer.User)@$($targetServer.Host)" $cmd 2>$null
        
        $pullCount = 0
        foreach ($remotePath in $remoteList) {
            if ($remotePath) {
                $relativePath = $remotePath.Replace($Config.RemotePath + "/", "").Replace("/", "\")
                $localPath = Join-Path $Config.LocalPath $relativePath
                $localDir = Split-Path $localPath -Parent
                
                if (-not (Test-Path $localDir)) { 
                    New-Item -ItemType Directory -Path $localDir -Force | Out-Null 
                }
                
                $null = & $Config.PscpPath -pw $targetServer.Password -q "$($targetServer.User)@$($targetServer.Host):$remotePath" $localPath 2>&1
                if ($LASTEXITCODE -eq 0) {
                    Write-Color "  [OK] $relativePath" "Green"
                    $pullCount++
                }
            }
        }
        
        Write-Color "`nPulled $pullCount files from $($targetServer.Name)" "Cyan"
    }
    
    "pull87" {
        & $PSCommandPath pull .87
    }
    
    "pull154" {
        & $PSCommandPath pull .154
    }
    
    # git fetch - 只檢查遠端狀態（不下載）
    "fetch" {
        Write-Color "[FETCH] Fetching remote status..." "Magenta"
        foreach ($server in $Config.Servers) {
            $results = Compare-Files -Server $server
            $remoteCount = $results.Same.Count + $results.Modified.Count + $results.Deleted.Count
            Write-Color "`n$($server.Name) ($($server.Host)):" "Cyan"
            Write-Color "  Remote files: $remoteCount" "White"
            Write-Color "  Same: $($results.Same.Count) | Modified: $($results.Modified.Count) | Remote-only: $($results.Deleted.Count)" "Gray"
        }
    }
    
    # git log - 查看遠端 log 檔案
    "log" {
        Write-Color "[LOG] Fetching remote log files..." "Magenta"
        
        $targetServer = $null
        if ($Arguments -and $Arguments.Count -gt 0) {
            $serverArg = $Arguments[0]
            foreach ($s in $Config.Servers) {
                if ($s.Name -eq $serverArg -or $s.Name -eq ".$serverArg" -or $serverArg -like "*$($s.Name)*") {
                    $targetServer = $s
                    break
                }
            }
        }
        if (-not $targetServer) { $targetServer = $Config.Servers[0] }
        
        Write-Color "`nLog files on $($targetServer.Name):" "Cyan"
        $cmd = "ls -lth $($Config.RemotePath)/log* 2>/dev/null | head -10"
        $result = & $Config.PlinkPath -ssh -pw $targetServer.Password -batch "$($targetServer.User)@$($targetServer.Host)" $cmd 2>$null
        if ($result) {
            foreach ($line in $result) { Write-Host "  $line" }
        }
        else {
            Write-Color "  No log files found" "Yellow"
        }
        
        # 顯示最新 log 的最後幾行
        Write-Color "`nLatest log tail:" "Cyan"
        $cmd = "tail -20 `$(ls -t $($Config.RemotePath)/log* 2>/dev/null | head -1) 2>/dev/null"
        $result = & $Config.PlinkPath -ssh -pw $targetServer.Password -batch "$($targetServer.User)@$($targetServer.Host)" $cmd 2>$null
        if ($result) {
            foreach ($line in $result) { Write-Host "  $line" -ForegroundColor Gray }
        }
    }
    
    # git reset --hard - 重置遠端（刪除遠端多餘檔案）
    { $_ -in "reset", "delete" } {
        Write-Color "[RESET] Removing files from remote that don't exist locally..." "Magenta"
        
        foreach ($server in $Config.Servers) {
            Write-Color "`nChecking $($server.Name) ($($server.Host))..." "Cyan"
            
            $results = Compare-Files -Server $server -Silent
            $toDelete = $results.Deleted
            
            if ($toDelete.Count -eq 0) {
                Write-Color "  No files to delete on $($server.Name)" "Green"
                continue
            }
            
            Write-Color "  Files to delete on $($server.Name):" "Yellow"
            foreach ($f in $toDelete) {
                Write-Color "    - $f" "Red"
            }
            
            $confirm = Read-Host "`n  Delete these $($toDelete.Count) files from $($server.Name)? (y/n)"
            if ($confirm -eq "y" -or $confirm -eq "Y") {
                $deleteCount = 0
                foreach ($f in $toDelete) {
                    $remotePath = "$($Config.RemotePath)/$f"
                    & $Config.PlinkPath -ssh -pw $server.Password -batch "$($server.User)@$($server.Host)" "rm -f '$remotePath'" 2>$null
                    if ($LASTEXITCODE -eq 0) {
                        Write-Color "    [DELETED] $f" "Red"
                        $deleteCount++
                    }
                    else {
                        Write-Color "    [FAILED] $f" "Yellow"
                    }
                }
                Write-Color "`n  Deleted $deleteCount files from $($server.Name)" "Cyan"
            }
            else {
                Write-Color "  Skipped deletion on $($server.Name)" "Yellow"
            }
        }
    }
    
    # git clone - 從遠端完整複製到本地
    "clone" {
        Write-Color "[CLONE] Full download from remote to local..." "Magenta"
        
        $targetServer = $null
        if ($Arguments -and $Arguments.Count -gt 0) {
            $serverArg = $Arguments[0]
            foreach ($s in $Config.Servers) {
                if ($s.Name -eq $serverArg -or $s.Name -eq ".$serverArg" -or $serverArg -like "*$($s.Name)*") {
                    $targetServer = $s
                    break
                }
            }
        }
        if (-not $targetServer) { $targetServer = $Config.Servers[0] }
        
        Write-Color "`nCloning from $($targetServer.Name) ($($targetServer.Host))..." "Cyan"
        Write-Color "This will overwrite local files with remote versions!" "Yellow"
        $confirm = Read-Host "Continue? (y/n)"
        
        if ($confirm -eq "y" -or $confirm -eq "Y") {
            $remoteFiles = Get-RemoteFiles -Server $targetServer
            $cloneCount = 0
            
            foreach ($f in $remoteFiles) {
                $remotePath = "$($Config.RemotePath)/$($f.Path)"
                $localPath = Join-Path $Config.LocalPath $f.Path.Replace("/", "\")
                $localDir = Split-Path $localPath -Parent
                
                if (-not (Test-Path $localDir)) { 
                    New-Item -ItemType Directory -Path $localDir -Force | Out-Null 
                }
                
                $null = & $Config.PscpPath -pw $targetServer.Password -q "$($targetServer.User)@$($targetServer.Host):$remotePath" $localPath 2>&1
                if ($LASTEXITCODE -eq 0) {
                    Write-Color "  [OK] $($f.Path)" "Green"
                    $cloneCount++
                }
            }
            Write-Color "`nCloned $cloneCount files from $($targetServer.Name)" "Cyan"
        }
    }
    
    # ===== Extra Commands (beyond Git) =====
    
    "sync" {
        Write-Color "[SYNC] Interactive sync (diff + push)" "Magenta"
        & $PSCommandPath diff
        Write-Host ""
        $confirm = Read-Host "Proceed with push? (y/n)"
        if ($confirm -eq "y" -or $confirm -eq "Y") {
            & $PSCommandPath push
        }
    }
    
    "issynced" {
        # Quick one-line sync status check
        $output = @()
        foreach ($server in $Config.Servers) {
            $results = Compare-Files -Server $server -Silent
            $needsPush = $results.New.Count + $results.Modified.Count
            if ($needsPush -eq 0) {
                $output += "$($server.Name): [OK] synced"
            }
            else {
                $output += "$($server.Name): [!] $needsPush pending"
            }
        }
        Write-Host ($output -join " | ")
    }
    
    "watch" {
        Write-Color "[WATCH] Auto-sync enabled - monitoring file changes..." "Magenta"
        Write-Color "Press Ctrl+C to stop`n" "Yellow"
        
        $watcher = New-Object System.IO.FileSystemWatcher
        $watcher.Path = $Config.LocalPath
        $watcher.IncludeSubdirectories = $true
        $watcher.EnableRaisingEvents = $true
        $watcher.NotifyFilter = [System.IO.NotifyFilters]::FileName -bor [System.IO.NotifyFilters]::LastWrite
        
        $lastSync = @{}
        $syncDelay = 2  # seconds to wait before syncing (debounce)
        
        $action = {
            $path = $Event.SourceEventArgs.FullPath
            $name = $Event.SourceEventArgs.Name
            $changeType = $Event.SourceEventArgs.ChangeType
            
            # Skip excluded files
            $skip = $false
            $excludePatterns = @("*.exe", "*.out", "a.out", "log*", "result\*", "statistics\*", "backup\*", ".git\*", "initial_D3Q19\*")
            foreach ($pattern in $excludePatterns) {
                if ($name -like $pattern) { $skip = $true; break }
            }
            
            # Only sync source files
            $ext = [System.IO.Path]::GetExtension($name)
            $syncExtensions = @(".cu", ".h", ".c", ".json", ".md", ".txt", ".ps1")
            $isVscode = $name -like ".vscode\*"
            $isGitignore = $name -eq ".gitignore"
            
            if (-not $skip -and ($syncExtensions -contains $ext -or $isVscode -or $isGitignore)) {
                $now = Get-Date
                if (-not $lastSync.ContainsKey($path) -or ($now - $lastSync[$path]).TotalSeconds -gt 2) {
                    $lastSync[$path] = $now
                    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $changeType : $name" -ForegroundColor Cyan
                    
                    # Trigger push in background
                    Start-Job -ScriptBlock {
                        param($scriptPath)
                        Start-Sleep -Seconds 1
                        & $scriptPath push 2>&1 | Out-Null
                    } -ArgumentList $using:PSCommandPath | Out-Null
                }
            }
        }
        
        Register-ObjectEvent $watcher "Changed" -Action $action | Out-Null
        Register-ObjectEvent $watcher "Created" -Action $action | Out-Null
        
        Write-Color "Watching: $($Config.LocalPath)" "White"
        Write-Color "Extensions: .cu .h .c .json .md .txt .ps1" "Gray"
        Write-Color "Auto-push to: .87 and .154`n" "Gray"
        
        try {
            while ($true) { Start-Sleep -Seconds 1 }
        }
        finally {
            Get-EventSubscriber | Unregister-Event
            $watcher.Dispose()
            Write-Color "`n[WATCH] Stopped" "Yellow"
        }
    }
    
    "autopush" {
        # Quick auto-push: check and push if needed (no interaction)
        $hasChanges = $false
        foreach ($server in $Config.Servers) {
            $results = Compare-Files -Server $server -Silent
            if ($results.New.Count -gt 0 -or $results.Modified.Count -gt 0) {
                $hasChanges = $true
                break
            }
        }
        if ($hasChanges) {
            Write-Color "[AUTO] Changes detected, pushing..." "Cyan"
            & $PSCommandPath push
        }
        else {
            Write-Color "[AUTO] No changes" "Green"
        }
    }
    
    "fullsync" {
        Write-Color "[FULLSYNC] Push + Reset (make remote match local exactly)" "Magenta"
        & $PSCommandPath push
        Write-Host ""
        & $PSCommandPath reset
    }
    
    default {
        Write-Host ""
        Write-Host "MobaXterm Sync Commands (Git-like)" -ForegroundColor Cyan
        Write-Host "===================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Usage: mobaxterm <command>" -ForegroundColor White
        Write-Host ""
        Write-Host "Git-like Commands:" -ForegroundColor Yellow
        Write-Host "  status      - Show sync status (like git status)"
        Write-Host "  add         - Show pending changes (like git add)"
        Write-Host "  diff        - Compare local vs remote (like git diff)"
        Write-Host "  push        - Upload to .87 and .154 (like git push)"
        Write-Host "  pull        - Download from remote (like git pull)"
        Write-Host "  pull .87    - Pull from .87 specifically"
        Write-Host "  pull .154   - Pull from .154 specifically"
        Write-Host "  fetch       - Check remote status without download (like git fetch)"
        Write-Host "  log         - View remote log files (like git log)"
        Write-Host "  log .154    - View logs from .154"
        Write-Host "  reset       - Delete remote-only files (like git reset --hard)"
        Write-Host "  clone       - Full download from remote (like git clone)"
        Write-Host "  clone .154  - Clone from .154"
        Write-Host ""
        Write-Host "Extra Commands:" -ForegroundColor Yellow
        Write-Host "  sync        - Interactive: diff -> confirm -> push"
        Write-Host "  fullsync    - Push + Reset (make remote = local exactly)"
        Write-Host "  issynced    - Quick one-line status check"
        Write-Host "  watch       - Auto-sync on file change (Ctrl+C to stop)"
        Write-Host "  autopush    - Push only if changes detected"
        Write-Host ""
        Write-Host "Aliases:" -ForegroundColor Yellow
        Write-Host "  check       - Same as diff"
        Write-Host "  delete      - Same as reset"
        Write-Host "  pull87      - Same as pull .87"
        Write-Host "  pull154     - Same as pull .154"
        Write-Host ""
        Write-Host "Examples:" -ForegroundColor Yellow
        Write-Host "  mobaxterm status       # Check sync state"
        Write-Host "  mobaxterm diff         # View differences"
        Write-Host "  mobaxterm push         # Upload changes"
        Write-Host "  mobaxterm pull .154    # Download from .154"
        Write-Host "  mobaxterm log          # View remote logs"
        Write-Host "  mobaxterm watch        # Start auto-sync"
        Write-Host ""
    }
}
