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
    # ?垢?郊嚗???.git??vscode ?楊霅舐??
    ExcludePatterns = @(".git/*", ".vscode/*", "a.out", "*.o", "*.exe")
    # ?郊???獢???
    SyncExtensions = @("*")
    SyncAll = $true  # ?郊???獢?垢撌乩??
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
    
    # ? .git ??.vscode ?桅?
    $excludeGrep = "grep -v '/.git/' | grep -v '/.vscode/'"
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
    
    # git diff - 瘥??砍??蝡臬榆??
    { $_ -in "diff", "check" } {
        Write-Color "[DIFF] Comparing local vs remote..." "Magenta"
        foreach ($server in $Config.Servers) {
            $results = Compare-Files -Server $server
            Show-CompareResults -Results $results -ServerName $server.Name
        }
    }
    
    # git status - 憿舐內?郊???
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
    
    # git add - 憿舐內敺??霈
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
        Write-Color "[PUSH] Syncing changes to remote servers..." "Magenta"
        
        $localFiles = Get-LocalFiles
        $localHash = @{}
        foreach ($f in $localFiles) { $localHash[$f.Path] = $f }
        
        foreach ($server in $Config.Servers) {
            Write-Color "`nPushing to $($server.Name) ($($server.Host))..." "Cyan"
            
            # 瘥?撌桃
            $results = Compare-Files -Server $server -Silent
            $toUpload = @()
            $toUpload += $results.New
            $toUpload += $results.Modified
            $toDelete = $results.Deleted
            
            if ($toUpload.Count -eq 0 -and $toDelete.Count -eq 0) {
                Write-Color "  [OK] Already synced, nothing to push" "Green"
                continue
            }
            
            # 銝?啣?/靽格??獢?
            $successCount = 0
            $failCount = 0
            
            foreach ($path in $toUpload) {
                $file = $localHash[$path]
                if (-not $file) { continue }
                
                $localPath = $file.FullPath
                $remoteDest = "$($server.User)@$($server.Host):$($Config.RemotePath)/$($file.Path)"
                
                $remoteDir = [System.IO.Path]::GetDirectoryName("$($Config.RemotePath)/$($file.Path)").Replace("\", "/")
                & $Config.PlinkPath -ssh -pw $server.Password -batch "$($server.User)@$($server.Host)" "mkdir -p '$remoteDir'" 2>$null
                
                $null = & $Config.PscpPath -pw $server.Password -q $localPath $remoteDest 2>&1
                if ($LASTEXITCODE -eq 0) {
                    Write-Color "  [UPLOAD] $($file.Path)" "Green"
                    $successCount++
                }
                else {
                    Write-Color "  [FAIL] $($file.Path)" "Red"
                    $failCount++
                }
            }
            
            # ?芷?垢憭?瑼?
            $deleteCount = 0
            foreach ($f in $toDelete) {
                $remotePath = "$($Config.RemotePath)/$f"
                & $Config.PlinkPath -ssh -pw $server.Password -batch "$($server.User)@$($server.Host)" "rm -f '$remotePath'" 2>$null
                if ($LASTEXITCODE -eq 0) {
                    Write-Color "  [DELETE] $f" "Red"
                    $deleteCount++
                }
            }
            
            Write-Color "`n$($server.Name): Uploaded=$successCount | Deleted=$deleteCount | Failed=$failCount" "Cyan"
            
            # 皜??垢蝛箄??冗嚗???.git嚗?
            $cleanupCmd = "find $($Config.RemotePath) -type d -empty ! -path '*/.git/*' -delete 2>/dev/null"
            & $Config.PlinkPath -ssh -pw $server.Password -batch "$($server.User)@$($server.Host)" $cleanupCmd 2>$null
        }
        
        Write-Color "`nPush completed!" "Green"
    }
    
    "pull" {
        Write-Color "[PULL] Downloading from remote (no delete)..." "Magenta"
        $targetServer = $null
        if ($Arguments -and $Arguments.Count -gt 0) {
            $serverArg = $Arguments[0]
            foreach ($s in $Config.Servers) {
                if ($s.Name -eq $serverArg -or $s.Name -eq ".$serverArg" -or $serverArg -like "*$($s.Name)*") {
                    $targetServer = $s; break
                }
            }
        }
        if (-not $targetServer) { $targetServer = $Config.Servers[0] }
        Write-Color "`nPulling from $($targetServer.Name)..." "Cyan"
        $results = Compare-Files -Server $targetServer -Silent
        $toDownload = @(); $toDownload += $results.Deleted; $toDownload += $results.Modified
        if ($toDownload.Count -eq 0) { Write-Color "  [OK] Nothing to pull" "Green" }
        else {
            $pullCount = 0
            foreach ($relativePath in $toDownload) {
                $remotePath = "$($Config.RemotePath)/$relativePath"
                $localPath = Join-Path $Config.LocalPath ($relativePath.Replace("/", "\"))
                $localDir = Split-Path $localPath -Parent
                if (-not (Test-Path $localDir)) { New-Item -ItemType Directory -Path $localDir -Force | Out-Null }
                $null = & $Config.PscpPath -pw $targetServer.Password -q "$($targetServer.User)@$($targetServer.Host):$remotePath" $localPath 2>&1
                if ($LASTEXITCODE -eq 0) { Write-Color "  [DOWNLOAD] $relativePath" "Green"; $pullCount++ }
            }
            Write-Color "`nDownloaded=$pullCount" "Cyan"
        }
    }
    
    "pull87" {
        & $PSCommandPath pull .87
    }
    
    "pull154" {
        & $PSCommandPath pull .154
    }
    
    # git fetch - 敺?蝡臭?頛蒂?芷?砍憭??辣嚗ync local to remote嚗?
    "fetch" {

        $targetServer = $null

        if ($Arguments -and $Arguments.Count -gt 0) {

            $serverArg = $Arguments[0]

            $targetServer = $Config.Servers | Where-Object { $_.Name -eq $serverArg -or $_.Host -like "*$serverArg*" } | Select-Object -First 1

        }

        if (-not $targetServer) { $targetServer = $Config.Servers[0] }

        

        Write-Color "[FETCH] Syncing from remote (with delete)..." "Magenta"

        Write-Color "Server: $($targetServer.Name) ($($targetServer.Host))" "Cyan"

        

        $results = Compare-Files -Server $targetServer

        $toDownload = @($results.Deleted) + @($results.Modified) | Where-Object { $_ }

        $toDelete = @($results.New) | Where-Object { $_ }

        

        # Download files from remote

        if ($toDownload.Count -gt 0) {

            Write-Color "Downloading $($toDownload.Count) file(s)..." "Yellow"

            foreach ($file in $toDownload) {

                $remotePath = "$($targetServer.User)@$($targetServer.Host):$($targetServer.RemotePath)/$file"

                $localPath = Join-Path $Config.LocalPath $file

                $localDir = Split-Path $localPath -Parent

                if (-not (Test-Path $localDir)) { New-Item -ItemType Directory -Path $localDir -Force | Out-Null }

                & $Config.PSCPPath -pw $targetServer.Password -r $remotePath $localPath 2>$null

                Write-Color "  Downloaded: $file" "Green"

            }

        }

        

        # Delete local files not on remote

        if ($toDelete.Count -gt 0) {

            Write-Color "Deleting $($toDelete.Count) local file(s) not on remote..." "Red"

            foreach ($file in $toDelete) {

                $localPath = Join-Path $Config.LocalPath $file

                if (Test-Path $localPath) {

                    Remove-Item $localPath -Force

                    Write-Color "  Deleted: $file" "Red"

                }

            }

        }

        

        if ($toDownload.Count -eq 0 -and $toDelete.Count -eq 0) {

            Write-Color "Local is in sync with remote." "Green"

        } else {

            Write-Color "Fetch complete: $($toDownload.Count) downloaded, $($toDelete.Count) deleted" "Green"

        }

    }

    

    "fetch87" {

        & $PSCommandPath fetch 87

    }

    

    "fetch154" {

        & $PSCommandPath fetch 154

    }
    # git log - ?亦??垢 log 瑼?
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
        
        # 憿舐內???log ??敺嗾銵?
        Write-Color "`nLatest log tail:" "Cyan"
        $cmd = "tail -20 `$(ls -t $($Config.RemotePath)/log* 2>/dev/null | head -1) 2>/dev/null"
        $result = & $Config.PlinkPath -ssh -pw $targetServer.Password -batch "$($targetServer.User)@$($targetServer.Host)" $cmd 2>$null
        if ($result) {
            foreach ($line in $result) { Write-Host "  $line" -ForegroundColor Gray }
        }
    }
    
    # git reset --hard - ?蔭?垢嚗?日?蝡臬?擗?獢?
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
    
    # git clone - 敺?蝡臬??渲?鋆賢?砍
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
    
    "watchpush" {
        # Background auto-upload: monitor local files and upload changes (persistent process)
        $pidFile = Join-Path $Config.LocalPath ".vscode\watchpush.pid"
        $logFile = Join-Path $Config.LocalPath ".vscode\watchpush.log"
        $daemonScript = Join-Path $Config.LocalPath ".vscode\watchpush-daemon.ps1"
        $subCommand = if ($Arguments.Count -gt 0) { $Arguments[0] } else { "" }
        
        switch ($subCommand) {
            "stop" {
                if (Test-Path $pidFile) {
                    $pids = Get-Content $pidFile -ErrorAction SilentlyContinue
                    foreach ($p in $pids) {
                        if ($p -match '^\d+$') {
                            $proc = Get-Process -Id $p -ErrorAction SilentlyContinue
                            if ($proc) {
                                Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
                                Write-Color "[WATCHPUSH] Stopped process $p" "Yellow"
                            }
                        }
                    }
                    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
                    Write-Color "[WATCHPUSH] Auto-upload stopped" "Green"
                }
                else {
                    Write-Color "[WATCHPUSH] No active watchpush process" "Yellow"
                }
            }
            
            "status" {
                Write-Color "`n=== WatchPush Status ===" "Cyan"
                if (Test-Path $pidFile) {
                    $pids = Get-Content $pidFile -ErrorAction SilentlyContinue
                    $active = @()
                    foreach ($p in $pids) {
                        if ($p -match '^\d+$') {
                            $proc = Get-Process -Id $p -ErrorAction SilentlyContinue
                            if ($proc) { $active += $p }
                        }
                    }
                    if ($active.Count -gt 0) {
                        Write-Color "[RUNNING] PIDs: $($active -join ', ')" "Green"
                        if (Test-Path $logFile) {
                            Write-Color "`nRecent Activity (last 15 lines):" "Yellow"
                            Get-Content $logFile -Tail 15 | ForEach-Object { Write-Host "  $_" }
                        }
                    }
                    else {
                        Write-Color "[STOPPED] No active process" "Yellow"
                        Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
                    }
                }
                else {
                    Write-Color "[STOPPED] WatchPush is not running" "Yellow"
                }
                Write-Host ""
            }
            
            "log" {
                if (Test-Path $logFile) {
                    Write-Color "=== WatchPush Log ===" "Cyan"
                    Get-Content $logFile -Tail 50 | ForEach-Object { Write-Host $_ }
                }
                else {
                    Write-Color "No log file found" "Yellow"
                }
            }
            
            "clear" {
                if (Test-Path $logFile) {
                    Remove-Item $logFile -Force
                    Write-Color "[WATCHPUSH] Log cleared" "Green"
                }
            }
            
            default {
                # Check if already running
                if (Test-Path $pidFile) {
                    $pids = Get-Content $pidFile -ErrorAction SilentlyContinue
                    foreach ($p in $pids) {
                        if ($p -match '^\d+$') {
                            $proc = Get-Process -Id $p -ErrorAction SilentlyContinue
                            if ($proc) {
                                Write-Color "[WATCHPUSH] Already running (PID: $p). Use 'mobaxterm watchpush stop' first." "Yellow"
                                return
                            }
                        }
                    }
                }
                
                $interval = 10  # Check interval in seconds
                if ($Arguments.Count -gt 0 -and $Arguments[0] -match '^\d+$') {
                    $interval = [int]$Arguments[0]
                }
                
                Write-Color "[WATCHPUSH] Starting background auto-upload (persistent)..." "Magenta"
                Write-Color "  Interval: ${interval}s" "White"
                Write-Color "  Targets: .87, .154" "White"
                Write-Color "  Log: $logFile" "Gray"
                Write-Color "`nCommands:" "Yellow"
                Write-Color "  mobaxterm watchpush status  - Check status & recent uploads" "Gray"
                Write-Color "  mobaxterm watchpush log     - View full log" "Gray"
                Write-Color "  mobaxterm watchpush stop    - Stop monitoring" "Gray"
                Write-Color "  mobaxterm bgstatus          - All background processes (push/pull/fetch/vtkrename)" "Gray"
                Write-Color "  mobaxterm syncstatus        - Combined push/pull status" "Gray"
                Write-Host ""
                
                # Clear old log (use UTF-8 without BOM)
                $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
                [System.IO.File]::WriteAllText($logFile, "", $utf8NoBom)
                
                # Prepare servers JSON
                $serversJson = $Config.Servers | ConvertTo-Json -Compress
                
                # Start independent PowerShell process
                $proc = Start-Process -FilePath "powershell.exe" -ArgumentList @(
                    "-NoProfile", "-WindowStyle", "Hidden", "-ExecutionPolicy", "Bypass",
                    "-File", "`"$daemonScript`"",
                    "-LocalPath", "`"$($Config.LocalPath)`"",
                    "-RemotePath", "`"$($Config.RemotePath)`"",
                    "-ServersJson", "'$serversJson'",
                    "-PlinkPath", "`"$($Config.PlinkPath)`"",
                    "-PscpPath", "`"$($Config.PscpPath)`"",
                    "-LogPath", "`"$logFile`"",
                    "-Interval", $interval
                ) -PassThru
                
                # Save process ID
                Start-Sleep -Milliseconds 500
                $proc.Id | Out-File $pidFile -Force
                Write-Color "[STARTED] Background process (PID: $($proc.Id))" "Green"
                
                Write-Color "`n[WATCHPUSH] Background auto-upload started!" "Green"
                Write-Color "Use 'mobaxterm watchpush status' to check progress" "Cyan"
            }
        }
    }
    
    "bgstatus" {
        # All background processes status (watchpush, watchpull, watchfetch, vtkrename)
        $services = @(
            @{ Name = "WatchPush"; Label = "[UPLOAD] WatchPush"; PidFile = ".vscode\watchpush.pid"; LogFile = ".vscode\watchpush.log"; Color = "Yellow" },
            @{ Name = "WatchPull"; Label = "[DOWNLOAD] WatchPull"; PidFile = ".vscode\watchpull.pid"; LogFile = ".vscode\watchpull.log"; Color = "Yellow" },
            @{ Name = "WatchFetch"; Label = "[SYNC+DELETE] WatchFetch"; PidFile = ".vscode\watchfetch.pid"; LogFile = ".vscode\watchfetch.log"; Color = "Red" },
            @{ Name = "VTKRenamer"; Label = "[VTK-RENAME] Auto-Renamer"; PidFile = ".vscode\vtk-renamer.pid"; LogFile = ".vscode\vtk-renamer.log"; Color = "Cyan" }
        )
        
        Write-Color "`n========== All Background Processes ==========" "Cyan"
        
        foreach ($svc in $services) {
            $pidFile = Join-Path $Config.LocalPath $svc.PidFile
            $logFile = Join-Path $Config.LocalPath $svc.LogFile
            
            Write-Color "`n$($svc.Label):" $svc.Color
            
            if (Test-Path $pidFile) {
                $pids = Get-Content $pidFile -ErrorAction SilentlyContinue
                $active = @()
                foreach ($p in $pids) {
                    if ($p -match '^\d+$') {
                        $proc = Get-Process -Id $p -ErrorAction SilentlyContinue
                        if ($proc) { $active += $p }
                    }
                }
                if ($active.Count -gt 0) {
                    Write-Color "  Status: RUNNING (PID: $($active -join ', '))" "Green"
                    if (Test-Path $logFile) {
                        $lastLines = Get-Content $logFile -Tail 2 -ErrorAction SilentlyContinue
                        if ($lastLines) {
                            foreach ($line in $lastLines) {
                                if ($line) { Write-Color "  $line" "Gray" }
                            }
                        }
                    }
                }
                else {
                    Write-Color "  Status: STOPPED" "Yellow"
                    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
                }
            }
            else {
                Write-Color "  Status: OFF" "Gray"
            }
        }
        
        Write-Color "`n=============================================" "Cyan"
        Write-Color "Commands:" "Yellow"
        Write-Color "  mobaxterm watchpush         - Start auto-upload" "Gray"
        Write-Color "  mobaxterm watchpull         - Start auto-download" "Gray"
        Write-Color "  mobaxterm watchfetch .87    - Start auto-sync with delete" "Gray"
        Write-Color "  mobaxterm vtkrename         - Start VTK renamer" "Gray"
        Write-Color "  mobaxterm <service> stop    - Stop service" "Gray"
        Write-Color "  mobaxterm <service> status  - Detailed status" "Gray"
        Write-Host ""
    }
    
    "syncstatus" {
        # Combined status for both watchpush and watchpull
        $pushPidFile = Join-Path $Config.LocalPath ".vscode\watchpush.pid"
        $pullPidFile = Join-Path $Config.LocalPath ".vscode\watchpull.pid"
        $pushLogFile = Join-Path $Config.LocalPath ".vscode\watchpush.log"
        $pullLogFile = Join-Path $Config.LocalPath ".vscode\watchpull.log"
        
        Write-Color "`n========== Sync Monitor Status ==========" "Cyan"
        
        # WatchPush status
        Write-Color "`n[UPLOAD] WatchPush:" "Yellow"
        if (Test-Path $pushPidFile) {
            $pids = Get-Content $pushPidFile -ErrorAction SilentlyContinue
            $active = @()
            foreach ($p in $pids) {
                if ($p -match '^\d+$') {
                    $proc = Get-Process -Id $p -ErrorAction SilentlyContinue
                    if ($proc) { $active += $p }
                }
            }
            if ($active.Count -gt 0) {
                Write-Color "  Status: RUNNING (PID: $($active -join ', '))" "Green"
                if (Test-Path $pushLogFile) {
                    $lastLine = Get-Content $pushLogFile -Tail 1
                    if ($lastLine) { Write-Color "  Last: $lastLine" "Gray" }
                }
            }
            else {
                Write-Color "  Status: STOPPED" "Yellow"
                Remove-Item $pushPidFile -Force -ErrorAction SilentlyContinue
            }
        }
        else {
            Write-Color "  Status: OFF" "Gray"
        }
        
        # WatchPull status
        Write-Color "`n[DOWNLOAD] WatchPull:" "Yellow"
        if (Test-Path $pullPidFile) {
            $pids = Get-Content $pullPidFile -ErrorAction SilentlyContinue
            $active = @()
            foreach ($p in $pids) {
                if ($p -match '^\d+$') {
                    $proc = Get-Process -Id $p -ErrorAction SilentlyContinue
                    if ($proc) { $active += $p }
                }
            }
            if ($active.Count -gt 0) {
                Write-Color "  Status: RUNNING (PID: $($active -join ', '))" "Green"
                if (Test-Path $pullLogFile) {
                    $lastLine = Get-Content $pullLogFile -Tail 1
                    if ($lastLine) { Write-Color "  Last: $lastLine" "Gray" }
                }
            }
            else {
                Write-Color "  Status: STOPPED" "Yellow"
                Remove-Item $pullPidFile -Force -ErrorAction SilentlyContinue
            }
        }
        else {
            Write-Color "  Status: OFF" "Gray"
        }
        
        Write-Color "`n=========================================" "Cyan"
        Write-Color "Commands:" "Yellow"
        Write-Color "  mobaxterm watchpush       - Start auto-upload" "Gray"
        Write-Color "  mobaxterm watchpush stop  - Stop auto-upload" "Gray"
        Write-Color "  mobaxterm watchpull       - Start auto-download" "Gray"
        Write-Color "  mobaxterm watchpull stop  - Stop auto-download" "Gray"
        Write-Host ""
    }
    
    "fullsync" {
        Write-Color "[FULLSYNC] Push + Reset (make remote match local exactly)" "Magenta"
        & $PSCommandPath push
        Write-Host ""
        & $PSCommandPath reset
    }
    
    "watchpull" {
        # Auto-download: monitor remote servers and download new files (persistent process)
        $pidFile = Join-Path $Config.LocalPath ".vscode\watchpull.pid"
        $logFile = Join-Path $Config.LocalPath ".vscode\watchpull.log"
        $daemonScript = Join-Path $Config.LocalPath ".vscode\watchpull-daemon.ps1"
        $subCommand = if ($Arguments.Count -gt 0) { $Arguments[0] } else { "" }
        
        switch ($subCommand) {
            "stop" {
                if (Test-Path $pidFile) {
                    $pids = Get-Content $pidFile -ErrorAction SilentlyContinue
                    foreach ($p in $pids) {
                        if ($p -match '^\d+$') {
                            $proc = Get-Process -Id $p -ErrorAction SilentlyContinue
                            if ($proc) {
                                Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
                                Write-Color "[WATCHPULL] Stopped process $p" "Yellow"
                            }
                        }
                    }
                    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
                    Write-Color "[WATCHPULL] Auto-download stopped" "Green"
                }
                else {
                    Write-Color "[WATCHPULL] No active watchpull process" "Yellow"
                }
            }
            
            "status" {
                Write-Color "`n=== WatchPull Status ===" "Cyan"
                if (Test-Path $pidFile) {
                    $pids = Get-Content $pidFile -ErrorAction SilentlyContinue
                    $active = @()
                    foreach ($p in $pids) {
                        if ($p -match '^\d+$') {
                            $proc = Get-Process -Id $p -ErrorAction SilentlyContinue
                            if ($proc) { $active += $p }
                        }
                    }
                    if ($active.Count -gt 0) {
                        Write-Color "[RUNNING] PIDs: $($active -join ', ')" "Green"
                        if (Test-Path $logFile) {
                            Write-Color "`nRecent Activity (last 20 lines):" "Yellow"
                            Get-Content $logFile -Tail 20 | ForEach-Object { Write-Host "  $_" }
                        }
                    }
                    else {
                        Write-Color "[STOPPED] No active process" "Yellow"
                        Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
                    }
                }
                else {
                    Write-Color "[STOPPED] WatchPull is not running" "Yellow"
                }
                Write-Host ""
            }
            
            "log" {
                if (Test-Path $logFile) {
                    Write-Color "=== WatchPull Log ===" "Cyan"
                    Get-Content $logFile -Tail 50 | ForEach-Object { Write-Host $_ }
                }
                else {
                    Write-Color "No log file found" "Yellow"
                }
            }
            
            "clear" {
                if (Test-Path $logFile) {
                    Remove-Item $logFile -Force
                    Write-Color "[WATCHPULL] Log cleared" "Green"
                }
            }
            
            default {
                # Start watchpull for specified server or both
                $targetServers = @()
                if ($subCommand -eq ".87" -or $subCommand -eq "87") {
                    $targetServers = @($Config.Servers | Where-Object { $_.Name -eq ".87" })
                }
                elseif ($subCommand -eq ".154" -or $subCommand -eq "154") {
                    $targetServers = @($Config.Servers | Where-Object { $_.Name -eq ".154" })
                }
                else {
                    $targetServers = $Config.Servers
                }
                
                # Check if already running
                if (Test-Path $pidFile) {
                    $pids = Get-Content $pidFile -ErrorAction SilentlyContinue
                    foreach ($p in $pids) {
                        if ($p -match '^\d+$') {
                            $proc = Get-Process -Id $p -ErrorAction SilentlyContinue
                            if ($proc) {
                                Write-Color "[WATCHPULL] Already running (PID: $p). Use 'mobaxterm watchpull stop' first." "Yellow"
                                return
                            }
                        }
                    }
                }
                
                $interval = 30  # Check interval in seconds
                if ($Arguments.Count -gt 1 -and $Arguments[1] -match '^\d+$') {
                    $interval = [int]$Arguments[1]
                }
                
                Write-Color "[WATCHPULL] Starting auto-download monitor (persistent)..." "Magenta"
                Write-Color "  Servers: $($targetServers.Name -join ', ')" "White"
                Write-Color "  Interval: ${interval}s" "White"
                Write-Color "  Log: $logFile" "Gray"
                Write-Color "`nCommands:" "Yellow"
                Write-Color "  mobaxterm watchpull status  - Check status & recent downloads" "Gray"
                Write-Color "  mobaxterm watchpull log     - View full log" "Gray"
                Write-Color "  mobaxterm watchpull stop    - Stop monitoring" "Gray"
                Write-Host ""
                
                # Clear old log (use UTF-8 without BOM)
                $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
                [System.IO.File]::WriteAllText($logFile, "", $utf8NoBom)
                
                # Start independent process for each server
                $processPids = @()
                foreach ($server in $targetServers) {
                    $proc = Start-Process -FilePath "powershell.exe" -ArgumentList @(
                        "-NoProfile", "-WindowStyle", "Hidden", "-ExecutionPolicy", "Bypass",
                        "-File", "`"$daemonScript`"",
                        "-LocalPath", "`"$($Config.LocalPath)`"",
                        "-RemotePath", "`"$($Config.RemotePath)`"",
                        "-ServerName", "`"$($server.Name)`"",
                        "-ServerHost", "`"$($server.Host)`"",
                        "-ServerUser", "`"$($server.User)`"",
                        "-ServerPass", "`"$($server.Password)`"",
                        "-PlinkPath", "`"$($Config.PlinkPath)`"",
                        "-PscpPath", "`"$($Config.PscpPath)`"",
                        "-LogPath", "`"$logFile`"",
                        "-Interval", $interval
                    ) -PassThru
                    
                    Start-Sleep -Milliseconds 500
                    $processPids += $proc.Id
                    Write-Color "[STARTED] $($server.Name) monitoring (PID: $($proc.Id))" "Green"
                }
                
                # Save process IDs
                $processPids | Out-File $pidFile -Force
                
                Write-Color "`n[WATCHPULL] Background monitoring started!" "Green"
                Write-Color "Use 'mobaxterm watchpull status' to check progress" "Cyan"
            }
        }
    }
    
    "watchfetch" {
        # Auto-download with delete: monitor remote and sync local to match (persistent process)
        $pidFile = Join-Path $Config.LocalPath ".vscode\watchfetch.pid"
        $logFile = Join-Path $Config.LocalPath ".vscode\watchfetch.log"
        $daemonScript = Join-Path $Config.LocalPath ".vscode\watchfetch-daemon.ps1"
        $subCommand = if ($Arguments.Count -gt 0) { $Arguments[0] } else { "" }
        
        switch ($subCommand) {
            "stop" {
                if (Test-Path $pidFile) {
                    $pids = Get-Content $pidFile -ErrorAction SilentlyContinue
                    foreach ($p in $pids) {
                        if ($p -match '^\d+$') {
                            $proc = Get-Process -Id $p -ErrorAction SilentlyContinue
                            if ($proc) {
                                Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
                                Write-Color "[WATCHFETCH] Stopped process $p" "Yellow"
                            }
                        }
                    }
                    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
                    Write-Color "[WATCHFETCH] Auto-fetch stopped" "Green"
                }
                else {
                    Write-Color "[WATCHFETCH] No active watchfetch process" "Yellow"
                }
            }
            
            "status" {
                Write-Color "`n=== WatchFetch Status (WITH DELETE) ===" "Cyan"
                if (Test-Path $pidFile) {
                    $pids = Get-Content $pidFile -ErrorAction SilentlyContinue
                    $active = @()
                    foreach ($p in $pids) {
                        if ($p -match '^\d+$') {
                            $proc = Get-Process -Id $p -ErrorAction SilentlyContinue
                            if ($proc) { $active += $p }
                        }
                    }
                    if ($active.Count -gt 0) {
                        Write-Color "[RUNNING] PIDs: $($active -join ', ')" "Green"
                        if (Test-Path $logFile) {
                            Write-Color "`nRecent Activity (last 20 lines):" "Yellow"
                            Get-Content $logFile -Tail 20 | ForEach-Object { Write-Host "  $_" }
                        }
                    }
                    else {
                        Write-Color "[STOPPED] No active process" "Yellow"
                        Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
                    }
                }
                else {
                    Write-Color "[STOPPED] WatchFetch is not running" "Yellow"
                }
                Write-Host ""
            }
            
            "log" {
                if (Test-Path $logFile) {
                    Write-Color "=== WatchFetch Log ===" "Cyan"
                    Get-Content $logFile -Tail 50 | ForEach-Object { Write-Host $_ }
                }
                else {
                    Write-Color "No log file found" "Yellow"
                }
            }
            
            "clear" {
                if (Test-Path $logFile) {
                    Remove-Item $logFile -Force
                    Write-Color "[WATCHFETCH] Log cleared" "Green"
                }
            }
            
            default {
                # Start watchfetch for specified server
                $targetServer = $null
                if ($subCommand -eq ".87" -or $subCommand -eq "87") {
                    $targetServer = $Config.Servers | Where-Object { $_.Name -eq ".87" } | Select-Object -First 1
                }
                elseif ($subCommand -eq ".154" -or $subCommand -eq "154") {
                    $targetServer = $Config.Servers | Where-Object { $_.Name -eq ".154" } | Select-Object -First 1
                }
                else {
                    $targetServer = $Config.Servers[0]  # Default to first server
                }
                
                # Check if already running
                if (Test-Path $pidFile) {
                    $pids = Get-Content $pidFile -ErrorAction SilentlyContinue
                    foreach ($p in $pids) {
                        if ($p -match '^\d+$') {
                            $proc = Get-Process -Id $p -ErrorAction SilentlyContinue
                            if ($proc) {
                                Write-Color "[WATCHFETCH] Already running (PID: $p). Use 'mobaxterm watchfetch stop' first." "Yellow"
                                return
                            }
                        }
                    }
                }
                
                $interval = 30
                if ($Arguments.Count -gt 1 -and $Arguments[1] -match '^\d+$') {
                    $interval = [int]$Arguments[1]
                }
                
                Write-Color "[WATCHFETCH] Starting auto-fetch monitor WITH DELETE (persistent)..." "Magenta"
                Write-Color "  Server: $($targetServer.Name)" "White"
                Write-Color "  Interval: ${interval}s" "White"
                Write-Color "  Mode: Download + Delete local files not on remote" "Red"
                Write-Color "  Log: $logFile" "Gray"
                Write-Color "`nCommands:" "Yellow"
                Write-Color "  mobaxterm watchfetch status  - Check status" "Gray"
                Write-Color "  mobaxterm watchfetch log     - View log" "Gray"
                Write-Color "  mobaxterm watchfetch stop    - Stop monitoring" "Gray"
                Write-Host ""
                
                # Clear old log
                $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
                [System.IO.File]::WriteAllText($logFile, "", $utf8NoBom)
                
                # Start daemon process
                $proc = Start-Process -FilePath "powershell.exe" -ArgumentList @(
                    "-NoProfile", "-WindowStyle", "Hidden", "-ExecutionPolicy", "Bypass",
                    "-File", "`"$daemonScript`"",
                    "-LocalPath", "`"$($Config.LocalPath)`"",
                    "-RemotePath", "`"$($Config.RemotePath)`"",
                    "-ServerName", "`"$($targetServer.Name)`"",
                    "-ServerHost", "`"$($targetServer.Host)`"",
                    "-ServerUser", "`"$($targetServer.User)`"",
                    "-ServerPass", "`"$($targetServer.Password)`"",
                    "-PlinkPath", "`"$($Config.PlinkPath)`"",
                    "-PscpPath", "`"$($Config.PscpPath)`"",
                    "-LogPath", "`"$logFile`"",
                    "-Interval", $interval
                ) -PassThru
                
                Start-Sleep -Milliseconds 500
                
                # Save PID
                $proc.Id | Out-File $pidFile -Force
                Write-Color "[STARTED] $($targetServer.Name) fetch monitoring (PID: $($proc.Id))" "Green"
                
                Write-Color "`n[WATCHFETCH] Background monitoring started!" "Green"
                Write-Color "WARNING: Local files not on remote will be DELETED!" "Red"
                Write-Color "Use 'mobaxterm watchfetch status' to check progress" "Cyan"
            }
        }
    }
    
    "autopull" {
        # Quick auto-pull: check and pull if needed (download only, no local delete)
        $targetServer = $Config.Servers[0]  # Default to .87
        if ($Arguments.Count -gt 0) {
            $arg = $Arguments[0]
            if ($arg -eq ".154" -or $arg -eq "154") {
                $targetServer = $Config.Servers | Where-Object { $_.Name -eq ".154" }
            }
            elseif ($arg -eq ".87" -or $arg -eq "87") {
                $targetServer = $Config.Servers | Where-Object { $_.Name -eq ".87" }
            }
        }
        
        $results = Compare-Files -Server $targetServer -Silent
        # 只下載遠端有的檔案，不刪除本地檔案
        $toDownload = @()
        $toDownload += $results.Deleted   # 遠端有本地沒有
        $toDownload += $results.Modified  # 遠端有更新
        
        if ($toDownload.Count -gt 0) {
            Write-Color "[AUTOPULL] $($toDownload.Count) files to download from $($targetServer.Name)" "Cyan"
            $pullCount = 0
            foreach ($relativePath in $toDownload) {
                $remotePath = "$($Config.RemotePath)/$relativePath"
                $localPath = Join-Path $Config.LocalPath ($relativePath.Replace("/", "\"))
                $localDir = Split-Path $localPath -Parent
                
                if (-not (Test-Path $localDir)) { 
                    New-Item -ItemType Directory -Path $localDir -Force | Out-Null 
                }
                
                $null = & $Config.PscpPath -pw $targetServer.Password -q "$($targetServer.User)@$($targetServer.Host):$remotePath" $localPath 2>&1
                if ($LASTEXITCODE -eq 0) {
                    Write-Color "  [DOWNLOAD] $relativePath" "Green"
                    $pullCount++
                }
            }
            Write-Color "[AUTOPULL] Downloaded $pullCount files" "Cyan"
        }
        else {
            Write-Color "[AUTOPULL] $($targetServer.Name) - No new files" "Green"
        }
    }
    
    "autofetch" {
        # Quick auto-fetch: download + delete local files not on remote (sync local to remote)
        $targetServer = $Config.Servers[0]  # Default to .87
        if ($Arguments.Count -gt 0) {
            $arg = $Arguments[0]
            if ($arg -eq ".154" -or $arg -eq "154") {
                $targetServer = $Config.Servers | Where-Object { $_.Name -eq ".154" }
            }
            elseif ($arg -eq ".87" -or $arg -eq "87") {
                $targetServer = $Config.Servers | Where-Object { $_.Name -eq ".87" }
            }
        }
        
        $results = Compare-Files -Server $targetServer -Silent
        $toDownload = @($results.Deleted) + @($results.Modified) | Where-Object { $_ }
        $toDelete = @($results.New) | Where-Object { $_ }
        
        $hasChanges = $false
        
        # Download from remote
        if ($toDownload.Count -gt 0) {
            $hasChanges = $true
            Write-Color "[AUTOFETCH] Downloading $($toDownload.Count) files from $($targetServer.Name)" "Cyan"
            foreach ($relativePath in $toDownload) {
                $remotePath = "$($Config.RemotePath)/$relativePath"
                $localPath = Join-Path $Config.LocalPath ($relativePath.Replace("/", "\"))
                $localDir = Split-Path $localPath -Parent
                
                if (-not (Test-Path $localDir)) { 
                    New-Item -ItemType Directory -Path $localDir -Force | Out-Null 
                }
                
                $null = & $Config.PscpPath -pw $targetServer.Password -q "$($targetServer.User)@$($targetServer.Host):$remotePath" $localPath 2>&1
                if ($LASTEXITCODE -eq 0) {
                    Write-Color "  [DOWNLOAD] $relativePath" "Green"
                }
            }
        }
        
        # Delete local files not on remote
        if ($toDelete.Count -gt 0) {
            $hasChanges = $true
            Write-Color "[AUTOFETCH] Deleting $($toDelete.Count) local files not on remote" "Red"
            foreach ($relativePath in $toDelete) {
                $localPath = Join-Path $Config.LocalPath ($relativePath.Replace("/", "\"))
                if (Test-Path $localPath) {
                    Remove-Item $localPath -Force
                    Write-Color "  [DELETE] $relativePath" "Red"
                }
            }
        }
        
        if (-not $hasChanges) {
            Write-Color "[AUTOFETCH] $($targetServer.Name) - Already in sync" "Green"
        } else {
            Write-Color "[AUTOFETCH] Complete: $($toDownload.Count) downloaded, $($toDelete.Count) deleted" "Cyan"
        }
    }
    
    "vtkrename" {
        # VTK file auto-renamer: monitor and rename VTK files to use zero-padding
        $pidFile = Join-Path $Config.LocalPath ".vscode\vtk-renamer.pid"
        $logFile = Join-Path $Config.LocalPath ".vscode\vtk-renamer.log"
        $renamerScript = Join-Path $Config.LocalPath ".vscode\vtk-renamer.ps1"
        $subCommand = if ($Arguments.Count -gt 0) { $Arguments[0] } else { "" }
        
        switch ($subCommand) {
            "stop" {
                if (Test-Path $pidFile) {
                    $processId = Get-Content $pidFile -ErrorAction SilentlyContinue
                    if ($processId -match '^\d+$') {
                        $proc = Get-Process -Id $processId -ErrorAction SilentlyContinue
                        if ($proc) {
                            Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
                            Write-Color "[VTK-RENAMER] Stopped process $processId" "Yellow"
                        }
                    }
                    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
                    Write-Color "[VTK-RENAMER] VTK renamer stopped" "Green"
                }
                else {
                    Write-Color "[VTK-RENAMER] No active renamer process" "Yellow"
                }
            }
            
            "status" {
                Write-Color "`n=== VTK Renamer Status ===" "Cyan"
                if (Test-Path $pidFile) {
                    $processId = Get-Content $pidFile -ErrorAction SilentlyContinue
                    if ($processId -match '^\d+$') {
                        $proc = Get-Process -Id $processId -ErrorAction SilentlyContinue
                        if ($proc) {
                            Write-Color "[RUNNING] PID: $processId" "Green"
                            if (Test-Path $logFile) {
                                Write-Color "`nRecent Activity (last 15 lines):" "Yellow"
                                Get-Content $logFile -Tail 15 | ForEach-Object { Write-Host "  $_" }
                            }
                        }
                        else {
                            Write-Color "[STOPPED] No active process" "Yellow"
                            Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
                        }
                    }
                }
                else {
                    Write-Color "[STOPPED] VTK renamer is not running" "Yellow"
                }
                Write-Host ""
            }
            
            "log" {
                if (Test-Path $logFile) {
                    Write-Color "=== VTK Renamer Log ===" "Cyan"
                    Get-Content $logFile -Tail 50 | ForEach-Object { Write-Host $_ }
                }
                else {
                    Write-Color "No log file found" "Yellow"
                }
            }
            
            "clear" {
                if (Test-Path $logFile) {
                    Remove-Item $logFile -Force
                    Write-Color "[VTK-RENAMER] Log cleared" "Green"
                }
            }
            
            default {
                # Start VTK renamer
                if (Test-Path $pidFile) {
                    $processId = Get-Content $pidFile -ErrorAction SilentlyContinue
                    if ($processId -match '^\d+$') {
                        $proc = Get-Process -Id $processId -ErrorAction SilentlyContinue
                        if ($proc) {
                            Write-Color "[VTK-RENAMER] Already running (PID: $processId). Use 'mobaxterm vtkrename stop' first." "Yellow"
                            return
                        }
                    }
                }
                
                $checkInterval = 5
                if ($Arguments.Count -gt 0 -and $Arguments[0] -match '^\d+$') {
                    $checkInterval = [int]$Arguments[0]
                }
                
                Write-Color "[VTK-RENAMER] Starting VTK file auto-renamer..." "Magenta"
                Write-Color "  Watch Path: $($Config.LocalPath)\result" "White"
                Write-Color "  Check Interval: ${checkInterval}s" "White"
                Write-Color "  Log: $logFile" "Gray"
                Write-Color "`nThis will rename:" "Yellow"
                Write-Color "  velocity_merged_1001.vtk → velocity_merged_001001.vtk" "Cyan"
                Write-Color "`nCommands:" "Yellow"
                Write-Color "  mobaxterm vtkrename status  - Check status" "Gray"
                Write-Color "  mobaxterm vtkrename log     - View log" "Gray"
                Write-Color "  mobaxterm vtkrename stop    - Stop renamer" "Gray"
                Write-Host ""
                
                # Clear old log
                if (Test-Path $logFile) {
                    Remove-Item $logFile -Force -ErrorAction SilentlyContinue
                }
                
                # Start renamer process
                $proc = Start-Process -FilePath "powershell.exe" -ArgumentList @(
                    "-NoProfile", "-WindowStyle", "Hidden", "-ExecutionPolicy", "Bypass",
                    "-File", "`"$renamerScript`"",
                    "-WatchPath", "`"$($Config.LocalPath)`"",
                    "-CheckInterval", $checkInterval
                ) -PassThru
                
                Start-Sleep -Milliseconds 500
                
                # Save PID
                $proc.Id | Out-File $pidFile -Force
                Write-Color "[STARTED] VTK renamer (PID: $($proc.Id))" "Green"
                
                Write-Color "`n[VTK-RENAMER] Background monitoring started!" "Green"
                Write-Color "Use 'mobaxterm vtkrename status' to check progress" "Cyan"
            }
        }
    }
    
    default {
        Write-Host ""
        Write-Host "MobaXterm Sync Commands (Git-like)" -ForegroundColor Cyan
        Write-Host "===================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Usage: mobaxterm <command>" -ForegroundColor White
        Write-Host ""
        Write-Host "=== Sync Command Summary ===" -ForegroundColor Yellow
        Write-Host "  FETCH series: Download + DELETE local (sync local to remote)" -ForegroundColor Red
        Write-Host "  PULL series:  Download only (no delete)" -ForegroundColor Green
        Write-Host "  PUSH series:  Upload + DELETE remote (sync remote to local)" -ForegroundColor Red
        Write-Host ""
        Write-Host "Download Commands (from remote):" -ForegroundColor Yellow
        Write-Host "  pull        - Download only, NO delete (safe)" -ForegroundColor Green
        Write-Host "  pull .87    - Pull from .87 only"
        Write-Host "  pull .154   - Pull from .154 only"
        Write-Host "  autopull    - Auto-pull if changes detected (no delete)"
        Write-Host "  watchpull   - Background auto-download (no delete)"
        Write-Host ""
        Write-Host "  fetch       - Download + DELETE local files not on remote" -ForegroundColor Red
        Write-Host "  fetch .87   - Fetch from .87 only"
        Write-Host "  fetch .154  - Fetch from .154 only"
        Write-Host "  autofetch   - Auto-fetch if changes detected (with delete)"
        Write-Host "  watchfetch  - Background auto-fetch (download + delete)"
        Write-Host ""
        Write-Host "Upload Commands (to remote):" -ForegroundColor Yellow
        Write-Host "  push        - Upload + DELETE remote files not in local" -ForegroundColor Red
        Write-Host "  autopush    - Auto-push if changes detected (with delete)"
        Write-Host "  watchpush   - Background auto-upload (with delete)"
        Write-Host ""
        Write-Host "Status & Info:" -ForegroundColor Yellow
        Write-Host "  status      - Show sync status"
        Write-Host "  diff        - Compare local vs remote"
        Write-Host "  log         - View remote log files"
        Write-Host "  bgstatus    - Check ALL background processes (push/pull/fetch/vtkrename)"
        Write-Host "  syncstatus  - Check sync background status (push/pull only)"
        Write-Host ""
        Write-Host "Other Commands:" -ForegroundColor Yellow
        Write-Host "  clone       - Full download from remote"
        Write-Host "  reset       - Delete remote-only files (no upload)"
        Write-Host "  sync        - Interactive: diff -> confirm -> push"
        Write-Host "  issynced    - Quick one-line status check"
        Write-Host ""
        Write-Host "VTK File Management:" -ForegroundColor Yellow
        Write-Host "  vtkrename         - Auto-rename VTK files to zero-padded format"
        Write-Host "  vtkrename status  - Check renamer status"
        Write-Host "  vtkrename log     - View rename log"
        Write-Host "  vtkrename stop    - Stop auto-renamer"
        Write-Host "  (Renames: velocity_merged_1001.vtk -> velocity_merged_001001.vtk)" -ForegroundColor Gray
        Write-Host ""
        Write-Host "Background Commands:" -ForegroundColor Yellow
        Write-Host "  watchpull/watchfetch/watchpush [.87|.154]  - Start monitoring"
        Write-Host "  watchpull/watchfetch/watchpush status      - Check status"
        Write-Host "  watchpull/watchfetch/watchpush log         - View log"
        Write-Host "  watchpull/watchfetch/watchpush stop        - Stop monitoring"
        Write-Host ""
        Write-Host "Aliases:" -ForegroundColor Yellow
        Write-Host "  pull87/pull154   - Same as pull .87/.154"
        Write-Host "  fetch87/fetch154 - Same as fetch .87/.154"
        Write-Host "  check            - Same as diff"
        Write-Host ""
        Write-Host "Examples:" -ForegroundColor Yellow
        Write-Host "  mobaxterm pull .87      # Safe download (no delete)"
        Write-Host "  mobaxterm fetch .87     # Download + delete local extras"
        Write-Host "  mobaxterm push          # Upload + delete remote extras"
        Write-Host "  mobaxterm watchpull     # Background safe download"
        Write-Host "  mobaxterm watchfetch    # Background sync (with delete)"
        Write-Host ""
    }
}

