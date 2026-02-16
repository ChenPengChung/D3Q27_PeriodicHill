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

# ========== Auto-setup 'mobaxterm' alias ==========
function Auto-SetupAlias {
    $scriptPath = $PSCommandPath

    # Ensure profile directory exists
    $profileDir = Split-Path -Parent $PROFILE
    if (-not (Test-Path $profileDir)) {
        New-Item -ItemType Directory -Path $profileDir -Force | Out-Null
    }

    # Ensure profile file exists
    if (-not (Test-Path $PROFILE)) {
        New-Item -ItemType File -Path $PROFILE -Force | Out-Null
    }

    # Check if alias already exists
    $profileContent = Get-Content $PROFILE -Raw -ErrorAction SilentlyContinue
    if (-not $profileContent -or -not $profileContent.Contains("function mobaxterm")) {
        $aliasCode = @"

# MobaXterm alias (auto-added)
function mobaxterm { & '$scriptPath' @args }
"@
        Add-Content -Path $PROFILE -Value $aliasCode
        Write-Host "[AUTO-SETUP] Added 'mobaxterm' function to $PROFILE" -ForegroundColor Green
        Write-Host "             Run '. `$PROFILE' or restart PowerShell to use." -ForegroundColor Gray
        Write-Host ""
    }
}

# Run auto-setup on first use (only in interactive console)
if ([Environment]::UserInteractive -and $Host.Name -eq 'ConsoleHost') {
    Auto-SetupAlias
}
# ========== End Auto-setup ==========

# Configuration
# Auto-detect: LocalPath from script location (cross-platform)
$_scriptDir = Split-Path -Parent $PSCommandPath
$_workspaceDir = Split-Path -Parent $_scriptDir
$_localFolderName = Split-Path -Leaf $_workspaceDir
$_isWindows = ($PSVersionTable.PSEdition -eq 'Desktop') -or ($IsWindows -eq $true)

$script:Config = @{
    LocalPath = $_workspaceDir
    RemotePath = "/home/chenpengchung/$_localFolderName"
    Servers = @(
        @{ Name = ".87"; Host = "140.114.58.87"; User = "chenpengchung"; Password = "1256" },
        @{ Name = ".89"; Host = "140.114.58.89"; User = "chenpengchung"; Password = "1256" },
        @{ Name = ".154"; Host = "140.114.58.154"; User = "chenpengchung"; Password = "1256" }
    )
    # 節點定義: Server -> Node -> GPU 類型
    Nodes = @{
        "89" = @(
            @{ Node = "0"; Label = ".89 direct"; GpuType = "V100-32G"; Description = "8x Tesla V100-SXM2-32GB" }
        )
        "87" = @(
            @{ Node = "2"; Label = ".87->ib2"; GpuType = "P100-16G"; Description = "8x Tesla P100-PCIE-16GB" },
            @{ Node = "3"; Label = ".87->ib3"; GpuType = "P100-16G"; Description = "8x Tesla P100-PCIE-16GB" },
            @{ Node = "5"; Label = ".87->ib5"; GpuType = "P100-16G"; Description = "8x Tesla P100-PCIE-16GB" },
            @{ Node = "6"; Label = ".87->ib6"; GpuType = "V100-16G"; Description = "8x Tesla V100-SXM2-16GB" }
        )
        "154" = @(
            @{ Node = "1"; Label = ".154->ib1"; GpuType = "P100-16G"; Description = "8x Tesla P100-PCIE-16GB" },
            @{ Node = "4"; Label = ".154->ib4"; GpuType = "P100-16G"; Description = "8x Tesla P100-PCIE-16GB" },
            @{ Node = "7"; Label = ".154->ib7"; GpuType = "P100-16G"; Description = "8x Tesla P100-PCIE-16GB" },
            @{ Node = "9"; Label = ".154->ib9"; GpuType = "P100-16G"; Description = "8x Tesla P100-PCIE-16GB" }
        )
    }
    NvccArch = "sm_35"
    MpiInclude = "/home/chenpengchung/openmpi-3.0.3/include"
    MpiLib = "/home/chenpengchung/openmpi-3.0.3/lib"
    DefaultGpuCount = 4
    IsWindows = $_isWindows
    PscpPath = if ($_isWindows) { "C:\Program Files\PuTTY\pscp.exe" } else { $null }
    PlinkPath = if ($_isWindows) { "C:\Program Files\PuTTY\plink.exe" } else { $null }
    SshPassword = "1256"
    SshOpts = "-o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new"
    # 排除的檔案，例如 .git 和 .vscode 設定檔等
    ExcludePatterns = @(".git/*", ".vscode/*", "a.out", "*.o", "*.exe",
        "*.swp", "*.swo", "*~", "__pycache__/*", "*.pyc", ".DS_Store", "*.vtu")
    # 同步的副檔名
    SyncExtensions = @("*")
    SyncAll = $true  # 同步所有類型的檔案
}

# ========== 跨平台 SSH/SCP 封裝 ==========
# Windows: plink/pscp (PuTTY)    macOS/Linux: sshpass + ssh/scp

function Invoke-Ssh {
    <#
    .SYNOPSIS
    Cross-platform SSH command execution wrapper.
    On Windows uses plink, on macOS/Linux uses sshpass+ssh.
    #>
    param(
        [hashtable]$Server,
        [string]$Command,
        [switch]$Batch,
        [switch]$Interactive,
        [string]$TtyCommand  # for -t option (interactive SSH sessions)
    )
    if ($Config.IsWindows) {
        if ($TtyCommand) {
            & $Config.PlinkPath -ssh -pw $Server.Password "$($Server.User)@$($Server.Host)" -t $TtyCommand
        } elseif ($Interactive) {
            & $Config.PlinkPath -ssh -pw $Server.Password "$($Server.User)@$($Server.Host)" $Command
        } else {
            & $Config.PlinkPath -ssh -pw $Server.Password -batch "$($Server.User)@$($Server.Host)" $Command 2>$null
        }
    } else {
        # macOS/Linux: use sshpass + ssh
        $sshOpts = $Config.SshOpts
        if ($TtyCommand) {
            sshpass -p $Server.Password ssh $sshOpts.Split(' ') -tt "$($Server.User)@$($Server.Host)" $TtyCommand
        } elseif ($Interactive) {
            sshpass -p $Server.Password ssh $sshOpts.Split(' ') -tt "$($Server.User)@$($Server.Host)" $Command
        } else {
            sshpass -p $Server.Password ssh $sshOpts.Split(' ') -o BatchMode=no "$($Server.User)@$($Server.Host)" $Command 2>$null
        }
    }
}

function Invoke-Scp {
    <#
    .SYNOPSIS
    Cross-platform SCP wrapper with retry logic. Direction: "upload" or "download".
    On Windows uses pscp, on macOS/Linux uses sshpass+scp.
    Retries up to 3 times on failure with 3-second backoff.
    #>
    param(
        [string]$Direction,      # "upload" or "download"
        [hashtable]$Server,
        [string]$LocalPath,
        [string]$RemotePath      # e.g. user@host:/path or just /remote/path
    )
    $maxRetries = 3
    for ($attempt = 1; $attempt -le $maxRetries; $attempt++) {
        if ($Config.IsWindows) {
            if ($Direction -eq "upload") {
                $remoteDest = "$($Server.User)@$($Server.Host):$RemotePath"
                $null = & $Config.PscpPath -pw $Server.Password -q -p $LocalPath $remoteDest 2>&1
            } else {
                $remoteSrc = "$($Server.User)@$($Server.Host):$RemotePath"
                $null = & $Config.PscpPath -pw $Server.Password -q -p $remoteSrc $LocalPath 2>&1
            }
        } else {
            # macOS/Linux: use sshpass + scp (-p preserves mtime/permissions)
            $sshOpts = $Config.SshOpts
            if ($Direction -eq "upload") {
                $remoteDest = "$($Server.User)@$($Server.Host):$RemotePath"
                $null = sshpass -p $Server.Password scp $sshOpts.Split(' ') -q -p $LocalPath $remoteDest 2>&1
            } else {
                $remoteSrc = "$($Server.User)@$($Server.Host):$RemotePath"
                $null = sshpass -p $Server.Password scp $sshOpts.Split(' ') -q -p $remoteSrc $LocalPath 2>&1
            }
        }
        if ($LASTEXITCODE -eq 0) { return }
        if ($attempt -lt $maxRetries) {
            Write-Host "  ⚠ SCP failed (attempt $attempt/$maxRetries), retrying in 3s..." -ForegroundColor Yellow
            Start-Sleep -Seconds 3
        }
    }
}

function Write-Color {
    param([string]$Text, [string]$Color = "White")
    Write-Host $Text -ForegroundColor $Color
}

# ========== GPU 相關輔助函數 ==========

function Get-ServerByName {
    param([string]$Name)
    $name = $Name.TrimStart(".")
    foreach ($s in $Config.Servers) {
        if ($s.Name -eq ".$name" -or $s.Name -eq $name) { return $s }
    }
    return $null
}

function Parse-GpuOutput {
    param([string]$RawOutput)
    if (-not $RawOutput -or $RawOutput -eq "OFFLINE" -or $RawOutput -match "error|fail") {
        return @{ Dots = @(); Free = 0; Total = 0; Offline = $true; Details = @() }
    }
    $dots = @(); $free = 0; $total = 0; $details = @()
    foreach ($line in $RawOutput -split "`n") {
        $line = $line.Trim()
        if (-not $line) { continue }
        $parts = $line -split ","
        if ($parts.Count -lt 2) { continue }
        $idx = $parts[0].Trim()
        $util = ($parts[1].Trim() -replace '[^0-9]','')
        if (-not $util) { continue }
        $total++
        $utilInt = [int]$util
        if ($utilInt -lt 10) {
            $free++
            $dots += "G"  # Green = free
            $details += @{ Index = $idx; Util = $utilInt; Free = $true }
        } else {
            $dots += "R"  # Red = busy
            $details += @{ Index = $idx; Util = $utilInt; Free = $false }
        }
    }
    if ($total -eq 0) { return @{ Dots = @(); Free = 0; Total = 0; Offline = $true; Details = @() } }
    return @{ Dots = $dots; Free = $free; Total = $total; Offline = $false; Details = $details }
}

function Query-GpuStatus {
    param(
        [string]$ServerKey,
        [string]$NodeNum
    )
    $server = Get-ServerByName $ServerKey
    if (-not $server) { return "OFFLINE" }

    try {
        if ($NodeNum -eq "0") {
            # 直連模式
            $cmd = "nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader"
            $result = Invoke-Ssh -Server $server -Command $cmd
        } else {
            # 跳板模式
            $cmd = "ssh -o ConnectTimeout=5 cfdlab-ib$NodeNum 'nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader'"
            $result = Invoke-Ssh -Server $server -Command $cmd
        }
        if ($result) { return ($result -join "`n") } else { return "OFFLINE" }
    } catch {
        return "OFFLINE"
    }
}

function Run-RemoteCommand {
    param(
        [string]$ServerKey,
        [string]$NodeNum,
        [string]$Command
    )
    $server = Get-ServerByName $ServerKey
    if (-not $server) {
        Write-Color "[ERROR] Unknown server: $ServerKey" "Red"
        return
    }

    if ($NodeNum -eq "0") {
        # 直連模式
        $fullCmd = "cd $($Config.RemotePath) && $Command"
        Invoke-Ssh -Server $server -Command $fullCmd
    } else {
        # 跳板模式
        $fullCmd = "ssh cfdlab-ib$NodeNum 'cd $($Config.RemotePath) && $Command'"
        Invoke-Ssh -Server $server -Command $fullCmd
    }
}

function Ensure-RemoteDir {
    param([hashtable]$Server)
    $cmd = "mkdir -p '$($Config.RemotePath)'"
    Invoke-Ssh -Server $Server -Command $cmd | Out-Null
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
    
    # 排除 .git 和 .vscode 目錄
    $excludeGrep = "grep -v '/.git/' | grep -v '/.vscode/' | grep -v '\.o$' | grep -v '\.swp$' | grep -v '\.swo$' | grep -v '~$' | grep -v '__pycache__' | grep -v '\.pyc$' | grep -v '\.DS_Store' | grep -v '\.vtu$'"
    $cmd = "find $($Config.RemotePath) -type f -exec md5sum {} \; 2>/dev/null | $excludeGrep"
    $result = Invoke-Ssh -Server $Server -Command $cmd
    
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

# ========== Git-style Progress Transfer Functions ==========

function Get-ServerEmoji {
    param([string]$Name)
    switch ($Name) {
        ".87"  { "🟢" }
        ".89"  { "🔵" }
        ".154" { "🟡" }
        default { "⚪" }
    }
}

function Get-ServerColor {
    param([string]$Name)
    switch ($Name) {
        ".87"  { "Green" }
        ".89"  { "Blue" }
        ".154" { "Yellow" }
        default { "White" }
    }
}

function Write-TransferHeader {
    param([hashtable]$Server, [string]$Verb, [int]$Index = 1, [int]$Total = 1)
    $color = Get-ServerColor $Server.Name
    $emoji = Get-ServerEmoji $Server.Name
    Write-Host "══════════════════════════════════════════════════" -ForegroundColor $color
    Write-Host " [$Index/$Total] $emoji $Verb $($Server.Name) ($($Server.Host))" -ForegroundColor $color
    Write-Host "══════════════════════════════════════════════════" -ForegroundColor $color
}

function Format-ProgressLine {
    <# Build a progress line with bar, %, count, speed, ETA, and filename #>
    param(
        [string]$Verb, [int]$Pct, [int]$Current, [int]$Total,
        [string]$FileName, [datetime]$PhaseStart, [bool]$IsInteractive
    )
    # Progress bar (20 chars)
    $barWidth = 20
    $filled = [math]::Floor($Pct * $barWidth / 100)
    $empty = $barWidth - $filled
    $bar = ("━" * $filled) + ("─" * $empty)

    # Speed & ETA
    $elapsed = ((Get-Date) - $PhaseStart).TotalSeconds
    $speedStr = "--"
    $etaStr = "--"
    if ($elapsed -gt 0 -and $Current -gt 0) {
        $fps = $Current / $elapsed
        $speedStr = "{0:F1} files/s" -f $fps
        $remaining = $Total - $Current
        if ($fps -gt 0) {
            $etaSec = [math]::Ceiling($remaining / $fps)
            if ($etaSec -ge 60) {
                $etaStr = "{0}m{1}s" -f [math]::Floor($etaSec / 60), ($etaSec % 60)
            } else {
                $etaStr = "${etaSec}s"
            }
        }
    }

    # Truncate filename if too long
    $maxNameLen = 30
    $shortName = if ($FileName.Length -gt $maxNameLen) { "..." + $FileName.Substring($FileName.Length - $maxNameLen + 3) } else { $FileName }

    if ($IsInteractive) {
        return "`r$bar $($Pct.ToString().PadLeft(3))% ($Current/$Total) $speedStr | ⏱ $etaStr  📁 $shortName".PadRight(100)
    } else {
        return "[$($Pct.ToString().PadLeft(3))%] ($Current/$Total) $FileName"
    }
}

function Write-TransferSummary {
    <# Print final stats line #>
    param([datetime]$StartTime, [int]$FileCount)
    $elapsed = (Get-Date) - $StartTime
    $speedStr = ""
    if ($elapsed.TotalSeconds -gt 0 -and $FileCount -gt 0) {
        $fps = $FileCount / $elapsed.TotalSeconds
        $speedStr = " | {0:F1} files/s" -f $fps
    }
    Write-Host ("Transfer complete. [{0:mm\:ss}]{1} ✔" -f $elapsed, $speedStr)
}

# ========== Sync History Logging ==========
$script:SyncHistoryFile = Join-Path $HOME ".sync-history.log"

function Write-SyncHistory {
    param([string]$Action, [string]$ServerName, [int]$FileCount, [int]$Adds, [int]$Dels)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm"
    $entry = "[$timestamp] ${Action} ${ServerName}: $FileCount file(s), +$Adds -$Dels lines"
    try { Add-Content -Path $script:SyncHistoryFile -Value $entry -ErrorAction Stop } catch {}
}

# ========== Code Diff Analysis Functions (GitHub-style) ==========

# Diff configuration
$script:DiffConfig = @{
    ContextLines = 3
    MaxFileSize = 10485760        # 10 MB
    NewFilePreviewLines = 20
    SmallDataMaxLines = 50
    MaxFilesFull = 100
    IgnoreWhitespace = $true
    SafetyDeleteWarn = 10
    SafetyLinesWarn = 1000
}

# Load .sync-config if exists
$_syncConfigPath = Join-Path $Config.LocalPath ".sync-config"
if (Test-Path $_syncConfigPath) {
    Get-Content $_syncConfigPath | Where-Object { $_ -match '^\s*(\w+)\s*=\s*(.+)$' -and $_ -notmatch '^\s*#' } | ForEach-Object {
        $key = $Matches[1].Trim()
        $val = $Matches[2].Trim().Trim('"')
        switch ($key) {
            'DIFF_CONTEXT_LINES'          { $script:DiffConfig.ContextLines = [int]$val }
            'DIFF_MAX_FILE_SIZE'          { $script:DiffConfig.MaxFileSize = [int]$val }
            'DIFF_NEW_FILE_PREVIEW_LINES' { $script:DiffConfig.NewFilePreviewLines = [int]$val }
            'DIFF_SMALL_DATA_MAX_LINES'   { $script:DiffConfig.SmallDataMaxLines = [int]$val }
            'DIFF_MAX_FILES_FULL'         { $script:DiffConfig.MaxFilesFull = [int]$val }
            'DIFF_IGNORE_WHITESPACE'      { $script:DiffConfig.IgnoreWhitespace = ($val -eq 'true') }
            'SAFETY_DELETE_WARN'          { $script:DiffConfig.SafetyDeleteWarn = [int]$val }
            'SAFETY_LINES_WARN'           { $script:DiffConfig.SafetyLinesWarn = [int]$val }
            'SYNC_HISTORY_FILE'           { $script:SyncHistoryFile = $val }
        }
    }
}

# Text file extensions for full diff
$script:TextExtensions = @(".cpp",".c",".h",".hpp",".cu",".cuh",".py",".sh",".zsh",".ps1",
    ".f90",".f95",".f03",".mk",".txt",".md",".rst",".json",".yaml",".yml",".toml")

function Test-TextFile {
    param([string]$FileName)
    $base = [System.IO.Path]::GetFileName($FileName)
    if ($base -eq "Makefile") { return $true }
    $ext = [System.IO.Path]::GetExtension($FileName).ToLower()
    return ($ext -in $script:TextExtensions)
}

function Test-BinaryFile {
    param([string]$FileName)
    $ext = [System.IO.Path]::GetExtension($FileName).ToLower()
    return ($ext -in @(".o",".out",".exe",".bin",".vtk",".vtu",".dat",".plt",".png",".jpg",".gif",".pdf",".zip",".tar",".gz",".bz2"))
}

function Format-FileSize {
    param([long]$Bytes)
    if ($Bytes -ge 1GB) { return "{0:F1} GB" -f ($Bytes / 1GB) }
    if ($Bytes -ge 1MB) { return "{0:F1} MB" -f ($Bytes / 1MB) }
    if ($Bytes -ge 1KB) { return "{0:F1} KB" -f ($Bytes / 1KB) }
    return "$Bytes B"
}

function Get-RemoteFileContent {
    param([hashtable]$Server, [string]$RelativePath)
    $cmd = "cat '$($Config.RemotePath)/$RelativePath'"
    $result = Invoke-Ssh -Server $Server -Command $cmd
    return $result
}

function Get-RemoteFileSize {
    param([hashtable]$Server, [string]$RelativePath)
    $cmd = "stat -c%s '$($Config.RemotePath)/$RelativePath' 2>/dev/null || echo 0"
    $result = Invoke-Ssh -Server $Server -Command $cmd
    if ($result) { return [long]($result | Select-Object -First 1) }
    return 0
}

# Show diff for a single file — returns hashtable { Adds; Dels }
function Show-FileDiff {
    param(
        [string]$Direction,     # push or pull
        [hashtable]$Server,
        [string]$FilePath,
        [string]$FileStatus,    # new / modified / deleted
        [string]$Mode           # full / summary / stat
    )
    $localFile = Join-Path $Config.LocalPath ($FilePath.Replace("/", [System.IO.Path]::DirectorySeparatorChar))
    $adds = 0; $dels = 0

    # ── New file ──
    if ($FileStatus -eq "new") {
        if ($Mode -eq "stat") {
            if (Test-Path $localFile) { $adds = (Get-Content $localFile -ErrorAction SilentlyContinue | Measure-Object).Count }
            Write-Host ("📄 {0,-50} " -f $FilePath) -NoNewline -ForegroundColor Blue
            Write-Host "+$adds" -ForegroundColor Green -NoNewline
            Write-Host " (new file)"
            return @{ Adds = $adds; Dels = 0 }
        }
        if (Test-TextFile $FilePath) {
            if (Test-Path $localFile) {
                $lines = Get-Content $localFile -ErrorAction SilentlyContinue
                $adds = ($lines | Measure-Object).Count
            }
            Write-Host ""
            Write-Host "📄 $FilePath" -ForegroundColor Blue -NoNewline
            Write-Host "  +$adds -0 (new file)" -ForegroundColor Green
            Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
            Write-Host "@@ -0,0 +1,$adds @@" -ForegroundColor Yellow
            $preview = $lines | Select-Object -First $script:DiffConfig.NewFilePreviewLines
            foreach ($l in $preview) { Write-Host "+$l" -ForegroundColor Green }
            if ($adds -gt $script:DiffConfig.NewFilePreviewLines) {
                $remaining = $adds - $script:DiffConfig.NewFilePreviewLines
                Write-Host "... ($remaining more lines)" -ForegroundColor DarkGray
            }
        } else {
            $fsize = if (Test-Path $localFile) { (Get-Item $localFile).Length } else { 0 }
            Write-Host ""
            Write-Host "📄 $FilePath (new file, $(Format-FileSize $fsize))" -ForegroundColor Blue
            $adds = 1
        }
        return @{ Adds = $adds; Dels = 0 }
    }

    # ── Deleted file ──
    if ($FileStatus -eq "deleted") {
        if ($Mode -eq "stat") {
            Write-Host ("📄 {0,-50} " -f $FilePath) -NoNewline -ForegroundColor Blue
            Write-Host "(deleted)" -ForegroundColor Red
            return @{ Adds = 0; Dels = 0 }
        }
        Write-Host ""
        Write-Host "📄 $FilePath" -ForegroundColor Blue -NoNewline
        Write-Host " (will be deleted)" -ForegroundColor Red
        return @{ Adds = 0; Dels = 0 }
    }

    # ── Modified file ──
    if (Test-BinaryFile $FilePath) {
        $localSize = if (Test-Path $localFile) { (Get-Item $localFile).Length } else { 0 }
        $remoteSize = Get-RemoteFileSize -Server $Server -RelativePath $FilePath
        Write-Host ""
        Write-Host "📄 $FilePath" -ForegroundColor Blue -NoNewline
        Write-Host " (binary: $(Format-FileSize $remoteSize) → $(Format-FileSize $localSize))" -ForegroundColor DarkGray
        return @{ Adds = 0; Dels = 0 }
    }

    if (-not (Test-TextFile $FilePath)) {
        $fsize = if (Test-Path $localFile) { (Get-Item $localFile).Length } else { 0 }
        if ($fsize -gt $script:DiffConfig.MaxFileSize) {
            $remoteSize = Get-RemoteFileSize -Server $Server -RelativePath $FilePath
            Write-Host ""
            Write-Host "📄 $FilePath" -ForegroundColor Blue -NoNewline
            Write-Host " (large: $(Format-FileSize $remoteSize) → $(Format-FileSize $fsize))" -ForegroundColor DarkGray
            return @{ Adds = 0; Dels = 0 }
        }
    }

    # Fetch remote content for comparison
    $remoteContent = Get-RemoteFileContent -Server $Server -RelativePath $FilePath
    if (-not $remoteContent) { $remoteContent = @() }
    $localContent = if (Test-Path $localFile) { Get-Content $localFile -ErrorAction SilentlyContinue } else { @() }
    if (-not $localContent) { $localContent = @() }

    # Determine old vs new based on direction
    if ($Direction -eq "push") {
        $oldContent = $remoteContent; $newContent = $localContent
    } else {
        $oldContent = $localContent; $newContent = $remoteContent
    }

    # Write to temp files and run diff
    $tmpOld = [System.IO.Path]::GetTempFileName()
    $tmpNew = [System.IO.Path]::GetTempFileName()
    try {
        $oldContent | Set-Content -Path $tmpOld -Encoding UTF8 -ErrorAction Stop
        $newContent | Set-Content -Path $tmpNew -Encoding UTF8 -ErrorAction Stop
    } catch {
        if (Test-Path $tmpOld) { Remove-Item $tmpOld -Force }
        if (Test-Path $tmpNew) { Remove-Item $tmpNew -Force }
        return @{ Adds = 0; Dels = 0 }
    }

    $diffArgs = @("-u", "--label", "a/$FilePath", "--label", "b/$FilePath")
    if ($script:DiffConfig.IgnoreWhitespace) { $diffArgs += "-w" }
    $diffArgs += @($tmpOld, $tmpNew)

    $diffOutput = $null
    try { $diffOutput = & diff @diffArgs 2>$null } catch {}

    Remove-Item $tmpOld, $tmpNew -Force -ErrorAction SilentlyContinue

    if (-not $diffOutput) {
        if ($Mode -eq "stat") {
            Write-Host ("📄 {0,-50} " -f $FilePath) -NoNewline -ForegroundColor Blue
            Write-Host "(whitespace only)" -ForegroundColor DarkGray
        }
        return @{ Adds = 0; Dels = 0 }
    }

    # Count +/- lines
    foreach ($line in $diffOutput) {
        if ($line -match '^\+[^+]') { $adds++ }
        elseif ($line -match '^\-[^-]') { $dels++ }
    }

    if ($Mode -eq "stat") {
        Write-Host ("📄 {0,-50} " -f $FilePath) -NoNewline -ForegroundColor Blue
        Write-Host "+$adds" -ForegroundColor Green -NoNewline
        Write-Host " -$dels" -ForegroundColor Red
        return @{ Adds = $adds; Dels = $dels }
    }

    # Full diff output
    Write-Host ""
    Write-Host "📄 $FilePath" -ForegroundColor Blue -NoNewline
    Write-Host ("  +$adds" ) -ForegroundColor Green -NoNewline
    Write-Host " -$dels" -ForegroundColor Red
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray

    $lineNo = 0
    $dataLines = 0
    $isText = Test-TextFile $FilePath
    foreach ($line in $diffOutput) {
        $lineNo++
        if ($lineNo -le 2) { continue }  # Skip --- +++ headers

        if (-not $isText) {
            $dataLines++
            if ($dataLines -gt $script:DiffConfig.SmallDataMaxLines) {
                Write-Host "... (truncated, showing first $($script:DiffConfig.SmallDataMaxLines) diff lines)" -ForegroundColor DarkGray
                break
            }
        }

        if ($line -match '^@@') { Write-Host $line -ForegroundColor Yellow }
        elseif ($line -match '^\+')  { Write-Host $line -ForegroundColor Green }
        elseif ($line -match '^\-')  { Write-Host $line -ForegroundColor Red }
        else { Write-Host $line -ForegroundColor DarkGray }
    }

    return @{ Adds = $adds; Dels = $dels }
}

# Full code diff analysis for a server before push/pull/fetch
# Returns $true if user confirms (or no confirm needed), $false if cancelled
function Invoke-CodeDiffAnalysis {
    param(
        [string]$Direction,     # push / pull / fetch
        [hashtable]$Server,
        [string]$Mode = "full", # full / summary / stat / no-diff
        [bool]$Confirm = $false
    )

    if ($Mode -eq "no-diff") { return $true }

    Write-Host ""
    Write-Host "🔍 Analyzing changes before $Direction..." -ForegroundColor White
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray

    # Get changed files
    $results = Compare-Files -Server $Server -Silent
    $newFiles = @(); $modifiedFiles = @(); $deletedFiles = @()

    if ($Direction -eq "push") {
        $newFiles = @($results.New)
        $modifiedFiles = @($results.Modified)
        $deletedFiles = @($results.Deleted) # remote-only → will be deleted on remote
    } else {
        $newFiles = @($results.Deleted)      # remote-only → new for local
        $modifiedFiles = @($results.Modified)
        $deletedFiles = @($results.New)      # local-only → will be deleted
        if ($Direction -eq "fetch") { $deletedFiles = @() } # fetch doesn't delete
    }

    $totalNew = $newFiles.Count
    $totalMod = $modifiedFiles.Count
    $totalDel = $deletedFiles.Count
    $totalFiles = $totalNew + $totalMod + $totalDel
    $script:DiffFileCount = $totalFiles

    if ($totalFiles -eq 0) {
        Write-Host ""
        Write-Host "  Already up to date." -ForegroundColor DarkGray
        Write-Host ""
        return $true
    }

    # Changes Summary
    Write-Host ""
    Write-Host "📊 Changes Summary:" -ForegroundColor White
    if ($totalMod -gt 0) { Write-Host "   📝 Modified: $totalMod files" }
    if ($totalNew -gt 0) { Write-Host "   ✨ New: $totalNew files" }
    if ($totalDel -gt 0) { Write-Host "   🗑️  Deleted: $totalDel files" }
    Write-Host ""

    # Safety warnings
    if ($totalDel -ge $script:DiffConfig.SafetyDeleteWarn) {
        Write-Host "⚠️  WARNING: $totalDel files will be deleted!" -ForegroundColor Red
    }

    if ($totalFiles -gt $script:DiffConfig.MaxFilesFull) {
        Write-Host "(Too many files ($totalFiles) — showing stat only)" -ForegroundColor DarkGray
        Write-Host ""
        $Mode = "stat"
    }

    $totalAdds = 0; $totalDels = 0

    if ($Mode -ne "summary") {
        foreach ($fp in $modifiedFiles) {
            $r = Show-FileDiff -Direction $Direction -Server $Server -FilePath $fp -FileStatus "modified" -Mode $Mode
            if ($r) { $totalAdds += $r.Adds; $totalDels += $r.Dels }
        }
        foreach ($fp in $newFiles) {
            $r = Show-FileDiff -Direction $Direction -Server $Server -FilePath $fp -FileStatus "new" -Mode $Mode
            if ($r) { $totalAdds += $r.Adds; $totalDels += $r.Dels }
        }
        foreach ($fp in $deletedFiles) {
            $r = Show-FileDiff -Direction $Direction -Server $Server -FilePath $fp -FileStatus "deleted" -Mode $Mode
            if ($r) { $totalAdds += $r.Adds; $totalDels += $r.Dels }
        }
    }

    $script:DiffTotalAdds = $totalAdds
    $script:DiffTotalDels = $totalDels

    # Code Changes Statistics
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
    Write-Host "📊 Code Changes Statistics:" -ForegroundColor White
    if ($totalAdds -gt 0) { Write-Host "   📈 Lines added: $totalAdds (+)" }
    if ($totalDels -gt 0) { Write-Host "   📉 Lines removed: $totalDels (-)" }
    $net = $totalAdds - $totalDels
    Write-Host ("   📐 Net change: {0:+#;-#;0} lines" -f $net)

    if (($totalAdds + $totalDels) -ge $script:DiffConfig.SafetyLinesWarn) {
        Write-Host ""
        Write-Host "⚠️  Large change: $($totalAdds + $totalDels) total line changes" -ForegroundColor Yellow
    }

    # File type breakdown
    $extGroups = @{}
    foreach ($fp in ($modifiedFiles + $newFiles)) {
        $ext = [System.IO.Path]::GetExtension($fp)
        if (-not $ext) { $ext = "(no ext)" }
        if ($extGroups.ContainsKey($ext)) { $extGroups[$ext]++ } else { $extGroups[$ext] = 1 }
    }
    if ($extGroups.Count -gt 0) {
        Write-Host ""
        Write-Host "   📁 Changed files by type:"
        foreach ($ext in $extGroups.Keys | Sort-Object) {
            Write-Host "      ${ext}: $($extGroups[$ext]) file(s)"
        }
    }
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray

    # Confirmation
    if ($Confirm) {
        Write-Host "⚠️  Review changes above before $Direction" -ForegroundColor Yellow
        $answer = Read-Host "Continue with ${Direction}? [Y/n]"
        if ($answer -in @("n","N","no","NO")) {
            Write-Host "Cancelled."
            return $false
        }
    }
    return $true
}

# ========== Diff Option Parsing Helper ==========

function Parse-DiffOptions {
    param([string[]]$Args_)
    $result = @{
        DiffMode = "full"
        Confirm  = $false
        Positional = @()
    }
    foreach ($a in $Args_) {
        switch ($a) {
            "--no-diff"       { $result.DiffMode = "no-diff"; $result.Confirm = $false }
            "--diff-summary"  { $result.DiffMode = "summary" }
            "--diff-stat"     { $result.DiffMode = "stat" }
            "--diff-full"     { $result.DiffMode = "full" }
            "--force"         { $result.Confirm = $false; $result.DiffMode = "no-diff" }
            "--quick"         { $result.DiffMode = "no-diff"; $result.Confirm = $false }
            default           { $result.Positional += $a }
        }
    }
    return $result
}

# ========== End Code Diff Analysis ==========

# ========== Per-file Line Diff Stats (computed before transfer) ==========

function Compute-FileLineDiffs {
    <#
    .SYNOPSIS Compute per-file line additions/deletions for transfer summary.
    Returns hashtable: @{ "filepath" = @{Adds=N; Dels=M}; ... ; _TotalAdds=N; _TotalDels=M }
    #>
    param(
        [hashtable]$Server,
        [string]$Direction,           # push / pull / fetch
        [string[]]$NewFiles,
        [string[]]$ModifiedFiles,
        [string[]]$DeletedFiles
    )

    $stats = @{ _TotalAdds = 0; _TotalDels = 0 }
    $host_ = $Server.Host
    $total = $NewFiles.Count + $ModifiedFiles.Count + $DeletedFiles.Count
    $idx = 0

    foreach ($fp in $ModifiedFiles) {
        $idx++
        if (-not (Test-TextFile $fp)) { continue }

        try {
            $localPath = Join-Path $Config.LocalPath ($fp.Replace("/", [System.IO.Path]::DirectorySeparatorChar))
            $remoteContent = Get-RemoteFileContent -Server $Server -RemotePath $fp

            if ($remoteContent -and (Test-Path $localPath)) {
                $tmpRemote = [System.IO.Path]::GetTempFileName()
                $tmpLocal = $localPath
                [System.IO.File]::WriteAllText($tmpRemote, ($remoteContent -join "`n"))

                if ($Direction -eq "push") {
                    # Push: local is new, remote is old
                    $diffOutput = diff -u $tmpRemote $tmpLocal 2>$null
                } else {
                    # Pull/Fetch: remote is new, local is old
                    $diffOutput = diff -u $tmpLocal $tmpRemote 2>$null
                }

                $adds = 0; $dels = 0
                if ($diffOutput) {
                    foreach ($dline in ($diffOutput | Select-Object -Skip 2)) {
                        if ($dline -match '^\+' -and $dline -notmatch '^\+\+\+') { $adds++ }
                        elseif ($dline -match '^-' -and $dline -notmatch '^---') { $dels++ }
                    }
                }
                $stats[$fp] = @{ Adds = $adds; Dels = $dels }
                $stats._TotalAdds += $adds
                $stats._TotalDels += $dels

                Remove-Item $tmpRemote -Force -ErrorAction SilentlyContinue
            }
        } catch {}
    }

    foreach ($fp in $NewFiles) {
        if (-not (Test-TextFile $fp)) { continue }
        try {
            if ($Direction -eq "push") {
                $localPath = Join-Path $Config.LocalPath ($fp.Replace("/", [System.IO.Path]::DirectorySeparatorChar))
                if (Test-Path $localPath) {
                    $lc = (Get-Content $localPath -ErrorAction Stop).Count
                    $stats[$fp] = @{ Adds = $lc; Dels = 0 }
                    $stats._TotalAdds += $lc
                }
            } else {
                $lc = Invoke-Ssh -Server $Server -Command "wc -l < '${($Config.RemotePath)}/${fp}'" 2>$null
                $lc = [int]($lc -replace '\D','')
                if ($lc -gt 0) {
                    $stats[$fp] = @{ Adds = $lc; Dels = 0 }
                    $stats._TotalAdds += $lc
                }
            }
        } catch {}
    }

    foreach ($fp in $DeletedFiles) {
        if (-not (Test-TextFile $fp)) { continue }
        try {
            if ($Direction -eq "push") {
                $lc = Invoke-Ssh -Server $Server -Command "wc -l < '${($Config.RemotePath)}/${fp}'" 2>$null
                $lc = [int]($lc -replace '\D','')
            } else {
                $localPath = Join-Path $Config.LocalPath ($fp.Replace("/", [System.IO.Path]::DirectorySeparatorChar))
                $lc = if (Test-Path $localPath) { (Get-Content $localPath -ErrorAction Stop).Count } else { 0 }
            }
            if ($lc -gt 0) {
                $stats[$fp] = @{ Adds = 0; Dels = $lc }
                $stats._TotalDels += $lc
            }
        } catch {}
    }

    return $stats
}

function Format-LineStatSuffix {
    param([hashtable]$Stats, [string]$FilePath)
    if ($Stats.ContainsKey($FilePath)) {
        $a = $Stats[$FilePath].Adds; $d = $Stats[$FilePath].Dels
        $parts = @()
        if ($a -gt 0) { $parts += "+$a" }
        if ($d -gt 0) { $parts += "-$d" }
        if ($parts.Count -gt 0) { return " | $($parts -join ' ')" }
    }
    return ""
}

# ========== End Per-file Line Diff Stats ==========

function Invoke-GitStylePush {
    <#
    .SYNOPSIS Git-style push with real-time progress for a single server.
    #>
    param(
        [hashtable]$Server,
        [switch]$DeleteRemoteExtras
    )
    $startTime = Get-Date
    $isInteractive = ($Host.Name -eq 'ConsoleHost') -and [Environment]::UserInteractive

    # Phase 1: Enumerate objects
    if ($isInteractive) {
        Write-Host "remote: Enumerating objects..." -ForegroundColor DarkGray -NoNewline
    } else {
        Write-Host "remote: Enumerating objects..." -ForegroundColor DarkGray
    }

    $results = Compare-Files -Server $Server -Silent
    $toUpload = @()
    $toUpload += $results.New
    $toUpload += $results.Modified
    $toDelete = @()
    if ($DeleteRemoteExtras) { $toDelete += $results.Deleted }
    $totalFiles = $toUpload.Count + $toDelete.Count

    if ($isInteractive) {
        Write-Host "`rremote: Enumerating objects: $totalFiles, done.       " -ForegroundColor DarkGray
    } else {
        Write-Host "remote: Enumerating objects: $totalFiles, done." -ForegroundColor DarkGray
    }

    if ($totalFiles -eq 0) {
        Write-Host "  Already up to date."
        $elapsed = (Get-Date) - $startTime
        Write-Host ("Transfer complete. [{0:mm\:ss}] ✔" -f $elapsed)
        return
    }

    # Phase 1.5: Compute per-file line diffs
    Write-Host "remote: Computing line changes..." -ForegroundColor DarkGray -NoNewline
    $lineStats = Compute-FileLineDiffs -Server $Server -Direction "push" `
        -NewFiles @($results.New) -ModifiedFiles @($results.Modified) -DeletedFiles @($toDelete)
    Write-Host "`rremote: Computing line changes: done.       " -ForegroundColor DarkGray

    # Phase 2: Transfer with progress
    $localFiles = Get-LocalFiles
    $localHash = @{}
    foreach ($f in $localFiles) { $localHash[$f.Path] = $f }
    $current = 0
    $newCount = 0; $modCount = 0; $delCount = 0
    $successCount = 0
    $formattedLines = @()
    $phaseStart = Get-Date

    Ensure-RemoteDir $Server

    foreach ($path in $toUpload) {
        $current++
        $file = $localHash[$path]
        if (-not $file) { continue }
        $pct = [math]::Floor($current * 100 / $totalFiles)

        $line = Format-ProgressLine -Verb "Uploading" -Pct $pct -Current $current -Total $totalFiles -FileName $path -PhaseStart $phaseStart -IsInteractive $isInteractive
        if ($isInteractive) { Write-Host $line -NoNewline } else { Write-Host $line }

        # Create remote directory
        $remoteDir = [System.IO.Path]::GetDirectoryName("$($Config.RemotePath)/$($file.Path)").Replace("\", "/")
        Invoke-Ssh -Server $Server -Command "mkdir -p '$remoteDir'" | Out-Null

        # Upload
        Invoke-Scp -Direction "upload" -Server $Server -LocalPath $file.FullPath -RemotePath "$($Config.RemotePath)/$($file.Path)"
        if ($LASTEXITCODE -eq 0) { $successCount++ }

        # Classify
        $suffix = Format-LineStatSuffix -Stats $lineStats -FilePath $path
        if ($path -in $results.New) {
            $newCount++
            $formattedLines += @{ Text = "  new file:   $path$suffix"; Color = "Green" }
        } else {
            $modCount++
            $formattedLines += @{ Text = "  modified:  $path$suffix"; Color = "White" }
        }
    }

    foreach ($f in $toDelete) {
        $current++
        $pct = [math]::Floor($current * 100 / $totalFiles)

        $line = Format-ProgressLine -Verb "Uploading" -Pct $pct -Current $current -Total $totalFiles -FileName "DELETE $f" -PhaseStart $phaseStart -IsInteractive $isInteractive
        if ($isInteractive) { Write-Host $line -NoNewline } else { Write-Host $line }

        $remotePath = "$($Config.RemotePath)/$f"
        Invoke-Ssh -Server $Server -Command "rm -f '$remotePath'" | Out-Null
        $delCount++
        $suffix = Format-LineStatSuffix -Stats $lineStats -FilePath $f
        $formattedLines += @{ Text = "  deleted:    $f$suffix"; Color = "Red" }
    }

    # Phase 3: Summary
    if ($isInteractive) {
        Write-Host ("`rUploading objects: 100% ($totalFiles/$totalFiles), done.".PadRight(100)) -ForegroundColor DarkGray
    } else {
        Write-Host "Uploading objects: 100% ($totalFiles/$totalFiles), done." -ForegroundColor DarkGray
    }

    foreach ($entry in $formattedLines) {
        Write-Host $entry.Text -ForegroundColor $entry.Color
    }

    $totalAdds = if ($lineStats.ContainsKey('_TotalAdds')) { $lineStats['_TotalAdds'] } else { 0 }
    $totalDels = if ($lineStats.ContainsKey('_TotalDels')) { $lineStats['_TotalDels'] } else { 0 }
    $summary = "$totalFiles file(s) changed"
    if ($totalAdds -gt 0) { $summary += ", $totalAdds insertion(+)" }
    if ($totalDels -gt 0) { $summary += ", $totalDels deletion(-)" }
    Write-Host $summary

    Write-TransferSummary -StartTime $startTime -FileCount $current

    # Cleanup empty directories
    $cleanupCmd = "find $($Config.RemotePath) -type d -empty ! -path '*/.git/*' -delete 2>/dev/null"
    Invoke-Ssh -Server $Server -Command $cleanupCmd | Out-Null
}

function Invoke-GitStylePull {
    <#
    .SYNOPSIS Git-style pull (download + delete local extras) with real-time progress for a single server.
    #>
    param(
        [hashtable]$Server
    )
    $startTime = Get-Date
    $isInteractive = ($Host.Name -eq 'ConsoleHost') -and [Environment]::UserInteractive

    # Phase 1: Enumerate
    if ($isInteractive) {
        Write-Host "remote: Enumerating objects..." -ForegroundColor DarkGray -NoNewline
    } else {
        Write-Host "remote: Enumerating objects..." -ForegroundColor DarkGray
    }

    $results = Compare-Files -Server $Server -Silent
    $toDownload = @($results.Deleted) + @($results.Modified) | Where-Object { $_ }
    $toDelete = @($results.New) | Where-Object { $_ }   # local-only → delete from local
    if (-not $toDownload) { $toDownload = @() }
    if (-not $toDelete) { $toDelete = @() }
    $totalFiles = $toDownload.Count + $toDelete.Count

    if ($isInteractive) {
        Write-Host "`rremote: Enumerating objects: $totalFiles, done.       " -ForegroundColor DarkGray
    } else {
        Write-Host "remote: Enumerating objects: $totalFiles, done." -ForegroundColor DarkGray
    }

    if ($totalFiles -eq 0) {
        Write-Host "  Already up to date."
        $elapsed = (Get-Date) - $startTime
        Write-Host ("Transfer complete. [{0:mm\:ss}] ✔" -f $elapsed)
        return
    }

    # Phase 1.5: Compute per-file line diffs
    Write-Host "remote: Computing line changes..." -ForegroundColor DarkGray -NoNewline
    $lineStats = Compute-FileLineDiffs -Server $Server -Direction "pull" `
        -NewFiles @($results.Deleted) -ModifiedFiles @($results.Modified) -DeletedFiles @($toDelete)
    Write-Host "`rremote: Computing line changes: done.       " -ForegroundColor DarkGray

    # Phase 2: Download with progress
    $current = 0; $newCount = 0; $modCount = 0; $delCount = 0
    $formattedLines = @()
    $phaseStart = Get-Date

    foreach ($relativePath in $toDownload) {
        $current++
        $pct = [math]::Floor($current * 100 / $totalFiles)

        $line = Format-ProgressLine -Verb "Pulling" -Pct $pct -Current $current -Total $totalFiles -FileName $relativePath -PhaseStart $phaseStart -IsInteractive $isInteractive
        if ($isInteractive) { Write-Host $line -NoNewline } else { Write-Host $line }

        $remotePath = "$($Config.RemotePath)/$relativePath"
        $localPath = Join-Path $Config.LocalPath ($relativePath.Replace("/", [System.IO.Path]::DirectorySeparatorChar))
        $localDir = Split-Path $localPath -Parent
        if (-not (Test-Path $localDir)) { New-Item -ItemType Directory -Path $localDir -Force | Out-Null }

        Invoke-Scp -Direction "download" -Server $Server -LocalPath $localPath -RemotePath $remotePath

        $suffix = Format-LineStatSuffix -Stats $lineStats -FilePath $relativePath
        if ($relativePath -in $results.Deleted) {
            $newCount++
            $formattedLines += @{ Text = "  new file:   $relativePath$suffix"; Color = "Green" }
        } else {
            $modCount++
            $formattedLines += @{ Text = "  modified:  $relativePath$suffix"; Color = "White" }
        }
    }

    # Phase 2.5: Delete local-only files (pull = mirror remote)
    foreach ($relativePath in $toDelete) {
        $current++
        $pct = [math]::Floor($current * 100 / $totalFiles)

        $line = Format-ProgressLine -Verb "Pulling" -Pct $pct -Current $current -Total $totalFiles -FileName "DELETE $relativePath" -PhaseStart $phaseStart -IsInteractive $isInteractive
        if ($isInteractive) { Write-Host $line -NoNewline } else { Write-Host $line }

        $localPath = Join-Path $Config.LocalPath ($relativePath.Replace("/", [System.IO.Path]::DirectorySeparatorChar))
        if (Test-Path $localPath) { Remove-Item $localPath -Force }
        $delCount++
        $suffix = Format-LineStatSuffix -Stats $lineStats -FilePath $relativePath
        $formattedLines += @{ Text = "  deleted:    $relativePath$suffix"; Color = "Red" }
    }

    # Phase 3: Summary
    if ($isInteractive) {
        Write-Host ("`rPulling objects: 100% ($totalFiles/$totalFiles), done.".PadRight(100)) -ForegroundColor DarkGray
    } else {
        Write-Host "Pulling objects: 100% ($totalFiles/$totalFiles), done." -ForegroundColor DarkGray
    }

    foreach ($entry in $formattedLines) {
        Write-Host $entry.Text -ForegroundColor $entry.Color
    }

    $totalAdds = if ($lineStats.ContainsKey('_TotalAdds')) { $lineStats['_TotalAdds'] } else { 0 }
    $totalDels = if ($lineStats.ContainsKey('_TotalDels')) { $lineStats['_TotalDels'] } else { 0 }
    $summary = "$totalFiles file(s) changed"
    if ($totalAdds -gt 0) { $summary += ", $totalAdds insertion(+)" }
    if ($totalDels -gt 0) { $summary += ", $totalDels deletion(-)" }
    Write-Host $summary

    Write-TransferSummary -StartTime $startTime -FileCount $current
}

function Invoke-GitStyleFetch {
    <#
    .SYNOPSIS Git-style fetch (download only, preserve local extras) with real-time progress.
    #>
    param(
        [hashtable]$Server
    )
    $startTime = Get-Date
    $isInteractive = ($Host.Name -eq 'ConsoleHost') -and [Environment]::UserInteractive

    # Phase 1: Enumerate
    if ($isInteractive) {
        Write-Host "remote: Enumerating objects..." -ForegroundColor DarkGray -NoNewline
    } else {
        Write-Host "remote: Enumerating objects..." -ForegroundColor DarkGray
    }

    $results = Compare-Files -Server $Server -Silent
    $toDownload = @()
    $toDownload += $results.Deleted    # remote-only → download
    $toDownload += $results.Modified   # content differs → download
    $totalFiles = $toDownload.Count

    if ($isInteractive) {
        Write-Host "`rremote: Enumerating objects: $totalFiles, done.       " -ForegroundColor DarkGray
    } else {
        Write-Host "remote: Enumerating objects: $totalFiles, done." -ForegroundColor DarkGray
    }

    if ($totalFiles -eq 0) {
        Write-Host "  Already up to date."
        $elapsed = (Get-Date) - $startTime
        Write-Host ("Transfer complete. [{0:mm\:ss}] ✔" -f $elapsed)
        return
    }

    # Phase 1.5: Compute per-file line diffs
    Write-Host "remote: Computing line changes..." -ForegroundColor DarkGray -NoNewline
    $lineStats = Compute-FileLineDiffs -Server $Server -Direction "pull" `
        -NewFiles @($results.Deleted) -ModifiedFiles @($results.Modified) -DeletedFiles @()
    Write-Host "`rremote: Computing line changes: done.       " -ForegroundColor DarkGray

    # Phase 2: Download with progress (no deletion — fetch preserves local-only files)
    $current = 0; $newCount = 0; $modCount = 0
    $formattedLines = @()
    $phaseStart = Get-Date

    foreach ($relativePath in $toDownload) {
        $current++
        $pct = [math]::Floor($current * 100 / $totalFiles)

        $line = Format-ProgressLine -Verb "Fetching" -Pct $pct -Current $current -Total $totalFiles -FileName $relativePath -PhaseStart $phaseStart -IsInteractive $isInteractive
        if ($isInteractive) { Write-Host $line -NoNewline } else { Write-Host $line }

        $remotePath = "$($Config.RemotePath)/$relativePath"
        $localPath = Join-Path $Config.LocalPath ($relativePath.Replace("/", [System.IO.Path]::DirectorySeparatorChar))
        $localDir = Split-Path $localPath -Parent
        if (-not (Test-Path $localDir)) { New-Item -ItemType Directory -Path $localDir -Force | Out-Null }

        Invoke-Scp -Direction "download" -Server $Server -LocalPath $localPath -RemotePath $remotePath

        $suffix = Format-LineStatSuffix -Stats $lineStats -FilePath $relativePath
        if ($relativePath -in $results.Deleted) {
            $newCount++
            $formattedLines += @{ Text = "  new file:   $relativePath$suffix"; Color = "Green" }
        } else {
            $modCount++
            $formattedLines += @{ Text = "  modified:  $relativePath$suffix"; Color = "White" }
        }
    }

    # Phase 3: Summary
    if ($isInteractive) {
        Write-Host ("`rFetching objects: 100% ($totalFiles/$totalFiles), done.".PadRight(100)) -ForegroundColor DarkGray
    } else {
        Write-Host "Fetching objects: 100% ($totalFiles/$totalFiles), done." -ForegroundColor DarkGray
    }

    foreach ($entry in $formattedLines) {
        Write-Host $entry.Text -ForegroundColor $entry.Color
    }

    $totalAdds = if ($lineStats.ContainsKey('_TotalAdds')) { $lineStats['_TotalAdds'] } else { 0 }
    $totalDels = if ($lineStats.ContainsKey('_TotalDels')) { $lineStats['_TotalDels'] } else { 0 }
    $summary = "$totalFiles file(s) changed"
    if ($totalAdds -gt 0) { $summary += ", $totalAdds insertion(+)" }
    if ($totalDels -gt 0) { $summary += ", $totalDels deletion(-)" }
    Write-Host $summary

    Write-TransferSummary -StartTime $startTime -FileCount $current
}

# ========== End Git-style Progress Functions ==========

# Command handlers
switch ($Command) {
    # ===== Git-like Commands =====
    
    # git diff - 比較本地與遠端差異 (GitHub-style code diff)
    { $_ -in "diff", "check" } {
        $opts = Parse-DiffOptions -Args_ $Arguments
        $diffMode = $opts.DiffMode
        # check if --summary/--stat/--full was passed directly
        foreach ($a in $Arguments) {
            switch ($a) {
                "--summary" { $diffMode = "summary" }
                "--stat"    { $diffMode = "stat" }
                "--full"    { $diffMode = "full" }
            }
        }
        foreach ($server in $Config.Servers) {
            Invoke-CodeDiffAnalysis -Direction "push" -Server $server -Mode $diffMode -Confirm $false
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
    
    # git add - 顯示待上傳的變更
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
        $opts = Parse-DiffOptions -Args_ $Arguments
        $opts.Confirm = $true  # push defaults to confirm
        $idx = 0
        $total = $Config.Servers.Count
        foreach ($server in $Config.Servers) {
            $idx++
            # Run diff analysis before transfer
            if ($opts.DiffMode -ne "no-diff") {
                if (-not (Invoke-CodeDiffAnalysis -Direction "push" -Server $server -Mode $opts.DiffMode -Confirm $opts.Confirm)) {
                    Write-Color "[PUSH] Cancelled for $($server.Name)" "Yellow"
                    continue
                }
            }
            Write-TransferHeader -Server $server -Verb "Push" -Index $idx -Total $total
            Invoke-GitStylePush -Server $server -DeleteRemoteExtras
            Write-SyncHistory -Action "PUSH" -ServerName $server.Name -FileCount $script:DiffFileCount -Adds $script:DiffTotalAdds -Dels $script:DiffTotalDels
        }
    }
    
    "pull" {
        $opts = Parse-DiffOptions -Args_ $Arguments
        $targetServers = @()
        foreach ($posArg in $opts.Positional) {
            foreach ($s in $Config.Servers) {
                if ($posArg -eq $s.Name -or $posArg -eq $s.Name.TrimStart('.') -or $posArg -eq ".$($s.Name.TrimStart('.'))") {
                    $targetServers = @($s); break
                }
            }
        }
        if ($targetServers.Count -eq 0) { $targetServers = $Config.Servers }  # Default: all
        foreach ($targetServer in $targetServers) {
            # Run diff analysis before transfer
            if ($opts.DiffMode -ne "no-diff") {
                if (-not (Invoke-CodeDiffAnalysis -Direction "pull" -Server $targetServer -Mode $opts.DiffMode -Confirm $opts.Confirm)) {
                    Write-Color "[PULL] $($targetServer.Name) Cancelled" "Yellow"
                    continue
                }
            }
            Write-TransferHeader -Server $targetServer -Verb "Pull"
            Invoke-GitStylePull -Server $targetServer
            Write-SyncHistory -Action "PULL" -ServerName $targetServer.Name -FileCount $script:DiffFileCount -Adds $script:DiffTotalAdds -Dels $script:DiffTotalDels
        }
    }
    
    "pull87" {
        & $PSCommandPath pull .87
    }
    
    "pull154" {
        & $PSCommandPath pull .154
    }
    
    # git fetch - 從遠端下載並刪除本地多餘檔案 (sync local to remote)
    "fetch" {
        $opts = Parse-DiffOptions -Args_ $Arguments
        $targetServers = @()
        foreach ($posArg in $opts.Positional) {
            foreach ($s in $Config.Servers) {
                if ($posArg -eq $s.Name -or $posArg -eq $s.Name.TrimStart('.') -or $posArg -eq ".$($s.Name.TrimStart('.'))") {
                    $targetServers = @($s); break
                }
            }
        }
        if ($targetServers.Count -eq 0) { $targetServers = $Config.Servers }  # Default: all
        foreach ($targetServer in $targetServers) {
            # Run diff analysis before transfer
            if ($opts.DiffMode -ne "no-diff") {
                if (-not (Invoke-CodeDiffAnalysis -Direction "fetch" -Server $targetServer -Mode $opts.DiffMode -Confirm $opts.Confirm)) {
                    Write-Color "[FETCH] $($targetServer.Name) Cancelled" "Yellow"
                    continue
                }
            }
            Write-TransferHeader -Server $targetServer -Verb "Fetch"
            Invoke-GitStyleFetch -Server $targetServer
            Write-SyncHistory -Action "FETCH" -ServerName $targetServer.Name -FileCount $script:DiffFileCount -Adds $script:DiffTotalAdds -Dels $script:DiffTotalDels
        }
    }

    "fetch87" { & $PSCommandPath fetch 87 }
    "fetch154" { & $PSCommandPath fetch 154 }
    # git log - 查看遠端 log 檔案
    "log" {
        Write-Color "[LOG] Fetching remote log files..." "Magenta"
        
        $targetServers = @()
        if ($Arguments -and $Arguments.Count -gt 0) {
            $serverArg = $Arguments[0]
            foreach ($s in $Config.Servers) {
                if ($s.Name -eq $serverArg -or $s.Name -eq ".$serverArg" -or $serverArg -like "*$($s.Name)*") {
                    $targetServers = @($s)
                    break
                }
            }
        }
        if ($targetServers.Count -eq 0) { $targetServers = $Config.Servers }  # Default: all
        
        foreach ($targetServer in $targetServers) {
            Write-Color "`nLog files on $($targetServer.Name):" "Cyan"
            $cmd = "ls -lth $($Config.RemotePath)/log* 2>/dev/null | head -10"
            $result = Invoke-Ssh -Server $targetServer -Command $cmd
            if ($result) {
                foreach ($line in $result) { Write-Host "  $line" }
            }
            else {
                Write-Color "  No log files found" "Yellow"
            }
            
            # 顯示最新 log 的最後幾行
            Write-Color "`nLatest log tail:" "Cyan"
            $cmd = "tail -20 `$(ls -t $($Config.RemotePath)/log* 2>/dev/null | head -1) 2>/dev/null"
            $result = Invoke-Ssh -Server $targetServer -Command $cmd
            if ($result) {
                foreach ($line in $result) { Write-Host "  $line" -ForegroundColor Gray }
            }
        }
    }
    
    # git reset --hard - 清理遠端，刪除遠端多餘檔案
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
                    Invoke-Ssh -Server $server -Command "rm -f '$remotePath'"
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
    
    # git clone - 從遠端完整下載到本地
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
                
                Invoke-Scp -Direction "download" -Server $targetServer -LocalPath $localPath -RemotePath $remotePath
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
            & $PSCommandPath push --no-diff
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
            $isVscode = ($name -like ".vscode/*" -or $name -like ".vscode\*")
            $isGitignore = $name -eq ".gitignore"
            
            if (-not $skip -and ($syncExtensions -contains $ext -or $isVscode -or $isGitignore)) {
                $now = Get-Date
                if (-not $lastSync.ContainsKey($path) -or ($now - $lastSync[$path]).TotalSeconds -gt 2) {
                    $lastSync[$path] = $now
                    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $changeType : $name" -ForegroundColor Cyan
                    
                    # Trigger push in background
                    Get-Job -State Completed -ErrorAction SilentlyContinue | Remove-Job -Force -ErrorAction SilentlyContinue
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
        $targetServers = $Config.Servers  # Default: all servers
        if ($Arguments.Count -gt 0) {
            $arg = $Arguments[0]
            foreach ($s in $Config.Servers) {
                if ($arg -eq $s.Name -or $arg -eq $s.Name.TrimStart('.')) {
                    $targetServers = @($s); break
                }
            }
        }
        $idx = 0
        $total = $targetServers.Count
        foreach ($server in $targetServers) {
            $idx++
            Write-TransferHeader -Server $server -Verb "Push (auto)" -Index $idx -Total $total
            Invoke-GitStylePush -Server $server -DeleteRemoteExtras
        }
    }
    
    "watchpush" {
        # Background auto-upload: monitor local files and upload changes (persistent process)
        $pidFile = Join-Path $Config.LocalPath ".vscode/watchpush.pid"
        $logFile = Join-Path $Config.LocalPath ".vscode/watchpush.log"
        $daemonScript = Join-Path $Config.LocalPath ".vscode/Pwshell_bg_watchpush.ps1"
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
                Write-Color "  Targets: $($Config.Servers.Name -join ', ')" "White"
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
                $_psExe = if ($Config.IsWindows) { "powershell.exe" } else { "pwsh" }
                $_psArgs = @(
                    "-NoProfile", "-ExecutionPolicy", "Bypass",
                    "-File", "`"$daemonScript`"",
                    "-LocalPath", "`"$($Config.LocalPath)`"",
                    "-RemotePath", "`"$($Config.RemotePath)`"",
                    "-ServersJson", "'$serversJson'",
                    "-PlinkPath", "`"$($Config.PlinkPath)`"",
                    "-PscpPath", "`"$($Config.PscpPath)`"",
                    "-LogPath", "`"$logFile`"",
                    "-Interval", $interval,
                    "-SshOpts", "`"$($Config.SshOpts)`""
                )
                if ($Config.IsWindows) { $_psArgs += @("-IsWindows"); $_psArgs = @("-WindowStyle", "Hidden") + $_psArgs }
                $proc = Start-Process -FilePath $_psExe -ArgumentList $_psArgs -PassThru
                
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
            @{ Name = "WatchPush"; Label = "[UPLOAD] WatchPush"; PidFile = (Join-Path ".vscode" "watchpush.pid"); LogFile = (Join-Path ".vscode" "watchpush.log"); Color = "Yellow" },
            @{ Name = "WatchPull"; Label = "[DOWNLOAD] WatchPull"; PidFile = (Join-Path ".vscode" "watchpull.pid"); LogFile = (Join-Path ".vscode" "watchpull.log"); Color = "Yellow" },
            @{ Name = "WatchFetch"; Label = "[SYNC+DELETE] WatchFetch"; PidFile = (Join-Path ".vscode" "watchfetch.pid"); LogFile = (Join-Path ".vscode" "watchfetch.log"); Color = "Red" },
            @{ Name = "VTKRenamer"; Label = "[VTK-RENAME] Auto-Renamer"; PidFile = (Join-Path ".vscode" "vtk-renamer.pid"); LogFile = (Join-Path ".vscode" "vtk-renamer.log"); Color = "Cyan" }
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
        $pushPidFile = Join-Path $Config.LocalPath ".vscode/watchpush.pid"
        $pullPidFile = Join-Path $Config.LocalPath ".vscode/watchpull.pid"
        $pushLogFile = Join-Path $Config.LocalPath ".vscode/watchpush.log"
        $pullLogFile = Join-Path $Config.LocalPath ".vscode/watchpull.log"
        
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
        & $PSCommandPath push --no-diff
        Write-Host ""
        & $PSCommandPath reset
    }
    
    "watchpull" {
        # Auto-download: monitor remote servers and download new files (persistent process)
        $pidFile = Join-Path $Config.LocalPath ".vscode/watchpull.pid"
        $logFile = Join-Path $Config.LocalPath ".vscode/watchpull.log"
        $daemonScript = Join-Path $Config.LocalPath ".vscode/Pwshell_bg_watchpull.ps1"
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
                # Start watchpull for specified server or all
                $targetServers = @()
                if ($subCommand -eq ".87" -or $subCommand -eq "87") {
                    $targetServers = @($Config.Servers | Where-Object { $_.Name -eq ".87" })
                }
                elseif ($subCommand -eq ".89" -or $subCommand -eq "89") {
                    $targetServers = @($Config.Servers | Where-Object { $_.Name -eq ".89" })
                }
                elseif ($subCommand -eq ".154" -or $subCommand -eq "154") {
                    $targetServers = @($Config.Servers | Where-Object { $_.Name -eq ".154" })
                }
                else {
                    $targetServers = $Config.Servers  # Default: all
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
                    $_psExe = if ($Config.IsWindows) { "powershell.exe" } else { "pwsh" }
                    $_psArgs = @(
                        "-NoProfile", "-ExecutionPolicy", "Bypass",
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
                        "-Interval", $interval,
                        "-SshOpts", "`"$($Config.SshOpts)`""
                    )
                    if ($Config.IsWindows) { $_psArgs += @("-IsWindows"); $_psArgs = @("-WindowStyle", "Hidden") + $_psArgs }
                    $proc = Start-Process -FilePath $_psExe -ArgumentList $_psArgs -PassThru
                    
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
        $pidFile = Join-Path $Config.LocalPath ".vscode/watchfetch.pid"
        $logFile = Join-Path $Config.LocalPath ".vscode/watchfetch.log"
        $daemonScript = Join-Path $Config.LocalPath ".vscode/Pwshell_bg_watchfetch.ps1"
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
                # Start watchfetch for specified server or all
                $targetServers = @()
                if ($subCommand -eq ".87" -or $subCommand -eq "87") {
                    $targetServers = @($Config.Servers | Where-Object { $_.Name -eq ".87" } | Select-Object -First 1)
                }
                elseif ($subCommand -eq ".89" -or $subCommand -eq "89") {
                    $targetServers = @($Config.Servers | Where-Object { $_.Name -eq ".89" } | Select-Object -First 1)
                }
                elseif ($subCommand -eq ".154" -or $subCommand -eq "154") {
                    $targetServers = @($Config.Servers | Where-Object { $_.Name -eq ".154" } | Select-Object -First 1)
                }
                else {
                    $targetServers = $Config.Servers  # Default: all
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
                Write-Color "  Servers: $($targetServers.Name -join ', ')" "White"
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
                
                # Start independent process for each server
                $processPids = @()
                foreach ($server in $targetServers) {
                    $_psExe = if ($Config.IsWindows) { "powershell.exe" } else { "pwsh" }
                    $_psArgs = @(
                        "-NoProfile", "-ExecutionPolicy", "Bypass",
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
                        "-Interval", $interval,
                        "-SshOpts", "`"$($Config.SshOpts)`""
                    )
                    if ($Config.IsWindows) { $_psArgs += @("-IsWindows"); $_psArgs = @("-WindowStyle", "Hidden") + $_psArgs }
                    $proc = Start-Process -FilePath $_psExe -ArgumentList $_psArgs -PassThru
                    
                    Start-Sleep -Milliseconds 500
                    $processPids += $proc.Id
                    Write-Color "[STARTED] $($server.Name) fetch monitoring (PID: $($proc.Id))" "Green"
                }
                
                # Save all PIDs
                $processPids | Out-File $pidFile -Force
                
                Write-Color "`n[WATCHFETCH] Background monitoring started!" "Green"
                Write-Color "WARNING: Local files not on remote will be DELETED!" "Red"
                Write-Color "Use 'mobaxterm watchfetch status' to check progress" "Cyan"
            }
        }
    }
    
    "autopull" {
        # Quick auto-pull: check and pull if needed (download only, no local delete)
        $targetServers = $Config.Servers  # Default: all
        if ($Arguments.Count -gt 0) {
            $arg = $Arguments[0]
            foreach ($s in $Config.Servers) {
                if ($arg -eq $s.Name -or $arg -eq $s.Name.TrimStart('.')) {
                    $targetServers = @($s); break
                }
            }
        }
        $idx = 0; $total = $targetServers.Count
        foreach ($server in $targetServers) {
            $idx++
            Write-TransferHeader -Server $server -Verb "Pull (auto)" -Index $idx -Total $total
            Invoke-GitStylePull -Server $server
        }
    }
    
    "autofetch" {
        # Quick auto-fetch: download + delete local files not on remote (sync local to remote)
        $targetServers = $Config.Servers  # Default: all
        if ($Arguments.Count -gt 0) {
            $arg = $Arguments[0]
            foreach ($s in $Config.Servers) {
                if ($arg -eq $s.Name -or $arg -eq $s.Name.TrimStart('.')) {
                    $targetServers = @($s); break
                }
            }
        }
        $idx = 0; $total = $targetServers.Count
        foreach ($server in $targetServers) {
            $idx++
            Write-TransferHeader -Server $server -Verb "Fetch (auto)" -Index $idx -Total $total
            Invoke-GitStyleFetch -Server $server
        }
    }
    
    "vtkrename" {
        # VTK file auto-renamer: monitor and rename VTK files to use zero-padding
        $pidFile = Join-Path $Config.LocalPath ".vscode/vtk-renamer.pid"
        $logFile = Join-Path $Config.LocalPath ".vscode/vtk-renamer.log"
        $renamerScript = Join-Path $Config.LocalPath ".vscode/Zsh_bg_renamer.ps1"
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
                $_psExe = if ($Config.IsWindows) { "powershell.exe" } else { "pwsh" }
                $_psArgs = @(
                    "-NoProfile", "-ExecutionPolicy", "Bypass",
                    "-File", "`"$renamerScript`"",
                    "-WatchPath", "`"$($Config.LocalPath)`"",
                    "-CheckInterval", $checkInterval
                )
                if ($Config.IsWindows) { $_psArgs = @("-WindowStyle", "Hidden") + $_psArgs }
                $proc = Start-Process -FilePath $_psExe -ArgumentList $_psArgs -PassThru
                
                Start-Sleep -Milliseconds 500
                
                # Save PID
                $proc.Id | Out-File $pidFile -Force
                Write-Color "[STARTED] VTK renamer (PID: $($proc.Id))" "Green"
                
                Write-Color "`n[VTK-RENAMER] Background monitoring started!" "Green"
                Write-Color "Use 'mobaxterm vtkrename status' to check progress" "Cyan"
            }
        }
    }

    # ========== GPU 狀態查詢命令 ==========

    "gpus" {
        # GPU 狀態總覽（所有伺服器）
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "         GPU Status Overview            " -ForegroundColor Cyan
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "  Querying all servers..." -ForegroundColor Gray
        Write-Host ""

        # 查詢 .89 直連
        Write-Host "  .89 (140.114.58.89) - 8x Tesla V100-SXM2-32GB" -ForegroundColor White
        $gpu89 = Query-GpuStatus "89" "0"
        $info89 = Parse-GpuOutput $gpu89
        Write-Host "    " -NoNewline
        if ($info89.Offline) {
            Write-Host "[OFFLINE]" -ForegroundColor DarkGray
        } else {
            foreach ($d in $info89.Dots) {
                if ($d -eq "G") { Write-Host " O " -NoNewline -ForegroundColor Green }
                else { Write-Host " X " -NoNewline -ForegroundColor Red }
            }
            $freeStr = "$($info89.Free)/$($info89.Total)"
            if ($info89.Free -eq 0) { Write-Host "  $freeStr" -ForegroundColor Red }
            elseif ($info89.Free -ge 4) { Write-Host "  $freeStr" -ForegroundColor Green }
            else { Write-Host "  $freeStr" -ForegroundColor Yellow }
        }
        Write-Host ""

        # 查詢 .87 節點
        Write-Host "  .87 (140.114.58.87) - Jump Server" -ForegroundColor White
        foreach ($node in $Config.Nodes["87"]) {
            $gpuOut = Query-GpuStatus "87" $node.Node
            $info = Parse-GpuOutput $gpuOut
            Write-Host "    ib$($node.Node) ($($node.GpuType)): " -NoNewline
            if ($info.Offline) {
                Write-Host "[OFFLINE/Maintenance]" -ForegroundColor DarkGray
            } else {
                foreach ($d in $info.Dots) {
                    if ($d -eq "G") { Write-Host "O " -NoNewline -ForegroundColor Green }
                    else { Write-Host "X " -NoNewline -ForegroundColor Red }
                }
                $freeStr = "$($info.Free)/$($info.Total)"
                if ($info.Free -eq 0) { Write-Host " $freeStr" -ForegroundColor Red }
                elseif ($info.Free -ge 4) { Write-Host " $freeStr" -ForegroundColor Green }
                else { Write-Host " $freeStr" -ForegroundColor Yellow }
            }
        }
        Write-Host ""

        # 查詢 .154 節點
        Write-Host "  .154 (140.114.58.154) - Jump Server" -ForegroundColor White
        foreach ($node in $Config.Nodes["154"]) {
            $gpuOut = Query-GpuStatus "154" $node.Node
            $info = Parse-GpuOutput $gpuOut
            Write-Host "    ib$($node.Node) ($($node.GpuType)): " -NoNewline
            if ($info.Offline) {
                Write-Host "[OFFLINE/Maintenance]" -ForegroundColor DarkGray
            } else {
                foreach ($d in $info.Dots) {
                    if ($d -eq "G") { Write-Host "O " -NoNewline -ForegroundColor Green }
                    else { Write-Host "X " -NoNewline -ForegroundColor Red }
                }
                $freeStr = "$($info.Free)/$($info.Total)"
                if ($info.Free -eq 0) { Write-Host " $freeStr" -ForegroundColor Red }
                elseif ($info.Free -ge 4) { Write-Host " $freeStr" -ForegroundColor Green }
                else { Write-Host " $freeStr" -ForegroundColor Yellow }
            }
        }

        Write-Host ""
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "  O=Free  X=Busy  Free/Total" -ForegroundColor DarkGray
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host ""
    }

    "gpu" {
        # GPU 詳細狀態（nvidia-smi 完整輸出）
        $target = if ($Arguments.Count -gt 0) { $Arguments[0] } else { "all" }
        $target = $target.TrimStart(".")

        Write-Host ""
        Write-Host "=== GPU Detailed Status ===" -ForegroundColor Cyan
        Write-Host ""

        switch ($target) {
            "89" {
                Write-Host ".89 (140.114.58.89) - 8x Tesla V100-SXM2-32GB" -ForegroundColor Yellow
                Write-Host ("-" * 50)
                $server = Get-ServerByName "89"
                $result = Invoke-Ssh -Server $server -Command "nvidia-smi"
                if ($result) { $result | ForEach-Object { Write-Host $_ } }
                else { Write-Host "[OFFLINE] Cannot connect" -ForegroundColor Red }
            }
            "87" {
                Write-Host ".87 Nodes Status" -ForegroundColor Yellow
                $server = Get-ServerByName "87"
                foreach ($node in $Config.Nodes["87"]) {
                    Write-Host ""
                    Write-Host "=== .87 ib$($node.Node) ===" -ForegroundColor Cyan
                    $result = Invoke-Ssh -Server $server -Command "ssh -o ConnectTimeout=3 cfdlab-ib$($node.Node) 'nvidia-smi'"
                    if ($result) { $result | ForEach-Object { Write-Host $_ } }
                    else { Write-Host "[OFFLINE/Maintenance]" -ForegroundColor Red }
                }
            }
            "154" {
                Write-Host ".154 Nodes Status" -ForegroundColor Yellow
                $server = Get-ServerByName "154"
                foreach ($node in $Config.Nodes["154"]) {
                    Write-Host ""
                    Write-Host "=== .154 ib$($node.Node) ===" -ForegroundColor Cyan
                    $result = Invoke-Ssh -Server $server -Command "ssh -o ConnectTimeout=3 cfdlab-ib$($node.Node) 'nvidia-smi'"
                    if ($result) { $result | ForEach-Object { Write-Host $_ } }
                    else { Write-Host "[OFFLINE/Maintenance]" -ForegroundColor Red }
                }
            }
            default {
                # all
                & $PSCommandPath gpu 89
                Write-Host ""
                & $PSCommandPath gpu 87
                Write-Host ""
                & $PSCommandPath gpu 154
            }
        }
        Write-Host ""
    }

    # ========== SSH / 遠端執行命令 ==========

    "ssh" {
        # SSH 連線（帶 GPU 狀態顯示）
        $combo = if ($Arguments.Count -gt 0) { $Arguments[0] } else { "87:3" }

        # 解析 server:node 格式
        $parts = $combo -split ":"
        if ($parts.Count -ne 2) {
            Write-Color "[ERROR] Invalid format. Use: 87:3 or 154:4 or 89:0" "Red"
            exit 1
        }

        $serverKey = $parts[0].TrimStart(".")
        $nodeNum = $parts[1]

        $server = Get-ServerByName $serverKey
        if (-not $server) {
            Write-Color "[ERROR] Unknown server: $serverKey. Use 87, 89 or 154." "Red"
            exit 1
        }

        # 顯示 GPU 狀態
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Cyan
        if ($nodeNum -eq "0") {
            Write-Host "  Connecting to .$serverKey (direct)" -ForegroundColor White
        } else {
            Write-Host "  Connecting to .$serverKey -> ib$nodeNum" -ForegroundColor White
        }
        Write-Host "========================================" -ForegroundColor Cyan

        $gpuOut = Query-GpuStatus $serverKey $nodeNum
        $info = Parse-GpuOutput $gpuOut

        Write-Host "  GPU Status: " -NoNewline
        if ($info.Offline) {
            Write-Host "[OFFLINE]" -ForegroundColor Red
        } else {
            foreach ($d in $info.Dots) {
                if ($d -eq "G") { Write-Host "O " -NoNewline -ForegroundColor Green }
                else { Write-Host "X " -NoNewline -ForegroundColor Red }
            }
            $freeStr = "$($info.Free)/$($info.Total) available"
            if ($info.Free -eq 0) { Write-Host " ($freeStr)" -ForegroundColor Red }
            elseif ($info.Free -ge 4) { Write-Host " ($freeStr)" -ForegroundColor Green }
            else { Write-Host " ($freeStr)" -ForegroundColor Yellow }
        }
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host ""

        # 確保遠端目錄存在
        Ensure-RemoteDir $server

        # 連線
        if ($nodeNum -eq "0") {
            Write-Color "[SSH] Connecting directly to .$serverKey..." "Cyan"
            Invoke-Ssh -Server $server -TtyCommand "cd $($Config.RemotePath); exec bash"
        } else {
            Write-Color "[SSH] Connecting via .$serverKey to ib$nodeNum..." "Cyan"
            Invoke-Ssh -Server $server -TtyCommand "ssh -t cfdlab-ib$nodeNum 'cd $($Config.RemotePath); exec bash'"
        }
    }

    "issh" {
        # 互動式 SSH 選擇器（調用 Pwshell_GPUconnect.ps1 -Interactive）
        $sshScript = Join-Path $Config.LocalPath ".vscode/Pwshell_GPUconnect.ps1"
        if (Test-Path $sshScript) {
            & $sshScript -Interactive
        } else {
            Write-Color "[ERROR] Pwshell_GPUconnect.ps1 not found" "Red"
        }
    }

    "run" {
        # 編譯並執行
        $combo = if ($Arguments.Count -gt 0) { $Arguments[0] } else { "87:3" }
        $gpuCount = if ($Arguments.Count -gt 1) { $Arguments[1] } else { $Config.DefaultGpuCount }

        $parts = $combo -split ":"
        if ($parts.Count -ne 2) {
            Write-Color "[ERROR] Invalid format. Use: run 87:3 [gpu_count]" "Red"
            exit 1
        }

        $serverKey = $parts[0].TrimStart(".")
        $nodeNum = $parts[1]

        Write-Color "[RUN] Compiling and running on .$serverKey ib$nodeNum with $gpuCount GPUs..." "Magenta"

        $buildCmd = "nvcc main.cu -arch=$($Config.NvccArch) -I$($Config.MpiInclude) -L$($Config.MpiLib) -lmpi -o a.out"
        $runCmd = "nohup mpirun -np $gpuCount ./a.out > log`$(date +%Y%m%d) 2>&1 &"
        $fullCmd = "$buildCmd && $runCmd"

        Run-RemoteCommand $serverKey $nodeNum $fullCmd
        Write-Color "[RUN] Job submitted!" "Green"
    }

    "jobs" {
        # 查看執行中的任務
        $combo = if ($Arguments.Count -gt 0) { $Arguments[0] } else { "87:3" }

        $parts = $combo -split ":"
        if ($parts.Count -ne 2) {
            Write-Color "[ERROR] Invalid format. Use: jobs 87:3" "Red"
            exit 1
        }

        $serverKey = $parts[0].TrimStart(".")
        $nodeNum = $parts[1]

        Write-Color "[JOBS] Checking running jobs on .$serverKey ib$nodeNum..." "Cyan"
        Run-RemoteCommand $serverKey $nodeNum "ps aux | grep a.out | grep -v grep || echo 'No running jobs'"
    }

    "kill" {
        # 終止執行中的任務
        $combo = if ($Arguments.Count -gt 0) { $Arguments[0] } else { "87:3" }

        $parts = $combo -split ":"
        if ($parts.Count -ne 2) {
            Write-Color "[ERROR] Invalid format. Use: kill 87:3" "Red"
            exit 1
        }

        $serverKey = $parts[0].TrimStart(".")
        $nodeNum = $parts[1]

        Write-Color "[KILL] Stopping jobs on .$serverKey ib$nodeNum..." "Red"
        Run-RemoteCommand $serverKey $nodeNum "pkill -f a.out || pkill -f mpirun || echo 'No jobs to kill'"
        Write-Color "[KILL] Done" "Green"
    }

    # ========== 伺服器別名命令 ==========

    # ========== 額外別名（與 Mac 對齊）==========

    # check = diff (Mac 兼容)
    "check" { & $PSCommandPath diff }

    # delete = reset (Mac 兼容)
    "delete" { & $PSCommandPath reset }

    # watch = watchpush (Mac 兼容)
    "watch" { & $PSCommandPath watchpush $Arguments }

    # pull 別名
    "pull87" { & $PSCommandPath pull .87 }
    "pull89" { & $PSCommandPath pull .89 }

    # fetch 別名
    "fetch89" { & $PSCommandPath fetch 89 }

    # log 別名
    "log87" { & $PSCommandPath log .87 }
    "log89" { & $PSCommandPath log .89 }
    "log154" { & $PSCommandPath log .154 }

    # diff 別名
    "diff87" {
        $server = $Config.Servers | Where-Object { $_.Name -eq ".87" }
        if ($server) { Invoke-CodeDiffAnalysis -Direction "push" -Server $server -Mode "full" -Confirm $false }
    }
    "diff89" {
        $server = $Config.Servers | Where-Object { $_.Name -eq ".89" }
        if ($server) { Invoke-CodeDiffAnalysis -Direction "push" -Server $server -Mode "full" -Confirm $false }
    }
    "diff154" {
        $server = $Config.Servers | Where-Object { $_.Name -eq ".154" }
        if ($server) { Invoke-CodeDiffAnalysis -Direction "push" -Server $server -Mode "full" -Confirm $false }
    }
    "diffall" { & $PSCommandPath diff }

    # push 別名
    "push87" {
        $opts = Parse-DiffOptions -Args_ $Arguments
        $opts.Confirm = $true
        $server = $Config.Servers | Where-Object { $_.Name -eq ".87" }
        if ($opts.DiffMode -ne "no-diff") {
            if (-not (Invoke-CodeDiffAnalysis -Direction "push" -Server $server -Mode $opts.DiffMode -Confirm $opts.Confirm)) {
                Write-Color "[PUSH] Cancelled" "Yellow"; return
            }
        }
        Write-TransferHeader -Server $server -Verb "Push"
        Invoke-GitStylePush -Server $server -DeleteRemoteExtras
        Write-SyncHistory -Action "PUSH" -ServerName $server.Name -FileCount $script:DiffFileCount -Adds $script:DiffTotalAdds -Dels $script:DiffTotalDels
    }
    "push89" {
        $opts = Parse-DiffOptions -Args_ $Arguments
        $opts.Confirm = $true
        $server = $Config.Servers | Where-Object { $_.Name -eq ".89" }
        if ($opts.DiffMode -ne "no-diff") {
            if (-not (Invoke-CodeDiffAnalysis -Direction "push" -Server $server -Mode $opts.DiffMode -Confirm $opts.Confirm)) {
                Write-Color "[PUSH] Cancelled" "Yellow"; return
            }
        }
        Write-TransferHeader -Server $server -Verb "Push"
        Invoke-GitStylePush -Server $server -DeleteRemoteExtras
        Write-SyncHistory -Action "PUSH" -ServerName $server.Name -FileCount $script:DiffFileCount -Adds $script:DiffTotalAdds -Dels $script:DiffTotalDels
    }
    "push154" {
        $opts = Parse-DiffOptions -Args_ $Arguments
        $opts.Confirm = $true
        $server = $Config.Servers | Where-Object { $_.Name -eq ".154" }
        if ($opts.DiffMode -ne "no-diff") {
            if (-not (Invoke-CodeDiffAnalysis -Direction "push" -Server $server -Mode $opts.DiffMode -Confirm $opts.Confirm)) {
                Write-Color "[PUSH] Cancelled" "Yellow"; return
            }
        }
        Write-TransferHeader -Server $server -Verb "Push"
        Invoke-GitStylePush -Server $server -DeleteRemoteExtras
        Write-SyncHistory -Action "PUSH" -ServerName $server.Name -FileCount $script:DiffFileCount -Adds $script:DiffTotalAdds -Dels $script:DiffTotalDels
    }
    "pushall" { & $PSCommandPath push @Arguments }

    # autopull 別名
    "autopull87" { & $PSCommandPath autopull .87 }
    "autopull89" { & $PSCommandPath autopull .89 }
    "autopull154" { & $PSCommandPath autopull .154 }

    # autofetch 別名
    "autofetch87" { & $PSCommandPath autofetch 87 }
    "autofetch89" { & $PSCommandPath autofetch 89 }
    "autofetch154" { & $PSCommandPath autofetch 154 }

    # autopush 別名
    "autopush87" { & $PSCommandPath autopush .87 }
    "autopush89" { & $PSCommandPath autopush .89 }
    "autopush154" { & $PSCommandPath autopush .154 }
    "autopushall" { & $PSCommandPath autopush }

    # ===== Code Diff Analysis Commands =====

    "sync-diff" {
        # Standalone diff (no transfer)
        $diffMode = "full"
        $positional = @()
        foreach ($a in $Arguments) {
            switch ($a) {
                "--summary" { $diffMode = "summary" }
                "--stat"    { $diffMode = "stat" }
                "--full"    { $diffMode = "full" }
                default     { $positional += $a }
            }
        }
        $targetServer = $Config.Servers[0]
        if ($positional.Count -gt 0) {
            foreach ($s in $Config.Servers) {
                $pa = $positional[0]
                if ($pa -eq $s.Name -or $pa -eq $s.Name.TrimStart('.') -or $pa -eq ".$($s.Name.TrimStart('.'))") {
                    $targetServer = $s; break
                }
            }
        }
        Invoke-CodeDiffAnalysis -Direction "push" -Server $targetServer -Mode $diffMode -Confirm $false
    }

    "sync-diff-summary" {
        $targetServer = $Config.Servers[0]
        if ($Arguments -and $Arguments.Count -gt 0) {
            foreach ($s in $Config.Servers) {
                $pa = $Arguments[0]
                if ($pa -eq $s.Name -or $pa -eq $s.Name.TrimStart('.') -or $pa -eq ".$($s.Name.TrimStart('.'))") {
                    $targetServer = $s; break
                }
            }
        }
        Invoke-CodeDiffAnalysis -Direction "push" -Server $targetServer -Mode "summary" -Confirm $false
    }

    "sync-diff-file" {
        # Show diff for a single file
        if (-not $Arguments -or $Arguments.Count -lt 1) {
            Write-Color "Usage: mobaxterm sync-diff-file <file> [server]" "Yellow"
            return
        }
        $filePath = $Arguments[0]
        $targetServer = $Config.Servers[0]
        if ($Arguments.Count -gt 1) {
            foreach ($s in $Config.Servers) {
                $pa = $Arguments[1]
                if ($pa -eq $s.Name -or $pa -eq $s.Name.TrimStart('.') -or $pa -eq ".$($s.Name.TrimStart('.'))") {
                    $targetServer = $s; break
                }
            }
        }
        Show-FileDiff -Direction "push" -Server $targetServer -FilePath $filePath -FileStatus "modified" -Mode "full"
    }

    "sync-log" {
        # Show sync history
        $histFile = $script:SyncHistoryFile
        if (-not $histFile) { $histFile = Join-Path $HOME ".sync-history.log" }
        if (Test-Path $histFile) {
            $lines = if ($Arguments -and $Arguments[0] -match '^\d+$') { [int]$Arguments[0] } else { 30 }
            Write-Color "=== Sync History (last $lines entries) ===" "Cyan"
            Get-Content $histFile -Tail $lines | ForEach-Object { Write-Host $_ }
        } else {
            Write-Color "No sync history found." "Yellow"
        }
    }

    "sync-stop" {
        # Stop all daemon processes
        Write-Color "[SYNC-STOP] Stopping all daemons..." "Yellow"
        foreach ($svcName in @("watchpush", "watchpull", "watchfetch", "vtk-renamer")) {
            $pidFile = Join-Path $Config.LocalPath ".vscode/$svcName.pid"
            if (Test-Path $pidFile) {
                $pids = Get-Content $pidFile -ErrorAction SilentlyContinue
                foreach ($p in $pids) {
                    if ($p -match '^\d+$') {
                        $proc = Get-Process -Id $p -ErrorAction SilentlyContinue
                        if ($proc) {
                            Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
                            Write-Color "  Stopped $svcName (PID: $p)" "Green"
                        }
                    }
                }
                Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
            }
        }
        Write-Color "[SYNC-STOP] All daemons stopped." "Green"
    }

    default {
        Write-Host ""
        Write-Host "MobaXterm Sync Commands (Git-like)" -ForegroundColor Cyan
        Write-Host "===================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Usage: mobaxterm <command> [options]" -ForegroundColor White
        Write-Host ""
        Write-Host "=== Sync Command Summary ===" -ForegroundColor Yellow
        Write-Host "  FETCH series: Download + DELETE local (sync local to remote)" -ForegroundColor Red
        Write-Host "  PULL series:  Download only (no delete)" -ForegroundColor Green
        Write-Host "  PUSH series:  Upload + DELETE remote (sync remote to local)" -ForegroundColor Red
        Write-Host ""
        Write-Host "Download Commands (from remote):" -ForegroundColor Yellow
        Write-Host "  pull             - Download only, NO delete (safe)" -ForegroundColor Green
        Write-Host "  pull .87/.89/.154 - Pull from specific server"
        Write-Host "  pull87/pull89/pull154 - Shorthand aliases"
        Write-Host "  autopull [server]  - Auto-pull if changes detected"
        Write-Host "  autopull87/89/154  - Shorthand aliases"
        Write-Host "  watchpull        - Background auto-download"
        Write-Host ""
        Write-Host "  fetch            - Download + DELETE local extras" -ForegroundColor Red
        Write-Host "  fetch .87/.89/.154 - Fetch from specific server"
        Write-Host "  fetch87/fetch89/fetch154 - Shorthand aliases"
        Write-Host "  autofetch [server] - Auto-fetch if changes detected"
        Write-Host "  autofetch87/89/154 - Shorthand aliases"
        Write-Host "  watchfetch       - Background auto-fetch"
        Write-Host ""
        Write-Host "Upload Commands (to remote):" -ForegroundColor Yellow
        Write-Host "  push             - Upload + DELETE remote extras" -ForegroundColor Red
        Write-Host "  push87/push89/push154 - Push to specific server"
        Write-Host "  pushall          - Push to all servers"
        Write-Host "  autopush [server] - Auto-push if changes detected"
        Write-Host "  autopush87/89/154/all - Shorthand aliases"
        Write-Host "  watchpush        - Background auto-upload"
        Write-Host ""
        Write-Host "Code Diff Analysis:" -ForegroundColor Yellow
        Write-Host "  sync-diff [server]       - Show code diff (no transfer)"
        Write-Host "  sync-diff --summary      - File list only (no code)"
        Write-Host "  sync-diff --stat         - File + line counts"
        Write-Host "  sync-diff-summary [srv]  - Quick summary"
        Write-Host "  sync-diff-file <file>    - Single file diff"
        Write-Host "  sync-log [N]             - Show sync history (last N)"
        Write-Host "  sync-stop                - Stop all daemons"
        Write-Host ""
        Write-Host "Diff Options (for push/pull/fetch):" -ForegroundColor Yellow
        Write-Host "  --no-diff       - Skip diff analysis"
        Write-Host "  --diff-summary  - Show summary only"
        Write-Host "  --diff-stat     - Show file statistics"
        Write-Host "  --diff-full     - Full code diff (default)"
        Write-Host "  --force         - Skip diff + no confirm"
        Write-Host "  --quick         - Same as --force"
        Write-Host ""
        Write-Host "GPU Status:" -ForegroundColor Yellow
        Write-Host "  gpus             - GPU status overview (all servers)"
        Write-Host "  gpu [89|87|154]  - Detailed GPU status (nvidia-smi)"
        Write-Host ""
        Write-Host "SSH & Remote Execution:" -ForegroundColor Yellow
        Write-Host "  ssh [87:3]       - SSH to server:node (with GPU status)"
        Write-Host "  issh             - Interactive SSH selector (GPU status menu)"
        Write-Host "  run [87:3] [gpu] - Compile and run on node"
        Write-Host "  jobs [87:3]      - Check running jobs on node"
        Write-Host "  kill [87:3]      - Kill running jobs on node"
        Write-Host ""
        Write-Host "Status & Info:" -ForegroundColor Yellow
        Write-Host "  status           - Show sync status"
        Write-Host "  diff             - Compare local vs remote (code diff)"
        Write-Host "  diff87/diff89/diff154/diffall - Diff specific server"
        Write-Host "  log [server]     - View remote log files"
        Write-Host "  log87/log89/log154 - Log from specific server"
        Write-Host "  issynced         - Quick one-line status check"
        Write-Host "  bgstatus         - Check all background processes"
        Write-Host "  syncstatus       - Check sync background status"
        Write-Host ""
        Write-Host "Other Commands:" -ForegroundColor Yellow
        Write-Host "  clone            - Full download from remote"
        Write-Host "  reset            - Delete remote-only files"
        Write-Host "  sync             - Interactive: diff -> confirm -> push"
        Write-Host "  fullsync         - Push + Reset (exact mirror)"
        Write-Host ""
        Write-Host "VTK File Management:" -ForegroundColor Yellow
        Write-Host "  vtkrename        - Auto-rename VTK files to zero-padded"
        Write-Host "  vtkrename status/log/stop - Manage VTK renamer"
        Write-Host ""
        Write-Host "Background Commands:" -ForegroundColor Yellow
        Write-Host "  watch<cmd> [server] - Start monitoring"
        Write-Host "  watch<cmd> status   - Check status"
        Write-Host "  watch<cmd> log      - View log"
        Write-Host "  watch<cmd> stop     - Stop monitoring"
        Write-Host ""
        Write-Host "Server/Node Combos:" -ForegroundColor Yellow
        Write-Host "  .89:0   - Direct connection to .89 (V100-32G)"
        Write-Host "  .87:2/3/5/6 - Via .87 to ib2/3/5/6"
        Write-Host "  .154:1/4/7/9 - Via .154 to ib1/4/7/9"
        Write-Host ""
        Write-Host "Examples:" -ForegroundColor Yellow
        Write-Host "  mobaxterm push87              # Push with code diff + confirm"
        Write-Host "  mobaxterm push87 --quick      # Push immediately, no diff"
        Write-Host "  mobaxterm push --diff-summary  # Push with summary only"
        Write-Host "  mobaxterm sync-diff 87        # Show diff for .87 (no transfer)"
        Write-Host "  mobaxterm sync-diff-file main.cu  # Diff single file"
        Write-Host "  mobaxterm sync-log            # Show sync history"
        Write-Host "  mobaxterm gpus                # Check all GPU status"
        Write-Host "  mobaxterm ssh 89:0            # SSH to .89 directly"
        Write-Host "  mobaxterm issh                # Interactive SSH menu"
        Write-Host "  mobaxterm run 87:3 4          # Compile & run with 4 GPUs"
        Write-Host ""
    }
}

