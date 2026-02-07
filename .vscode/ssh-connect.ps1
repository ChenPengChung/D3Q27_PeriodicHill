# SSH Connect Script
# Usage: ./ssh-connect.ps1 <serverCombo> [panel]
# serverCombo format: "87:3" or "154:4"

param(
    [Parameter(Mandatory=$true)]
    [string]$ServerCombo
)

$Config = @{
    PlinkPath = "C:\Program Files\PuTTY\plink.exe"
    Password = "1256"
    Username = "chenpengchung"
    RemotePath = "/home/chenpengchung/D3Q27_PeriodicHill"
    Servers = @{
        "87" = "140.114.58.87"
        "154" = "140.114.58.154"
    }
}

# Parse serverCombo (e.g., "87:3" -> server=87, node=3)
$parts = $ServerCombo -split ":"
if ($parts.Count -ne 2) {
    Write-Host "[ERROR] Invalid format. Use: 87:3 or 154:4" -ForegroundColor Red
    exit 1
}

$serverKey = $parts[0]
$nodeNum = $parts[1]

if (-not $Config.Servers.ContainsKey($serverKey)) {
    Write-Host "[ERROR] Unknown server: $serverKey. Use 87 or 154." -ForegroundColor Red
    exit 1
}

$masterIP = $Config.Servers[$serverKey]
$childNode = "cfdlab-ib$nodeNum"

Write-Host "[SSH] Connecting via .$serverKey ($masterIP) to $childNode..." -ForegroundColor Cyan

# Execute SSH
& $Config.PlinkPath -ssh -pw $Config.Password "$($Config.Username)@$masterIP" -t "ssh -t $childNode 'cd $($Config.RemotePath); exec bash'"
