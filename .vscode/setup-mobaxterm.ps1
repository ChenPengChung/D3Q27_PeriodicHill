# MobaXterm Sync - PowerShell Profile Setup
# 執行此腳本來設定 mobaxterm 命令

$profilePath = $PROFILE
$scriptPath = 'c:\Users\88697.CHENPENGCHUNG12\Desktop\GitHub-PeriodicHill\D3Q27_PeriodicHill\.vscode\mobaxterm.ps1'

Write-Host "MobaXterm Sync - Profile Setup" -ForegroundColor Cyan
Write-Host ""

# 檢查 profile 是否存在
if (-not (Test-Path $profilePath)) {
    Write-Host "PowerShell Profile does not exist, creating..." -ForegroundColor Yellow
    $profileDir = Split-Path $profilePath -Parent
    if (-not (Test-Path $profileDir)) {
        New-Item -ItemType Directory -Path $profileDir -Force | Out-Null
    }
    New-Item -ItemType File -Path $profilePath -Force | Out-Null
}

# 檢查是否已經設定
$existingProfile = Get-Content $profilePath -Raw -ErrorAction SilentlyContinue
if ($existingProfile -and $existingProfile.Contains("function mobaxterm")) {
    Write-Host "mobaxterm command already configured in Profile" -ForegroundColor Green
}
else {
    # 使用單引號 here-string 避免變數展開問題
    $profileContent = @'

# MobaXterm Sync Commands
function mobaxterm {
    param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments)
    & 'c:\Users\88697.CHENPENGCHUNG12\Desktop\GitHub-PeriodicHill\D3Q27_PeriodicHill\.vscode\mobaxterm.ps1' @Arguments
}
'@
    Add-Content -Path $profilePath -Value $profileContent
    Write-Host "Added mobaxterm command to Profile" -ForegroundColor Green
}

Write-Host ""
Write-Host "Profile location: $profilePath" -ForegroundColor Gray
Write-Host ""
Write-Host "Restart PowerShell or run: . `$PROFILE" -ForegroundColor Yellow
Write-Host ""
Write-Host "Usage:" -ForegroundColor Cyan
Write-Host "   mobaxterm check   - Check differences"
Write-Host "   mobaxterm add .   - Show changes"
Write-Host "   mobaxterm push    - Push to remote"
Write-Host "   mobaxterm pull    - Pull from remote"
Write-Host ""
