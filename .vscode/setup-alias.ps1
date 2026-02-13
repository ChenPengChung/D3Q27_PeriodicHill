<#
.SYNOPSIS
    Setup 'mobaxterm' command alias for PowerShell
.DESCRIPTION
    Run this script once to add 'mobaxterm' function to your PowerShell profile.
    After setup, you can use: mobaxterm <command> [arguments]
.EXAMPLE
    .\setup-alias.ps1
#>

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$MobaxtermPath = Join-Path $ScriptDir "mobaxterm.ps1"

# 確保 Profile 目錄存在
$ProfileDir = Split-Path -Parent $PROFILE
if (-not (Test-Path $ProfileDir)) {
    New-Item -ItemType Directory -Path $ProfileDir -Force | Out-Null
}

# 確保 Profile 檔案存在
if (-not (Test-Path $PROFILE)) {
    New-Item -ItemType File -Path $PROFILE -Force | Out-Null
}

# 要添加的 function
$FunctionCode = @"

# ========== MobaXterm Alias ==========
function mobaxterm {
    & '$MobaxtermPath' @args
}
# ========== End MobaXterm Alias ==========
"@

# 檢查是否已經添加過
$ProfileContent = Get-Content $PROFILE -Raw -ErrorAction SilentlyContinue
if ($ProfileContent -and $ProfileContent.Contains("MobaXterm Alias")) {
    Write-Host "[INFO] 'mobaxterm' alias already exists in your profile." -ForegroundColor Yellow
    Write-Host "       Profile: $PROFILE" -ForegroundColor Gray
} else {
    # 添加到 Profile
    Add-Content -Path $PROFILE -Value $FunctionCode
    Write-Host "[SUCCESS] 'mobaxterm' alias added to your PowerShell profile!" -ForegroundColor Green
    Write-Host "          Profile: $PROFILE" -ForegroundColor Gray
}

Write-Host ""
Write-Host "To activate now, run:" -ForegroundColor Cyan
Write-Host "  . `$PROFILE" -ForegroundColor White
Write-Host ""
Write-Host "Or restart PowerShell, then you can use:" -ForegroundColor Cyan
Write-Host "  mobaxterm gpus" -ForegroundColor White
Write-Host "  mobaxterm ssh 87:3" -ForegroundColor White
Write-Host "  mobaxterm watchpull .89" -ForegroundColor White
Write-Host "  mobaxterm push" -ForegroundColor White
Write-Host ""
