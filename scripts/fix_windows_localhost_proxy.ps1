param(
    [string]$Distro = "Ubuntu",
    [int[]]$Ports = @(3100, 8100, 8101)
)

$ErrorActionPreference = "Stop"

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator
)
if (-not $isAdmin) {
    throw "Run this script from Administrator PowerShell. Windows portproxy requires admin rights."
}

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw "wsl.exe is unavailable."
}

$ipOutput = & wsl.exe -d $Distro -- hostname -I
if ($LASTEXITCODE -ne 0) {
    throw "Could not query WSL IP for distro '$Distro'."
}

$wslIp = ($ipOutput -split '\s+' | Where-Object { $_ -match '^\d+\.\d+\.\d+\.\d+$' } | Select-Object -First 1)
if (-not $wslIp) {
    throw "Could not parse WSL IPv4 address from: $ipOutput"
}

Write-Host "WSL distro: $Distro"
Write-Host "WSL IPv4:   $wslIp"

foreach ($port in $Ports) {
    Write-Host "Mapping 127.0.0.1:$port -> ${wslIp}:$port"
    & netsh interface portproxy delete v4tov4 listenaddress=127.0.0.1 listenport=$port | Out-Null
    & netsh interface portproxy add v4tov4 listenaddress=127.0.0.1 listenport=$port connectaddress=$wslIp connectport=$port | Out-Null
}

Write-Host ""
Write-Host "Current Windows portproxy mappings:"
& netsh interface portproxy show v4tov4

Write-Host ""
Write-Host "Done. Test from normal PowerShell:"
Write-Host "  Invoke-WebRequest http://localhost:3100 -UseBasicParsing"
