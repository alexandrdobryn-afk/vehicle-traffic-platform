param(
    [string]$Distro = "Ubuntu",
    [string]$Profile = "",
    [switch]$Cpu,
    [switch]$Build,
    [switch]$Foreground
)

$ErrorActionPreference = "Stop"
$launcher = Join-Path $PSScriptRoot "scripts\start.sh"

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw "WSL is not installed or wsl.exe is unavailable."
}

$linuxLauncher = (& wsl.exe -d $Distro -- wslpath -a -u $launcher).Trim()
if ($LASTEXITCODE -ne 0 -or -not $linuxLauncher) {
    throw "Cannot resolve the project path inside WSL distribution '$Distro'."
}

$arguments = @("-d", $Distro, "--", "bash", $linuxLauncher)
if ($Cpu) { $arguments += "--cpu" }
if ($Build) { $arguments += "--build" }
if ($Foreground) { $arguments += "--foreground" }
if ($Profile) { $arguments += @("--profile", $Profile) }

& wsl.exe @arguments
exit $LASTEXITCODE
