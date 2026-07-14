param(
    [string]$Distro = "Ubuntu",
    [string]$Profile = "",
    [switch]$Cpu,
    [switch]$AllowCpuFallback,
    [switch]$Build,
    [switch]$VerifyCudaContainers,
    [switch]$Foreground,
    [switch]$Detached
)

$ErrorActionPreference = "Stop"
$launcher = Join-Path $PSScriptRoot "scripts\start.sh"

function ConvertTo-WslPathFallback([string]$Path) {
    $resolved = [System.IO.Path]::GetFullPath($Path)
    if ($resolved -notmatch '^([A-Za-z]):\\(.*)$') {
        throw "Cannot convert non-drive path to WSL path: $resolved"
    }
    $drive = $Matches[1].ToLowerInvariant()
    $rest = $Matches[2] -replace '\\', '/'
    return "/mnt/$drive/$rest"
}

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw "WSL is not installed or wsl.exe is unavailable."
}

if ($Foreground -and $Detached) {
    throw "Use either -Foreground or -Detached, not both."
}

$useForeground = $Foreground -or -not $Detached

$linuxLauncher = ConvertTo-WslPathFallback $launcher

function New-WslLaunchArguments {
    $args = @("-d", $Distro, "--", "bash", $linuxLauncher)
    if ($Cpu) { $args += "--cpu" }
    if ($AllowCpuFallback) { $args += "--allow-cpu-fallback" }
    if ($Build) { $args += "--build" }
    if ($VerifyCudaContainers) { $args += "--verify-cuda-containers" }
    if ($useForeground) { $args += "--foreground" }
    if ($Profile) { $args += @("--profile", $Profile) }
    return $args
}

function Invoke-BevpLaunch {
    $args = New-WslLaunchArguments
    & wsl.exe @args
    return $LASTEXITCODE
}

if ($useForeground) {
    Write-Host "Starting BEVP in foreground mode. Keep this window open while testing." -ForegroundColor Yellow
    if ($Cpu) {
        Write-Warning "CPU mode was explicitly requested. CUDA will be disabled."
    } elseif ($AllowCpuFallback) {
        Write-Warning "CUDA will be attempted first, but CPU fallback is allowed."
    } else {
        Write-Host "CUDA is required for this launch. Startup will stop instead of silently falling back to CPU." -ForegroundColor Cyan
    }
} else {
    Write-Warning "Detached mode does not keep WSL alive on this Windows/WSL setup. Use -Foreground for stable local testing."
}

$exitCode = Invoke-BevpLaunch

if (($exitCode -eq 42) -and -not $Cpu -and -not $AllowCpuFallback) {
    Write-Warning "WSL/Docker/GPU reported a recoverable wedge. Normal launch will not terminate WSL automatically."
    Write-Host "Run the explicit repair command from Administrator PowerShell if you want to reset WSL/Docker/GPU state:" -ForegroundColor Yellow
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\scripts\repair_wsl_docker.ps1 -CheckCuda" -ForegroundColor Yellow
}

exit $exitCode
