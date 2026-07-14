param(
    [string]$Distro = "Ubuntu",
    [switch]$WriteRecommendedWslConfig,
    [switch]$CheckCuda,
    [string]$CudaSmokeImage = "nvidia/cuda:12.1.1-base-ubuntu22.04"
)

$ErrorActionPreference = "Stop"

function Test-IsAdmin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Run-Step($Title, [scriptblock]$Action) {
    Write-Host ""
    Write-Host "== $Title ==" -ForegroundColor Cyan
    & $Action
}

function Find-NvidiaSmi {
    $cmd = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $system32 = Join-Path $env:WINDIR "System32\nvidia-smi.exe"
    if (Test-Path -LiteralPath $system32) { return $system32 }
    return $null
}

function Quote-NativeArgument([string]$Value) {
    if ($null -eq $Value) { return '""' }
    if ($Value -match '^[A-Za-z0-9_./:=+@%-]+$') { return $Value }
    return '"' + ($Value -replace '"', '\"') + '"'
}

function Invoke-NativeProcess {
    param(
        [string]$FilePath,
        [string[]]$Arguments = @(),
        [int]$TimeoutSeconds = 60,
        [switch]$AllowNonZero
    )

    $stdout = [System.IO.Path]::GetTempFileName()
    $stderr = [System.IO.Path]::GetTempFileName()
    $argumentLine = ($Arguments | ForEach-Object { Quote-NativeArgument $_ }) -join " "

    try {
        $process = Start-Process -FilePath $FilePath -ArgumentList $argumentLine -NoNewWindow -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            try { $process.Kill() } catch { }
            throw "$FilePath did not finish within $TimeoutSeconds seconds."
        }

        $out = Get-Content -LiteralPath $stdout -Raw -ErrorAction SilentlyContinue
        $err = Get-Content -LiteralPath $stderr -Raw -ErrorAction SilentlyContinue
        if ($out) { Write-Host ($out.TrimEnd()) }
        if ($err) { Write-Host ($err.TrimEnd()) -ForegroundColor DarkYellow }

        if (($process.ExitCode -ne 0) -and -not $AllowNonZero) {
            throw "$FilePath exited with code $($process.ExitCode)."
        }
        return $process.ExitCode
    }
    finally {
        Remove-Item -LiteralPath $stdout, $stderr -Force -ErrorAction SilentlyContinue
    }
}

function Invoke-WslBash {
    param(
        [string]$Script,
        [int]$TimeoutSeconds = 60,
        [switch]$Root,
        [switch]$AllowNonZero
    )

    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Script))
    $args = @("-d", $Distro)
    if ($Root) { $args += @("-u", "root") }
    $args += @("--", "bash", "-lc", "echo $encoded | base64 -d | bash")
    return Invoke-NativeProcess -FilePath "wsl.exe" -Arguments $args -TimeoutSeconds $TimeoutSeconds -AllowNonZero:$AllowNonZero
}

if (-not (Test-IsAdmin)) {
    throw "Run this script from PowerShell opened as Administrator."
}

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw "wsl.exe is not available."
}

Run-Step "WSL version" {
    Invoke-NativeProcess -FilePath "wsl.exe" -Arguments @("--version") -TimeoutSeconds 20
}

Run-Step "Shutdown WSL VM" {
    Invoke-NativeProcess -FilePath "wsl.exe" -Arguments @("--shutdown") -TimeoutSeconds 30
    Start-Sleep -Seconds 3
}

Run-Step "Restart WSLService" {
    Restart-Service WSLService -Force
    Start-Sleep -Seconds 5
    Invoke-NativeProcess -FilePath "sc.exe" -Arguments @("query", "WSLService") -TimeoutSeconds 20
}

if ($WriteRecommendedWslConfig) {
    Run-Step "Write recommended .wslconfig" {
        $path = Join-Path $env:USERPROFILE ".wslconfig"
        if (Test-Path -LiteralPath $path) {
            $backup = "$path.bak-$(Get-Date -Format yyyyMMdd-HHmmss)"
            Copy-Item -LiteralPath $path -Destination $backup
            Write-Host "Backup: $backup"
        }
        @"
[experimental]
autoMemoryReclaim=dropCache
sparseVhd=true
"@ | Set-Content -LiteralPath $path -Encoding ASCII
        Write-Host "Wrote $path"
        Invoke-NativeProcess -FilePath "wsl.exe" -Arguments @("--shutdown") -TimeoutSeconds 30
        Start-Sleep -Seconds 3
    }
}

Run-Step "WSL distro status" {
    $listExit = Invoke-NativeProcess -FilePath "wsl.exe" -Arguments @("-l", "-v") -TimeoutSeconds 20 -AllowNonZero
    $statusExit = Invoke-NativeProcess -FilePath "wsl.exe" -Arguments @("--status") -TimeoutSeconds 20 -AllowNonZero
    if (($listExit -ne 0) -or ($statusExit -ne 0)) {
        Write-Warning "WSL status commands returned an error. Continuing because direct distro commands may still work."
    }
}

Run-Step "Ensure Docker is active in $Distro" {
    Invoke-WslBash -Root -TimeoutSeconds 90 -Script @'
set -Eeuo pipefail
systemctl enable docker >/dev/null 2>&1 || true
systemctl start docker >/dev/null 2>&1 || service docker start
timeout 30s docker info >/dev/null
if command -v systemctl >/dev/null 2>&1; then
  systemctl is-active docker || true
fi
'@
}

if ($CheckCuda) {
    Run-Step "Windows NVIDIA driver" {
        $nvidiaSmi = Find-NvidiaSmi
        if (-not $nvidiaSmi) {
            throw "nvidia-smi.exe was not found. Install or repair the Windows NVIDIA driver with WSL CUDA support."
        }
        Invoke-NativeProcess -FilePath $nvidiaSmi -Arguments @("-L") -TimeoutSeconds 20
    }

    Run-Step "WSL NVIDIA device visibility" {
        Invoke-WslBash -TimeoutSeconds 20 -Script @'
set -Eeuo pipefail
if command -v nvidia-smi >/dev/null 2>&1; then
  smi="$(command -v nvidia-smi)"
elif [ -x /usr/lib/wsl/lib/nvidia-smi ]; then
  smi="/usr/lib/wsl/lib/nvidia-smi"
else
  echo "nvidia-smi is missing in WSL. The Windows driver may not be mapped into WSL." >&2
  exit 11
fi
timeout 8s "$smi" -L
'@
    }

    Run-Step "Check for Linux NVIDIA display driver packages" {
        Invoke-WslBash -TimeoutSeconds 30 -AllowNonZero -Script @'
set -Eeuo pipefail
bad="$(dpkg-query -W -f='${Package}\n' 'nvidia-driver-*' 'cuda-drivers' 'cuda' 2>/dev/null || true)"
if [ -n "$bad" ]; then
  echo "WARNING: Linux NVIDIA driver/meta packages detected in WSL:"
  echo "$bad"
  echo "CUDA on WSL should use the Windows NVIDIA display driver. Do not install Linux display drivers inside WSL."
else
  echo "No Linux NVIDIA display driver meta packages detected."
fi
'@
    }

    Run-Step "Configure NVIDIA Docker runtime" {
        Invoke-WslBash -Root -TimeoutSeconds 90 -Script @'
set -Eeuo pipefail
if ! command -v nvidia-ctk >/dev/null 2>&1; then
  echo "nvidia-ctk is not installed; install NVIDIA Container Toolkit inside WSL to use Docker GPU containers." >&2
  exit 13
fi
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker >/dev/null 2>&1 || service docker restart
timeout 30s docker info >/dev/null
'@
    }

    Run-Step "Optional NVIDIA CDI spec" {
        $cdiExit = Invoke-WslBash -Root -TimeoutSeconds 60 -AllowNonZero -Script @'
set -Eeuo pipefail
mkdir -p /etc/cdi
nvidia-ctk cdi generate --mode=wsl --device-name-strategy=type-index --output=/etc/cdi/nvidia.yaml
test -s /etc/cdi/nvidia.yaml
grep -q 'nvidia.com/gpu' /etc/cdi/nvidia.yaml
systemctl restart docker >/dev/null 2>&1 || service docker restart
'@
        if ($cdiExit -ne 0) {
            Write-Warning "CDI generation failed. Continuing because BEVP uses Docker Compose GPU reservations through the NVIDIA runtime."
        }
    }

    Run-Step "Docker NVIDIA runtime" {
        Invoke-WslBash -TimeoutSeconds 30 -Script @'
set -Eeuo pipefail
docker info --format '{{json .Runtimes}}'
docker info --format '{{json .Runtimes}}' | grep -qi '"nvidia"'
'@
    }

    Run-Step "Docker GPU container smoke test" {
        $script = @"
set -Eeuo pipefail
smoke_name=bevp-repair-cuda-smoke
echo "Using CUDA smoke image: $CudaSmokeImage"
timeout 20s docker rm -f "`$smoke_name" >/dev/null 2>&1 || true
smoke_status=0
timeout 240s docker run --name "`$smoke_name" --runtime=nvidia --gpus all --pull=missing "$CudaSmokeImage" nvidia-smi -L || smoke_status=`$?
timeout 20s docker rm -f "`$smoke_name" >/dev/null 2>&1 || true
exit "`$smoke_status"
"@
        Invoke-WslBash -Root -TimeoutSeconds 300 -Script $script
    }
}

Run-Step "Docker disk usage" {
    Invoke-WslBash -TimeoutSeconds 45 -Script @'
set -Eeuo pipefail
timeout 30s docker system df
'@
}

Write-Host ""
if ($CheckCuda) {
    Write-Host "Repair complete. CUDA was validated through Docker GPU container execution." -ForegroundColor Green
} else {
    Write-Host "Repair complete. Docker/WSL was checked; run again with -CheckCuda to validate GPU containers." -ForegroundColor Green
}
Write-Host "Start BEVP from a normal project PowerShell window:"
Write-Host '  .\start.cmd -Build -Foreground'
Write-Host "The project launcher defaults to foreground mode; keep that window open."
Write-Host "Use .\start.cmd -Cpu -Build -Foreground only for explicit CPU emergency mode."
