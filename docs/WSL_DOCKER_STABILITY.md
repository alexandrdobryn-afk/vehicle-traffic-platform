# WSL and Docker stability

## Verified symptom

On this Windows + WSL setup, BEVP containers can be healthy and then disappear because the WSL VM or Docker daemon is stopped from the Windows side. Docker logs show a graceful daemon termination, not a BEVP backend crash:

```text
Processing signal 'terminated'
Daemon shutdown complete
```

`docker compose restart: unless-stopped` handles application/container crashes, but it cannot keep containers alive if WSL stops the entire Docker daemon.

## Root cause

Microsoft documents that enabling `systemd` in WSL does not make systemd services keep the WSL instance alive. Docker Engine inside Ubuntu is therefore not a persistent Windows service. If BEVP is started through a short-lived detached WSL command, the command can exit and WSL can later stop the VM.

On this machine there is also a Windows-side WSL API issue: `wsl --version` and direct `wsl -d Ubuntu` can work, while `wsl -l -v` / `wsl --status` can fail with:

```text
Wsl/EnumerateDistros/Service/E_ACCESSDENIED
```

That requires a Windows-side WSL repair, not a backend code fix.

## Normal stable launch

Use the project launcher without `-Detached`. It defaults to foreground mode and requires CUDA unless CPU is explicitly requested:

```powershell
Set-Location "C:\Users\Admin\Desktop\CV drone\drone-vision-platform"
.\start.cmd -Build -Foreground
```

Keep the terminal open while testing. Use detached mode only when you know WSL will stay alive independently:

```powershell
.\start.cmd -Build -Detached
```

CPU is an emergency-only explicit mode:

```powershell
.\start.cmd -Cpu -Build -Foreground
```

## CUDA-first policy

BEVP test and training runs are NVIDIA-first. The launcher no longer silently falls back to CPU. If CUDA preflight fails, startup stops before rebuilding/running the stack.

The preflight checks:

1. Docker responds inside the selected WSL distribution.
2. Docker reports the `nvidia` runtime.
3. GPU compose overlay is selected.
4. `bevp-backend` receives NVIDIA Docker `DeviceRequests`.
5. The runtime-status API reads a cached runtime snapshot maintained by a background monitor. CUDA and ONNX Runtime probing run in a child Python process group with a watchdog, outside the FastAPI request path.

The GPU overlay gives GPU access only to the inference backend and training worker. The training API is a control plane and does not receive Docker GPU `DeviceRequests`; frontend startup is not blocked by training API readiness.

The normal launcher deliberately does not use direct WSL `nvidia-smi` as the main gate. On this machine, direct WSL `nvidia-smi` can wedge even when the Docker runtime is the thing we actually need to prove. The repair script still checks WSL `nvidia-smi`, but it does so with a Windows-side watchdog and then validates the real target path with a Docker GPU container.

The launcher also avoids running the CUDA proof through `docker exec` inside the live backend. Strict disposable-container CUDA probes are available, but they are opt-in because a wedged WSL/Docker GPU runtime can hang container creation itself:

```powershell
.\start.cmd -Build -Foreground -VerifyCudaContainers
```

This follows the CUDA-on-WSL boundary: install the NVIDIA driver on Windows, do not install a Linux display driver in WSL, and use NVIDIA Container Toolkit for Docker GPU containers.

## Repair WSL access denied

Open PowerShell as Administrator and run:

```powershell
Set-Location "C:\Users\Admin\Desktop\CV drone\drone-vision-platform"
powershell -ExecutionPolicy Bypass -File .\scripts\repair_wsl_docker.ps1
```

For CUDA repair and validation:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\repair_wsl_docker.ps1 -CheckCuda
```

With `-CheckCuda`, repair is only considered complete after a disposable Docker GPU smoke test succeeds:

```text
docker run --rm --runtime=nvidia --gpus all ... nvidia-smi -L
```

The default smoke image is `nvidia/cuda:12.1.1-base-ubuntu22.04`. Override it only when the image tag is unavailable:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\repair_wsl_docker.ps1 -CheckCuda -CudaSmokeImage nvidia/cuda:12.4.1-base-ubuntu22.04
```

Optional WSL disk/memory cleanup config:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\repair_wsl_docker.ps1 -WriteRecommendedWslConfig
```

This writes `%USERPROFILE%\.wslconfig` with:

```ini
[experimental]
autoMemoryReclaim=dropCache
sparseVhd=true
```

## If Docker still disappears

1. Run the repair script from Administrator PowerShell.
2. Start BEVP in foreground mode.
3. If Docker/WSL is still wedged, explicitly run `wsl --terminate Ubuntu` for the selected BEVP distro. Use `wsl --shutdown` only when you intentionally want to stop every WSL distro.
4. If `wsl -l -v` still returns `E_ACCESSDENIED`, reboot Windows. At that point WSLService access is broken outside the project.

The Windows launcher does not terminate WSL automatically during normal startup. Destructive recovery is kept in the explicit repair path so BEVP does not kill unrelated WSL work.

## If containers are healthy but Windows localhost is refused

If `docker compose ps` inside Ubuntu shows BEVP services healthy, and URLs work through the WSL IP but Windows `http://localhost:3100` is refused, Windows localhost forwarding is broken outside the app.

Open Administrator PowerShell and run:

```powershell
Set-Location "C:\Users\Admin\Desktop\CV drone\drone-vision-platform"
powershell -ExecutionPolicy Bypass -File .\scripts\fix_windows_localhost_proxy.ps1
```

This maps `127.0.0.1:3100`, `127.0.0.1:8100`, and `127.0.0.1:8101` to the current Ubuntu WSL IP. Re-run it if WSL is restarted and receives a new IP address.
