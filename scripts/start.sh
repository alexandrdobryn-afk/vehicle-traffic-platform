#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

force_cpu=false
allow_cpu_fallback=false
build=false
follow_logs=false
profile=""
compose_timeout_seconds=300
verify_cuda_containers=false
EXIT_WSL_RECOVERY=42
EXIT_CUDA_REQUIRED=43

while (($#)); do
  case "$1" in
    --cpu) force_cpu=true ;;
    --allow-cpu-fallback) allow_cpu_fallback=true ;;
    --build) build=true ;;
    --foreground) follow_logs=true ;;
    --verify-cuda-containers) verify_cuda_containers=true ;;
    --profile)
      shift
      profile="${1:-}"
      [[ -n "$profile" ]] || { echo "ERROR: --profile requires a value." >&2; exit 2; }
      ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

if ! timeout 15s docker info >/dev/null 2>&1; then
  echo "WARNING: Docker is not responding in this WSL distribution (15s timeout)." >&2
  echo "WARNING: Trying to start/restart Docker inside WSL before failing." >&2
  if command -v systemctl >/dev/null 2>&1; then
    sudo -n systemctl start docker >/dev/null 2>&1 || true
  fi
  sudo -n service docker start >/dev/null 2>&1 || true
  if ! timeout 30s docker info >/dev/null 2>&1; then
    echo "ERROR: Docker is still not responding after the local restart attempt." >&2
    echo "Run from Administrator PowerShell:" >&2
    echo "  powershell -ExecutionPolicy Bypass -File .\\scripts\\repair_wsl_docker.ps1 -CheckCuda" >&2
    exit "$EXIT_WSL_RECOVERY"
  fi
fi

gpu_available=false
gpu_failure_reason=""

docker_nvidia_runtime_ready() {
  timeout 10s docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -qi '"nvidia"'
}

if ! $force_cpu; then
  echo "Checking NVIDIA Docker runtime..."
  if docker_nvidia_runtime_ready; then
    gpu_available=true
    echo "NVIDIA Docker runtime detected."
  else
    gpu_failure_reason="Docker does not report the nvidia runtime. Run the repair script once to configure NVIDIA Container Toolkit for Docker."
    if ! $allow_cpu_fallback; then
      echo "ERROR: CUDA is required, but NVIDIA/WSL/Docker GPU preflight failed." >&2
      echo "Reason: $gpu_failure_reason" >&2
      echo "" >&2
      echo "Run from Administrator PowerShell:" >&2
      echo "  powershell -ExecutionPolicy Bypass -File .\\scripts\\repair_wsl_docker.ps1 -CheckCuda" >&2
      echo "" >&2
      echo "For an explicit CPU-only emergency launch, use:" >&2
      echo "  .\\start.cmd -Cpu -Build -Foreground" >&2
      exit "$EXIT_CUDA_REQUIRED"
    fi
    echo "WARNING: NVIDIA GPU probe failed; continuing on CPU because --allow-cpu-fallback was set." >&2
    echo "WARNING: $gpu_failure_reason" >&2
  fi
fi

compose=(docker compose -f docker-compose.yml)
if $gpu_available; then
  compose+=(-f docker-compose.gpu.yml)
  echo "Compute mode: NVIDIA CUDA (GPU overlay enabled)."
else
  echo "Compute mode: CPU."
fi

if [[ -n "$profile" ]]; then
  compose+=(--profile "$profile")
fi

handle_compose_failure() {
  local status="$1"
  local action="$2"
  local project_dir_quoted
  printf -v project_dir_quoted '%q' "$PROJECT_DIR"
  if (( status == 124 )); then
    echo "ERROR: docker compose ${action} did not finish within ${compose_timeout_seconds}s." >&2
    echo "Docker/WSL may be wedged. Run from Administrator PowerShell:" >&2
    echo "  powershell -ExecutionPolicy Bypass -File .\\scripts\\repair_wsl_docker.ps1 -CheckCuda" >&2
    exit "$EXIT_WSL_RECOVERY"
  fi
  echo "ERROR: docker compose ${action} returned an application/service error (exit $status)." >&2
  echo "This is not automatically classified as a WSL/GPU wedge. Check the unhealthy service logs above or run:" >&2
  echo "  wsl -d Ubuntu -- bash -lc \"cd ${project_dir_quoted} && docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs --tail=120\"" >&2
  exit 1
}

run_cuda_image_probe() {
  local image="$1"
  local label="$2"
  local probe_name="bevp-cuda-probe-${label//[^A-Za-z0-9]/-}-$$"
  local probe_output probe_status=0

  echo "Verifying CUDA in disposable ${label} container..."
  probe_output="$(
    timeout 75s docker run --rm --name "$probe_name" --runtime=nvidia --gpus all \
      --entrypoint python "$image" \
      -c 'import json, sys, torch; ok=torch.cuda.is_available(); data={"cuda_available": ok, "device_count": torch.cuda.device_count(), "device_name": torch.cuda.get_device_name(0) if ok else None}; print(json.dumps(data), flush=True); sys.exit(0 if ok else 7)' 2>&1
  )" || probe_status=$?
  timeout 15s docker rm -f "$probe_name" >/dev/null 2>&1 || true

  if (( probe_status != 0 )); then
    echo "ERROR: CUDA is not usable in the ${label} image." >&2
    if [[ -n "$probe_output" ]]; then
      echo "$probe_output" >&2
    else
      echo "The CUDA probe produced no output before it failed or timed out." >&2
    fi
    echo "This probe runs in a disposable container so the live BEVP containers are not killed by the test." >&2
    if (( probe_status == 124 )); then
      exit "$EXIT_WSL_RECOVERY"
    fi
    exit "$EXIT_CUDA_REQUIRED"
  fi

  echo "CUDA verified in ${label} image: $probe_output"
}

if $build; then
  compose_timeout_seconds=1800
  build_status=0
  timeout "${compose_timeout_seconds}s" "${compose[@]}" build || build_status=$?
  if (( build_status != 0 )); then
    handle_compose_failure "$build_status" "build"
  fi
fi

if $gpu_available && { $verify_cuda_containers || [[ "${BEVP_VERIFY_CUDA_CONTAINERS:-}" == "1" ]]; }; then
  run_cuda_image_probe "bevp-backend:latest" "backend"
  run_cuda_image_probe "bevp-training-worker:latest" "training-worker"
elif $gpu_available; then
  echo "Skipping heavy CUDA container smoke probes during normal launch."
  echo "Use --verify-cuda-containers or the repair script for explicit Docker GPU smoke validation."
fi

up_args=(up -d --remove-orphans)
compose_status=0
timeout "${compose_timeout_seconds}s" "${compose[@]}" "${up_args[@]}" || compose_status=$?
if (( compose_status != 0 )); then
  handle_compose_failure "$compose_status" "up"
fi

print_startup_diagnostics() {
  echo "" >&2
  echo "=== docker compose ps ===" >&2
  timeout 20s "${compose[@]}" ps >&2 || echo "docker compose ps timed out" >&2
  echo "" >&2
  echo "=== backend logs ===" >&2
  timeout 20s "${compose[@]}" logs --tail=80 backend >&2 || echo "backend logs timed out" >&2
  echo "" >&2
  echo "=== frontend logs ===" >&2
  timeout 20s "${compose[@]}" logs --tail=80 frontend >&2 || echo "frontend logs timed out" >&2
  echo "" >&2
  echo "=== training-api logs ===" >&2
  timeout 20s "${compose[@]}" logs --tail=80 training-api >&2 || echo "training-api logs timed out" >&2
  echo "" >&2
  echo "=== postgres logs ===" >&2
  timeout 20s "${compose[@]}" logs --tail=40 postgres >&2 || echo "postgres logs timed out" >&2
  echo "" >&2
  echo "=== redis logs ===" >&2
  timeout 20s "${compose[@]}" logs --tail=40 redis >&2 || echo "redis logs timed out" >&2
}

container_health_status() {
  local container="$1"
  timeout 8s docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$container" 2>/dev/null || true
}

wait_for_container_health() {
  local container="$1"
  local timeout_seconds="${2:-180}"
  local status=""
  local waited=0
  while (( waited < timeout_seconds )); do
    status="$(container_health_status "$container")"
    case "$status" in
      healthy|running)
        echo "$container is $status."
        return 0
        ;;
      exited|dead)
        echo "ERROR: $container is $status." >&2
        return 1
        ;;
    esac
    sleep 2
    waited=$((waited + 2))
  done
  echo "ERROR: $container did not become healthy within ${timeout_seconds}s. Last status: ${status:-missing}" >&2
  return 1
}

wait_for_url() {
  local name="$1"
  local url="$2"
  local timeout_seconds="${3:-120}"
  local waited=0
  while (( waited < timeout_seconds )); do
    if curl -fsS --max-time 5 "$url" >/dev/null 2>&1; then
      echo "$name is reachable: $url"
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
  done
  echo "ERROR: $name did not answer within ${timeout_seconds}s: $url" >&2
  return 1
}

if ! wait_for_container_health bevp-postgres 120 \
  || ! wait_for_container_health bevp-redis 120 \
  || ! wait_for_container_health bevp-backend 240 \
  || ! wait_for_container_health bevp-frontend 180 \
  || ! wait_for_url "Inference API readiness" "http://localhost:8100/api/v1/health/ready" 120 \
  || ! wait_for_url "Modules API" "http://localhost:8100/api/v1/modules" 120 \
  || ! wait_for_url "Frontend modules page" "http://localhost:3100/modules" 120; then
  print_startup_diagnostics
  exit 1
fi

training_status="READY"
if ! wait_for_container_health bevp-training-api 60 \
  || ! wait_for_url "Training API readiness" "http://localhost:8101/api/v1/training/health/ready" 60; then
  training_status="DEGRADED"
  echo "WARNING: Training API is degraded. Core BEVP UI and inference are running." >&2
  echo "Open the UI and check the Training section logs if training is needed now." >&2
fi

worker_status="$(container_health_status bevp-training-worker)"
if [[ "$worker_status" != "running" && "$worker_status" != "healthy" ]]; then
  echo "WARNING: Training worker is ${worker_status:-missing}. Training jobs may not run until it recovers." >&2
fi

if $gpu_available; then
  if ! timeout 20s docker inspect bevp-backend --format '{{json .HostConfig.DeviceRequests}}' 2>/dev/null | grep -qi nvidia; then
    echo "ERROR: bevp-backend started without NVIDIA Docker DeviceRequests." >&2
    echo "The GPU compose overlay was selected, but Docker did not attach the GPU device." >&2
    exit "$EXIT_WSL_RECOVERY"
  fi

  echo "NVIDIA DeviceRequests verified for bevp-backend."
fi

echo "BEVP core: READY | Frontend: READY | Inference backend: READY | Training API: ${training_status}"
echo "UI: http://localhost:3100 | API: http://localhost:8100/docs | Training: http://localhost:8101/docs"
wsl_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
if [[ -n "$wsl_ip" ]]; then
  echo "If Windows localhost forwarding is broken, use: http://${wsl_ip}:3100"
fi

if $follow_logs; then
  echo "Following logs. Keep this window open while testing. Press Ctrl+C to stop following logs; containers remain running."
  "${compose[@]}" logs -f backend frontend training-api training-worker
fi
