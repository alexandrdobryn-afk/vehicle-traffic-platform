#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

force_cpu=false
build=false
follow_logs=false
profile=""

while (($#)); do
  case "$1" in
    --cpu) force_cpu=true ;;
    --build) build=true ;;
    --foreground) follow_logs=true ;;
    --profile)
      shift
      profile="${1:-}"
      [[ -n "$profile" ]] || { echo "ERROR: --profile requires a value." >&2; exit 2; }
      ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker is not running in this WSL distribution." >&2
  exit 1
fi

gpu_available=false
if ! $force_cpu \
  && command -v nvidia-smi >/dev/null 2>&1 \
  && nvidia-smi -L >/dev/null 2>&1 \
  && docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -qi '"nvidia"'; then
  gpu_available=true
fi

compose=(docker compose -f docker-compose.yml)
if $gpu_available; then
  compose+=(-f docker-compose.gpu.yml)
  echo "Compute mode: NVIDIA CUDA (GPU overlay enabled)."
else
  echo "Compute mode: CPU (compatible fallback)."
  if ! $force_cpu && command -v nvidia-smi >/dev/null 2>&1; then
    echo "WARNING: NVIDIA GPU was detected, but NVIDIA Container Toolkit is unavailable." >&2
  fi
fi

if [[ -n "$profile" ]]; then
  compose+=(--profile "$profile")
fi

up_args=(up -d --remove-orphans)
if $build; then
  up_args+=(--build)
fi

"${compose[@]}" "${up_args[@]}"

if $gpu_available; then
  verified=false
  for _ in {1..30}; do
    if docker exec vtp-backend python -c \
      'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' \
      >/dev/null 2>&1; then
      verified=true
      break
    fi
    sleep 1
  done

  if ! $verified; then
    echo "ERROR: GPU mode was requested, but CUDA is unavailable inside vtp-backend." >&2
    echo "Check: docker logs vtp-backend" >&2
    exit 1
  fi

  gpu_name="$(docker exec vtp-backend python -c \
    'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null)"
  echo "CUDA verified inside vtp-backend: $gpu_name"
fi

echo "VTP is running. Frontend: http://localhost:3000"

if $follow_logs; then
  exec "${compose[@]}" logs -f
fi
