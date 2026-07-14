"""Cached hardware capability detection.

HTTP request handlers must not initialize CUDA or ONNX Runtime directly. On
Windows + WSL2 a broken GPU passthrough can block low-level driver calls. This
module therefore keeps a cached runtime snapshot that is refreshed by a
background monitor. Callers read the latest snapshot and never run hardware
probing in the FastAPI request path.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict


_CAPABILITIES_CACHE: Dict[str, Any] | None = None
_CAPABILITIES_CACHE_AT = 0.0
_CAPABILITIES_CACHE_LOCK = threading.Lock()
_MONITOR_STARTED = False
_MONITOR_LOCK = threading.Lock()
_MONITOR_INTERVAL_SECONDS = float(os.environ.get("BEVP_RUNTIME_MONITOR_INTERVAL_SECONDS", "30"))
_PROBE_TIMEOUT_SECONDS = float(os.environ.get("BEVP_RUNTIME_PROBE_TIMEOUT_SECONDS", "8"))


def _unknown_capabilities(error: str = "runtime monitor has not completed its first probe") -> Dict[str, Any]:
    return {
        "cpu": {"available": True},
        "cuda": {
            "available": False,
            "devices": [],
            "state": "unknown",
            "probe_error": error,
        },
        "torch_version": None,
        "onnxruntime_version": None,
        "onnxruntime_providers": [],
        "container_runtime": os.environ.get("AI_RUNTIME", "cpu"),
        "checked_at": None,
        "age_seconds": None,
        "duration_ms": None,
        "stale": True,
    }


def _probe_runtime_in_child() -> Dict[str, Any]:
    code = r"""
import json
import sys
import time

started = time.monotonic()
result = {
    "torch_version": None,
    "cuda_available": False,
    "cuda_devices": [],
    "onnxruntime_version": None,
    "onnxruntime_providers": [],
    "errors": [],
}

try:
    import torch

    result["torch_version"] = torch.__version__
    result["cuda_available"] = bool(torch.cuda.is_available())
    if result["cuda_available"]:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            result["cuda_devices"].append({
                "index": index,
                "name": properties.name,
                "memory_mb": round(properties.total_memory / 1024 / 1024),
                "compute_capability": f"{properties.major}.{properties.minor}",
            })
except Exception as exc:
    result["errors"].append(f"torch: {exc}")

try:
    import onnxruntime as ort

    result["onnxruntime_version"] = ort.__version__
    result["onnxruntime_providers"] = list(ort.get_available_providers())
except Exception as exc:
    result["errors"].append(f"onnxruntime: {exc}")

result["duration_ms"] = round((time.monotonic() - started) * 1000)
print(json.dumps(result), flush=True)
sys.exit(0)
"""
    try:
        process = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except Exception as exc:
        return {
            "torch_version": None,
            "cuda_available": False,
            "cuda_devices": [],
            "onnxruntime_version": None,
            "onnxruntime_providers": [],
            "duration_ms": None,
            "error": f"runtime probe could not start: {exc}",
            "state": "probe_start_failed",
        }

    deadline = time.monotonic() + _PROBE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            try:
                stdout, stderr = process.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", "runtime probe finished but output collection timed out"

            if return_code != 0:
                return {
                    "torch_version": None,
                    "cuda_available": False,
                    "cuda_devices": [],
                    "onnxruntime_version": None,
                    "onnxruntime_providers": [],
                    "duration_ms": None,
                    "error": (stderr or stdout or "runtime probe failed").strip(),
                    "state": "failed",
                }

            try:
                parsed = json.loads(stdout.strip().splitlines()[-1])
            except Exception:
                return {
                    "torch_version": None,
                    "cuda_available": False,
                    "cuda_devices": [],
                    "onnxruntime_version": None,
                    "onnxruntime_providers": [],
                    "duration_ms": None,
                    "error": "runtime probe returned invalid JSON",
                    "state": "invalid_output",
                }

            errors = parsed.get("errors") or []
            return {
                "torch_version": parsed.get("torch_version"),
                "cuda_available": bool(parsed.get("cuda_available")),
                "cuda_devices": parsed.get("cuda_devices") or [],
                "onnxruntime_version": parsed.get("onnxruntime_version"),
                "onnxruntime_providers": parsed.get("onnxruntime_providers") or [],
                "duration_ms": parsed.get("duration_ms"),
                "error": "; ".join(errors) if errors else None,
                "state": "available" if parsed.get("cuda_available") else "unavailable",
            }
        time.sleep(0.1)

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass

    try:
        process.wait(timeout=0.25)
        state = "timeout"
        error = "runtime probe watchdog timeout"
    except subprocess.TimeoutExpired:
        state = "probe_stuck"
        error = "runtime probe watchdog timeout; child process did not exit after SIGKILL"

    return {
        "torch_version": None,
        "cuda_available": False,
        "cuda_devices": [],
        "onnxruntime_version": None,
        "onnxruntime_providers": [],
        "duration_ms": round(_PROBE_TIMEOUT_SECONDS * 1000),
        "error": error,
        "state": state,
    }


def _publish_probe_result(probe: Dict[str, Any]) -> None:
    global _CAPABILITIES_CACHE, _CAPABILITIES_CACHE_AT

    now = time.monotonic()
    checked_at = datetime.now(timezone.utc).isoformat()
    capabilities = {
        "cpu": {"available": True},
        "cuda": {
            "available": bool(probe["cuda_available"]),
            "devices": probe["cuda_devices"],
            "state": probe["state"],
            "probe_error": probe.get("error"),
        },
        "torch_version": probe["torch_version"],
        "onnxruntime_version": probe["onnxruntime_version"],
        "onnxruntime_providers": probe["onnxruntime_providers"],
        "container_runtime": os.environ.get("AI_RUNTIME", "cpu"),
        "checked_at": checked_at,
        "age_seconds": 0,
        "duration_ms": probe.get("duration_ms"),
        "stale": False,
    }
    with _CAPABILITIES_CACHE_LOCK:
        _CAPABILITIES_CACHE = capabilities
        _CAPABILITIES_CACHE_AT = now


def _runtime_monitor_loop() -> None:
    while True:
        _publish_probe_result(_probe_runtime_in_child())
        time.sleep(max(5.0, _MONITOR_INTERVAL_SECONDS))


def start_runtime_monitor() -> None:
    global _MONITOR_STARTED

    with _MONITOR_LOCK:
        if _MONITOR_STARTED:
            return
        thread = threading.Thread(
            target=_runtime_monitor_loop,
            name="bevp-runtime-monitor",
            daemon=True,
        )
        thread.start()
        _MONITOR_STARTED = True


def get_runtime_capabilities() -> Dict[str, Any]:
    start_runtime_monitor()
    with _CAPABILITIES_CACHE_LOCK:
        if _CAPABILITIES_CACHE is None:
            return _unknown_capabilities()
        capabilities = dict(_CAPABILITIES_CACHE)
        age_seconds = max(0.0, time.monotonic() - _CAPABILITIES_CACHE_AT)

    capabilities["age_seconds"] = round(age_seconds, 1)
    capabilities["stale"] = age_seconds > max(_MONITOR_INTERVAL_SECONDS * 3, 90.0)
    return capabilities


def wait_for_runtime_probe(timeout_seconds: float = 10.0) -> Dict[str, Any]:
    start_runtime_monitor()
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while time.monotonic() < deadline:
        with _CAPABILITIES_CACHE_LOCK:
            if _CAPABILITIES_CACHE is not None:
                break
        time.sleep(0.1)
    return get_runtime_capabilities()


def resolve_execution_policy(
    requested: str = "auto",
    gpu_device_index: int = 0,
    fallback: str = "fail_closed",
) -> Dict[str, Any]:
    capabilities = get_runtime_capabilities()
    cuda_devices = capabilities["cuda"]["devices"]
    cuda_ready = capabilities["cuda"]["available"] and any(
        device["index"] == gpu_device_index for device in cuda_devices
    )

    fallback_reason = None
    if requested == "cpu":
        resolved = "cpu"
    elif requested == "cuda" and cuda_ready:
        resolved = "cuda"
    elif requested == "cuda" and fallback == "allow_cpu":
        resolved = "cpu"
        fallback_reason = "requested_cuda_is_unavailable"
    elif requested == "cuda":
        resolved = None
    else:
        resolved = "cuda" if cuda_ready else "cpu"
        if not cuda_ready and capabilities["cuda"].get("state") in {"unknown", "timeout", "probe_stuck"}:
            fallback_reason = f"cuda_probe_{capabilities['cuda'].get('state')}"

    return {
        "requested": requested,
        "resolved": resolved,
        "device": f"cuda:{gpu_device_index}" if resolved == "cuda" else resolved,
        "available": resolved is not None,
        "fallback_policy": fallback,
        "fallback_reason": fallback_reason,
        "gpu_device_index": gpu_device_index,
        "precision": "fp32",
        "capabilities": capabilities,
    }
