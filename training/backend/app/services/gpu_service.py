import json
import logging
import os
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List

import psutil

from app.schemas.schemas import GPUInfo

logger = logging.getLogger(__name__)

_GPU_CACHE: List[GPUInfo] | None = None
_GPU_CACHE_AT = 0.0
_GPU_CACHE_TTL_SECONDS = 30.0
_CUDA_PROBE_TIMEOUT_SECONDS = 8.0


def _probe_torch_gpus() -> Dict[str, Any]:
    code = r"""
import json
import sys

result = {"gpus": []}
try:
    import torch

    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            used_bytes = total_bytes - free_bytes
            result["gpus"].append({
                "id": index,
                "name": props.name,
                "memory_total_mb": total_bytes / 1e6,
                "memory_used_mb": used_bytes / 1e6,
                "memory_free_mb": free_bytes / 1e6,
                "utilization_pct": 0,
                "temperature": None,
                "is_available": (free_bytes / 1e6) > 512,
            })
except Exception as exc:
    result["error"] = str(exc)

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
        return {"gpus": [], "error": f"torch GPU probe could not start: {exc}"}

    deadline = time.monotonic() + _CUDA_PROBE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            try:
                stdout, stderr = process.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", "torch GPU probe finished but output collection timed out"

            if return_code != 0:
                return {"gpus": [], "error": (stderr or stdout or "torch GPU probe failed").strip()}
            try:
                return json.loads(stdout.strip().splitlines()[-1])
            except Exception:
                return {"gpus": [], "error": "torch GPU probe returned invalid JSON"}
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
        return {"gpus": [], "error": "torch GPU probe watchdog timeout"}
    except subprocess.TimeoutExpired:
        return {
            "gpus": [],
            "error": "torch GPU probe watchdog timeout; child process did not exit after SIGKILL",
        }


class GPUService:

    def get_gpu_info(self) -> List[GPUInfo]:
        """Return info for all available GPUs without probing CUDA in this API process."""
        global _GPU_CACHE, _GPU_CACHE_AT

        now = time.monotonic()
        if _GPU_CACHE is not None and (now - _GPU_CACHE_AT) < _GPU_CACHE_TTL_SECONDS:
            return _GPU_CACHE

        probe = _probe_torch_gpus()
        if probe.get("error"):
            logger.warning("Training GPU probe failed: %s", probe["error"])
        gpus = [GPUInfo(**item) for item in probe.get("gpus", [])]

        _GPU_CACHE = gpus
        _GPU_CACHE_AT = now
        return gpus

    def get_system_stats(self) -> dict:
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage(str(__import__("pathlib").Path(__file__).parent))
        return {
            "cpu_pct": cpu,
            "ram_total_gb": round(mem.total / 1e9, 1),
            "ram_used_gb": round(mem.used / 1e9, 1),
            "ram_pct": mem.percent,
            "disk_total_gb": round(disk.total / 1e9, 1),
            "disk_used_gb": round(disk.used / 1e9, 1),
            "disk_pct": disk.percent,
        }

    def is_gpu_available(self) -> bool:
        gpus = self.get_gpu_info()
        return any(g.is_available for g in gpus)


gpu_service = GPUService()
