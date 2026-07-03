"""Hardware capability detection and honest execution-provider resolution."""

import os
from typing import Any, Dict


def get_runtime_capabilities() -> Dict[str, Any]:
    cuda_available = False
    cuda_devices = []
    torch_version = None
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                cuda_devices.append({
                    "index": index,
                    "name": properties.name,
                    "memory_mb": round(properties.total_memory / 1024 / 1024),
                    "compute_capability": f"{properties.major}.{properties.minor}",
                })
    except Exception:
        pass

    ort_version = None
    ort_providers = []
    try:
        import onnxruntime as ort

        ort_version = ort.__version__
        ort_providers = ort.get_available_providers()
    except Exception:
        pass

    return {
        "cpu": {"available": True},
        "cuda": {
            "available": cuda_available,
            "devices": cuda_devices,
        },
        "torch_version": torch_version,
        "onnxruntime_version": ort_version,
        "onnxruntime_providers": ort_providers,
        "container_runtime": os.environ.get("AI_RUNTIME", "cpu"),
    }


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
