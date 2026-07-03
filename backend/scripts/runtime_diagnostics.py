"""Print machine-readable accelerator diagnostics from inside the backend."""

import json

import onnxruntime as ort
import torch


cuda_available = torch.cuda.is_available()
print(json.dumps({
    "torch_version": torch.__version__,
    "cuda_available": cuda_available,
    "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
    "cuda_device_count": torch.cuda.device_count(),
    "onnxruntime_version": ort.__version__,
    "onnxruntime_providers": ort.get_available_providers(),
}, sort_keys=True))
