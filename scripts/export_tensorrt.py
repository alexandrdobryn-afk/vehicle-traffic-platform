"""Export approved drone-vision YOLO artifacts to TensorRT engines."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "backend" / "models"


def export_yolo_artifact(weights: Path, half: bool = True) -> Path:
    if not weights.exists():
        raise FileNotFoundError(weights)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT export requires a CUDA-capable NVIDIA GPU")
    from ultralytics import YOLO

    model = YOLO(str(weights))
    model.export(format="engine", half=half, device=0)
    engine_path = weights.with_suffix(".engine")
    if not engine_path.exists():
        raise RuntimeError(f"TensorRT export did not produce {engine_path}")
    return engine_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role",
        choices=["object_detector", "object_segmenter"],
        default="object_detector",
    )
    parser.add_argument(
        "--weights",
        default="production.pt",
        help="Artifact filename inside backend/models/<role> or an absolute path",
    )
    parser.add_argument("--fp32", action="store_true", help="Disable FP16 export")
    args = parser.parse_args()

    requested = Path(args.weights)
    weights = requested if requested.is_absolute() else MODELS_DIR / args.role / requested
    engine = export_yolo_artifact(weights, half=not args.fp32)
    print(f"Exported TensorRT engine: {engine}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
