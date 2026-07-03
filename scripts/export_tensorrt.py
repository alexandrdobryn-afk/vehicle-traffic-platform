#!/usr/bin/env python3
"""
Export YOLO models to TensorRT .engine for maximum GPU inference speed.
Run this on the target GPU machine before deploying.

Requirements:
- NVIDIA GPU with CUDA
- TensorRT 8.x or 10.x
- Ultralytics >= 8.2

Usage:
    python export_tensorrt.py --model yolo11n --half
    python export_tensorrt.py --model yolov8n_plate --input backend/models/plate_detector/yolov8n_plate.pt
"""
import argparse
import os
import sys


def export_vehicle_detector(size: str = "n", half: bool = True):
    from ultralytics import YOLO
    models_dir = os.path.join(os.path.dirname(__file__), "..", "backend", "models", "vehicle_detector")
    pt_path = os.path.join(models_dir, f"yolo11{size}.pt")

    if not os.path.exists(pt_path):
        print(f"Downloading yolo11{size}.pt...")
        model = YOLO(f"yolo11{size}.pt")
    else:
        model = YOLO(pt_path)

    print(f"Exporting yolo11{size} to TensorRT (half={half})...")
    engine_path = model.export(
        format="engine",
        half=half,
        device=0,
        workspace=4,   # GB
        simplify=True,
    )
    dest = os.path.join(models_dir, f"yolo11{size}.engine")
    if engine_path != dest:
        import shutil
        shutil.copy(engine_path, dest)
    print(f"✅ Saved: {dest}")


def export_plate_detector(pt_path: str, half: bool = True):
    from ultralytics import YOLO
    if not os.path.exists(pt_path):
        print(f"❌ Not found: {pt_path}")
        return

    model = YOLO(pt_path)
    print(f"Exporting plate detector to TensorRT...")
    engine_path = model.export(format="engine", half=half, device=0, workspace=2, simplify=True)
    dest = pt_path.replace(".pt", ".engine")
    if engine_path != dest:
        import shutil
        shutil.copy(engine_path, dest)
    print(f"✅ Saved: {dest}")


def check_gpu():
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ No CUDA GPU detected. TensorRT export requires NVIDIA GPU.")
            sys.exit(1)
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA: {torch.version.cuda}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        print("❌ PyTorch not installed")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO models to TensorRT")
    parser.add_argument("--target", choices=["vehicle", "plate", "all"], default="all")
    parser.add_argument("--size", choices=["n", "s", "m"], default="n", help="YOLO size for vehicle detector")
    parser.add_argument("--plate_pt", default="backend/models/plate_detector/yolov8n_plate.pt")
    parser.add_argument("--half", action="store_true", default=True, help="FP16 precision")
    parser.add_argument("--full_precision", action="store_true", help="Use FP32 instead of FP16")
    args = parser.parse_args()

    check_gpu()
    half = not args.full_precision

    if args.target in ("vehicle", "all"):
        export_vehicle_detector(size=args.size, half=half)

    if args.target in ("plate", "all"):
        plate_path = os.path.abspath(args.plate_pt)
        if os.path.exists(plate_path):
            export_plate_detector(plate_path, half=half)
        else:
            print(f"⚠️  Plate detector not found at {plate_path}, skipping")

    print("\n✅ Export complete. Restart the platform to use TensorRT models.")
