"""
Export Worker — converts trained PyTorch models to ONNX and TensorRT.
"""
import os
import logging
from pathlib import Path
from app.celery_app import celery_app
from app.config import settings

logger = logging.getLogger(__name__)


@celery_app.task(name="app.workers.export_worker.export_onnx", bind=True)
def export_onnx(self, model_version_id: int):
    """Export a registered model to ONNX format."""
    db = _get_db()
    try:
        from app.models.database import ModelVersion
        mv = db.query(ModelVersion).filter(ModelVersion.id == model_version_id).first()
        if not mv or not mv.weights_path or not os.path.exists(mv.weights_path):
            return {"success": False, "error": "Weights not found"}

        model_type = mv.model_type
        onnx_path = str(Path(mv.weights_path).parent / f"{mv.name}.onnx")

        if mv.onnx_path and os.path.exists(mv.onnx_path):
            return {"success": True, "onnx_path": mv.onnx_path, "reused": True}

        if model_type in ("vehicle_detector", "plate_detector"):
            onnx_path = _export_yolo_onnx(mv.weights_path, onnx_path)
        else:
            return {
                "success": False,
                "error": "Training did not produce the required ONNX artifact; retrain the model",
            }

        if onnx_path and os.path.exists(onnx_path):
            mv.onnx_path = onnx_path
            db.commit()
            logger.info(f"ONNX export complete: {onnx_path}")
            return {"success": True, "onnx_path": onnx_path}

        return {"success": False, "error": "Export produced no output"}
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        db.close()


@celery_app.task(name="app.workers.export_worker.export_tensorrt", bind=True)
def export_tensorrt(self, model_version_id: int, half: bool = True):
    """Export YOLO model to TensorRT .engine."""
    db = _get_db()
    try:
        from app.models.database import ModelVersion
        mv = db.query(ModelVersion).filter(ModelVersion.id == model_version_id).first()
        if not mv:
            return {"success": False, "error": "Model not found"}

        if mv.model_type not in ("vehicle_detector", "plate_detector"):
            return {"success": False, "error": "TensorRT export only supported for YOLO models"}

        weights = mv.weights_path
        if not weights or not os.path.exists(weights):
            return {"success": False, "error": "Weights not found"}

        import torch
        if not torch.cuda.is_available():
            return {"success": False, "error": "CUDA GPU required for TensorRT export"}

        from ultralytics import YOLO
        model = YOLO(weights)
        engine_path = model.export(
            format="engine",
            half=half,
            device=0,
            workspace=4,
            simplify=True,
        )

        if engine_path and os.path.exists(engine_path):
            mv.trt_path = str(engine_path)
            db.commit()
            logger.info(f"TensorRT export complete: {engine_path}")
            return {"success": True, "trt_path": str(engine_path)}

        return {"success": False, "error": "TRT export produced no output"}
    except Exception as e:
        logger.error(f"TRT export failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        db.close()


# ─── Helpers ──────────────────────────────────────────────────────

def _export_yolo_onnx(weights_path: str, onnx_path: str) -> str:
    from ultralytics import YOLO
    model = YOLO(weights_path)
    out = model.export(format="onnx", simplify=True, opset=11)
    if out and os.path.exists(str(out)):
        import shutil
        shutil.copy(str(out), onnx_path)
    return onnx_path


def _export_torch_onnx(weights_path: str, onnx_path: str, arch: str) -> str:
    import torch
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Recreate model architecture
    if arch == "mobilenetv3":
        from torchvision import models
        model = models.mobilenet_v3_small(pretrained=False)
        # We don't know num_classes here without DB — use checkpoint
    elif arch == "resnet18":
        from torchvision import models
        model = models.resnet18(pretrained=False)
    else:
        from torchvision import models
        model = models.efficientnet_b0(pretrained=False)

    state = torch.load(weights_path, map_location=device)
    try:
        model.load_state_dict(state)
    except Exception:
        # May have different classifier head — try strict=False
        model.load_state_dict(state, strict=False)

    model.eval().to(device)
    dummy = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=11,
    )
    return onnx_path


def _get_db():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    engine = create_engine(settings.DATABASE_SYNC_URL)
    Session = sessionmaker(bind=engine)
    return Session()
