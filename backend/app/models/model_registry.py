"""
Model Registry — central config for all AI models.
Edit paths here to swap models without changing service code.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import importlib.util
from functools import lru_cache
import os
import yaml
import logging

logger = logging.getLogger(__name__)

MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")


@lru_cache(maxsize=None)
def _runtime_package_available(model_type: str) -> bool:
    """Probe imports, not only package metadata, so health never claims a broken adapter."""
    try:
        if model_type == "rfdetr":
            from rfdetr import RFDETRNano  # noqa: F401
            return True
        if model_type == "paddleocr":
            import paddle  # noqa: F401
            from paddleocr import PaddleOCR  # noqa: F401
            return True
        if model_type == "fastalpr":
            from fast_plate_ocr import LicensePlateRecognizer  # noqa: F401
            return True
        if model_type == "openimagemodels":
            from open_image_models.detection.core.yolo_v9.inference import (  # noqa: F401
                YoloV9ObjectDetector,
            )
            return True
        if model_type == "easyocr":
            import easyocr  # noqa: F401
            return True
    except Exception as exc:
        logger.warning("Runtime probe failed for %s: %s", model_type, exc)
        return False
    return False


@dataclass
class ModelConfig:
    name: str
    model_type: str          # yolo, rfdetr, paddleocr, lprnet, easyocr, onnx
    format: str              # pt, onnx, engine, paddle
    path: Optional[str]
    classes: List[str] = field(default_factory=list)
    input_size: Optional[tuple] = None
    fp16: bool = False
    @property
    def available(self) -> bool:
        """Evaluate availability at use time so deployments are visible."""
        package = {
            "rfdetr": "rfdetr",
            "paddleocr": "paddleocr",
            "easyocr": "easyocr",
            "fastalpr": "fast_plate_ocr",
            "openimagemodels": "open_image_models",
        }.get(self.model_type)
        if self.path:
            return os.path.isfile(self.path) and (
                not package or _runtime_package_available(self.model_type)
            )
        return bool(package and importlib.util.find_spec(package) and _runtime_package_available(self.model_type))

    def resolve(self) -> "ModelConfig":
        return self


# ─── Vehicle Detector ───────────────────────────────────────────
VEHICLE_DETECTOR_PRODUCTION_TRT = ModelConfig(
    name="vehicle_detector_production_trt",
    model_type="yolo",
    format="engine",
    path=f"{MODELS_DIR}/vehicle_detector/production.engine",
    classes=["car", "motorcycle", "bus", "truck", "van"],
    input_size=(640, 640),
    fp16=True,
).resolve()

VEHICLE_DETECTOR_PRODUCTION_PT = ModelConfig(
    name="vehicle_detector_production",
    model_type="yolo",
    format="pt",
    path=f"{MODELS_DIR}/vehicle_detector/production.pt",
    classes=["car", "motorcycle", "bus", "truck", "van"],
    input_size=(640, 640),
).resolve()

VEHICLE_DETECTOR_YOLO11N = ModelConfig(
    name="yolo11n_vehicle",
    model_type="yolo",
    format="pt",
    path=f"{MODELS_DIR}/vehicle_detector/yolo11n.pt",
    classes=["car", "motorcycle", "bus", "truck"],
    input_size=(640, 640),
    fp16=False,
).resolve()

VEHICLE_DETECTOR_YOLO11S = ModelConfig(
    name="yolo11s_vehicle",
    model_type="yolo",
    format="pt",
    path=f"{MODELS_DIR}/vehicle_detector/yolo11s.pt",
    classes=["car", "motorcycle", "bus", "truck"],
    input_size=(640, 640),
).resolve()

VEHICLE_DETECTOR_YOLO11N_TRT = ModelConfig(
    name="yolo11n_vehicle_trt",
    model_type="yolo",
    format="engine",
    path=f"{MODELS_DIR}/vehicle_detector/yolo11n.engine",
    classes=["car", "motorcycle", "bus", "truck"],
    input_size=(640, 640),
    fp16=True,
).resolve()

VEHICLE_DETECTOR_YOLO26N = ModelConfig(
    name="yolo26n_vehicle",
    model_type="yolo",
    format="pt",
    path=f"{MODELS_DIR}/vehicle_detector/yolo26n.pt",
    classes=["car", "motorcycle", "bus", "truck"],
    input_size=(640, 640),
    fp16=False,
).resolve()

VEHICLE_DETECTOR_YOLO26S = ModelConfig(
    name="yolo26s_vehicle",
    model_type="yolo",
    format="pt",
    path=f"{MODELS_DIR}/vehicle_detector/yolo26s.pt",
    classes=["car", "motorcycle", "bus", "truck"],
    input_size=(640, 640),
    fp16=False,
).resolve()

VEHICLE_DETECTOR_YOLO26N_TRT = ModelConfig(
    name="yolo26n_vehicle_trt",
    model_type="yolo",
    format="engine",
    path=f"{MODELS_DIR}/vehicle_detector/yolo26n.engine",
    classes=["car", "motorcycle", "bus", "truck"],
    input_size=(640, 640),
    fp16=True,
).resolve()

VEHICLE_DETECTOR_RFDETR_NANO = ModelConfig(
    name="rfdetr_nano_vehicle",
    model_type="rfdetr",
    format="auto",
    path=f"{MODELS_DIR}/vehicle_detector/rf-detr-nano.pth",
    classes=["car", "motorcycle", "bus", "truck"],
).resolve()

VEHICLE_DETECTOR_RFDETR_MEDIUM = ModelConfig(
    name="rfdetr_medium_vehicle",
    model_type="rfdetr",
    format="auto",
    path=f"{MODELS_DIR}/vehicle_detector/rf-detr-medium.pth",
    classes=["car", "motorcycle", "bus", "truck"],
).resolve()

# Legacy path kept for deployments that already placed a local checkpoint there.
VEHICLE_DETECTOR_RFDETR = ModelConfig(
    name="rfdetr_vehicle_local",
    model_type="rfdetr",
    format="pt",
    path=f"{MODELS_DIR}/vehicle_detector/rf_detr.pt",
    classes=["car", "motorcycle", "bus", "truck"],
).resolve()

# ─── Plate Detector ─────────────────────────────────────────────
PLATE_DETECTOR_PRODUCTION_TRT = ModelConfig(
    name="plate_detector_production_trt",
    model_type="yolo",
    format="engine",
    path=f"{MODELS_DIR}/plate_detector/production.engine",
    classes=["license_plate"],
    input_size=(640, 640),
    fp16=True,
).resolve()

PLATE_DETECTOR_PRODUCTION_PT = ModelConfig(
    name="plate_detector_production",
    model_type="yolo",
    format="pt",
    path=f"{MODELS_DIR}/plate_detector/production.pt",
    classes=["license_plate"],
    input_size=(640, 640),
).resolve()

PLATE_DETECTOR_YOLO8N = ModelConfig(
    name="yolov8n_plate",
    model_type="yolo",
    format="pt",
    path=f"{MODELS_DIR}/plate_detector/yolov8n_plate.pt",
    classes=["license_plate"],
    input_size=(640, 640),
).resolve()

PLATE_DETECTOR_OPENIMAGEMODELS = ModelConfig(
    name="openimagemodels_plate_384",
    model_type="openimagemodels",
    format="onnx",
    path=f"{MODELS_DIR}/plate_detector/openimagemodels_yolov9t_384.onnx",
    classes=["license_plate"],
    input_size=(384, 384),
).resolve()

PLATE_DETECTOR_YOLO11N = ModelConfig(
    name="yolo11n_plate",
    model_type="yolo",
    format="pt",
    path=f"{MODELS_DIR}/plate_detector/yolo11n_plate.pt",
    classes=["license_plate"],
    input_size=(640, 640),
).resolve()

PLATE_DETECTOR_YOLO8N_TRT = ModelConfig(
    name="yolov8n_plate_trt",
    model_type="yolo",
    format="engine",
    path=f"{MODELS_DIR}/plate_detector/yolov8n_plate.engine",
    classes=["license_plate"],
    input_size=(640, 640),
    fp16=True,
).resolve()

# ─── OCR ────────────────────────────────────────────────────────
OCR_PRODUCTION = ModelConfig(
    name="ocr_production",
    model_type="lprnet",
    format="onnx",
    path=f"{MODELS_DIR}/ocr/production.onnx",
).resolve()

OCR_PADDLEOCR = ModelConfig(
    name="paddleocr",
    model_type="paddleocr",
    format="paddle",
    path=None,
).resolve()

OCR_LPRNET = ModelConfig(
    name="lprnet_ua",
    model_type="lprnet",
    format="onnx",
    path=f"{MODELS_DIR}/ocr/lprnet_ua.onnx",
).resolve()

OCR_EASYOCR = ModelConfig(
    name="easyocr",
    model_type="easyocr",
    format="pt",
    path=None,
).resolve()

OCR_FASTALPR = ModelConfig(
    name="fastalpr",
    model_type="fastalpr",
    format="onnx",
    path=None,
).resolve()

# ─── Color Classifier ────────────────────────────────────────────
COLOR_CLASSIFIER_PRODUCTION = ModelConfig(
    name="color_classifier_production",
    model_type="onnx",
    format="onnx",
    path=f"{MODELS_DIR}/color_classifier/production.onnx",
    classes=[
        "black", "white", "gray", "silver", "red", "blue", "green",
        "yellow", "orange", "brown", "beige", "unknown",
    ],
    input_size=(224, 224),
).resolve()

COLOR_CLASSIFIER_MOBILENET = ModelConfig(
    name="mobilenetv3_vehicle_color",
    model_type="onnx",
    format="onnx",
    path=f"{MODELS_DIR}/color_classifier/mobilenetv3_color.onnx",
    classes=[
        "black", "white", "gray", "silver",
        "red", "blue", "green", "yellow",
        "orange", "brown", "beige", "unknown"
    ],
    input_size=(224, 224),
).resolve()

# ─── Vehicle Brand / Logo Detector ───────────────────────────────────────────
BRAND_CLASSIFIER_BRANDEYE = ModelConfig(
    name="brandeye_logo_detector",
    model_type="yolo",
    format="pt",
    path=f"{MODELS_DIR}/brand_classifier/brandeye.pt",
    input_size=(960, 960),
).resolve()


class ModelRegistry:
    """
    Selects the right model per AI mode.
    Handles fallback if preferred model files are missing.
    """

    def get_vehicle_detector(self, ai_mode: str) -> ModelConfig:
        production = [VEHICLE_DETECTOR_PRODUCTION_TRT, VEHICLE_DETECTOR_PRODUCTION_PT]
        if ai_mode == "speed":
            candidates = production + [VEHICLE_DETECTOR_YOLO11N_TRT, VEHICLE_DETECTOR_YOLO11N]
        elif ai_mode == "balanced":
            candidates = production + [VEHICLE_DETECTOR_YOLO11N_TRT, VEHICLE_DETECTOR_YOLO11S, VEHICLE_DETECTOR_YOLO11N]
        elif ai_mode == "quality":
            candidates = production + [VEHICLE_DETECTOR_RFDETR_MEDIUM, VEHICLE_DETECTOR_RFDETR, VEHICLE_DETECTOR_YOLO11S, VEHICLE_DETECTOR_YOLO11N]
        elif ai_mode == "practical":
            candidates = [VEHICLE_DETECTOR_YOLO26N_TRT, VEHICLE_DETECTOR_YOLO26N]
        elif ai_mode == "max_accuracy":
            candidates = [VEHICLE_DETECTOR_RFDETR_MEDIUM]
        elif ai_mode == "edge_onnx":
            candidates = [VEHICLE_DETECTOR_YOLO26N_TRT, VEHICLE_DETECTOR_YOLO26N]
        else:  # hybrid and unknown strings
            candidates = production + [VEHICLE_DETECTOR_YOLO11N_TRT, VEHICLE_DETECTOR_YOLO11S, VEHICLE_DETECTOR_YOLO11N]

        return self._first_available(candidates, "vehicle_detector")

    def get_named_vehicle_detector(self, model_name: str, ai_mode: str) -> ModelConfig:
        requested = {
            "yolo11n_vehicle": VEHICLE_DETECTOR_YOLO11N,
            "yolo11s_vehicle": VEHICLE_DETECTOR_YOLO11S,
            "yolo11n_vehicle_trt": VEHICLE_DETECTOR_YOLO11N_TRT,
            "yolo26n_vehicle": VEHICLE_DETECTOR_YOLO26N,
            "yolo26s_vehicle": VEHICLE_DETECTOR_YOLO26S,
            "yolo26n_vehicle_trt": VEHICLE_DETECTOR_YOLO26N_TRT,
            "rfdetr_nano_vehicle": VEHICLE_DETECTOR_RFDETR_NANO,
            "rfdetr_medium_vehicle": VEHICLE_DETECTOR_RFDETR_MEDIUM,
            "rfdetr_vehicle_local": VEHICLE_DETECTOR_RFDETR,
        }.get(model_name)
        if requested and requested.available:
            return requested
        raise RuntimeError(f"Requested vehicle detector '{model_name}' is not installed")

    def get_plate_detector(self, ai_mode: str) -> ModelConfig:
        if ai_mode in ("speed", "edge_onnx"):
            candidates = [PLATE_DETECTOR_OPENIMAGEMODELS]
        elif ai_mode in ("quality", "max_accuracy"):
            candidates = [PLATE_DETECTOR_YOLO11N]
        else:
            candidates = [PLATE_DETECTOR_PRODUCTION_PT, PLATE_DETECTOR_PRODUCTION_TRT, PLATE_DETECTOR_YOLO8N, PLATE_DETECTOR_YOLO8N_TRT]

        return self._first_available(candidates, "plate_detector")

    def get_named_plate_detector(self, model_name: str, ai_mode: str) -> ModelConfig:
        requested = {
            "yolov8n_plate": PLATE_DETECTOR_YOLO8N,
            "yolov8n_plate_trt": PLATE_DETECTOR_YOLO8N_TRT,
            "openimagemodels_plate_384": PLATE_DETECTOR_OPENIMAGEMODELS,
            "yolo11n_plate": PLATE_DETECTOR_YOLO11N,
            "plate_detector_production": PLATE_DETECTOR_PRODUCTION_PT,
            "plate_detector_production_trt": PLATE_DETECTOR_PRODUCTION_TRT,
        }.get(model_name)
        if requested and requested.available:
            return requested
        raise RuntimeError(f"Requested plate detector '{model_name}' is not installed")

    def get_ocr_engine(self, ai_mode: str) -> ModelConfig:
        if ai_mode == "speed":
            candidates = [OCR_PRODUCTION, OCR_LPRNET, OCR_EASYOCR]
        elif ai_mode in ("practical", "max_accuracy"):
            candidates = [OCR_PADDLEOCR]
        elif ai_mode == "edge_onnx":
            candidates = [OCR_FASTALPR]
        else:
            candidates = [OCR_PRODUCTION, OCR_EASYOCR, OCR_LPRNET]

        return self._first_available(candidates, "ocr")

    def get_named_ocr_engine(self, engine: str, ai_mode: str) -> ModelConfig:
        requested = {
            "lprnet": OCR_PRODUCTION if OCR_PRODUCTION.available else OCR_LPRNET,
            "easyocr": OCR_EASYOCR,
            "paddleocr": OCR_PADDLEOCR,
            "fastalpr": OCR_FASTALPR,
        }.get(engine)
        if requested and requested.available:
            return requested
        raise RuntimeError(f"Requested OCR engine '{engine}' is not installed")

    def get_color_classifier(self) -> ModelConfig:
        candidates = [COLOR_CLASSIFIER_PRODUCTION, COLOR_CLASSIFIER_MOBILENET]
        return self._first_available(candidates, "color_classifier")

    def get_brand_classifier(self) -> ModelConfig:
        return BRAND_CLASSIFIER_BRANDEYE

    def _first_available(self, candidates: List[ModelConfig], role: str) -> ModelConfig:
        for m in candidates:
            if m.available:
                logger.info(f"[ModelRegistry] {role}: using {m.name} ({m.format})")
                return m
        # Optional services may use an explicit algorithmic fallback. Required
        # services fail closed when their loader receives this unavailable config.
        logger.error(f"[ModelRegistry] {role}: no usable model is installed")
        return candidates[-1]

    def print_status(self):
        all_models = [
            VEHICLE_DETECTOR_YOLO11N, VEHICLE_DETECTOR_YOLO11S,
            VEHICLE_DETECTOR_YOLO11N_TRT, VEHICLE_DETECTOR_YOLO26N,
            VEHICLE_DETECTOR_YOLO26S, VEHICLE_DETECTOR_YOLO26N_TRT,
            VEHICLE_DETECTOR_RFDETR_NANO, VEHICLE_DETECTOR_RFDETR_MEDIUM,
            VEHICLE_DETECTOR_RFDETR,
            PLATE_DETECTOR_YOLO8N, PLATE_DETECTOR_YOLO8N_TRT,
            OCR_PADDLEOCR, OCR_FASTALPR, OCR_LPRNET, OCR_EASYOCR,
            COLOR_CLASSIFIER_MOBILENET,
            BRAND_CLASSIFIER_BRANDEYE,
        ]
        print("\n=== Model Registry Status ===")
        for m in all_models:
            status = "✅" if m.available else "❌"
            print(f"  {status} {m.name:<40} {m.path}")
        print()


model_registry = ModelRegistry()
