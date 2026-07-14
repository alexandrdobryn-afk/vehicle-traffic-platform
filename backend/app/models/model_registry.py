"""Central registry for drone/aerial computer-vision model artifacts."""

from dataclasses import dataclass, field
from functools import lru_cache
import importlib.util
import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")


@lru_cache(maxsize=None)
def _runtime_package_available(model_type: str) -> bool:
    """Probe imports so health does not report broken optional adapters as ready."""
    try:
        if model_type == "rfdetr":
            from rfdetr import RFDETRNano  # noqa: F401
            return True
        if model_type == "paddleocr":
            import paddle  # noqa: F401
            from paddleocr import PaddleOCR  # noqa: F401
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
    model_type: str
    format: str
    path: Optional[str]
    path_options: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    input_size: Optional[tuple] = None
    fp16: bool = False

    @property
    def artifact_path(self) -> Optional[str]:
        candidates = [candidate for candidate in [self.path, *self.path_options] if candidate]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        return self.path

    @property
    def available(self) -> bool:
        package = {
            "rfdetr": "rfdetr",
            "paddleocr": "paddleocr",
            "easyocr": "easyocr",
        }.get(self.model_type)
        artifact_path = self.artifact_path
        if artifact_path:
            if not os.path.isfile(artifact_path):
                return False
            if self.model_type == "classifier" and artifact_path.lower().endswith(".onnx"):
                if not os.path.isfile(os.path.join(os.path.dirname(artifact_path), "labels.json")):
                    return False
            return (
                not package or _runtime_package_available(self.model_type)
            )
        return bool(package and importlib.util.find_spec(package) and _runtime_package_available(self.model_type))

    def resolve(self) -> "ModelConfig":
        return self


OBJECT_DETECTOR_PRODUCTION = ModelConfig(
    name="object_detector_production",
    model_type="yolo",
    format="pt",
    path=f"{MODELS_DIR}/object_detector/production.pt",
    classes=[],
    input_size=(1024, 1024),
).resolve()

OBJECT_DETECTOR_YOLO11N = ModelConfig(
    name="yolo11n_object",
    model_type="yolo",
    format="pt",
    path=f"{MODELS_DIR}/object_detector/yolo11n.pt",
    input_size=(1024, 1024),
).resolve()

OBJECT_DETECTOR_YOLO11S = ModelConfig(
    name="yolo11s_object",
    model_type="yolo",
    format="pt",
    path=f"{MODELS_DIR}/object_detector/yolo11s.pt",
    input_size=(1024, 1024),
).resolve()

OBJECT_DETECTOR_RFDETR_NANO = ModelConfig(
    name="rfdetr_nano_object",
    model_type="rfdetr",
    format="pth",
    path=f"{MODELS_DIR}/object_detector/rf-detr-nano.pth",
).resolve()

OBJECT_DETECTOR_RFDETR_MEDIUM = ModelConfig(
    name="rfdetr_medium_object",
    model_type="rfdetr",
    format="pth",
    path=f"{MODELS_DIR}/object_detector/rf-detr-medium.pth",
).resolve()

OBJECT_DETECTOR_RTDETR = ModelConfig(
    name="rtdetr_aerial",
    model_type="rtdetr",
    format="pt",
    path=f"{MODELS_DIR}/object_detector/rtdetr-l.pt",
    input_size=(1280, 1280),
).resolve()

OBJECT_SEGMENTER_PRODUCTION = ModelConfig(
    name="object_segmenter_production",
    model_type="yolo",
    format="pt",
    path=f"{MODELS_DIR}/object_segmenter/production.pt",
    classes=[],
    input_size=(1024, 1024),
).resolve()

OBJECT_CLASSIFIER_PRODUCTION = ModelConfig(
    name="object_classifier_production",
    model_type="classifier",
    format="onnx/pt",
    path=f"{MODELS_DIR}/object_classifier/production.onnx",
    path_options=[f"{MODELS_DIR}/object_classifier/production.pt"],
    classes=[],
    input_size=(224, 224),
).resolve()

TEXT_OCR_PADDLEOCR = ModelConfig(
    name="paddleocr_text",
    model_type="paddleocr",
    format="paddle",
    path=None,
).resolve()

TEXT_OCR_EASYOCR = ModelConfig(
    name="easyocr_text",
    model_type="easyocr",
    format="pt",
    path=None,
).resolve()


class ModelRegistry:
    """Resolve effective aerial model artifacts per runtime mode."""

    def get_object_detector(self, ai_mode: str) -> ModelConfig:
        if ai_mode == "speed":
            candidates = [OBJECT_DETECTOR_PRODUCTION, OBJECT_DETECTOR_YOLO11N]
        elif ai_mode in {"quality", "max_accuracy"}:
            candidates = [
                OBJECT_DETECTOR_PRODUCTION,
                OBJECT_DETECTOR_RFDETR_MEDIUM,
                OBJECT_DETECTOR_RTDETR,
                OBJECT_DETECTOR_YOLO11S,
                OBJECT_DETECTOR_YOLO11N,
            ]
        elif ai_mode == "practical":
            candidates = [
                OBJECT_DETECTOR_PRODUCTION,
                OBJECT_DETECTOR_YOLO11S,
                OBJECT_DETECTOR_YOLO11N,
            ]
        else:
            candidates = [
                OBJECT_DETECTOR_PRODUCTION,
                OBJECT_DETECTOR_YOLO11S,
                OBJECT_DETECTOR_YOLO11N,
            ]
        return self._first_available(candidates, "object_detector")

    def get_named_object_detector(self, model_name: str, ai_mode: str) -> ModelConfig:
        requested = {
            "object_detector_production": OBJECT_DETECTOR_PRODUCTION,
            "yolo11n_object": OBJECT_DETECTOR_YOLO11N,
            "yolo11s_object": OBJECT_DETECTOR_YOLO11S,
            "rfdetr_nano_object": OBJECT_DETECTOR_RFDETR_NANO,
            "rfdetr_medium_object": OBJECT_DETECTOR_RFDETR_MEDIUM,
            "rtdetr_aerial": OBJECT_DETECTOR_RTDETR,
        }.get(model_name)
        if requested and requested.available:
            return requested
        raise RuntimeError(f"Requested object detector '{model_name}' is not installed")

    def get_object_segmenter(self) -> ModelConfig:
        return OBJECT_SEGMENTER_PRODUCTION

    def get_object_classifier(self) -> ModelConfig:
        return OBJECT_CLASSIFIER_PRODUCTION

    def get_text_ocr_engine(self, ai_mode: str) -> ModelConfig:
        candidates = [TEXT_OCR_PADDLEOCR, TEXT_OCR_EASYOCR]
        return self._first_available(candidates, "text_ocr")

    def _first_available(self, candidates: List[ModelConfig], role: str) -> ModelConfig:
        for model in candidates:
            if model.available:
                logger.info("[ModelRegistry] %s: using %s (%s)", role, model.name, model.format)
                return model
        logger.error("[ModelRegistry] %s: no usable model is installed", role)
        return candidates[-1]

    def all_models(self) -> List[ModelConfig]:
        return [
            OBJECT_DETECTOR_PRODUCTION,
            OBJECT_DETECTOR_YOLO11N,
            OBJECT_DETECTOR_YOLO11S,
            OBJECT_DETECTOR_RFDETR_NANO,
            OBJECT_DETECTOR_RFDETR_MEDIUM,
            OBJECT_DETECTOR_RTDETR,
            OBJECT_SEGMENTER_PRODUCTION,
            OBJECT_CLASSIFIER_PRODUCTION,
            TEXT_OCR_PADDLEOCR,
            TEXT_OCR_EASYOCR,
        ]

    def print_status(self):
        print("\n=== Drone Vision Model Registry Status ===")
        for model in self.all_models():
            status = "OK" if model.available else "MISS"
            print(f"  {status:<4} {model.name:<35} {model.path}")
        print()


model_registry = ModelRegistry()
