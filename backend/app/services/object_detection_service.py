import cv2
import numpy as np
from typing import List, Optional, Sequence
from dataclasses import dataclass
import logging
from app.config import settings, AIMode
from app.services.aerial_processing_service import AerialProcessingService, TileConfig

logger = logging.getLogger(__name__)

@dataclass
class Detection:
    bbox: List[int]          # [x1, y1, x2, y2]
    confidence: float
    class_name: str
    class_id: int


class ObjectDetectionService:
    """
    Multi-model aerial object detector.
    Supports YOLO-family, RT-DETR and RF-DETR artifacts behind one adapter.
    """

    def __init__(
        self,
        ai_mode: AIMode = AIMode.BALANCED,
        model_config=None,
        execution_device: Optional[str] = None,
        accepted_classes: Optional[Sequence[str]] = None,
        aerial_config: Optional[dict] = None,
    ):
        self.ai_mode = ai_mode
        self.model_config = model_config
        self.execution_device = execution_device
        self.accepted_classes = (
            {str(name).strip().lower() for name in accepted_classes if str(name).strip()}
            if accepted_classes is not None else set()
        )
        self.aerial = AerialProcessingService(TileConfig.from_mapping(aerial_config))
        self.yolo_model = None
        self.rfdetr_model = None
        self.device = self._get_device()
        self._load_models()

    def _get_device(self) -> str:
        if self.execution_device:
            return self.execution_device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _load_models(self):
        """Load models based on AI mode."""
        if self.model_config and self.model_config.model_type == "rfdetr":
            self._load_rfdetr()
        elif self.model_config and self.model_config.model_type == "rtdetr":
            self._load_rtdetr()
        else:
            self._load_yolo()

    def _load_rtdetr(self):
        """Load an Ultralytics RT-DETR artifact through the common result adapter."""
        try:
            from ultralytics import RTDETR
            if not self.model_config or not self.model_config.available:
                raise FileNotFoundError("RT-DETR weights are not installed")
            self.yolo_model = RTDETR(self.model_config.path)
            logger.info("RT-DETR model loaded: %s", self.model_config.path)
        except Exception as exc:
            raise RuntimeError(f"RT-DETR is unavailable: {exc}") from exc

    def _load_yolo(self):
        """Load the exact artifact selected by the model registry."""
        try:
            from ultralytics import YOLO
            if not self.model_config or not self.model_config.available:
                raise FileNotFoundError(
                    "Object detector weights are missing. Run scripts/download_models.py"
                )
            self.yolo_model = YOLO(self.model_config.path)
            logger.info(
                "YOLO model loaded: %s (%s)",
                self.model_config.path,
                self.model_config.format,
            )
        except Exception as e:
            logger.error(f"Failed to load YOLO: {e}")
            raise RuntimeError(f"Object detector is unavailable: {e}") from e

    def _load_rfdetr(self):
        """Load RF-DETR model for quality mode."""
        try:
            model_path = self.model_config.artifact_path if self.model_config else None
            model_name = (self.model_config.name if self.model_config else "").lower()
            from rfdetr import RFDETRBase, RFDETRMedium, RFDETRNano

            if "medium" in model_name:
                self.rfdetr_model = RFDETRMedium(pretrain_weights=model_path)
            elif "nano" in model_name:
                self.rfdetr_model = RFDETRNano(pretrain_weights=model_path)
            else:
                # Local/custom checkpoints use the base class. Official
                # package variants auto-download their COCO-pretrained weights.
                kwargs = {"pretrain_weights": model_path} if model_path else {}
                self.rfdetr_model = RFDETRBase(**kwargs)
            logger.info("RF-DETR loaded: %s", self.model_config.name if self.model_config else "auto")
            underlying = getattr(self.rfdetr_model, "model", None)
            if underlying is not None and hasattr(underlying, "to"):
                underlying.to(self.device)
            self._load_yolo_fallback()
        except ImportError:
            raise RuntimeError("RF-DETR package is not installed")
        except Exception as e:
            logger.error(f"Failed to load RF-DETR: {e}")
            raise RuntimeError(f"RF-DETR is unavailable: {e}") from e

    def _load_yolo_fallback(self):
        """Load a known YOLO fallback behind RF-DETR when available."""
        try:
            from ultralytics import YOLO
            from app.models.model_registry import OBJECT_DETECTOR_YOLO11S, OBJECT_DETECTOR_YOLO11N
            fallback = OBJECT_DETECTOR_YOLO11S if OBJECT_DETECTOR_YOLO11S.available else OBJECT_DETECTOR_YOLO11N
            if fallback.available:
                self.yolo_model = YOLO(fallback.path)
                logger.info("YOLO fallback loaded behind RF-DETR: %s", fallback.path)
        except Exception as exc:
            logger.warning("Could not load YOLO fallback behind RF-DETR: %s", exc)

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float = None,
        use_rfdetr: bool = False
    ) -> List[Detection]:
        """
        Run object detection on a frame.
        Returns list of Detection objects.
        """
        if confidence_threshold is None:
            confidence_threshold = settings.OBJECT_CONFIDENCE_THRESHOLD

        analysis_frame = self.aerial.prepare(frame)
        if self.rfdetr_model is not None and (use_rfdetr or self.yolo_model is None):
            detections = self.aerial.infer_tiled(
                analysis_frame, lambda tile: self._detect_rfdetr(tile, confidence_threshold)
            )
        elif self.yolo_model is not None:
            detections = self.aerial.infer_tiled(
                analysis_frame, lambda tile: self._detect_yolo(tile, confidence_threshold)
            )
        else:
            logger.error("No detection model available")
            return []
        if self.aerial.config.scale > 1.0:
            scale = self.aerial.config.scale
            for detection in detections:
                detection.bbox = [int(round(value / scale)) for value in detection.bbox]
        return self.aerial.object_agnostic_nms(detections, self.aerial.config.nms_iou)

    def _accept_class(self, class_name: str) -> bool:
        """An empty class set means class-agnostic inference."""
        return not self.accepted_classes or class_name.lower() in self.accepted_classes

    def _detect_yolo(self, frame: np.ndarray, conf: float) -> List[Detection]:
        try:
            predict_kwargs = {
                "conf": conf,
                "iou": self.aerial.config.nms_iou,
                "verbose": False,
                "agnostic_nms": True,
            }
            if not self.model_config or self.model_config.format != "engine":
                predict_kwargs["device"] = self.device
                # Keep FP32 across CPU and CUDA. FP16/TensorRT are separate,
                # benchmark-gated backends and must never be enabled silently.
                predict_kwargs["half"] = False
            results = self.yolo_model(frame, **predict_kwargs)

            detections = []
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = str(result.names.get(cls_id, "unknown")).lower()

                    if not self._accept_class(cls_name):
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf_val = float(box.conf[0])

                    detections.append(Detection(
                        bbox=[x1, y1, x2, y2],
                        confidence=conf_val,
                        class_name=cls_name,
                        class_id=cls_id
                    ))

            return detections
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return []

    def _detect_rfdetr(self, frame: np.ndarray, conf: float) -> List[Detection]:
        try:
            # RF-DETR inference
            results = self.rfdetr_model.predict(frame, threshold=conf)
            detections = []
            if hasattr(results, "xyxy"):
                boxes = getattr(results, "xyxy", [])
                confidences = getattr(results, "confidence", [])
                class_ids = getattr(results, "class_id", [])
                for idx, box in enumerate(boxes):
                    cls_id = int(class_ids[idx]) if idx < len(class_ids) else -1
                    cls_name = f"class_{cls_id}" if cls_id >= 0 else "unknown"
                    if not self._accept_class(cls_name):
                        continue
                    detections.append(Detection(
                        bbox=list(map(int, box.tolist() if hasattr(box, "tolist") else box)),
                        confidence=float(confidences[idx]) if idx < len(confidences) else 0.0,
                        class_name=cls_name,
                        class_id=cls_id
                    ))
                return detections

            # Backward-compatible dict/list output handling for custom loaders.
            for det in results:
                label = det.get("label") or det.get("class_name")
                if label and self._accept_class(str(label)):
                    detections.append(Detection(
                        bbox=list(map(int, det.get("box") or det.get("bbox"))),
                        confidence=float(det.get("score") or det.get("confidence", 0.0)),
                        class_name=label,
                        class_id=int(det.get("class_id", 0)),
                    ))
            return detections
        except Exception as e:
            logger.error(f"RF-DETR detection error: {e}")
            return self._detect_yolo(frame, conf) if self.yolo_model is not None else []

    def crop_object(self, frame: np.ndarray, bbox: List[int], padding: int = 10) -> np.ndarray:
        """Crop object region with optional padding."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        return frame[y1:y2, x1:x2].copy()

    def should_use_rfdetr(self, frame: np.ndarray, yolo_detections: List[Detection]) -> bool:
        """
        In HYBRID mode, decide whether to use RF-DETR for re-check.
        Triggers on: low confidence detections, crowded scenes.
        """
        if not yolo_detections:
            return True
        avg_conf = sum(d.confidence for d in yolo_detections) / len(yolo_detections)
        return avg_conf < 0.55 or len(yolo_detections) > 8
