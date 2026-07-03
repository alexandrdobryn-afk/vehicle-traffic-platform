import cv2
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PlateDetection:
    bbox: List[int]        # [x1, y1, x2, y2] relative to vehicle crop
    confidence: float
    crop: Optional[np.ndarray] = None


class PlateDetectionService:
    """
    Detects license plates inside vehicle crops.
    Uses YOLO model trained specifically for plate detection.
    Uses classical CV only when no configured model exists. A configured model
    never silently turns into another detector.
    """

    def __init__(
        self,
        model_config=None,
        confidence_threshold: float = 0.40,
        execution_device: Optional[str] = None,
    ):
        self.model = None
        self.confidence_threshold = confidence_threshold
        self.model_config = model_config
        self.execution_device = execution_device
        self.device = self._get_device()
        self._load_model()

    def _get_device(self) -> str:
        if self.execution_device:
            return self.execution_device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _load_model(self):
        if self.model_config is None or not self.model_config.available:
            logger.warning("No plate detector model available, using classical CV fallback")
            return

        try:
            if self.model_config.model_type == "openimagemodels":
                import onnxruntime as ort
                from open_image_models.detection.core.yolo_v9.inference import (
                    YoloV9ObjectDetector,
                )

                available_providers = ort.get_available_providers()
                providers = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if self.device.startswith("cuda") and "CUDAExecutionProvider" in available_providers
                    else ["CPUExecutionProvider"]
                )
                self.model = YoloV9ObjectDetector(
                    model_path=self.model_config.path,
                    class_labels=["License Plate"],
                    conf_thresh=self.confidence_threshold,
                    providers=providers,
                )
            else:
                from ultralytics import YOLO
                self.model = YOLO(self.model_config.path)
            logger.info(f"Plate detector loaded: {self.model_config.name}")
        except Exception as e:
            logger.error(f"Failed to load plate detector: {e}")
            self.model = None
            raise RuntimeError(
                f"Plate detector '{self.model_config.name}' failed to initialize"
            ) from e

    def detect(self, vehicle_crop: np.ndarray) -> List[PlateDetection]:
        """Detect plates in vehicle crop."""
        if self.model is not None:
            if self.model_config.model_type == "openimagemodels":
                return self._detect_openimagemodels(vehicle_crop)
            return self._detect_yolo(vehicle_crop)
        else:
            return self._detect_classical(vehicle_crop)

    def _detect_openimagemodels(self, vehicle_crop: np.ndarray) -> List[PlateDetection]:
        try:
            h, w = vehicle_crop.shape[:2]
            detections = []
            for result in self.model.predict(vehicle_crop):
                box = result.bounding_box.clamp(w, h)
                x1, y1, x2, y2 = tuple(box)
                plate_crop = vehicle_crop[y1:y2, x1:x2].copy()
                if plate_crop.size == 0:
                    continue
                detections.append(PlateDetection(
                    bbox=[x1, y1, x2, y2],
                    confidence=float(result.confidence),
                    crop=plate_crop,
                ))
            detections.sort(key=lambda d: d.confidence, reverse=True)
            return detections[:3]
        except Exception as e:
            logger.error(f"OpenImageModels plate detection error: {e}")
            return []

    def _detect_yolo(self, vehicle_crop: np.ndarray) -> List[PlateDetection]:
        try:
            predict_kwargs = {
                "conf": self.confidence_threshold,
                "verbose": False,
            }
            if not self.model_config or self.model_config.format != "engine":
                predict_kwargs.update(device=self.device, half=False)
            results = self.model(vehicle_crop, **predict_kwargs)
            detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    plate_crop = vehicle_crop[y1:y2, x1:x2].copy()

                    if plate_crop.size == 0:
                        continue

                    detections.append(PlateDetection(
                        bbox=[x1, y1, x2, y2],
                        confidence=conf,
                        crop=plate_crop
                    ))

            # Sort by confidence desc
            detections.sort(key=lambda d: d.confidence, reverse=True)
            return detections[:3]  # Max 3 candidates per vehicle

        except Exception as e:
            logger.error(f"YOLO plate detection error: {e}")
            return []

    def _detect_classical(self, vehicle_crop: np.ndarray) -> List[PlateDetection]:
        """
        Classical CV plate detection as fallback.
        Works on plates with high contrast edges.
        """
        try:
            gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # Focus on lower half of vehicle (plates are usually lower)
            roi_y_start = int(h * 0.4)
            roi = gray[roi_y_start:, :]

            # Edge detection
            blurred = cv2.GaussianBlur(roi, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 200)

            # Morphological to connect edges
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            detections = []
            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect = cw / max(ch, 1)
                area = cw * ch

                # Plate-like aspect ratio and size
                if 2.0 <= aspect <= 7.0 and 500 <= area <= (w * h * 0.15):
                    # Adjust Y back to full crop coordinates
                    abs_y = roi_y_start + y
                    plate_crop = vehicle_crop[abs_y:abs_y + ch, x:x + cw].copy()

                    if plate_crop.size == 0:
                        continue

                    detections.append(PlateDetection(
                        bbox=[x, abs_y, x + cw, abs_y + ch],
                        confidence=0.35,  # Low confidence for classical CV
                        crop=plate_crop
                    ))

            # Sort by area desc (biggest plate first)
            detections.sort(key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]), reverse=True)
            return detections[:2]

        except Exception as e:
            logger.error(f"Classical plate detection error: {e}")
            return []

    def get_best_plate(self, detections: List[PlateDetection]) -> Optional[PlateDetection]:
        """Return highest-confidence plate."""
        if not detections:
            return None
        return max(detections, key=lambda d: d.confidence)
