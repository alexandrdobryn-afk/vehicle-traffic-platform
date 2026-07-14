"""Production instance-segmentation adapter with compact polygon output."""

from typing import Dict, List
import logging

import numpy as np

logger = logging.getLogger(__name__)


class SegmentationService:
    def __init__(self, model_config, execution_device: str = "cpu", threshold: float = 0.35):
        self.model_config = model_config
        self.threshold = float(threshold)
        self.model = None
        if not model_config.available:
            return
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_config.path)
            self.device = execution_device
        except Exception as exc:
            logger.warning("Segmentation model load failed: %s", exc)

    @property
    def available(self) -> bool:
        return self.model is not None

    def segment(self, frame: np.ndarray) -> List[Dict]:
        if not self.available or frame is None or frame.size == 0:
            return []
        result = self.model.predict(frame, conf=self.threshold, device=self.device, verbose=False)[0]
        if result.boxes is None or result.masks is None:
            return []
        names = result.names or {}
        polygons = result.masks.xy
        segments: List[Dict] = []
        for index, polygon in enumerate(polygons):
            if polygon is None or len(polygon) < 3:
                continue
            box = result.boxes[index]
            class_id = int(box.cls.item())
            # Limit payload size while preserving polygon shape.
            step = max(1, len(polygon) // 96)
            points = [[round(float(x), 1), round(float(y), 1)] for x, y in np.asarray(polygon)[::step]]
            segments.append({
                "class_id": class_id,
                "class_name": str(names.get(class_id, class_id)),
                "confidence": round(float(box.conf.item()), 4),
                "bbox": [round(float(v), 1) for v in box.xyxy[0].tolist()],
                "polygon": points,
            })
        return segments
