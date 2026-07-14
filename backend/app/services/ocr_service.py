"""Optional OCR adapters for text visible on aerial crops."""

from __future__ import annotations

from typing import Dict, List
import logging

import numpy as np

logger = logging.getLogger(__name__)


class OCRService:
    def __init__(self, model_config, execution_device: str = "cpu", threshold: float = 0.35):
        self.model_config = model_config
        self.execution_device = execution_device
        self.threshold = float(threshold)
        self.engine = None
        self.engine_name = getattr(model_config, "name", "auto")

        if not getattr(model_config, "available", False):
            return

        try:
            if model_config.model_type == "paddleocr":
                from paddleocr import PaddleOCR

                self.engine = PaddleOCR(
                    use_angle_cls=True,
                    lang="en",
                    use_gpu=execution_device.startswith("cuda"),
                    show_log=False,
                )
            elif model_config.model_type == "easyocr":
                import easyocr

                self.engine = easyocr.Reader(["en"], gpu=execution_device.startswith("cuda"))
        except Exception as exc:
            logger.warning("OCR adapter load failed: %s", exc)
            self.engine = None

    @property
    def available(self) -> bool:
        return self.engine is not None

    def recognize(self, crop: np.ndarray) -> Dict:
        if not self.available or crop is None or crop.size == 0:
            return {"text": "", "confidence": 0.0, "items": [], "status": "unavailable"}

        try:
            if self.model_config.model_type == "paddleocr":
                return self._recognize_paddle(crop)
            if self.model_config.model_type == "easyocr":
                return self._recognize_easyocr(crop)
        except Exception as exc:
            logger.warning("OCR recognition failed: %s", exc)
        return {"text": "", "confidence": 0.0, "items": [], "status": "failed"}

    def _recognize_paddle(self, crop: np.ndarray) -> Dict:
        result = self.engine.ocr(crop, cls=True)
        items: List[Dict] = []
        for page in result or []:
            for item in page or []:
                if not item or len(item) < 2:
                    continue
                text, score = item[1]
                confidence = float(score)
                if confidence < self.threshold:
                    continue
                items.append({
                    "text": str(text),
                    "confidence": round(confidence, 4),
                    "box": [[round(float(x), 1), round(float(y), 1)] for x, y in item[0]],
                })
        return self._pack_items(items)

    def _recognize_easyocr(self, crop: np.ndarray) -> Dict:
        result = self.engine.readtext(crop)
        items = []
        for box, text, score in result or []:
            confidence = float(score)
            if confidence < self.threshold:
                continue
            items.append({
                "text": str(text),
                "confidence": round(confidence, 4),
                "box": [[round(float(x), 1), round(float(y), 1)] for x, y in box],
            })
        return self._pack_items(items)

    @staticmethod
    def _pack_items(items: List[Dict]) -> Dict:
        if not items:
            return {"text": "", "confidence": 0.0, "items": [], "status": "empty"}
        text = " ".join(item["text"] for item in items).strip()
        confidence = sum(item["confidence"] for item in items) / max(len(items), 1)
        return {
            "text": text,
            "confidence": round(float(confidence), 4),
            "items": items,
            "status": "recognized",
        }
