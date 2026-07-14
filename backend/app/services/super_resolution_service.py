"""Crop-level super-resolution adapter with a dependency-free OpenCV fallback."""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np


class SuperResolutionService:
    def __init__(self, engine: str = "auto", min_object_size_px: int = 32, max_crops_per_frame: int = 8):
        self.engine = engine or "auto"
        self.min_object_size_px = max(4, int(min_object_size_px))
        self.max_crops_per_frame = max(1, int(max_crops_per_frame))
        self._processed_this_frame = 0

    @property
    def available(self) -> bool:
        return True

    def begin_frame(self):
        self._processed_this_frame = 0

    def enhance(self, crop: np.ndarray) -> Tuple[np.ndarray, Dict]:
        if crop is None or crop.size == 0:
            return crop, {"status": "empty", "scale": 1.0, "engine": "opencv_lanczos"}
        h, w = crop.shape[:2]
        smallest = min(h, w)
        if self._processed_this_frame >= self.max_crops_per_frame:
            return crop, {"status": "budget_exhausted", "scale": 1.0, "engine": "opencv_lanczos"}
        if smallest >= self.min_object_size_px:
            return crop, {"status": "skipped_large_enough", "scale": 1.0, "engine": "opencv_lanczos"}

        scale = min(4.0, max(2.0, self.min_object_size_px / max(smallest, 1)))
        enhanced = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=0.8)
        enhanced = cv2.addWeighted(enhanced, 1.35, blurred, -0.35, 0)
        self._processed_this_frame += 1
        return enhanced, {
            "status": "enhanced",
            "scale": round(float(scale), 3),
            "engine": "opencv_lanczos",
            "source_size": [int(w), int(h)],
            "output_size": [int(enhanced.shape[1]), int(enhanced.shape[0])],
        }
