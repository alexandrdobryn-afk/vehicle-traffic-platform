import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class QualityResult:
    is_good: bool
    blur_score: float
    brightness_score: float
    contrast_score: float
    overall_score: float
    reason: str = ""


class FrameQualityService:
    """Filters frames before OCR to avoid processing blurry/dark/bad images."""

    def __init__(
        self,
        blur_threshold: float = 80.0,
        min_brightness: float = 40.0,
        max_brightness: float = 220.0,
        min_contrast: float = 30.0,
        min_plate_width: int = 60,
        min_plate_height: int = 20,
    ):
        self.blur_threshold = blur_threshold
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast
        self.min_plate_width = min_plate_width
        self.min_plate_height = min_plate_height

    def check_frame(self, frame: np.ndarray) -> QualityResult:
        """Check overall frame quality."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blur_score = self._compute_blur(gray)
        brightness = self._compute_brightness(gray)
        contrast = self._compute_contrast(gray)

        blur_ok = blur_score >= self.blur_threshold
        brightness_ok = self.min_brightness <= brightness <= self.max_brightness
        contrast_ok = contrast >= self.min_contrast

        reasons = []
        if not blur_ok:
            reasons.append(f"blur={blur_score:.1f}")
        if not brightness_ok:
            reasons.append(f"brightness={brightness:.1f}")
        if not contrast_ok:
            reasons.append(f"contrast={contrast:.1f}")

        is_good = blur_ok and brightness_ok and contrast_ok
        overall = (
            min(blur_score / self.blur_threshold, 1.0) * 0.4
            + min(contrast / 80.0, 1.0) * 0.3
            + (1.0 - abs(brightness - 128) / 128) * 0.3
        )

        return QualityResult(
            is_good=is_good,
            blur_score=blur_score,
            brightness_score=brightness,
            contrast_score=contrast,
            overall_score=float(np.clip(overall, 0, 1)),
            reason=", ".join(reasons) if reasons else "ok"
        )

    def check_plate_crop(self, plate_crop: np.ndarray) -> QualityResult:
        """Check plate crop quality specifically."""
        h, w = plate_crop.shape[:2]

        if w < self.min_plate_width or h < self.min_plate_height:
            return QualityResult(
                is_good=False,
                blur_score=0,
                brightness_score=0,
                contrast_score=0,
                overall_score=0,
                reason=f"plate too small: {w}x{h}"
            )

        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY) if len(plate_crop.shape) == 3 else plate_crop

        # For plates, use stricter blur threshold
        blur_score = self._compute_blur(gray)
        brightness = self._compute_brightness(gray)
        contrast = self._compute_contrast(gray)

        # Check aspect ratio (plates are wide)
        aspect = w / max(h, 1)
        aspect_ok = 1.5 <= aspect <= 8.0

        blur_ok = blur_score >= self.blur_threshold * 0.6  # slightly relaxed for crops
        brightness_ok = 30 <= brightness <= 225
        contrast_ok = contrast >= 20

        reasons = []
        if not blur_ok:
            reasons.append(f"blur={blur_score:.1f}")
        if not brightness_ok:
            reasons.append(f"brightness={brightness:.1f}")
        if not contrast_ok:
            reasons.append(f"contrast={contrast:.1f}")
        if not aspect_ok:
            reasons.append(f"aspect={aspect:.1f}")

        is_good = blur_ok and brightness_ok and contrast_ok and aspect_ok
        overall = (
            min(blur_score / self.blur_threshold, 1.0) * 0.5
            + min(contrast / 60.0, 1.0) * 0.3
            + (1.0 - abs(brightness - 128) / 128) * 0.2
        )

        return QualityResult(
            is_good=is_good,
            blur_score=blur_score,
            brightness_score=brightness,
            contrast_score=contrast,
            overall_score=float(np.clip(overall, 0, 1)),
            reason=", ".join(reasons) if reasons else "ok"
        )

    def _compute_blur(self, gray: np.ndarray) -> float:
        """Laplacian variance — higher = sharper."""
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _compute_brightness(self, gray: np.ndarray) -> float:
        return float(gray.mean())

    def _compute_contrast(self, gray: np.ndarray) -> float:
        return float(gray.std())

    def preprocess_plate_for_ocr(self, plate_crop: np.ndarray) -> np.ndarray:
        """Enhance plate image before OCR."""
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY) if len(plate_crop.shape) == 3 else plate_crop

        # Upscale if too small
        h, w = gray.shape
        if w < 200:
            scale = 200 / w
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        # CLAHE equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Denoise
        gray = cv2.fastNlMeansDenoising(gray, h=10)

        # Back to BGR for PaddleOCR
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
