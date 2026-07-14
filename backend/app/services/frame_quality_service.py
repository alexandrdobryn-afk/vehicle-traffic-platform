import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class QualityResult:
    is_good: bool
    blur_score: float
    brightness_score: float
    contrast_score: float
    overall_score: float
    reason: str = ""


class FrameQualityService:
    """Filters frames before inference to avoid processing blurry/dark/bad images."""

    def __init__(
        self,
        blur_threshold: float = 80.0,
        min_brightness: float = 40.0,
        max_brightness: float = 220.0,
        min_contrast: float = 30.0,
    ):
        self.blur_threshold = blur_threshold
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast

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

    def _compute_blur(self, gray: np.ndarray) -> float:
        """Laplacian variance — higher = sharper."""
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _compute_brightness(self, gray: np.ndarray) -> float:
        return float(gray.mean())

    def _compute_contrast(self, gray: np.ndarray) -> float:
        return float(gray.std())
