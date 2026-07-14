"""High-resolution and small-object processing primitives.

Tiling is deliberately model-independent.  A detector callback runs on each
tile and detections are projected back into source coordinates before global
object-agnostic NMS. This makes the same implementation usable by YOLO, DETR
and future ONNX/TensorRT adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, TypeVar

import cv2
import numpy as np

T = TypeVar("T")


@dataclass(frozen=True)
class TileConfig:
    enabled: bool = False
    size: int = 1024
    overlap: float = 0.2
    nms_iou: float = 0.5
    scale: float = 1.0
    enhance: bool = False

    @classmethod
    def from_mapping(cls, value: dict | None) -> "TileConfig":
        value = value or {}
        return cls(
            enabled=bool(value.get("enabled", False)),
            size=max(320, min(2048, int(value.get("size", 1024)))),
            overlap=max(0.0, min(0.5, float(value.get("overlap", 0.2)))),
            nms_iou=max(0.1, min(0.9, float(value.get("nms_iou", 0.5)))),
            scale=max(1.0, min(2.0, float(value.get("scale", 1.0)))),
            enhance=bool(value.get("enhance", False)),
        )


class AerialProcessingService:
    def __init__(self, config: TileConfig):
        self.config = config

    def prepare(self, frame: np.ndarray) -> np.ndarray:
        result = frame
        if self.config.scale > 1.0:
            result = cv2.resize(
                result,
                None,
                fx=self.config.scale,
                fy=self.config.scale,
                interpolation=cv2.INTER_CUBIC,
            )
        if self.config.enhance:
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            lightness, a, b = cv2.split(lab)
            lightness = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(lightness)
            result = cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2BGR)
        return result

    def infer_tiled(self, frame: np.ndarray, callback: Callable[[np.ndarray], Sequence[T]]) -> list[T]:
        height, width = frame.shape[:2]
        tile_size = min(self.config.size, max(height, width))
        if not self.config.enabled or (height <= tile_size and width <= tile_size):
            return list(callback(frame))

        stride = max(1, int(tile_size * (1.0 - self.config.overlap)))
        predictions: list[T] = []
        for y in self._positions(height, tile_size, stride):
            for x in self._positions(width, tile_size, stride):
                tile = frame[y:min(y + tile_size, height), x:min(x + tile_size, width)]
                for prediction in callback(tile):
                    prediction.bbox = [
                        int(prediction.bbox[0] + x), int(prediction.bbox[1] + y),
                        int(prediction.bbox[2] + x), int(prediction.bbox[3] + y),
                    ]
                    predictions.append(prediction)
        return self.object_agnostic_nms(predictions, self.config.nms_iou)

    @staticmethod
    def _positions(length: int, tile: int, stride: int) -> Iterable[int]:
        if length <= tile:
            return (0,)
        positions = list(range(0, max(1, length - tile + 1), stride))
        final = length - tile
        if positions[-1] != final:
            positions.append(final)
        return positions

    @staticmethod
    def object_agnostic_nms(predictions: Sequence[T], iou_threshold: float) -> list[T]:
        kept: list[T] = []
        remaining = sorted(predictions, key=lambda item: float(item.confidence), reverse=True)
        while remaining:
            candidate = remaining.pop(0)
            replaced = False
            suppressed = False
            for index, kept_item in enumerate(kept):
                if not _same_object_overlap(candidate, kept_item, iou_threshold):
                    continue
                candidate_area = _area(candidate.bbox)
                kept_area = _area(kept_item.bbox)
                candidate_conf = float(candidate.confidence)
                kept_conf = float(kept_item.confidence)
                if candidate_area > kept_area and candidate_conf >= kept_conf - 0.18:
                    kept[index] = candidate
                    replaced = True
                suppressed = True
                break
            if not suppressed and not replaced:
                kept.append(candidate)
        return kept

    @staticmethod
    def class_aware_nms(predictions: Sequence[T], iou_threshold: float) -> list[T]:
        """Backward-compatible alias; the platform now suppresses by object."""
        return AerialProcessingService.object_agnostic_nms(predictions, iou_threshold)


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _same_object_overlap(left: T, right: T, iou_threshold: float) -> bool:
    left_box = left.bbox
    right_box = right.bbox
    left_group = _class_group(getattr(left, "class_name", "unknown"))
    right_group = _class_group(getattr(right, "class_name", "unknown"))

    x1, y1 = max(left_box[0], right_box[0]), max(left_box[1], right_box[1])
    x2, y2 = min(left_box[2], right_box[2]), min(left_box[3], right_box[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return False
    left_area = _area(left_box)
    right_area = _area(right_box)
    union = left_area + right_area - intersection
    iou = intersection / union if union > 0 else 0.0

    # Exact or near-exact duplicate boxes are almost always detector/tile
    # duplicates, even if the model flipped the class label.
    if iou >= iou_threshold:
        return True

    # Containment is more dangerous: a person inside a car, a roof on a
    # building, or a sign on a road can be valid separate objects. Only merge
    # contained boxes when they are in the same coarse physical group.
    if left_group != right_group:
        return False
    contained_overlap = intersection / max(1.0, min(left_area, right_area))
    return contained_overlap >= 0.68


def _class_group(class_name: str) -> str:
    normalized = (class_name or "unknown").strip().lower() or "unknown"
    vehicle_like = {
        "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    }
    return "vehicle_like" if normalized in vehicle_like else normalized
