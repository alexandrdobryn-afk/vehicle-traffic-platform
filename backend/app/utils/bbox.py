from typing import List, Tuple
import numpy as np


def iou(box1: List[int], box2: List[int]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def clip_bbox(bbox: List[int], frame_w: int, frame_h: int) -> List[int]:
    x1, y1, x2, y2 = bbox
    return [max(0, x1), max(0, y1), min(frame_w, x2), min(frame_h, y2)]


def bbox_area(bbox: List[int]) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def bbox_center(bbox: List[int]) -> Tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


def scale_bbox(bbox: List[int], scale: float) -> List[int]:
    cx, cy = bbox_center(bbox)
    w = (bbox[2] - bbox[0]) * scale
    h = (bbox[3] - bbox[1]) * scale
    return [int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2)]
