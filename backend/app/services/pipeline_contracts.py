"""Stable contracts for pluggable drone/aerial computer-vision components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class Prediction:
    bbox: tuple[int, int, int, int]
    confidence: float
    class_name: str
    class_id: int
    mask: Any | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineContext:
    source_id: int
    run_id: str
    frame_index: int
    timestamp: str
    task_profile: str = "generic_objects"
    metadata: Mapping[str, Any] = field(default_factory=dict)


class Detector(Protocol):
    def detect(self, frame: np.ndarray, confidence_threshold: float | None = None) -> Sequence[Prediction]: ...


class Tracker(Protocol):
    def update(self, detections: Sequence[Prediction], frame: np.ndarray) -> Sequence[Any]: ...


class Classifier(Protocol):
    def classify(self, crop: np.ndarray) -> Mapping[str, Any]: ...


class Segmenter(Protocol):
    def segment(self, frame: np.ndarray) -> Sequence[Prediction]: ...


class FrameProcessor(Protocol):
    def process(self, frame: np.ndarray, context: PipelineContext) -> np.ndarray: ...
