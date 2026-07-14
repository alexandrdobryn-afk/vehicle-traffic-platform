"""Tracker-independent constant-velocity Kalman prediction for object boxes."""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import time

import numpy as np

from app.services.tracking_service import TrackedObject


@dataclass
class _TrackFilter:
    state: np.ndarray
    covariance: np.ndarray
    class_name: str
    confidence: float
    missing: int = 0
    updated_at: float = 0.0


class KalmanPredictionService:
    """Smooth observed boxes and bridge short detector/tracker dropouts."""

    def __init__(self, max_prediction_frames: int = 8, smoothing: bool = True):
        self.max_prediction_frames = max(0, int(max_prediction_frames))
        self.smoothing = bool(smoothing)
        self._filters: Dict[int, _TrackFilter] = {}
        self._measurement = np.zeros((4, 8), dtype=np.float32)
        self._measurement[:, :4] = np.eye(4, dtype=np.float32)
        self._measurement_noise = np.eye(4, dtype=np.float32) * 4.0
        self._process_noise = np.eye(8, dtype=np.float32) * 0.05

    @staticmethod
    def _box_to_measurement(box: List[int]) -> np.ndarray:
        x1, y1, x2, y2 = [float(v) for v in box]
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2, max(1, x2 - x1), max(1, y2 - y1)], dtype=np.float32)

    @staticmethod
    def _state_to_box(state: np.ndarray, shape: Tuple[int, ...]) -> List[int]:
        cx, cy, width, height = state[:4]
        frame_h, frame_w = shape[:2]
        width, height = max(1.0, float(width)), max(1.0, float(height))
        return [
            max(0, int(round(cx - width / 2))),
            max(0, int(round(cy - height / 2))),
            min(frame_w, int(round(cx + width / 2))),
            min(frame_h, int(round(cy + height / 2))),
        ]

    @staticmethod
    def _transition(dt: float) -> np.ndarray:
        transition = np.eye(8, dtype=np.float32)
        transition[0, 4] = transition[1, 5] = transition[2, 6] = transition[3, 7] = dt
        return transition

    def update(self, tracks: List[TrackedObject], frame_shape: Tuple[int, ...]) -> List[TrackedObject]:
        now = time.monotonic()
        observed = set()
        output: List[TrackedObject] = []

        for track in tracks:
            observed.add(track.track_id)
            measurement = self._box_to_measurement(track.bbox)
            item = self._filters.get(track.track_id)
            if item is None:
                state = np.zeros(8, dtype=np.float32)
                state[:4] = measurement
                item = _TrackFilter(state, np.eye(8, dtype=np.float32) * 10.0, track.class_name, track.confidence, updated_at=now)
                self._filters[track.track_id] = item
            else:
                dt = min(1.0, max(1 / 120, now - item.updated_at))
                transition = self._transition(dt)
                item.state = transition @ item.state
                item.covariance = transition @ item.covariance @ transition.T + self._process_noise
                innovation = measurement - self._measurement @ item.state
                innovation_cov = self._measurement @ item.covariance @ self._measurement.T + self._measurement_noise
                gain = item.covariance @ self._measurement.T @ np.linalg.inv(innovation_cov)
                item.state = item.state + gain @ innovation
                item.covariance = (np.eye(8, dtype=np.float32) - gain @ self._measurement) @ item.covariance
                item.class_name, item.confidence = track.class_name, track.confidence
                item.updated_at = now
            item.missing = 0
            if self.smoothing:
                track.bbox = self._state_to_box(item.state, frame_shape)
            track.predicted = False
            track.motion_vector = [round(float(item.state[4]), 3), round(float(item.state[5]), 3)]
            output.append(track)

        for track_id, item in list(self._filters.items()):
            if track_id in observed:
                continue
            item.missing += 1
            if item.missing > self.max_prediction_frames:
                del self._filters[track_id]
                continue
            dt = min(1.0, max(1 / 120, now - item.updated_at))
            transition = self._transition(dt)
            item.state = transition @ item.state
            item.covariance = transition @ item.covariance @ transition.T + self._process_noise
            item.updated_at = now
            box = self._state_to_box(item.state, frame_shape)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            output.append(TrackedObject(
                track_id=track_id,
                bbox=box,
                confidence=max(0.05, item.confidence * (0.88 ** item.missing)),
                class_name=item.class_name,
                time_since_update=item.missing,
                predicted=True,
                motion_vector=[round(float(item.state[4]), 3), round(float(item.state[5]), 3)],
            ))
        return output

    def reset(self) -> None:
        self._filters.clear()
