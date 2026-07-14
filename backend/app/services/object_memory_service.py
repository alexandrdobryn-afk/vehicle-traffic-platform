"""Object-level memory that sits above frame trackers.

Trackers answer "what ID did this algorithm assign right now?" Object memory
answers "which physical object is this across ID resets and report rows?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import hashlib

import numpy as np

from app.services.tracking_service import TrackedObject


@dataclass
class ObjectCard:
    memory_id: int
    class_group: str
    bbox: List[int]
    first_frame: int
    last_frame: int
    observations: int = 0
    confidence_sum: float = 0.0
    best_confidence: float = 0.0
    best_bbox: List[int] = field(default_factory=list)
    trajectory: List[List[float]] = field(default_factory=list)
    size_history: List[List[int]] = field(default_factory=list)
    velocity: List[float] = field(default_factory=lambda: [0.0, 0.0])
    direction_degrees: Optional[float] = None
    appearance_signature: List[float] = field(default_factory=list)
    embedding_model: Optional[str] = None
    embedding_hash: Optional[str] = None
    merged_track_ids: List[int] = field(default_factory=list)
    class_votes: Dict[str, float] = field(default_factory=dict)

    @property
    def mean_confidence(self) -> float:
        return self.confidence_sum / max(1, self.observations)

    @property
    def speed_pixels_per_frame(self) -> float:
        return float(math.hypot(self.velocity[0], self.velocity[1]))


class ObjectMemoryService:
    """Maintains canonical object cards above ByteTrack/BoT-SORT/OC-SORT."""

    def __init__(
        self,
        max_gap_frames: int = 90,
        merge_threshold: float = 0.48,
        duplicate_iou: float = 0.45,
        duplicate_contained: float = 0.78,
        appearance_threshold: float = 0.68,
        embedding_enabled: bool = True,
        embedding_model: str = "histogram",
        store_embedding_vector: bool = False,
    ):
        self.max_gap_frames = max(1, int(max_gap_frames))
        self.merge_threshold = float(merge_threshold)
        self.duplicate_iou = float(duplicate_iou)
        self.duplicate_contained = float(duplicate_contained)
        self.appearance_threshold = float(appearance_threshold)
        self.embedding_enabled = bool(embedding_enabled)
        self.embedding_model = embedding_model
        self.store_embedding_vector = bool(store_embedding_vector)
        self._frame_index = 0
        self._next_memory_id = 1
        self._cards: Dict[int, ObjectCard] = {}
        self._raw_to_memory: Dict[Tuple[int, str], int] = {}
        self.last_stats: Dict[str, int] = {}

    def update(
        self,
        tracks: List[TrackedObject],
        frame: Optional[np.ndarray] = None,
    ) -> List[TrackedObject]:
        self._frame_index += 1
        self._forget_stale()
        stats = {
            "input_tracks": len(tracks),
            "same_frame_duplicates": 0,
            "merged_tracks": 0,
            "new_cards": 0,
        }
        assigned: Dict[int, TrackedObject] = {}
        output: List[TrackedObject] = []

        for track in sorted(tracks, key=self._track_rank, reverse=True):
            group = self._class_group(track.class_name)
            signature = self._track_embedding(track, frame)
            raw_key = (int(track.track_id), group)
            memory_id = self._raw_to_memory.get(raw_key)

            duplicate_id = self._assigned_duplicate_id(track, assigned)
            if duplicate_id is not None:
                self._raw_to_memory[raw_key] = duplicate_id
                stats["same_frame_duplicates"] += 1
                self._update_card(duplicate_id, track, signature)
                continue

            if memory_id is None or memory_id not in self._cards:
                candidate_id, score = self._best_candidate(track, group, signature)
                if candidate_id is not None and score >= self.merge_threshold:
                    memory_id = candidate_id
                    stats["merged_tracks"] += 1
                else:
                    memory_id = self._create_card(group, track, signature)
                    stats["new_cards"] += 1
                self._raw_to_memory[raw_key] = memory_id

            if memory_id in assigned and not self._same_frame_duplicate(track, assigned[memory_id]):
                memory_id = self._create_card(group, track, signature)
                self._raw_to_memory[raw_key] = memory_id
                stats["new_cards"] += 1

            if memory_id in assigned and self._same_frame_duplicate(track, assigned[memory_id]):
                stats["same_frame_duplicates"] += 1
                self._update_card(memory_id, track, signature)
                continue

            self._update_card(memory_id, track, signature)
            card = self._cards[memory_id]
            track.track_id = memory_id
            track.motion_vector = card.velocity
            track.appearance_signature = card.appearance_signature
            assigned[memory_id] = track
            output.append(track)

        stats["output_tracks"] = len(output)
        self.last_stats = stats
        return sorted(output, key=lambda item: item.track_id)

    def card_summary(self, memory_id: int) -> Optional[dict]:
        card = self._cards.get(memory_id)
        if card is None:
            return None
        return {
            "memory_id": card.memory_id,
            "observations": card.observations,
            "mean_confidence": round(card.mean_confidence, 4),
            "best_confidence": round(card.best_confidence, 4),
            "best_bbox": card.best_bbox,
            "last_bbox": card.bbox,
            "trajectory": card.trajectory[-30:],
            "speed_pixels_per_frame": round(card.speed_pixels_per_frame, 3),
            "direction_degrees": card.direction_degrees,
            "size": card.size_history[-1] if card.size_history else None,
            "embedding": {
                "enabled": self.embedding_enabled,
                "model": card.embedding_model or self.embedding_model,
                "dim": len(card.appearance_signature),
                "hash": card.embedding_hash,
                **({"vector": card.appearance_signature} if self.store_embedding_vector else {}),
            },
            "merged_track_ids": list(card.merged_track_ids),
            "class_votes": dict(card.class_votes),
        }

    def reset(self):
        self._frame_index = 0
        self._next_memory_id = 1
        self._cards.clear()
        self._raw_to_memory.clear()
        self.last_stats = {}

    def _create_card(
        self,
        group: str,
        track: TrackedObject,
        signature: List[float],
    ) -> int:
        memory_id = self._next_memory_id
        self._next_memory_id += 1
        self._cards[memory_id] = ObjectCard(
            memory_id=memory_id,
            class_group=group,
            bbox=list(track.bbox),
            first_frame=self._frame_index,
            last_frame=self._frame_index,
            best_bbox=list(track.bbox),
            appearance_signature=signature,
            embedding_model=self.embedding_model if signature else None,
            embedding_hash=self._embedding_hash(signature) if signature else None,
        )
        return memory_id

    def _update_card(
        self,
        memory_id: int,
        track: TrackedObject,
        signature: List[float],
    ):
        card = self._cards[memory_id]
        previous_center = self._center(card.bbox)
        current_center = self._center(track.bbox)
        velocity = [
            float(current_center[0] - previous_center[0]),
            float(current_center[1] - previous_center[1]),
        ]
        card.velocity = velocity
        if velocity[0] or velocity[1]:
            card.direction_degrees = round((math.degrees(math.atan2(velocity[1], velocity[0])) + 360) % 360, 2)
        card.bbox = list(track.bbox)
        card.last_frame = self._frame_index
        card.observations += 1
        card.confidence_sum += float(track.confidence)
        if float(track.confidence) >= card.best_confidence:
            card.best_confidence = float(track.confidence)
            card.best_bbox = list(track.bbox)
        center = [round(current_center[0], 2), round(current_center[1], 2)]
        card.trajectory.append(center)
        card.trajectory = card.trajectory[-300:]
        card.size_history.append([
            max(0, int(track.bbox[2] - track.bbox[0])),
            max(0, int(track.bbox[3] - track.bbox[1])),
        ])
        card.size_history = card.size_history[-60:]
        raw_id = int(track.track_id)
        if raw_id not in card.merged_track_ids:
            card.merged_track_ids.append(raw_id)
            card.merged_track_ids = card.merged_track_ids[-20:]
        card.class_votes[track.class_name] = card.class_votes.get(track.class_name, 0.0) + float(track.confidence)
        if signature:
            if card.appearance_signature:
                blended = np.asarray(card.appearance_signature, dtype=np.float32) * 0.72 + np.asarray(signature, dtype=np.float32) * 0.28
                card.appearance_signature = self._normalize_vector(blended).round(5).tolist()
            else:
                card.appearance_signature = signature
            card.embedding_model = self.embedding_model
            card.embedding_hash = self._embedding_hash(card.appearance_signature)

    def _best_candidate(
        self,
        track: TrackedObject,
        group: str,
        signature: List[float],
    ) -> Tuple[Optional[int], float]:
        best_id = None
        best_score = 0.0
        for memory_id, card in self._cards.items():
            gap = self._frame_index - card.last_frame
            if card.class_group != group or gap < 1 or gap > self.max_gap_frames:
                continue
            score = self._association_score(track, card, gap, signature)
            if score > best_score:
                best_id = memory_id
                best_score = score
        return best_id, best_score

    def _association_score(
        self,
        track: TrackedObject,
        card: ObjectCard,
        gap: int,
        signature: List[float],
    ) -> float:
        iou = self._bbox_iou(track.bbox, card.bbox)
        contained = self._contained_overlap(track.bbox, card.bbox)
        motion = self._motion_score(track.bbox, card, gap)
        center = self._center_score(track.bbox, card.bbox)
        appearance = self._appearance_similarity(signature, card.appearance_signature)

        if iou >= 0.08:
            return 0.55 + min(0.35, iou)
        if contained >= 0.55:
            return 0.50 + min(0.35, contained * 0.35)
        if motion >= 0.38 and appearance >= self.appearance_threshold:
            return motion * 0.62 + appearance * 0.38
        if center >= 0.42 and (appearance >= self.appearance_threshold or gap <= 2):
            return center * 0.70 + appearance * 0.30
        if appearance >= 0.88 and gap <= max(8, self.max_gap_frames // 4):
            return 0.52 + appearance * 0.20
        return 0.0

    def _motion_score(self, bbox: List[int], card: ObjectCard, gap: int) -> float:
        previous_center = self._center(card.bbox)
        predicted_center = (
            previous_center[0] + card.velocity[0] * max(1, gap),
            previous_center[1] + card.velocity[1] * max(1, gap),
        )
        current_center = self._center(bbox)
        distance = float(math.hypot(current_center[0] - predicted_center[0], current_center[1] - predicted_center[1]))
        gate = max(80.0, self._max_box_extent(bbox, card.bbox) * 1.5 + 45.0 * max(1, gap))
        if distance > gate or not self._compatible_area(bbox, card.bbox):
            return 0.0
        return 1.0 - (distance / gate)

    @classmethod
    def _center_score(cls, left: List[int], right: List[int]) -> float:
        distance = float(math.hypot(cls._center(left)[0] - cls._center(right)[0], cls._center(left)[1] - cls._center(right)[1]))
        gate = max(70.0, cls._max_box_extent(left, right) * 1.1)
        if distance > gate or not cls._compatible_area(left, right):
            return 0.0
        return 1.0 - (distance / gate)

    def _same_frame_duplicate(self, left: TrackedObject, right: TrackedObject) -> bool:
        if self._class_group(left.class_name) != self._class_group(right.class_name):
            return False
        return (
            self._bbox_iou(left.bbox, right.bbox) >= self.duplicate_iou
            or self._contained_overlap(left.bbox, right.bbox) >= self.duplicate_contained
        )

    def _assigned_duplicate_id(
        self,
        track: TrackedObject,
        assigned: Dict[int, TrackedObject],
    ) -> Optional[int]:
        for memory_id, assigned_track in assigned.items():
            if self._same_frame_duplicate(track, assigned_track):
                return memory_id
        return None

    def _track_embedding(
        self,
        track: TrackedObject,
        frame: Optional[np.ndarray],
    ) -> List[float]:
        if not self.embedding_enabled:
            return []
        return track.appearance_signature or self._appearance_signature(frame, track.bbox)

    def _forget_stale(self):
        stale = [
            memory_id for memory_id, card in self._cards.items()
            if self._frame_index - card.last_frame > self.max_gap_frames
        ]
        for memory_id in stale:
            del self._cards[memory_id]
        if stale:
            stale_set = set(stale)
            self._raw_to_memory = {
                key: value for key, value in self._raw_to_memory.items()
                if value not in stale_set
            }

    @staticmethod
    def _class_group(class_name: str) -> str:
        normalized = (class_name or "unknown").strip().lower() or "unknown"
        vehicle_like = {"bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"}
        return "vehicle_like" if normalized in vehicle_like else normalized

    @staticmethod
    def _track_rank(track: TrackedObject) -> tuple:
        width = max(0, track.bbox[2] - track.bbox[0])
        height = max(0, track.bbox[3] - track.bbox[1])
        return (not track.predicted, float(track.confidence), width * height)

    @staticmethod
    def _center(box: List[int]) -> Tuple[float, float]:
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    @staticmethod
    def _bbox_area(box: List[int]) -> float:
        return float(max(0, box[2] - box[0]) * max(0, box[3] - box[1]))

    @classmethod
    def _bbox_iou(cls, left: List[int], right: List[int]) -> float:
        x1, y1 = max(left[0], right[0]), max(left[1], right[1])
        x2, y2 = min(left[2], right[2]), min(left[3], right[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = cls._bbox_area(left) + cls._bbox_area(right) - intersection
        return float(intersection / union) if union > 0 else 0.0

    @classmethod
    def _contained_overlap(cls, left: List[int], right: List[int]) -> float:
        x1, y1 = max(left[0], right[0]), max(left[1], right[1])
        x2, y2 = min(left[2], right[2]), min(left[3], right[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        return float(intersection / max(1.0, min(cls._bbox_area(left), cls._bbox_area(right))))

    @classmethod
    def _compatible_area(cls, left: List[int], right: List[int]) -> bool:
        left_area = cls._bbox_area(left)
        right_area = cls._bbox_area(right)
        ratio = left_area / max(1.0, right_area)
        return 0.08 <= ratio <= 12.0

    @staticmethod
    def _max_box_extent(left: List[int], right: List[int]) -> float:
        return float(max(
            max(1, left[2] - left[0]),
            max(1, left[3] - left[1]),
            max(1, right[2] - right[0]),
            max(1, right[3] - right[1]),
        ))

    @staticmethod
    def _appearance_signature(frame: Optional[np.ndarray], bbox: List[int]) -> List[float]:
        if frame is None or frame.size == 0:
            return []
        frame_h, frame_w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, min(frame_w, x1)), max(0, min(frame_h, y1))
        x2, y2 = max(0, min(frame_w, x2)), max(0, min(frame_h, y2))
        if x2 <= x1 or y2 <= y1:
            return []
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return []
        histograms = [
            np.histogram(crop[:, :, channel], bins=12, range=(0, 256))[0].astype(np.float32)
            for channel in range(min(3, crop.shape[2]))
        ]
        return ObjectMemoryService._normalize_vector(np.concatenate(histograms)).round(5).tolist()

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-9 else vector

    @staticmethod
    def _appearance_similarity(left: List[float], right: List[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        left_vector = np.asarray(left, dtype=np.float32)
        right_vector = np.asarray(right, dtype=np.float32)
        denom = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
        return float(np.dot(left_vector, right_vector) / denom) if denom > 0 else 0.0

    @staticmethod
    def _embedding_hash(vector: List[float]) -> Optional[str]:
        if not vector:
            return None
        data = np.asarray(vector, dtype=np.float32)
        return hashlib.sha1(data.tobytes()).hexdigest()[:12]
