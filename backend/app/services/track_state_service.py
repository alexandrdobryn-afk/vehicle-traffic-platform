import json
import asyncio
from typing import Dict, Optional, List, Any, Tuple
from dataclasses import dataclass, field, asdict, is_dataclass
from datetime import datetime
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class ActiveTrack:
    track_id: int
    camera_id: int
    object_class: str
    bbox: List[int]
    class_history: List = field(default_factory=list)
    first_seen: str = ""
    last_seen: str = ""
    frame_count: int = 0
    db_track_id: Optional[int] = None  # PostgreSQL row id
    best_crop_path: Optional[str] = None
    best_detection_conf: float = 0.0
    best_crop_score: float = 0.0
    processing_run_id: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    trajectory: List[List[float]] = field(default_factory=list)
    speed_pixels_per_second: float = 0.0
    direction_degrees: Optional[float] = None
    state: str = "active"
    attributes: Dict[str, Any] = field(default_factory=dict)
    first_video_timestamp_seconds: Optional[float] = None
    last_video_timestamp_seconds: Optional[float] = None
    predicted: bool = False
    motion_vector: List[float] = field(default_factory=list)
    events_fired: Dict[str, bool] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        try:
            start = datetime.fromisoformat(self.first_seen.replace("Z", "+00:00"))
            end = datetime.fromisoformat(self.last_seen.replace("Z", "+00:00"))
            return max(0.0, (end - start).total_seconds())
        except (TypeError, ValueError):
            return 0.0

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "camera_id": self.camera_id,
            "object_class": self.object_class,
            "detection_confidence": round(self.best_detection_conf, 3),
            "bbox": self.bbox,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "frame_count": self.frame_count,
            "db_track_id": self.db_track_id,
            "duration_seconds": self.duration_seconds,
            "best_crop_path": self.best_crop_path,
            "processing_run_id": self.processing_run_id,
            "diagnostics": self.diagnostics,
            "trajectory": self.trajectory,
            "speed_pixels_per_second": round(self.speed_pixels_per_second, 3),
            "direction_degrees": self.direction_degrees,
            "state": self.state,
            "attributes": self.attributes,
            "last_bbox": self.bbox,
            "first_video_timestamp_seconds": self.first_video_timestamp_seconds,
            "last_video_timestamp_seconds": self.last_video_timestamp_seconds,
            "predicted": self.predicted,
            "motion_vector": self.motion_vector,
            "events_fired": self.events_fired,
        }

    def to_ws_dict(self) -> dict:
        """Minimal dict for WebSocket broadcast."""
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "object_class": self.object_class,
            "detection_confidence": round(self.best_detection_conf, 3),
            "trajectory": self.trajectory[-30:],
            "speed_pixels_per_second": round(self.speed_pixels_per_second, 3),
            "direction_degrees": self.direction_degrees,
            "state": self.state,
            "attributes": self.attributes,
            "predicted": self.predicted,
            "motion_vector": self.motion_vector,
        }


class TrackStateService:
    """
    Manages active object tracks in memory + Redis.
    Coordinates between pipeline and database writes.
    """

    def __init__(
        self,
        camera_id: int,
        redis_client=None,
        ttl: int = 120,
        processing_run_id: str = "",
        max_missing_frames: int = 15,
    ):
        self.camera_id = camera_id
        self.redis = redis_client
        self.ttl = ttl
        self.processing_run_id = processing_run_id
        self.max_missing_frames = max(1, max_missing_frames)
        self._tracks: Dict[int, ActiveTrack] = {}
        self._missing_frames: Dict[int, int] = {}
        self._lock = asyncio.Lock()

    async def get_track(self, track_id: int) -> Optional[ActiveTrack]:
        async with self._lock:
            return self._tracks.get(track_id)

    async def get_all_active(self) -> List[ActiveTrack]:
        async with self._lock:
            return list(self._tracks.values())

    async def upsert_track(
        self,
        track_id: int,
        object_class: str,
        bbox: List[int],
        detection_conf: float,
        now: str,
        predicted: bool = False,
        motion_vector: Optional[List[float]] = None,
        video_timestamp_seconds: Optional[float] = None,
    ) -> Tuple[ActiveTrack, bool]:
        """
        Insert or update a track. Returns (track, is_new).
        """
        async with self._lock:
            is_new = track_id not in self._tracks

            if is_new:
                track = ActiveTrack(
                    track_id=track_id,
                    camera_id=self.camera_id,
                    object_class=object_class,
                    bbox=bbox,
                    first_seen=now,
                    last_seen=now,
                    best_detection_conf=detection_conf,
                    processing_run_id=self.processing_run_id,
                    first_video_timestamp_seconds=video_timestamp_seconds,
                    last_video_timestamp_seconds=video_timestamp_seconds,
                )
                self._tracks[track_id] = track
            else:
                track = self._tracks[track_id]
                previous_bbox = track.bbox
                previous_time = track.last_seen
                track.bbox = bbox
                if not predicted:
                    track.last_seen = now
                try:
                    elapsed = (
                        datetime.fromisoformat(now.replace("Z", "+00:00"))
                        - datetime.fromisoformat(previous_time.replace("Z", "+00:00"))
                    ).total_seconds()
                    if elapsed > 0:
                        old_center = ((previous_bbox[0] + previous_bbox[2]) / 2, (previous_bbox[1] + previous_bbox[3]) / 2)
                        new_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                        dx, dy = new_center[0] - old_center[0], new_center[1] - old_center[1]
                        track.speed_pixels_per_second = math.hypot(dx, dy) / elapsed
                        if dx or dy:
                            track.direction_degrees = round((math.degrees(math.atan2(dy, dx)) + 360) % 360, 2)
                except (TypeError, ValueError):
                    pass
                if not predicted and detection_conf > track.best_detection_conf:
                    track.best_detection_conf = detection_conf
                if video_timestamp_seconds is not None:
                    track.last_video_timestamp_seconds = video_timestamp_seconds

            track.predicted = predicted
            track.motion_vector = list(motion_vector or [])
            track.state = "predicted" if predicted else "active"

            center = [round((bbox[0] + bbox[2]) / 2, 2), round((bbox[1] + bbox[3]) / 2, 2)]
            track.trajectory.append(center)
            if len(track.trajectory) > 300:
                track.trajectory = track.trajectory[-300:]

            if not predicted:
                track.class_history.append((object_class, float(detection_conf)))
            if len(track.class_history) > 15:
                track.class_history = track.class_history[-15:]
            class_scores: Dict[str, float] = {}
            for observed_class, observed_confidence in track.class_history:
                class_scores[observed_class] = class_scores.get(observed_class, 0.0) + observed_confidence
            track.object_class = max(class_scores, key=class_scores.get)

            if not predicted:
                track.frame_count += 1
            self._missing_frames[track_id] = 0
            return track, is_new

    async def update_attributes(self, track_id: int, **attributes: Any):
        async with self._lock:
            if track_id in self._tracks:
                self._tracks[track_id].attributes.update(attributes)

    async def mark_event_fired(self, track_id: int, event_type: str):
        async with self._lock:
            if track_id in self._tracks:
                self._tracks[track_id].events_fired[event_type] = True

    async def update_diagnostics(self, track_id: int, **diagnostics: Any):
        async with self._lock:
            if track_id in self._tracks:
                self._tracks[track_id].diagnostics.update(diagnostics)

    async def update_crops(
        self,
        track_id: int,
        crop_path: Optional[str] = None,
    ):
        async with self._lock:
            if track_id in self._tracks:
                t = self._tracks[track_id]
                if crop_path:
                    t.best_crop_path = crop_path

    async def set_db_id(self, track_id: int, db_id: int):
        async with self._lock:
            if track_id in self._tracks:
                self._tracks[track_id].db_track_id = db_id

    async def remove_stale_tracks(
        self,
        active_track_ids: List[int],
        now_ts: float,
    ) -> List[ActiveTrack]:
        """Remove tracks only after a grace window, preserving tracker continuity."""
        removed = []
        async with self._lock:
            active = set(active_track_ids)
            for tid in self._tracks:
                if tid in active:
                    self._missing_frames[tid] = 0
                else:
                    self._missing_frames[tid] = self._missing_frames.get(tid, 0) + 1
            stale = [
                tid for tid in self._tracks
                if self._missing_frames.get(tid, 0) > self.max_missing_frames
            ]
            for tid in stale:
                track = self._tracks.pop(tid)
                track.state = "lost"
                removed.append(track)
                self._missing_frames.pop(tid, None)
        return removed

    async def remove_all_tracks(self) -> List[ActiveTrack]:
        """Atomically remove and return every active track at a finite source EOF."""
        async with self._lock:
            removed = list(self._tracks.values())
            for track in removed:
                track.state = "completed"
            self._tracks.clear()
            self._missing_frames.clear()
            return removed

    async def sync_to_redis(self, track: ActiveTrack):
        """Persist track to Redis for cross-service access."""
        if self.redis is None:
            return
        try:
            key = f"track:{self.camera_id}:{track.track_id}"
            data = json.dumps(track.to_dict())
            await self.redis.setex(key, self.ttl, data)
        except Exception as e:
            logger.debug(f"Redis sync error: {e}")

    async def sync_many_to_redis(self, tracks: List[ActiveTrack]):
        """Persist a frame's track state in one Redis round trip."""
        if self.redis is None or not tracks:
            return
        try:
            pipeline = self.redis.pipeline(transaction=False)
            for track in tracks:
                key = f"track:{self.camera_id}:{track.track_id}"
                pipeline.setex(key, self.ttl, json.dumps(track.to_dict()))
            await pipeline.execute()
        except Exception as e:
            logger.debug(f"Redis batch sync error: {e}")

    async def remove_from_redis(self, track: ActiveTrack):
        if self.redis is None:
            return
        try:
            key = f"track:{self.camera_id}:{track.track_id}"
            await self.redis.delete(key)
        except Exception as e:
            logger.debug(f"Redis remove error: {e}")

    def get_active_count(self) -> int:
        return len(self._tracks)
