import json
import asyncio
from typing import Dict, Optional, List, Any, Tuple
from dataclasses import dataclass, field, asdict, is_dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ActiveTrack:
    track_id: int
    camera_id: int
    vehicle_class: str
    bbox: List[int]
    color: str = "unknown"
    color_confidence: float = 0.0
    color_history: List = field(default_factory=list)
    vehicle_make: str = "unknown"
    make_confidence: float = 0.0
    brand_history: List = field(default_factory=list)
    class_history: List = field(default_factory=list)
    plate: Optional[str] = None
    plate_status: str = "searching"
    plate_confidence: float = 0.0
    ocr_candidates: List = field(default_factory=list)
    first_seen: str = ""
    last_seen: str = ""
    frame_count: int = 0
    db_track_id: Optional[int] = None  # PostgreSQL row id
    best_vehicle_crop_path: Optional[str] = None
    best_plate_crop_path: Optional[str] = None
    best_detection_conf: float = 0.0
    best_vehicle_crop_score: float = 0.0
    best_plate_crop_score: float = 0.0
    processing_run_id: str = ""
    recognition_diagnostics: Dict[str, Any] = field(default_factory=dict)

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
            "vehicle_class": self.vehicle_class,
            "detection_confidence": round(self.best_detection_conf, 3),
            "bbox": self.bbox,
            "color": self.color,
            "color_confidence": self.color_confidence,
            "vehicle_make": self.vehicle_make,
            "make_confidence": self.make_confidence,
            "plate": self.plate,
            "plate_status": self.plate_status,
            "plate_confidence": self.plate_confidence,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "frame_count": self.frame_count,
            "db_track_id": self.db_track_id,
            "duration_seconds": self.duration_seconds,
            "best_vehicle_crop_path": self.best_vehicle_crop_path,
            "best_plate_crop_path": self.best_plate_crop_path,
            "processing_run_id": self.processing_run_id,
            "recognition_diagnostics": self.recognition_diagnostics,
            "ocr_candidates": [
                asdict(candidate) if is_dataclass(candidate) else candidate
                for candidate in self.ocr_candidates
            ],
        }

    def to_ws_dict(self) -> dict:
        """Minimal dict for WebSocket broadcast."""
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "vehicle_class": self.vehicle_class,
            "plate": self.plate,
            "plate_status": self.plate_status,
            "plate_confidence": round(self.plate_confidence, 3),
            "color": self.color,
            "color_confidence": round(self.color_confidence, 3),
            "vehicle_make": self.vehicle_make,
            "make_confidence": round(self.make_confidence, 3),
            "recognition_diagnostics": self.recognition_diagnostics,
        }


class TrackStateService:
    """
    Manages active vehicle tracks in memory + Redis.
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
        vehicle_class: str,
        bbox: List[int],
        detection_conf: float,
        now: str,
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
                    vehicle_class=vehicle_class,
                    bbox=bbox,
                    first_seen=now,
                    last_seen=now,
                    best_detection_conf=detection_conf,
                    processing_run_id=self.processing_run_id,
                )
                self._tracks[track_id] = track
            else:
                track = self._tracks[track_id]
                track.bbox = bbox
                track.last_seen = now
                if detection_conf > track.best_detection_conf:
                    track.best_detection_conf = detection_conf

            track.class_history.append((vehicle_class, float(detection_conf)))
            if len(track.class_history) > 15:
                track.class_history = track.class_history[-15:]
            class_scores: Dict[str, float] = {}
            for observed_class, observed_confidence in track.class_history:
                class_scores[observed_class] = class_scores.get(observed_class, 0.0) + observed_confidence
            track.vehicle_class = max(class_scores, key=class_scores.get)

            track.frame_count += 1
            self._missing_frames[track_id] = 0
            return track, is_new

    async def update_color(
        self,
        track_id: int,
        color: str,
        confidence: float,
        color_history: list,
    ):
        async with self._lock:
            if track_id in self._tracks:
                t = self._tracks[track_id]
                t.color = color
                t.color_confidence = confidence
                t.color_history = color_history

    async def update_brand(
        self,
        track_id: int,
        vehicle_make: str,
        confidence: float,
        brand_history: list,
    ):
        async with self._lock:
            if track_id in self._tracks:
                track = self._tracks[track_id]
                track.vehicle_make = vehicle_make
                track.make_confidence = confidence
                track.brand_history = brand_history

    async def update_plate(
        self,
        track_id: int,
        plate: Optional[str],
        plate_status: str,
        plate_confidence: float,
        ocr_candidates: list,
    ):
        async with self._lock:
            if track_id in self._tracks:
                t = self._tracks[track_id]
                t.plate = plate
                t.plate_status = plate_status
                t.plate_confidence = plate_confidence
                t.ocr_candidates = ocr_candidates

    async def update_diagnostics(self, track_id: int, **diagnostics: Any):
        async with self._lock:
            if track_id in self._tracks:
                self._tracks[track_id].recognition_diagnostics.update(diagnostics)

    async def update_crops(
        self,
        track_id: int,
        vehicle_crop_path: Optional[str] = None,
        plate_crop_path: Optional[str] = None,
    ):
        async with self._lock:
            if track_id in self._tracks:
                t = self._tracks[track_id]
                if vehicle_crop_path:
                    t.best_vehicle_crop_path = vehicle_crop_path
                if plate_crop_path:
                    t.best_plate_crop_path = plate_crop_path

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
                removed.append(self._tracks.pop(tid))
                self._missing_frames.pop(tid, None)
        return removed

    async def remove_all_tracks(self) -> List[ActiveTrack]:
        """Atomically remove and return every active track at a finite source EOF."""
        async with self._lock:
            removed = list(self._tracks.values())
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
