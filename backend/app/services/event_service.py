import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

event_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)


class EventService:
    """Consumes runtime events and persists generic aerial object tracks."""

    def __init__(self, db_session_factory, redis_client=None):
        self.db_factory = db_session_factory
        self.redis = redis_client
        self._running = False

    async def start(self):
        self._running = True
        asyncio.create_task(self._consumer_loop())
        logger.info("EventService started")

    async def stop(self):
        self._running = False

    async def _consumer_loop(self):
        while self._running:
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=1.0)
                await self._process_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                logger.error("EventService consumer error: %s", exc)

    async def _process_event(self, event: dict):
        try:
            async with self.db_factory() as db:
                from app.models.database import Event

                payload = event.get("payload_json", {})
                object_track_id = None
                if payload.get("track_id"):
                    object_track_id = await self._upsert_object_track(db, payload)

                db_event = Event(
                    camera_id=event["camera_id"],
                    object_track_id=object_track_id,
                    event_type=event["event_type"],
                    payload_json=payload,
                    frame_path=event.get("frame_path"),
                )
                db.add(db_event)
                await db.commit()
        except Exception as exc:
            logger.error("Event persist error: %s", exc)

    async def _upsert_object_track(self, db, payload: dict) -> Optional[int]:
        try:
            from app.models.database import ObjectTrack
            from sqlalchemy import select

            source_id = payload.get("camera_id")
            track_id = payload.get("track_id")
            run_id = payload.get("processing_run_id") or "legacy"
            result = await db.execute(select(ObjectTrack).where(
                ObjectTrack.source_id == source_id,
                ObjectTrack.track_id == track_id,
                ObjectTrack.processing_run_id == run_id,
            ))
            track = result.scalar_one_or_none()
            values = {
                "object_class": payload.get("object_class") or "unknown",
                "confidence": float(payload.get("detection_confidence", 0.0)),
                "trajectory": payload.get("trajectory") or [],
                "speed_pixels_per_second": float(payload.get("speed_pixels_per_second", 0.0)),
                "direction_degrees": payload.get("direction_degrees"),
                "state": payload.get("state", "active"),
                "attributes": {
                    **(payload.get("diagnostics") or {}),
                    **(payload.get("attributes") or {}),
                    "predicted": bool(payload.get("predicted", False)),
                    "motion_vector": payload.get("motion_vector") or [],
                },
                "best_crop_path": payload.get("best_crop_path"),
                "last_bbox": payload.get("last_bbox") or [],
                "first_video_timestamp_seconds": payload.get("first_video_timestamp_seconds"),
                "last_video_timestamp_seconds": payload.get("last_video_timestamp_seconds"),
                "first_seen": self._parse_timestamp(payload.get("first_seen")) or datetime.now(timezone.utc),
                "last_seen": self._parse_timestamp(payload.get("last_seen")) or datetime.now(timezone.utc),
                "duration_seconds": float(payload.get("duration_seconds", 0.0)),
            }
            if track is None:
                track = ObjectTrack(source_id=source_id, track_id=track_id, processing_run_id=run_id, **values)
                db.add(track)
            else:
                for key, value in values.items():
                    setattr(track, key, value)
            await db.flush()
            return track.id
        except Exception as exc:
            logger.error("Upsert object track error: %s", exc)
            return None

    @staticmethod
    def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            return None
