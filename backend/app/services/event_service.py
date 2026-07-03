import asyncio
import httpx
import logging
from typing import Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Global async queue for events (non-blocking pipeline)
event_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)


class EventService:
    """
    Consumes events from queue and:
    - Writes to PostgreSQL
    - Checks watchlist
    - Dispatches alerts (webhook, Telegram, email)
    """

    def __init__(self, db_session_factory, redis_client=None):
        self.db_factory = db_session_factory
        self.redis = redis_client
        self._running = False
        self._watchlist_cache: dict = {}
        self._watchlist_updated = 0.0

    async def start(self):
        self._running = True
        asyncio.create_task(self._consumer_loop())
        asyncio.create_task(self._watchlist_sync_loop())
        logger.info("EventService started")

    async def stop(self):
        self._running = False

    async def _consumer_loop(self):
        """Drain event queue and persist to DB."""
        while self._running:
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=1.0)
                await self._process_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"EventService consumer error: {e}")

    async def _process_event(self, event: dict):
        try:
            async with self.db_factory() as db:
                from app.models.database import Event, VehicleTrack
                from sqlalchemy import select

                # Ensure VehicleTrack exists in DB
                db_track_id = event.get("vehicle_track_id")
                payload = event.get("payload_json", {})

                if db_track_id is None and payload.get("track_id"):
                    db_track_id = await self._upsert_vehicle_track(db, payload)

                db_event = Event(
                    camera_id=event["camera_id"],
                    vehicle_track_id=db_track_id,
                    event_type=event["event_type"],
                    payload_json=payload,
                    frame_path=event.get("frame_path"),
                )
                db.add(db_event)
                await db.commit()

                # Watchlist check
                plate = payload.get("plate")
                if plate and event["event_type"] in ("plate_verified", "vehicle_entered"):
                    await self._check_watchlist(plate, event["camera_id"], db_track_id, payload)

        except Exception as e:
            logger.error(f"Event persist error: {e}")

    async def _upsert_vehicle_track(self, db, payload: dict) -> Optional[int]:
        """Insert or update vehicle track in PostgreSQL."""
        try:
            from app.models.database import VehicleTrack
            from sqlalchemy import select

            camera_id = payload.get("camera_id")
            track_id = payload.get("track_id")
            processing_run_id = payload.get("processing_run_id") or "legacy"

            result = await db.execute(
                select(VehicleTrack).where(
                    VehicleTrack.camera_id == camera_id,
                    VehicleTrack.track_id == track_id,
                    VehicleTrack.processing_run_id == processing_run_id,
                )
            )
            existing = result.scalar_one_or_none()
            first_seen = self._parse_timestamp(payload.get("first_seen"))
            last_seen = self._parse_timestamp(payload.get("last_seen"))

            if existing:
                existing.final_plate = payload.get("plate") or existing.final_plate
                existing.plate_status = payload.get("plate_status", existing.plate_status)
                existing.final_plate_confidence = payload.get("plate_confidence", existing.final_plate_confidence)
                existing.color = payload.get("color", existing.color)
                existing.color_confidence = payload.get("color_confidence", existing.color_confidence)
                existing.vehicle_make = payload.get("vehicle_make", existing.vehicle_make)
                existing.make_confidence = payload.get("make_confidence", existing.make_confidence)
                existing.first_seen = first_seen or existing.first_seen
                existing.last_seen = last_seen or datetime.now(timezone.utc)
                existing.duration_seconds = float(
                    payload.get("duration_seconds", existing.duration_seconds or 0.0)
                )
                existing.best_vehicle_crop_path = (
                    payload.get("best_vehicle_crop_path") or existing.best_vehicle_crop_path
                )
                existing.best_plate_crop_path = (
                    payload.get("best_plate_crop_path") or existing.best_plate_crop_path
                )
                existing.recognition_diagnostics = payload.get(
                    "recognition_diagnostics", existing.recognition_diagnostics
                )
                db_track = existing
            else:
                db_track = VehicleTrack(
                    camera_id=camera_id,
                    track_id=track_id,
                    processing_run_id=processing_run_id,
                    vehicle_class=payload.get("vehicle_class", "car"),
                    final_plate=payload.get("plate"),
                    plate_status=payload.get("plate_status", "searching"),
                    final_plate_confidence=payload.get("plate_confidence", 0.0),
                    color=payload.get("color", "unknown"),
                    color_confidence=payload.get("color_confidence", 0.0),
                    vehicle_make=payload.get("vehicle_make", "unknown"),
                    make_confidence=payload.get("make_confidence", 0.0),
                    first_seen=first_seen or datetime.now(timezone.utc),
                    last_seen=last_seen or datetime.now(timezone.utc),
                    duration_seconds=float(payload.get("duration_seconds", 0.0)),
                    best_vehicle_crop_path=payload.get("best_vehicle_crop_path"),
                    best_plate_crop_path=payload.get("best_plate_crop_path"),
                    recognition_diagnostics=payload.get("recognition_diagnostics") or {},
                )
                db.add(db_track)

            await db.flush()
            await self._replace_plate_candidates(
                db,
                db_track.id,
                payload.get("ocr_candidates") or [],
                payload.get("best_plate_crop_path"),
            )
            await db.commit()
            return db_track.id
        except Exception as e:
            logger.error(f"Upsert vehicle track error: {e}")
            return None

    async def _replace_plate_candidates(
        self,
        db,
        vehicle_track_id: int,
        candidates: list,
        crop_path: Optional[str],
    ):
        if not candidates:
            return
        from app.models.database import PlateCandidate
        from sqlalchemy import delete

        await db.execute(
            delete(PlateCandidate).where(
                PlateCandidate.vehicle_track_id == vehicle_track_id
            )
        )
        for candidate in candidates:
            db.add(PlateCandidate(
                vehicle_track_id=vehicle_track_id,
                plate_text=candidate.get("text", ""),
                raw_ocr_text=candidate.get("raw_text"),
                confidence=float(candidate.get("confidence", 0.0)),
                regex_valid=bool(candidate.get("regex_valid", False)),
                regex_score=float(candidate.get("regex_score", 0.0)),
                image_quality_score=float(candidate.get("image_quality_score", 0.0)),
                crop_path=crop_path,
            ))

    @staticmethod
    def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            return None

    async def _check_watchlist(
        self,
        plate: str,
        camera_id: int,
        track_id: Optional[int],
        payload: dict,
    ):
        """Check if plate matches watchlist and dispatch alerts."""
        import time
        if time.time() - self._watchlist_updated > 30:
            await self._refresh_watchlist()

        entry = self._watchlist_cache.get(plate.upper())
        if not entry:
            return

        logger.warning(f"🚨 WATCHLIST MATCH: {plate} on camera {camera_id}")

        alert_data = {
            "type": "watchlist_match",
            "plate": plate,
            "camera_id": camera_id,
            "description": entry.get("description"),
            "channels": entry.get("alert_channels", ["frontend"]),
            "payload": payload,
        }

        channels = entry.get("alert_channels", ["frontend"])

        if "frontend" in channels:
            from app.services.websocket_service import websocket_manager
            await websocket_manager.broadcast_alert(alert_data)

        if "webhook" in channels:
            asyncio.create_task(self._send_webhook(alert_data))

        if "telegram" in channels:
            asyncio.create_task(self._send_telegram(alert_data))

        # Log watchlist event
        await event_queue.put({
            "camera_id": camera_id,
            "vehicle_track_id": track_id,
            "event_type": "watchlist_match",
            "payload_json": alert_data,
            "frame_path": None,
        })

    async def _refresh_watchlist(self):
        import time
        try:
            async with self.db_factory() as db:
                from app.models.database import WatchlistEntry
                from sqlalchemy import select
                result = await db.execute(
                    select(WatchlistEntry).where(WatchlistEntry.is_active == True)
                )
                entries = result.scalars().all()
                self._watchlist_cache = {
                    e.plate_number.upper(): {
                        "description": e.description,
                        "alert_channels": e.alert_channels,
                    }
                    for e in entries
                }
                self._watchlist_updated = time.time()
        except Exception as e:
            logger.error(f"Watchlist refresh error: {e}")

    async def _watchlist_sync_loop(self):
        while self._running:
            await self._refresh_watchlist()
            await asyncio.sleep(30)

    async def _send_webhook(self, data: dict):
        from app.config import settings
        if not settings.WEBHOOK_URL:
            return
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(settings.WEBHOOK_URL, json=data)
        except Exception as e:
            logger.warning(f"Webhook send failed: {e}")

    async def _send_telegram(self, data: dict):
        from app.config import settings
        if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
            return
        try:
            text = (
                f"🚨 Watchlist Match!\n"
                f"Plate: {data['plate']}\n"
                f"Camera: {data['camera_id']}\n"
                f"Description: {data.get('description', 'N/A')}"
            )
            url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(url, json={
                    "chat_id": settings.TELEGRAM_CHAT_ID,
                    "text": text,
                })
        except Exception as e:
            logger.warning(f"Telegram send failed: {e}")
