import asyncio
import logging
import os
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.models.database import init_db, get_db, User, AppSettings, Camera, engine
from app.api.routers import (
    auth_router, cameras_router, videos_router, streams_router, tracks_router,
    events_router, watchlist_router, analytics_router,
    settings_router, health_router,
    gemini_router,
)
from app.services.camera_manager import camera_manager
from app.services.websocket_service import websocket_manager
from app.services.event_service import EventService
from app.services.gemini_training_service import gemini_training_service
from app.utils.auth import hash_password, decode_token
from app.schemas.schemas import AppSettingsSchema
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Global services
redis_client = None
event_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    global redis_client, event_service

    if settings.APP_ENV.lower() == "production":
        if settings.SECRET_KEY.startswith("change-me-") or len(settings.SECRET_KEY) < 32:
            raise RuntimeError("Production requires a unique SECRET_KEY of at least 32 characters")
        if settings.INITIAL_ADMIN_PASSWORD == "admin123":
            raise RuntimeError("Production requires a unique INITIAL_ADMIN_PASSWORD")
        if "vtp_pass" in settings.DATABASE_URL:
            raise RuntimeError("Production requires a unique PostgreSQL password")

    logger.info("🚀 Starting Vehicle Traffic Platform...")

    # Create storage directories
    for path in [settings.STORAGE_PATH, settings.CROPS_PATH, settings.FRAMES_PATH, settings.VIDEOS_PATH]:
        os.makedirs(path, exist_ok=True)

    # Init database
    await init_db()
    await _migrate_global_recognition_settings()
    camera_manager.set_source_finished_callback(_mark_recorded_source_finished)
    logger.info("✅ Database initialized")

    # Seed default admin user if not exists
    await _seed_admin()

    # Init Redis
    try:
        redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        await redis_client.ping()
        camera_manager.set_redis(redis_client)
        logger.info("✅ Redis connected")
    except Exception as e:
        logger.warning(f"⚠️  Redis not available: {e}. Continuing without Redis.")
        redis_client = None

    # Start event service
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker

    engine_for_events = create_async_engine(settings.DATABASE_URL, echo=False)
    AsyncSessionForEvents = sessionmaker(
        engine_for_events, class_=AsyncSession, expire_on_commit=False
    )
    gemini_training_service.set_session_factory(AsyncSessionForEvents)
    camera_manager.set_frame_analysis_callback(gemini_training_service.consider_frame)

    event_service = EventService(
        db_session_factory=AsyncSessionForEvents,
        redis_client=redis_client,
    )
    await event_service.start()
    logger.info("✅ Event service started")

    # Auto-start cameras that were previously active
    await _restore_cameras()

    logger.info("✅ Platform ready")

    yield

    # Shutdown
    logger.info("🛑 Shutting down...")
    if event_service:
        await event_service.stop()
    await gemini_training_service.shutdown()
    for stats in camera_manager.get_all_stats():
        await camera_manager.stop_camera(stats["camera_id"])
    if redis_client:
        await redis_client.close()


app = FastAPI(
    title="Vehicle Traffic Platform",
    version="1.0.0",
    description="Professional vehicle traffic analysis platform",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(auth_router)
app.include_router(cameras_router)
app.include_router(videos_router)
app.include_router(streams_router)
app.include_router(tracks_router)
app.include_router(events_router)
app.include_router(watchlist_router)
app.include_router(analytics_router)
app.include_router(settings_router)
app.include_router(gemini_router)
app.include_router(health_router)

# Static files for crops/frames
app.mount("/storage", StaticFiles(directory=settings.STORAGE_PATH), name="storage")


# ═══ WEBSOCKET ════════════════════════════════════════════════════
@app.websocket("/ws/live/{camera_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    camera_id: int,
    token: str = Query(...),
):
    # Authenticate WebSocket
    try:
        decode_token(token)
    except Exception:
        await websocket.close(code=4001)
        return

    await websocket_manager.connect(camera_id, websocket)
    try:
        while True:
            # Keep alive — client can send ping
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        pass
    finally:
        await websocket_manager.disconnect(camera_id, websocket)


# ═══ STARTUP HELPERS ══════════════════════════════════════════════
async def _seed_admin():
    """Create default admin if no users exist."""
    from sqlalchemy import select
    from app.models.database import engine

    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with AsyncSessionLocal() as db:
        from sqlalchemy import func
        count = await db.scalar(select(func.count(User.id)))
        if count == 0:
            admin = User(
                email=settings.INITIAL_ADMIN_EMAIL,
                password_hash=hash_password(settings.INITIAL_ADMIN_PASSWORD),
                role="admin",
            )
            db.add(admin)
            await db.commit()
            logger.info("Initial admin created: %s", settings.INITIAL_ADMIN_EMAIL)


async def _mark_recorded_source_finished(camera_id: int, status: str):
    """Persist finite-source completion without coupling CameraManager to SQLAlchemy."""
    from sqlalchemy import select
    from app.models.database import engine, Camera

    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Camera).where(Camera.id == camera_id))
        camera = result.scalar_one_or_none()
        if camera and camera.source_type == "file":
            camera.is_active = False
            camera.status = status
            await db.commit()


async def _migrate_global_recognition_settings():
    """Materialize recognition defaults per source and copy legacy values once."""
    from sqlalchemy import select

    source_keys = {
        "vehicle_confidence_threshold", "plate_confidence_threshold",
        "ocr_threshold", "plate_regex_profile", "input_resolution",
        "ocr_voting_window", "track_missing_grace_frames",
        "minimum_plate_width", "minimum_plate_height",
    }
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(AppSettings).where(AppSettings.key == "global"))
        settings_row = result.scalar_one_or_none()
        result = await db.execute(select(Camera))
        changed = 0
        defaults = AppSettingsSchema().model_dump()
        legacy = settings_row.value if settings_row and settings_row.value else {}
        for camera in result.scalars().all():
            config = dict(camera.pipeline_config or {})
            before = dict(config)
            for key in source_keys:
                if key not in config:
                    config[key] = legacy.get(key, defaults[key])
            config.setdefault("frame_skip", 0)
            if config != before:
                camera.pipeline_config = config
                changed += 1
        if changed:
            await db.commit()
            logger.info("Migrated legacy global recognition settings into %s sources", changed)


async def _restore_cameras():
    """Re-start cameras that had is_active=True."""
    from sqlalchemy import select
    from app.models.database import engine, Camera
    from app.utils.auth import decrypt_url

    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with AsyncSessionLocal() as db:
        settings_result = await db.execute(
            select(AppSettings).where(AppSettings.key == "global")
        )
        settings_row = settings_result.scalar_one_or_none()
        global_settings = {
            **AppSettingsSchema().model_dump(),
            **((settings_row.value or {}) if settings_row else {}),
        }
        result = await db.execute(select(Camera).where(Camera.is_active == True))
        cameras = result.scalars().all()
        for cam in cameras:
            try:
                url = cam.source_file_path if cam.source_type == "file" else decrypt_url(cam.rtsp_url_encrypted)
                defaults = AppSettingsSchema().model_dump()
                source_keys = {
                    "vehicle_confidence_threshold", "plate_confidence_threshold",
                    "ocr_threshold", "plate_regex_profile", "input_resolution",
                    "ocr_voting_window", "track_missing_grace_frames",
                    "minimum_plate_width", "minimum_plate_height",
                }
                config = cam.pipeline_config or {}
                runtime_settings = {key: config.get(key, defaults[key]) for key in source_keys}
                frame_skip = config.get("frame_skip")
                if isinstance(frame_skip, int) and 1 <= frame_skip <= 10:
                    runtime_settings["frame_skip"] = frame_skip
                runtime_settings["pipeline_mode"] = cam.pipeline_mode or "automatic"
                runtime_settings["pipeline_config"] = config
                runtime_settings["save_crops"] = bool(cam.save_crops)
                runtime_settings["anonymization_mode"] = bool(cam.anonymization)
                for key in (
                    "execution_provider", "runtime_fallback",
                    "gpu_device_index", "inference_precision",
                ):
                    runtime_settings[key] = global_settings[key]
                await camera_manager.start_camera(
                    camera_id=cam.id,
                    rtsp_url=url,
                    ai_mode=cam.ai_mode,
                    max_fps=cam.max_fps,
                    runtime_settings=runtime_settings,
                    source_type=cam.source_type,
                    snapshot_interval_seconds=cam.snapshot_interval_seconds,
                )
                logger.info(f"✅ Camera {cam.id} ({cam.name}) auto-started")
            except Exception as e:
                logger.warning(f"⚠️  Failed to auto-start camera {cam.id}: {e}")
