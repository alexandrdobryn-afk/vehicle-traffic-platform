import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.models.database import init_db, get_db
from app.api.routers import (
    datasets_router, annotations_router, jobs_router,
    registry_router, gpu_router, arch_router,
)
from app.utils.auth import decode_token

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

redis_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client

    if settings.APP_ENV.lower() == "production" and (
        settings.SECRET_KEY.startswith("change-me-") or len(settings.SECRET_KEY) < 32
    ):
        raise RuntimeError("Production requires a unique SECRET_KEY of at least 32 characters")
    if settings.APP_ENV.lower() == "production" and "vtp_pass" in settings.DATABASE_URL:
        raise RuntimeError("Production requires a unique PostgreSQL password")

    logger.info("🚀 Starting VTP Training Platform...")

    # Create data directories
    for path in [
        settings.DATASETS_PATH, settings.UPLOADS_PATH,
        settings.MODELS_PATH, settings.EXPORTS_PATH, settings.LOGS_PATH,
    ]:
        os.makedirs(path, exist_ok=True)

    # Init DB tables
    await init_db()
    logger.info("✅ Training DB initialized")

    # Redis
    try:
        redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        await redis_client.ping()
        logger.info("✅ Redis connected")
    except Exception as e:
        logger.warning(f"⚠️  Redis unavailable: {e}")
        redis_client = None

    logger.info(f"✅ Training API ready on port {settings.PORT}")
    yield

    if redis_client:
        await redis_client.close()


app = FastAPI(
    title="VTP Training Platform",
    version="1.0.0",
    description="AI model training and lifecycle management",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(datasets_router)
app.include_router(annotations_router)
app.include_router(jobs_router)
app.include_router(registry_router)
app.include_router(gpu_router)
app.include_router(arch_router)

# Static files for dataset images
app.mount("/data", StaticFiles(directory=settings.DATA_ROOT), name="data")


# ═══ WEBSOCKET — Live Training Progress ══════════════════════════
@app.websocket("/ws/training/{job_id}")
async def training_ws(
    websocket: WebSocket,
    job_id: int,
    token: str = Query(...),
):
    """
    WebSocket for realtime training progress.
    Subscribes to Redis pub/sub channel for the job.
    """
    try:
        decode_token(token)
    except Exception:
        await websocket.close(code=4001)
        return

    await websocket.accept()
    logger.info(f"WS client connected to job {job_id}")

    # Subscribe to Redis channel
    if redis_client:
        pubsub = redis_client.pubsub()
        channel = f"training:job:{job_id}"
        await pubsub.subscribe(channel)

        try:
            # Send current progress immediately
            key = f"training:progress:{job_id}"
            current = await redis_client.get(key)
            if current:
                await websocket.send_text(current)

            # Stream updates
            async def listen():
                async for message in pubsub.listen():
                    if message["type"] == "message":
                        try:
                            await websocket.send_text(message["data"])
                        except Exception:
                            break

            listen_task = asyncio.create_task(listen())

            # Keep alive
            while True:
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                    if data == "ping":
                        await websocket.send_text("pong")
                except asyncio.TimeoutError:
                    await websocket.send_text(json.dumps({"type": "heartbeat"}))
                except WebSocketDisconnect:
                    break

            listen_task.cancel()
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.close()
    else:
        # Fallback: poll DB directly
        try:
            while True:
                await asyncio.sleep(2)
                async for db in get_db():
                    from sqlalchemy import select
                    from app.models.database import TrainingJob, TrainingMetrics
                    result = await db.execute(
                        select(TrainingJob).where(TrainingJob.id == job_id)
                    )
                    job = result.scalar_one_or_none()
                    if job:
                        m_result = await db.execute(
                            select(TrainingMetrics)
                            .where(TrainingMetrics.job_id == job_id)
                            .order_by(TrainingMetrics.epoch.desc())
                            .limit(1)
                        )
                        latest = m_result.scalar_one_or_none()
                        payload = {
                            "job_id": job_id,
                            "status": job.status,
                            "current_epoch": job.current_epoch,
                            "total_epochs": job.total_epochs,
                            "progress_pct": job.progress_pct,
                            "latest_metrics": {
                                "map50": latest.map50 if latest else None,
                                "train_loss": latest.train_loss if latest else None,
                            } if latest else None,
                        }
                        await websocket.send_text(json.dumps(payload, default=str))

                        if job.status in ("completed", "failed", "cancelled"):
                            break
        except WebSocketDisconnect:
            pass

    logger.info(f"WS client disconnected from job {job_id}")


# ═══ HEALTH ══════════════════════════════════════════════════════
@app.get("/api/v1/training/health")
async def health():
    redis_ok = False
    if redis_client:
        try:
            await redis_client.ping()
            redis_ok = True
        except Exception:
            pass

    # Check Celery workers
    workers_ok = False
    try:
        from app.celery_app import celery_app
        inspect = celery_app.control.inspect(timeout=2)
        active = inspect.active()
        workers_ok = active is not None
    except Exception:
        pass

    return {
        "status": "ok",
        "redis": "ok" if redis_ok else "unavailable",
        "celery_workers": "ok" if workers_ok else "no workers",
        "version": "1.0.0",
    }
