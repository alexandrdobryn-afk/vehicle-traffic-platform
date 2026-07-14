import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.orm import configure_mappers

from app.config import settings
from app.models.database import engine, init_db, get_db
from app.api.routers import (
    datasets_router, annotations_router, jobs_router,
    registry_router, gpu_router, arch_router,
    active_learning_router, evaluation_router,
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
    if settings.APP_ENV.lower() == "production" and "bevp_pass" in settings.DATABASE_URL:
        raise RuntimeError("Production requires a unique PostgreSQL password")

    logger.info("рџљЂ Starting BEVP Training Platform...")

    # Configure every ORM relationship before advertising readiness. Metadata
    # creation alone does not catch ambiguous relationships.
    configure_mappers()

    # Create data directories
    for path in [
        settings.DATASETS_PATH, settings.UPLOADS_PATH,
        settings.MODELS_PATH, settings.EXPORTS_PATH, settings.LOGS_PATH,
    ]:
        os.makedirs(path, exist_ok=True)

    # Init DB tables
    await init_db()
    logger.info("вњ… Training DB initialized")

    # Redis
    try:
        redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        await redis_client.ping()
        logger.info("вњ… Redis connected")
    except Exception as e:
        logger.warning(f"вљ пёЏ  Redis unavailable: {e}")
        redis_client = None

    logger.info(f"вњ… Training API ready on port {settings.PORT}")
    yield

    if redis_client:
        await redis_client.aclose()


app = FastAPI(
    title="BEVP Training Platform",
    version="1.0.0",
    description="AI model training and lifecycle management",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_origin_regex=r"http://172\.\d+\.\d+\.\d+:3100" if settings.APP_ENV.lower() == "development" else None,
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
app.include_router(active_learning_router)
app.include_router(evaluation_router)

# Static files for dataset images
app.mount("/data", StaticFiles(directory=settings.DATA_ROOT), name="data")


def _celery_workers_ready() -> bool:
    try:
        from app.celery_app import celery_app

        inspect = celery_app.control.inspect(timeout=2)
        active = inspect.active()
        return active is not None
    except Exception:
        return False


async def _ensure_redis_client():
    global redis_client

    if redis_client is None:
        redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    try:
        await asyncio.wait_for(redis_client.ping(), timeout=2)
        return redis_client
    except Exception:
        try:
            await redis_client.close()
        except Exception:
            pass
        redis_client = None
        raise


# === WEBSOCKET — Live Training Progress ==========================
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


# === HEALTH ======================================================
@app.get("/api/v1/training/health/live")
async def health_live():
    return {"status": "ok", "service": "training-api"}


async def _training_ready_state() -> dict:
    orm_ok = True
    try:
        configure_mappers()
    except Exception:
        logger.exception("Training ORM mapper readiness check failed")
        orm_ok = False

    database_ok = False
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        database_ok = True
    except Exception:
        logger.exception("Training database readiness check failed")

    redis_ok = False
    try:
        await _ensure_redis_client()
        redis_ok = True
    except Exception:
        logger.exception("Training Redis readiness check failed")

    startup_ready = orm_ok and database_ok and redis_ok
    return {
        "status": "ok" if startup_ready else "degraded",
        "orm": "ok" if orm_ok else "error",
        "database": "ok" if database_ok else "error",
        "redis": "ok" if redis_ok else "unavailable",
        "version": "1.0.0",
        "ready": startup_ready,
    }


@app.get("/api/v1/training/health/ready")
async def health_ready(response: Response):
    state = await _training_ready_state()
    if not state["ready"]:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {key: value for key, value in state.items() if key != "ready"}


@app.get("/api/v1/training/health/workers")
async def health_workers():
    workers_ok = await asyncio.to_thread(_celery_workers_ready)
    return {
        "status": "ok" if workers_ok else "degraded",
        "celery_workers": "ok" if workers_ok else "no workers",
        "note": "Worker status is diagnostic only and does not control Compose readiness.",
    }


@app.get("/api/v1/training/health")
async def health(response: Response):
    state = await _training_ready_state()

    # Worker status is intentionally diagnostic only. Compose uses /health/ready.
    workers_ok = await asyncio.to_thread(_celery_workers_ready)
    if not state["ready"]:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ok" if state["ready"] and workers_ok else "degraded",
        "orm": state["orm"],
        "database": state["database"],
        "redis": state["redis"],
        "celery_workers": "ok" if workers_ok else "no workers",
        "version": state["version"],
    }
