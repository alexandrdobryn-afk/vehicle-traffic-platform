"""Video Worker — frame extraction runs on CPU queue."""
import os
import logging
from app.celery_app import celery_app
from app.config import settings

logger = logging.getLogger(__name__)


@celery_app.task(name="app.workers.video_worker.extract_frames_task", bind=True)
def extract_frames_task(self, dataset_id: int, video_id: int, config: dict):
    """Extract frames from uploaded video — runs in CPU worker."""
    try:
        from app.schemas.schemas import VideoExtractConfig
        from app.services.dataset_service import dataset_service
        import asyncio

        cfg = VideoExtractConfig(**config)
        async def run():
            from sqlalchemy.ext.asyncio import AsyncSession
            from sqlalchemy.orm import sessionmaker
            from app.models.database import DatasetVideo, engine
            session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
            async with session_factory() as db:
                try:
                    return await dataset_service.extract_frames(db, dataset_id, video_id, cfg)
                except Exception:
                    video = await db.get(DatasetVideo, video_id)
                    if video:
                        video.status = "error"
                        await db.commit()
                    raise

        count = asyncio.run(run())
        return {"extracted": count, "video_id": video_id}
    except Exception as e:
        logger.error(f"Frame extraction failed: {e}", exc_info=True)
        return {"error": str(e)}
