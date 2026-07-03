"""Video Worker — frame extraction runs on CPU queue."""
import os
import logging
from app.celery_app import celery_app
from app.config import settings

logger = logging.getLogger(__name__)


@celery_app.task(name="app.workers.video_worker.extract_frames_task", bind=True)
def extract_frames_task(self, dataset_id: int, video_id: int, config: dict):
    """Extract frames from uploaded video — runs in CPU worker."""
    db = _get_db()
    try:
        from app.schemas.schemas import VideoExtractConfig
        from app.services.dataset_service import dataset_service
        import asyncio

        cfg = VideoExtractConfig(**config)
        # Run async service in sync context
        loop = asyncio.new_event_loop()
        count = loop.run_until_complete(
            dataset_service.extract_frames(db, dataset_id, video_id, cfg)
        )
        loop.close()
        return {"extracted": count, "video_id": video_id}
    except Exception as e:
        logger.error(f"Frame extraction failed: {e}", exc_info=True)
        from app.models.database import DatasetVideo
        v = db.query(DatasetVideo).filter(DatasetVideo.id == video_id).first()
        if v:
            v.status = "error"
            db.commit()
        return {"error": str(e)}
    finally:
        db.close()


def _get_db():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    engine = create_engine(settings.DATABASE_SYNC_URL)
    Session = sessionmaker(bind=engine)
    return Session()
