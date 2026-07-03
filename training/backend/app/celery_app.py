from celery import Celery
from app.config import settings

celery_app = Celery(
    "vtp_training",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "app.workers.training_worker",
        "app.workers.export_worker",
        "app.workers.video_worker",
        "app.workers.validation_worker",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,       # one task at a time per worker
    task_soft_time_limit=3600 * 8,      # 8 hours soft limit
    task_time_limit=3600 * 10,          # 10 hours hard limit
    result_expires=3600 * 24 * 7,       # keep results 7 days
    task_routes={
        "app.workers.training_worker.*": {"queue": "gpu"},
        "app.workers.export_worker.*": {"queue": "gpu"},
        "app.workers.validation_worker.*": {"queue": "gpu"},
        "app.workers.video_worker.*": {"queue": "cpu"},
    },
)
