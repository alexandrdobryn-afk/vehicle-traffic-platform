from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, List


class TrainingSettings(BaseSettings):
    # App
    APP_NAME: str = "BEVP Training Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    APP_ENV: str = "development"
    PORT: int = 8001

    # Auth — shared with main backend
    SECRET_KEY: str = Field(default="change-me-in-production-super-secret-key-32chars")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24

    # Database — same PostgreSQL, separate schema
    DATABASE_URL: str = "postgresql+asyncpg://bevp_user:bevp_pass@postgres:5432/bevp_db"
    DATABASE_SYNC_URL: str = "postgresql://bevp_user:bevp_pass@postgres:5432/bevp_db"

    # Redis — same Redis instance, different DB index
    REDIS_URL: str = "redis://redis:6379/1"
    CELERY_BROKER_URL: str = "redis://redis:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/1"

    # Storage
    DATA_ROOT: str = "/app/data"
    DATASETS_PATH: str = "/app/data/datasets"
    UPLOADS_PATH: str = "/app/data/uploads"
    MODELS_PATH: str = "/app/data/models"
    TRAINED_MODELS_PATH: str = "/app/data/models/new"
    EXPORTS_PATH: str = "/app/data/exports"
    LOGS_PATH: str = "/app/data/logs"

    # Inference backend models dir (for deploy)
    INFERENCE_MODELS_PATH: str = "/app/inference_models"
    SOURCE_STORAGE_PATH: str = "/app/source_storage"

    # Training defaults
    DEFAULT_EPOCHS: int = 100
    DEFAULT_BATCH_SIZE: int = 16
    DEFAULT_IMG_SIZE: int = 640
    DEFAULT_LEARNING_RATE: float = 0.01
    MAX_CONCURRENT_JOBS: int = 1

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3100",
        "http://frontend:3000",
    ]

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = TrainingSettings()
