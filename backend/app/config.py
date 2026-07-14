from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from enum import Enum


class AIMode(str, Enum):
    SPEED = "speed"
    BALANCED = "balanced"
    QUALITY = "quality"
    HYBRID = "hybrid"
    PRACTICAL = "practical"
    MAX_ACCURACY = "max_accuracy"
    EDGE_ONNX = "edge_onnx"


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Bird's-Eye Vision Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    APP_ENV: str = "development"
    SECRET_KEY: str = Field(default="change-me-in-production-super-secret-key-32chars")
    INITIAL_ADMIN_EMAIL: str = "admin@bevp.local"
    INITIAL_ADMIN_PASSWORD: str = "admin123"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24h

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://bevp_user:bevp_pass@postgres:5432/bevp_db"
    DATABASE_SYNC_URL: str = "postgresql://bevp_user:bevp_pass@postgres:5432/bevp_db"

    # Redis
    REDIS_URL: str = "redis://redis:6379/0"
    REDIS_ACTIVE_TRACK_TTL: int = 120
    REDIS_CAMERA_STATUS_TTL: int = 10

    # Storage
    STORAGE_PATH: str = "/app/storage"
    CROPS_PATH: str = "/app/storage/crops"
    FRAMES_PATH: str = "/app/storage/frames"
    VIDEOS_PATH: str = "/app/storage/videos"
    MAX_VIDEO_UPLOAD_BYTES: int = 2 * 1024 * 1024 * 1024

    # Encryption
    FERNET_KEY: str = Field(default="")  # Generated on startup if empty

    # AI
    DEFAULT_AI_MODE: AIMode = AIMode.BALANCED
    OBJECT_CONFIDENCE_THRESHOLD: float = 0.35
    FRAME_SKIP_SPEED: int = 3
    FRAME_SKIP_BALANCED: int = 2
    FRAME_SKIP_QUALITY: int = 1

    # Video
    MJPEG_QUALITY: int = 75
    MAX_STREAM_FPS: int = 25
    STREAM_BUFFER_SIZE: int = 2

    # Alerts
    WEBHOOK_URL: Optional[str] = None
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None

    # CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:3100",
        "http://127.0.0.1:3000",
        "http://frontend:3000",
    ]

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
