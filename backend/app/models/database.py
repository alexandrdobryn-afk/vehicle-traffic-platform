from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, JSON, inspect, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func
from app.config import settings


class Base(DeclarativeBase):
    pass


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    rtsp_url_encrypted = Column(Text, nullable=False)
    source_type = Column(String(20), nullable=False, default="rtsp", server_default="rtsp")
    source_file_path = Column(Text, nullable=True)
    source_file_name = Column(String(255), nullable=True)
    source_duration_seconds = Column(Float, nullable=True)
    source_fps = Column(Float, nullable=True)
    snapshot_interval_seconds = Column(Float, nullable=False, default=1.0, server_default="1.0")
    location = Column(String(500), nullable=True)
    status = Column(String(50), default="offline")  # online, offline, error
    ai_mode = Column(String(50), default="balanced")
    pipeline_mode = Column(String(20), nullable=False, default="automatic", server_default="automatic")
    pipeline_config = Column(JSON, nullable=True, default=dict)
    priority = Column(Integer, default=1)
    max_fps = Column(Integer, default=25)
    is_active = Column(Boolean, default=False)
    save_crops = Column(Boolean, default=True)
    anonymization = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    tracks = relationship("VehicleTrack", back_populates="camera")
    events = relationship("Event", back_populates="camera")


class VehicleTrack(Base):
    __tablename__ = "vehicle_tracks"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    track_id = Column(Integer, nullable=False)
    vehicle_class = Column(String(50), nullable=False)
    final_plate = Column(String(20), nullable=True)
    plate_status = Column(String(30), default="searching")  # searching, candidate, verified, low_confidence, invalid
    final_plate_confidence = Column(Float, default=0.0)
    color = Column(String(30), default="unknown")
    color_confidence = Column(Float, default=0.0)
    vehicle_make = Column(String(80), nullable=False, default="unknown", server_default="unknown")
    make_confidence = Column(Float, nullable=False, default=0.0, server_default="0")
    first_seen = Column(DateTime(timezone=True), server_default=func.now())
    last_seen = Column(DateTime(timezone=True), server_default=func.now())
    duration_seconds = Column(Float, default=0.0)
    best_vehicle_crop_path = Column(String(500), nullable=True)
    best_plate_crop_path = Column(String(500), nullable=True)
    processing_run_id = Column(String(64), nullable=False, default="legacy", server_default="legacy")
    recognition_diagnostics = Column(JSON, nullable=True, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    camera = relationship("Camera", back_populates="tracks")
    plate_candidates = relationship("PlateCandidate", back_populates="vehicle_track")
    events = relationship("Event", back_populates="vehicle_track")


class PlateCandidate(Base):
    __tablename__ = "plate_candidates"

    id = Column(Integer, primary_key=True, index=True)
    vehicle_track_id = Column(Integer, ForeignKey("vehicle_tracks.id"), nullable=False)
    plate_text = Column(String(20), nullable=False)
    raw_ocr_text = Column(String(100), nullable=True)
    confidence = Column(Float, default=0.0)
    regex_valid = Column(Boolean, default=False)
    regex_score = Column(Float, default=0.0)
    image_quality_score = Column(Float, default=0.0)
    frame_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    crop_path = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    vehicle_track = relationship("VehicleTrack", back_populates="plate_candidates")


class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    vehicle_track_id = Column(Integer, ForeignKey("vehicle_tracks.id"), nullable=True)
    event_type = Column(String(50), nullable=False)
    payload_json = Column(JSON, nullable=True)
    frame_path = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    camera = relationship("Camera", back_populates="events")
    vehicle_track = relationship("VehicleTrack", back_populates="events")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), default="viewer")  # admin, operator, viewer
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())


class WatchlistEntry(Base):
    __tablename__ = "watchlist"

    id = Column(Integer, primary_key=True, index=True)
    plate_number = Column(String(20), nullable=False, index=True)
    description = Column(Text, nullable=True)
    alert_channels = Column(JSON, default=["frontend"])  # frontend, webhook, email, telegram
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())


class SystemLog(Base):
    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True, index=True)
    level = Column(String(20), nullable=False)  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    source = Column(String(100), nullable=False)
    message = Column(Text, nullable=False)
    payload_json = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AppSettings(Base):
    __tablename__ = "app_settings"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(100), unique=True, nullable=False)
    value = Column(JSON, nullable=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())


# Database engine setup
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_size=10,
    max_overflow=20
)


async def get_db():
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_ensure_camera_source_columns)
        await conn.run_sync(_ensure_vehicle_track_columns)


def _ensure_camera_source_columns(sync_conn):
    """Upgrade databases created by the legacy create_all startup path."""
    inspector = inspect(sync_conn)
    if not inspector.has_table("cameras"):
        return

    columns = {column["name"] for column in inspector.get_columns("cameras")}
    if "source_type" not in columns:
        sync_conn.execute(text(
            "ALTER TABLE cameras ADD COLUMN source_type VARCHAR(20) NOT NULL DEFAULT 'rtsp'"
        ))
    if "snapshot_interval_seconds" not in columns:
        sync_conn.execute(text(
            "ALTER TABLE cameras ADD COLUMN snapshot_interval_seconds FLOAT NOT NULL DEFAULT 1.0"
        ))
    if "source_file_path" not in columns:
        sync_conn.execute(text("ALTER TABLE cameras ADD COLUMN source_file_path TEXT"))
    if "source_file_name" not in columns:
        sync_conn.execute(text("ALTER TABLE cameras ADD COLUMN source_file_name VARCHAR(255)"))
    if "source_duration_seconds" not in columns:
        sync_conn.execute(text("ALTER TABLE cameras ADD COLUMN source_duration_seconds FLOAT"))
    if "source_fps" not in columns:
        sync_conn.execute(text("ALTER TABLE cameras ADD COLUMN source_fps FLOAT"))
    if "pipeline_mode" not in columns:
        sync_conn.execute(text(
            "ALTER TABLE cameras ADD COLUMN pipeline_mode VARCHAR(20) NOT NULL DEFAULT 'automatic'"
        ))
    if "pipeline_config" not in columns:
        sync_conn.execute(text("ALTER TABLE cameras ADD COLUMN pipeline_config JSON"))


def _ensure_vehicle_track_columns(sync_conn):
    """Upgrade legacy create_all databases with recognition persistence fields."""
    inspector = inspect(sync_conn)
    if not inspector.has_table("vehicle_tracks"):
        return
    columns = {column["name"] for column in inspector.get_columns("vehicle_tracks")}
    if "processing_run_id" not in columns:
        sync_conn.execute(text(
            "ALTER TABLE vehicle_tracks ADD COLUMN processing_run_id VARCHAR(64) NOT NULL DEFAULT 'legacy'"
        ))
    if "recognition_diagnostics" not in columns:
        sync_conn.execute(text(
            "ALTER TABLE vehicle_tracks ADD COLUMN recognition_diagnostics JSON"
        ))
    if "vehicle_make" not in columns:
        sync_conn.execute(text(
            "ALTER TABLE vehicle_tracks ADD COLUMN vehicle_make VARCHAR(80) NOT NULL DEFAULT 'unknown'"
        ))
    if "make_confidence" not in columns:
        sync_conn.execute(text(
            "ALTER TABLE vehicle_tracks ADD COLUMN make_confidence FLOAT NOT NULL DEFAULT 0"
        ))
