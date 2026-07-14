from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, JSON, inspect, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func
from app.config import settings


class Base(DeclarativeBase):
    pass


class Project(Base):
    __tablename__ = "cv_projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, unique=True)
    slug = Column(String(100), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    task_profile = Column(String(50), nullable=False, default="generic_objects", server_default="generic_objects")
    target_classes = Column(JSON, nullable=False, default=list)
    status = Column(String(30), nullable=False, default="active", server_default="active")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    sources = relationship("Camera", back_populates="project")
    pipelines = relationship("PipelineDefinition", back_populates="project")


class PipelineDefinition(Base):
    __tablename__ = "cv_pipelines"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("cv_projects.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    version = Column(Integer, nullable=False, default=1, server_default="1")
    task_profile = Column(String(50), nullable=False, default="generic_objects", server_default="generic_objects")
    config = Column(JSON, nullable=False, default=dict)
    is_default = Column(Boolean, nullable=False, default=False, server_default="false")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    project = relationship("Project", back_populates="pipelines")


class EvaluationRun(Base):
    __tablename__ = "cv_evaluation_runs"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("cv_projects.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    task_type = Column(String(30), nullable=False)
    model_name = Column(String(255), nullable=False)
    dataset_ref = Column(String(500), nullable=False)
    status = Column(String(30), nullable=False, default="pending", server_default="pending")
    config = Column(JSON, nullable=False, default=dict)
    metrics = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("cv_projects.id", ondelete="SET NULL"), nullable=True, index=True)
    pipeline_id = Column(Integer, ForeignKey("cv_pipelines.id", ondelete="SET NULL"), nullable=True, index=True)
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
    task_profile = Column(String(50), nullable=False, default="aerial_small_objects", server_default="aerial_small_objects")
    pipeline_mode = Column(String(20), nullable=False, default="automatic", server_default="automatic")
    pipeline_config = Column(JSON, nullable=True, default=dict)
    priority = Column(Integer, default=1)
    max_fps = Column(Integer, default=25)
    is_active = Column(Boolean, default=False)
    save_crops = Column(Boolean, default=True)
    anonymization = Column(Boolean, default=False)
    gemini_enabled = Column(Boolean, nullable=False, default=False, server_default="false")
    gemini_verify_predictions = Column(Boolean, nullable=False, default=True, server_default="true")
    gemini_collect_training = Column(Boolean, nullable=False, default=True, server_default="true")
    gemini_sample_interval_seconds = Column(Integer, nullable=False, default=30, server_default="30")
    gemini_max_candidates_per_run = Column(Integer, nullable=False, default=25, server_default="25")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    object_tracks = relationship("ObjectTrack", back_populates="source")
    events = relationship("Event", back_populates="camera")
    project = relationship("Project", back_populates="sources")
    pipeline = relationship("PipelineDefinition")


class ObjectTrack(Base):
    __tablename__ = "object_tracks"

    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(Integer, ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True)
    processing_run_id = Column(String(64), nullable=False, index=True)
    track_id = Column(Integer, nullable=False)
    object_class = Column(String(100), nullable=False, index=True)
    confidence = Column(Float, nullable=False, default=0.0)
    trajectory = Column(JSON, nullable=False, default=list)
    speed_pixels_per_second = Column(Float, nullable=False, default=0.0)
    direction_degrees = Column(Float, nullable=True)
    state = Column(String(30), nullable=False, default="active")
    attributes = Column(JSON, nullable=False, default=dict)
    best_crop_path = Column(String(500), nullable=True)
    last_bbox = Column(JSON, nullable=False, default=list)
    first_video_timestamp_seconds = Column(Float, nullable=True)
    last_video_timestamp_seconds = Column(Float, nullable=True)
    first_seen = Column(DateTime(timezone=True), server_default=func.now())
    last_seen = Column(DateTime(timezone=True), server_default=func.now())
    duration_seconds = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    source = relationship("Camera", back_populates="object_tracks")


class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    object_track_id = Column(Integer, ForeignKey("object_tracks.id"), nullable=True)
    event_type = Column(String(50), nullable=False)
    payload_json = Column(JSON, nullable=True)
    frame_path = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    camera = relationship("Camera", back_populates="events")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), default="viewer")  # admin, operator, viewer
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


class GeminiReviewCandidate(Base):
    __tablename__ = "gemini_review_candidates"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True)
    processing_run_id = Column(String(64), nullable=False, index=True)
    track_id = Column(Integer, nullable=True, index=True)
    status = Column(String(30), nullable=False, default="processing", server_default="processing", index=True)
    selection_reason = Column(String(100), nullable=False, default="representative_track")
    frame_path = Column(String(500), nullable=False)
    mask_path = Column(String(500), nullable=True)
    frame_width = Column(Integer, nullable=False)
    frame_height = Column(Integer, nullable=False)
    target_model_type = Column(String(50), nullable=False, default="object_segmenter")
    local_predictions = Column(JSON, nullable=True, default=list)
    gemini_verification = Column(JSON, nullable=True)
    proposed_annotations = Column(JSON, nullable=True, default=list)
    raw_response = Column(JSON, nullable=True)
    provider_model = Column(String(100), nullable=True)
    response_id = Column(String(255), nullable=True)
    usage_metadata = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    reviewed_by = Column(String(255), nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
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
        await conn.run_sync(_ensure_event_object_columns)
        await conn.run_sync(_ensure_object_track_columns)


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
    if "project_id" not in columns:
        sync_conn.execute(text("ALTER TABLE cameras ADD COLUMN project_id INTEGER"))
        sync_conn.execute(text("CREATE INDEX IF NOT EXISTS ix_cameras_project_id ON cameras (project_id)"))
    if "pipeline_id" not in columns:
        sync_conn.execute(text("ALTER TABLE cameras ADD COLUMN pipeline_id INTEGER"))
        sync_conn.execute(text("CREATE INDEX IF NOT EXISTS ix_cameras_pipeline_id ON cameras (pipeline_id)"))
    if "task_profile" not in columns:
        sync_conn.execute(text(
            "ALTER TABLE cameras ADD COLUMN task_profile VARCHAR(50) NOT NULL DEFAULT 'aerial_small_objects'"
        ))
    gemini_columns = {
        "gemini_enabled": "BOOLEAN NOT NULL DEFAULT false",
        "gemini_verify_predictions": "BOOLEAN NOT NULL DEFAULT true",
        "gemini_collect_training": "BOOLEAN NOT NULL DEFAULT true",
        "gemini_sample_interval_seconds": "INTEGER NOT NULL DEFAULT 30",
        "gemini_max_candidates_per_run": "INTEGER NOT NULL DEFAULT 25",
    }
    for column_name, definition in gemini_columns.items():
        if column_name not in columns:
            sync_conn.execute(text(
                f"ALTER TABLE cameras ADD COLUMN {column_name} {definition}"
            ))

def _ensure_event_object_columns(sync_conn):
    inspector = inspect(sync_conn)
    if not inspector.has_table("events"):
        return
    columns = {column["name"] for column in inspector.get_columns("events")}
    if "object_track_id" not in columns:
        sync_conn.execute(text("ALTER TABLE events ADD COLUMN object_track_id INTEGER"))
        sync_conn.execute(text("CREATE INDEX IF NOT EXISTS ix_events_object_track_id ON events (object_track_id)"))


def _ensure_object_track_columns(sync_conn):
    """Keep databases created before the object registry view compatible."""
    inspector = inspect(sync_conn)
    if not inspector.has_table("object_tracks"):
        return
    columns = {column["name"] for column in inspector.get_columns("object_tracks")}
    additions = {
        "last_bbox": "JSON NOT NULL DEFAULT '[]'",
        "first_video_timestamp_seconds": "FLOAT",
        "last_video_timestamp_seconds": "FLOAT",
    }
    for column_name, definition in additions.items():
        if column_name not in columns:
            sync_conn.execute(text(
                f"ALTER TABLE object_tracks ADD COLUMN {column_name} {definition}"
            ))
