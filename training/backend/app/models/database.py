from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime,
    Text, ForeignKey, JSON, BigInteger, Enum as SAEnum
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func
from app.config import settings
import enum


class Base(DeclarativeBase):
    pass


# ─── Enums ────────────────────────────────────────────────────────

class DatasetStatus(str, enum.Enum):
    CREATING = "creating"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"


class AnnotationType(str, enum.Enum):
    BBOX = "bbox"
    CLASSIFICATION = "classification"
    OCR = "ocr"


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    PREPARING = "preparing"
    TRAINING = "training"
    VALIDATION = "validation"
    EXPORT = "export"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelType(str, enum.Enum):
    VEHICLE_DETECTOR = "vehicle_detector"
    PLATE_DETECTOR = "plate_detector"
    OCR = "ocr"
    COLOR_CLASSIFIER = "color_classifier"


class ModelFormat(str, enum.Enum):
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"


class DeployStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DEPLOYED = "deployed"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


# ─── Dataset ──────────────────────────────────────────────────────

class Dataset(Base):
    __tablename__ = "tr_datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    model_type = Column(String(50), nullable=False)   # vehicle_detector, plate_detector, etc.
    annotation_type = Column(String(50), nullable=False)  # bbox, classification, ocr
    classes = Column(JSON, default=[])                # ["car", "truck", ...]
    status = Column(String(30), default=DatasetStatus.CREATING)
    version = Column(String(20), default="1.0")
    image_count = Column(Integer, default=0)
    video_count = Column(Integer, default=0)
    annotation_count = Column(Integer, default=0)
    storage_path = Column(String(500), nullable=True)
    author_id = Column(Integer, nullable=True)
    author_email = Column(String(255), nullable=True)
    tags = Column(JSON, default=[])
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    images = relationship("DatasetImage", back_populates="dataset", cascade="all, delete-orphan")
    splits = relationship("DatasetSplit", back_populates="dataset", cascade="all, delete-orphan")


class DatasetImage(Base):
    __tablename__ = "tr_dataset_images"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("tr_datasets.id", ondelete="CASCADE"), nullable=False)
    filename = Column(String(500), nullable=False)
    original_filename = Column(String(500), nullable=True)
    file_path = Column(String(1000), nullable=False)
    file_size = Column(BigInteger, default=0)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    source = Column(String(50), default="upload")   # upload, video_frame, augmented
    source_video_id = Column(Integer, ForeignKey("tr_dataset_videos.id"), nullable=True)
    source_frame_idx = Column(Integer, nullable=True)
    split = Column(String(10), nullable=True)        # train, val, test
    is_annotated = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    dataset = relationship("Dataset", back_populates="images")
    annotations = relationship("Annotation", back_populates="image", cascade="all, delete-orphan")


class DatasetVideo(Base):
    __tablename__ = "tr_dataset_videos"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("tr_datasets.id", ondelete="CASCADE"), nullable=False)
    filename = Column(String(500), nullable=False)
    file_path = Column(String(1000), nullable=False)
    file_size = Column(BigInteger, default=0)
    duration_seconds = Column(Float, nullable=True)
    fps = Column(Float, nullable=True)
    total_frames = Column(Integer, nullable=True)
    extracted_frames = Column(Integer, default=0)
    status = Column(String(30), default="uploaded")   # uploaded, extracting, done, error
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class DatasetSplit(Base):
    __tablename__ = "tr_dataset_splits"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("tr_datasets.id", ondelete="CASCADE"), nullable=False)
    split_name = Column(String(20), nullable=False)   # train, val, test
    image_count = Column(Integer, default=0)
    ratio = Column(Float, default=0.0)
    seed = Column(Integer, default=42)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    dataset = relationship("Dataset", back_populates="splits")


# ─── Annotations ──────────────────────────────────────────────────

class Annotation(Base):
    __tablename__ = "tr_annotations"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("tr_dataset_images.id", ondelete="CASCADE"), nullable=False)
    annotation_type = Column(String(30), nullable=False)   # bbox, classification, ocr
    class_name = Column(String(100), nullable=True)
    class_id = Column(Integer, nullable=True)
    # BBox (normalized YOLO format: cx, cy, w, h)
    x_center = Column(Float, nullable=True)
    y_center = Column(Float, nullable=True)
    bbox_width = Column(Float, nullable=True)
    bbox_height = Column(Float, nullable=True)
    # OCR
    ocr_text = Column(String(50), nullable=True)
    # Classification
    label = Column(String(100), nullable=True)
    # Metadata
    confidence = Column(Float, nullable=True)   # for auto-annotations
    is_auto = Column(Boolean, default=False)
    is_verified = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    image = relationship("DatasetImage", back_populates="annotations")


# ─── Training Jobs ────────────────────────────────────────────────

class TrainingJob(Base):
    __tablename__ = "tr_jobs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    model_type = Column(String(50), nullable=False)
    architecture = Column(String(100), nullable=False)   # yolo11n, mobilenetv3, etc.
    dataset_id = Column(Integer, ForeignKey("tr_datasets.id"), nullable=False)
    status = Column(String(30), default=JobStatus.QUEUED)
    celery_task_id = Column(String(255), nullable=True)
    # Hyperparameters
    hyperparams = Column(JSON, default={})
    augmentation_config = Column(JSON, default={})
    # Progress
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer, default=100)
    progress_pct = Column(Float, default=0.0)
    eta_seconds = Column(Integer, nullable=True)
    # Results
    best_metrics = Column(JSON, nullable=True)
    final_metrics = Column(JSON, nullable=True)
    output_model_id = Column(Integer, ForeignKey("tr_model_versions.id"), nullable=True)
    # Runtime
    gpu_device = Column(String(50), nullable=True)
    error_message = Column(Text, nullable=True)
    log_path = Column(String(500), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    # Author
    author_id = Column(Integer, nullable=True)
    author_email = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    dataset = relationship("Dataset")
    metrics_history = relationship("TrainingMetrics", back_populates="job", cascade="all, delete-orphan")
    model_versions = relationship(
        "ModelVersion",
        foreign_keys="ModelVersion.job_id",
        back_populates="job",
    )
    output_model = relationship(
        "ModelVersion",
        foreign_keys=[output_model_id],
        uselist=False,
        post_update=True,
    )


class TrainingMetrics(Base):
    __tablename__ = "tr_metrics"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("tr_jobs.id", ondelete="CASCADE"), nullable=False)
    epoch = Column(Integer, nullable=False)
    # Detection metrics
    train_loss = Column(Float, nullable=True)
    val_loss = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    map50 = Column(Float, nullable=True)
    map50_95 = Column(Float, nullable=True)
    # Classification metrics
    accuracy = Column(Float, nullable=True)
    # OCR metrics
    char_accuracy = Column(Float, nullable=True)
    plate_accuracy = Column(Float, nullable=True)
    # System
    gpu_memory_mb = Column(Float, nullable=True)
    gpu_utilization = Column(Float, nullable=True)
    lr = Column(Float, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    job = relationship("TrainingJob", back_populates="metrics_history")


# ─── Model Registry ───────────────────────────────────────────────

class ModelVersion(Base):
    __tablename__ = "tr_model_versions"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    model_type = Column(String(50), nullable=False)
    architecture = Column(String(100), nullable=False)
    version = Column(String(20), nullable=False)        # v1.0, v1.1, etc.
    version_number = Column(Integer, default=1)
    description = Column(Text, nullable=True)
    changelog = Column(Text, nullable=True)
    # Source
    job_id = Column(Integer, ForeignKey("tr_jobs.id"), nullable=True)
    dataset_id = Column(Integer, ForeignKey("tr_datasets.id"), nullable=True)
    dataset_name = Column(String(255), nullable=True)
    # Files
    weights_path = Column(String(1000), nullable=True)
    onnx_path = Column(String(1000), nullable=True)
    trt_path = Column(String(1000), nullable=True)
    # Metrics
    metrics = Column(JSON, nullable=True)
    hyperparams = Column(JSON, nullable=True)
    artifact_metadata = Column(JSON, nullable=True)
    # Deploy status
    deploy_status = Column(String(30), default=DeployStatus.PENDING)
    is_production = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    # Validation
    validation_passed = Column(Boolean, nullable=True)
    auto_test_results = Column(JSON, nullable=True)
    benchmark_vs_prev = Column(JSON, nullable=True)
    # Author
    author_id = Column(Integer, nullable=True)
    author_email = Column(String(255), nullable=True)
    approved_by = Column(String(255), nullable=True)
    deployed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    job = relationship(
        "TrainingJob",
        foreign_keys=[job_id],
        back_populates="model_versions",
    )
    deploy_logs = relationship("DeployLog", back_populates="model_version", cascade="all, delete-orphan")


class DeployLog(Base):
    __tablename__ = "tr_deploy_logs"

    id = Column(Integer, primary_key=True, index=True)
    model_version_id = Column(Integer, ForeignKey("tr_model_versions.id", ondelete="CASCADE"))
    action = Column(String(50), nullable=False)    # approved, deployed, rejected, rolled_back
    actor_email = Column(String(255), nullable=True)
    comment = Column(Text, nullable=True)
    previous_model_id = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    model_version = relationship("ModelVersion", back_populates="deploy_logs")


# ─── DB Engine ────────────────────────────────────────────────────

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_size=10,
    max_overflow=20,
)


async def get_db():
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
        # create_all does not evolve existing installations. Keep this additive
        # compatibility migration here because the original Docker setup did
        # not run Alembic on startup.
        from sqlalchemy import text
        await conn.execute(text(
            "ALTER TABLE tr_model_versions "
            "ADD COLUMN IF NOT EXISTS artifact_metadata JSON"
        ))
