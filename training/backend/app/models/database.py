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
    DRAFT = "draft"
    READY = "ready"
    ANNOTATION_IN_PROGRESS = "annotation_in_progress"
    REVIEW_REQUIRED = "review_required"
    READY_FOR_TRAINING = "ready_for_training"
    FROZEN = "frozen"
    USED_FOR_TRAINING = "used_for_training"
    ARCHIVED = "archived"
    PROCESSING = "processing"
    ERROR = "error"


class AnnotationType(str, enum.Enum):
    BBOX = "bbox"
    SEGMENTATION = "segmentation"
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
    OBJECT_DETECTOR = "object_detector"
    OBJECT_SEGMENTER = "object_segmenter"
    OBJECT_CLASSIFIER = "object_classifier"
    TEXT_OCR = "text_ocr"


class ModelFormat(str, enum.Enum):
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"


class DeployStatus(str, enum.Enum):
    PENDING = "pending"
    CANDIDATE = "candidate"
    APPROVED = "approved"
    DEPLOYED = "deployed"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class FrameStatus(str, enum.Enum):
    UNLABELED = "unlabeled"
    AUTO_LABELED = "auto_labeled"
    NEEDS_REVIEW = "needs_review"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    REJECTED = "rejected"
    HARD_NEGATIVE = "hard_negative"
    TRAINING_READY = "training_ready"


class ActiveLearningStatus(str, enum.Enum):
    OPEN = "open"
    IN_REVIEW = "in_review"
    ANNOTATED = "annotated"
    APPROVED = "approved"
    REJECTED = "rejected"
    SKIPPED = "skipped"


class GateResult(str, enum.Enum):
    APPROVED = "approved"
    CANDIDATE = "candidate"
    REJECTED = "rejected"


# ─── Dataset ──────────────────────────────────────────────────────

class Dataset(Base):
    __tablename__ = "tr_datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    model_type = Column(String(50), nullable=False)
    annotation_type = Column(String(50), nullable=False)  # bbox, classification, ocr
    classes = Column(JSON, default=[])
    status = Column(String(30), default=DatasetStatus.CREATING)
    version = Column(String(20), default="1.0")
    parent_dataset_id = Column(Integer, ForeignKey("tr_datasets.id", ondelete="SET NULL"), nullable=True, index=True)
    content_hash = Column(String(64), nullable=True, index=True)
    is_frozen = Column(Boolean, nullable=False, default=False, server_default="false")
    frozen_at = Column(DateTime(timezone=True), nullable=True)
    lineage = Column(JSON, nullable=False, default=dict)
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
    frame_status = Column(String(30), nullable=False, default="unlabeled", server_default="unlabeled")
    review_priority = Column(Float, nullable=False, default=0.0, server_default="0")
    review_reason = Column(String(100), nullable=True)
    scene_tags = Column(JSON, nullable=False, default=list)
    quality_tags = Column(JSON, nullable=False, default=list)
    frame_metadata = Column(JSON, nullable=False, default=dict)
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
    # Instance segmentation (normalized polygon points and optional source mask)
    polygon = Column(JSON, nullable=True)
    mask_path = Column(String(1000), nullable=True)
    provenance = Column(JSON, nullable=True)
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
    training_mode = Column(String(50), nullable=False, default="full_finetune", server_default="full_finetune")
    # Hyperparameters
    hyperparams = Column(JSON, default={})
    augmentation_config = Column(JSON, default={})
    tile_config = Column(JSON, nullable=False, default=dict)
    evaluation_policy = Column(JSON, nullable=False, default=dict)
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
    # Text OCR metrics
    char_accuracy = Column(Float, nullable=True)
    text_accuracy = Column(Float, nullable=True)
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
    gate_result = Column(String(30), nullable=True)
    gate_reasons = Column(JSON, nullable=False, default=list)
    evaluation_report_id = Column(Integer, nullable=True)
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


class EvaluationReport(Base):
    __tablename__ = "tr_evaluation_reports"

    id = Column(Integer, primary_key=True, index=True)
    model_version_id = Column(Integer, ForeignKey("tr_model_versions.id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_id = Column(Integer, ForeignKey("tr_datasets.id", ondelete="SET NULL"), nullable=True, index=True)
    job_id = Column(Integer, ForeignKey("tr_jobs.id", ondelete="SET NULL"), nullable=True, index=True)
    summary = Column(JSON, nullable=False, default=dict)
    per_class_metrics = Column(JSON, nullable=False, default=dict)
    slice_metrics = Column(JSON, nullable=False, default=dict)
    speed_metrics = Column(JSON, nullable=False, default=dict)
    confusion_matrix = Column(JSON, nullable=True)
    gate_result = Column(String(30), nullable=False, default="candidate", server_default="candidate")
    gate_reasons = Column(JSON, nullable=False, default=list)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    errors = relationship("EvaluationError", back_populates="report", cascade="all, delete-orphan")


class EvaluationError(Base):
    __tablename__ = "tr_evaluation_errors"

    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, ForeignKey("tr_evaluation_reports.id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_image_id = Column(Integer, ForeignKey("tr_dataset_images.id", ondelete="SET NULL"), nullable=True, index=True)
    error_type = Column(String(50), nullable=False)
    class_name = Column(String(100), nullable=True)
    confidence = Column(Float, nullable=True)
    priority_score = Column(Float, nullable=False, default=0.0, server_default="0")
    bbox = Column(JSON, nullable=True)
    details = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    report = relationship("EvaluationReport", back_populates="errors")


class ActiveLearningItem(Base):
    __tablename__ = "tr_active_learning_items"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("tr_datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    image_id = Column(Integer, ForeignKey("tr_dataset_images.id", ondelete="CASCADE"), nullable=False, index=True)
    model_version_id = Column(Integer, ForeignKey("tr_model_versions.id", ondelete="SET NULL"), nullable=True, index=True)
    evaluation_error_id = Column(Integer, ForeignKey("tr_evaluation_errors.id", ondelete="SET NULL"), nullable=True, index=True)
    reason = Column(String(80), nullable=False)
    priority_score = Column(Float, nullable=False, default=0.0, server_default="0")
    status = Column(String(30), nullable=False, default="open", server_default="open", index=True)
    suggested_class = Column(String(100), nullable=True)
    source = Column(String(80), nullable=False, default="manual", server_default="manual")
    details = Column(JSON, nullable=False, default=dict)
    reviewer_email = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())


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
        await conn.execute(text(
            "ALTER TABLE tr_annotations ADD COLUMN IF NOT EXISTS polygon JSON"
        ))
        await conn.execute(text(
            "ALTER TABLE tr_annotations ADD COLUMN IF NOT EXISTS mask_path VARCHAR(1000)"
        ))
        await conn.execute(text(
            "ALTER TABLE tr_annotations ADD COLUMN IF NOT EXISTS provenance JSON"
        ))
        await conn.execute(text(
            "ALTER TABLE tr_datasets ADD COLUMN IF NOT EXISTS parent_dataset_id INTEGER"
        ))
        await conn.execute(text(
            "ALTER TABLE tr_datasets ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64)"
        ))
        await conn.execute(text(
            "ALTER TABLE tr_datasets ADD COLUMN IF NOT EXISTS is_frozen BOOLEAN NOT NULL DEFAULT false"
        ))
        await conn.execute(text(
            "ALTER TABLE tr_datasets ADD COLUMN IF NOT EXISTS frozen_at TIMESTAMPTZ"
        ))
        await conn.execute(text(
            "ALTER TABLE tr_datasets ADD COLUMN IF NOT EXISTS lineage JSON NOT NULL DEFAULT '{}'::json"
        ))
        await conn.execute(text(
            "ALTER TABLE tr_dataset_images ADD COLUMN IF NOT EXISTS frame_status VARCHAR(30) NOT NULL DEFAULT 'unlabeled'"
        ))
        await conn.execute(text(
            "ALTER TABLE tr_dataset_images ADD COLUMN IF NOT EXISTS review_priority FLOAT NOT NULL DEFAULT 0"
        ))
        await conn.execute(text(
            "ALTER TABLE tr_dataset_images ADD COLUMN IF NOT EXISTS review_reason VARCHAR(100)"
        ))
        await conn.execute(text(
            "ALTER TABLE tr_dataset_images ADD COLUMN IF NOT EXISTS scene_tags JSON NOT NULL DEFAULT '[]'::json"
        ))
        await conn.execute(text(
            "ALTER TABLE tr_dataset_images ADD COLUMN IF NOT EXISTS quality_tags JSON NOT NULL DEFAULT '[]'::json"
        ))
        await conn.execute(text(
            "ALTER TABLE tr_dataset_images ADD COLUMN IF NOT EXISTS frame_metadata JSON NOT NULL DEFAULT '{}'::json"
        ))
        await conn.execute(text(
            "ALTER TABLE tr_jobs ADD COLUMN IF NOT EXISTS training_mode VARCHAR(50) NOT NULL DEFAULT 'full_finetune'"
        ))
        await conn.execute(text(
            "ALTER TABLE tr_jobs ADD COLUMN IF NOT EXISTS tile_config JSON NOT NULL DEFAULT '{}'::json"
        ))
        await conn.execute(text(
            "ALTER TABLE tr_jobs ADD COLUMN IF NOT EXISTS evaluation_policy JSON NOT NULL DEFAULT '{}'::json"
        ))
        await conn.execute(text(
            "ALTER TABLE tr_model_versions ADD COLUMN IF NOT EXISTS gate_result VARCHAR(30)"
        ))
        await conn.execute(text(
            "ALTER TABLE tr_model_versions ADD COLUMN IF NOT EXISTS gate_reasons JSON NOT NULL DEFAULT '[]'::json"
        ))
        await conn.execute(text(
            "ALTER TABLE tr_model_versions ADD COLUMN IF NOT EXISTS evaluation_report_id INTEGER"
        ))
