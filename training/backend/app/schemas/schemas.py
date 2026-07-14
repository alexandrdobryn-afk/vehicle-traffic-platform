from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ModelTypeEnum(str, Enum):
    OBJECT_DETECTOR = "object_detector"
    OBJECT_SEGMENTER = "object_segmenter"
    OBJECT_CLASSIFIER = "object_classifier"


class AnnotationTypeEnum(str, Enum):
    BBOX = "bbox"
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"


class JobStatusEnum(str, Enum):
    QUEUED = "queued"
    PREPARING = "preparing"
    TRAINING = "training"
    VALIDATION = "validation"
    EXPORT = "export"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DeployStatusEnum(str, Enum):
    PENDING = "pending"
    CANDIDATE = "candidate"
    APPROVED = "approved"
    DEPLOYED = "deployed"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class FrameStatusEnum(str, Enum):
    UNLABELED = "unlabeled"
    AUTO_LABELED = "auto_labeled"
    NEEDS_REVIEW = "needs_review"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    REJECTED = "rejected"
    HARD_NEGATIVE = "hard_negative"
    TRAINING_READY = "training_ready"


class TrainingModeEnum(str, Enum):
    BASELINE_INFERENCE = "baseline_inference"
    HEAD_FINETUNE = "head_finetune"
    FULL_FINETUNE = "full_finetune"
    TILED_TRAINING = "tiled_training"
    HARD_NEGATIVE_TRAINING = "hard_negative_training"
    SEMI_SUPERVISED = "semi_supervised"
    CONTINUAL_RETRAINING = "continual_retraining"


class ActiveLearningStatusEnum(str, Enum):
    OPEN = "open"
    IN_REVIEW = "in_review"
    ANNOTATED = "annotated"
    APPROVED = "approved"
    REJECTED = "rejected"
    SKIPPED = "skipped"


class GateResultEnum(str, Enum):
    APPROVED = "approved"
    CANDIDATE = "candidate"
    REJECTED = "rejected"


# ─── Dataset ──────────────────────────────────────────────────────

class DatasetCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    model_type: ModelTypeEnum
    annotation_type: AnnotationTypeEnum
    classes: List[str] = []
    tags: List[str] = []


class DatasetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    classes: Optional[List[str]] = None
    tags: Optional[List[str]] = None


class DatasetResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    model_type: str
    annotation_type: str
    classes: List[str]
    status: str
    version: str
    parent_dataset_id: Optional[int] = None
    content_hash: Optional[str] = None
    is_frozen: bool = False
    frozen_at: Optional[datetime] = None
    lineage: dict = Field(default_factory=dict)
    image_count: int
    video_count: int
    annotation_count: int
    author_email: Optional[str]
    tags: List[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DatasetImageResponse(BaseModel):
    id: int
    dataset_id: int
    filename: str
    file_path: str
    width: Optional[int]
    height: Optional[int]
    source: str
    split: Optional[str]
    is_annotated: bool
    frame_status: str = "unlabeled"
    review_priority: float = 0.0
    review_reason: Optional[str] = None
    scene_tags: List[str] = Field(default_factory=list)
    quality_tags: List[str] = Field(default_factory=list)
    frame_metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

    class Config:
        from_attributes = True


class DatasetSplitConfig(BaseModel):
    train_ratio: float = Field(default=0.8, ge=0.1, le=0.9)
    val_ratio: float = Field(default=0.2, ge=0.05, le=0.5)
    test_ratio: float = Field(default=0.0, ge=0.0, le=0.3)
    seed: int = 42


# ─── Annotations ──────────────────────────────────────────────────

class AnnotationCreate(BaseModel):
    annotation_type: AnnotationTypeEnum
    class_name: Optional[str] = None
    class_id: Optional[int] = None
    # BBox (YOLO normalized: 0..1)
    x_center: Optional[float] = Field(None, ge=0, le=1)
    y_center: Optional[float] = Field(None, ge=0, le=1)
    bbox_width: Optional[float] = Field(None, ge=0, le=1)
    bbox_height: Optional[float] = Field(None, ge=0, le=1)
    polygon: Optional[List[List[float]]] = None
    mask_path: Optional[str] = None
    provenance: Optional[Dict[str, Any]] = None
    # OCR
    ocr_text: Optional[str] = None
    # Classification
    label: Optional[str] = None
    confidence: Optional[float] = None
    is_auto: bool = False
    is_verified: Optional[bool] = None


class AnnotationResponse(BaseModel):
    id: int
    image_id: int
    annotation_type: str
    class_name: Optional[str]
    class_id: Optional[int]
    x_center: Optional[float]
    y_center: Optional[float]
    bbox_width: Optional[float]
    bbox_height: Optional[float]
    polygon: Optional[List[List[float]]]
    mask_path: Optional[str]
    provenance: Optional[Dict[str, Any]]
    ocr_text: Optional[str]
    label: Optional[str]
    confidence: Optional[float]
    is_auto: bool
    is_verified: bool
    created_at: datetime

    class Config:
        from_attributes = True


class GeminiCandidateImport(BaseModel):
    candidate_ids: List[int] = Field(..., min_length=1, max_length=500)
    dataset_id: Optional[int] = None
    dataset_name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    target_model_type: ModelTypeEnum = ModelTypeEnum.OBJECT_SEGMENTER


# ─── Augmentation ─────────────────────────────────────────────────

class AugmentationConfig(BaseModel):
    enabled: bool = True
    rotation: float = Field(default=10.0, ge=0, le=180)
    scale_min: float = Field(default=0.8, ge=0.1, le=1.0)
    scale_max: float = Field(default=1.2, ge=1.0, le=3.0)
    brightness: float = Field(default=0.2, ge=0, le=1)
    contrast: float = Field(default=0.2, ge=0, le=1)
    saturation: float = Field(default=0.2, ge=0, le=1)
    hue: float = Field(default=0.05, ge=0, le=0.5)
    noise: float = Field(default=0.01, ge=0, le=0.1)
    blur: float = Field(default=0.1, ge=0, le=1)
    motion_blur: bool = True
    mosaic: float = Field(default=0.5, ge=0, le=1)
    mixup: float = Field(default=0.1, ge=0, le=1)
    random_crop: bool = True
    horizontal_flip: float = Field(default=0.5, ge=0, le=1)


# ─── Hyperparameters ──────────────────────────────────────────────

class HyperParams(BaseModel):
    epochs: int = Field(default=100, ge=1, le=1000)
    batch_size: int = Field(default=16, ge=1, le=128)
    img_size: int = Field(default=640, ge=320, le=1280)
    learning_rate: float = Field(default=0.01, ge=1e-6, le=1.0)
    weight_decay: float = Field(default=5e-4, ge=0, le=0.1)
    optimizer: str = Field(default="SGD")
    scheduler: str = Field(default="cosine")
    warmup_epochs: int = Field(default=3, ge=0, le=20)
    patience: int = Field(default=50, ge=1, le=200)
    device: str = Field(default="auto")
    workers: int = Field(default=4, ge=0, le=16)
    pretrained: bool = True
    half: bool = False


class TileTrainingConfig(BaseModel):
    enabled: bool = False
    tile_size: int = Field(default=1024, ge=320, le=4096)
    overlap: float = Field(default=0.2, ge=0.0, le=0.75)
    include_empty_tiles: bool = True
    max_empty_tile_ratio: float = Field(default=0.25, ge=0.0, le=1.0)
    min_bbox_area: float = Field(default=0.00001, ge=0.0, le=1.0)
    min_visibility: float = Field(default=0.2, ge=0.0, le=1.0)


class EvaluationPolicy(BaseModel):
    auto_validate_after_training: bool = True
    min_ap_small_delta: float = Field(default=0.05, ge=-1.0, le=1.0)
    max_recall_small_drop: float = Field(default=0.0, ge=-1.0, le=1.0)
    max_fp_per_frame_increase_ratio: float = Field(default=0.10, ge=0.0, le=10.0)
    max_old_holdout_drop: float = Field(default=0.02, ge=0.0, le=1.0)
    min_fps: float = Field(default=0.0, ge=0.0)
    max_latency_p95_ms: Optional[float] = Field(default=None, ge=1.0)
    min_recall_small: float = Field(default=0.0, ge=0.0, le=1.0)
    max_fp_per_frame: Optional[float] = Field(default=None, ge=0.0)
    small_object_area_threshold: float = Field(default=0.01, ge=0.0, le=1.0)
    error_iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    error_confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    max_error_items: int = Field(default=300, ge=0, le=5000)


# ─── Training Job ─────────────────────────────────────────────────

class TrainingJobCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    model_type: ModelTypeEnum
    architecture: str                   # yolo11n, yolo11s, mobilenetv3, etc.
    base_model_id: Optional[str] = Field(default=None, max_length=255)
    dataset_id: int
    training_mode: TrainingModeEnum = TrainingModeEnum.FULL_FINETUNE
    hyperparams: HyperParams = HyperParams()
    augmentation: AugmentationConfig = AugmentationConfig()
    tile_config: TileTrainingConfig = TileTrainingConfig()
    evaluation_policy: EvaluationPolicy = EvaluationPolicy()
    description: Optional[str] = None


class TrainingJobResponse(BaseModel):
    id: int
    name: str
    model_type: str
    architecture: str
    dataset_id: int
    status: str
    celery_task_id: Optional[str]
    training_mode: str = "full_finetune"
    hyperparams: Optional[Dict[str, Any]]
    base_model: Optional[Dict[str, Any]] = None
    tile_config: Dict[str, Any] = Field(default_factory=dict)
    evaluation_policy: Dict[str, Any] = Field(default_factory=dict)
    current_epoch: int
    total_epochs: int
    progress_pct: float
    eta_seconds: Optional[int]
    best_metrics: Optional[Dict[str, Any]]
    final_metrics: Optional[Dict[str, Any]]
    gpu_device: Optional[str]
    error_message: Optional[str]
    author_email: Optional[str]
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True

    @model_validator(mode="after")
    def populate_base_model(self):
        self.base_model = (self.hyperparams or {}).get("base_model")
        return self


class TrainingMetricsResponse(BaseModel):
    id: int
    job_id: int
    epoch: int
    train_loss: Optional[float]
    val_loss: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    map50: Optional[float]
    map50_95: Optional[float]
    accuracy: Optional[float]
    char_accuracy: Optional[float]
    text_accuracy: Optional[float]
    gpu_memory_mb: Optional[float]
    gpu_utilization: Optional[float]
    lr: Optional[float]
    timestamp: datetime

    class Config:
        from_attributes = True


# ─── GPU ──────────────────────────────────────────────────────────

class GPUInfo(BaseModel):
    id: int
    name: str
    memory_total_mb: float
    memory_used_mb: float
    memory_free_mb: float
    utilization_pct: float
    temperature: Optional[float]
    is_available: bool


# ─── Model Registry ───────────────────────────────────────────────

class ModelVersionResponse(BaseModel):
    id: int
    name: str
    model_type: str
    architecture: str
    version: str
    version_number: int
    description: Optional[str]
    changelog: Optional[str]
    dataset_name: Optional[str]
    weights_path: Optional[str]
    onnx_path: Optional[str]
    trt_path: Optional[str]
    metrics: Optional[Dict[str, Any]]
    hyperparams: Optional[Dict[str, Any]]
    artifact_metadata: Optional[Dict[str, Any]]
    deploy_status: str
    is_production: bool
    validation_passed: Optional[bool]
    auto_test_results: Optional[Dict[str, Any]]
    benchmark_vs_prev: Optional[Dict[str, Any]]
    gate_result: Optional[str] = None
    gate_reasons: List[str] = Field(default_factory=list)
    evaluation_report_id: Optional[int] = None
    author_email: Optional[str]
    approved_by: Optional[str]
    deployed_at: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


class DeployApproval(BaseModel):
    comment: Optional[str] = None
    run_auto_tests: bool = True


class RollbackRequest(BaseModel):
    model_version_id: int
    comment: Optional[str] = None


# ─── WebSocket Progress ───────────────────────────────────────────

class TrainingProgressWS(BaseModel):
    job_id: int
    status: str
    current_epoch: int
    total_epochs: int
    progress_pct: float
    eta_seconds: Optional[int]
    latest_metrics: Optional[Dict[str, Any]]
    gpu_utilization: Optional[float]
    gpu_memory_mb: Optional[float]
    message: Optional[str]


# ─── Video Extract ────────────────────────────────────────────────

class VideoExtractConfig(BaseModel):
    fps: float = Field(default=1.0, ge=0.1, le=30.0,
                       description="Frames per second to extract")
    interval_seconds: Optional[float] = Field(
        default=None,
        ge=0.25,
        le=3600.0,
        description="Extract one frame every N seconds. Overrides fps when set.",
    )
    max_frames: Optional[int] = Field(None, ge=1, le=10000)
    start_time: float = Field(default=0.0, ge=0)
    end_time: Optional[float] = None
    quality: int = Field(default=85, ge=50, le=100)


# ─── Auto Annotation ──────────────────────────────────────────────

class AutoAnnotateConfig(BaseModel):
    model_version_id: Optional[int] = None    # use specific model or current production
    confidence_threshold: float = Field(default=0.5, ge=0.1, le=1.0)
    image_ids: Optional[List[int]] = None     # None = all un-annotated
    overwrite_existing: bool = False


class FrameStatusUpdate(BaseModel):
    status: FrameStatusEnum
    review_reason: Optional[str] = Field(default=None, max_length=100)
    review_priority: Optional[float] = Field(default=None, ge=0, le=100)
    scene_tags: Optional[List[str]] = None
    quality_tags: Optional[List[str]] = None


class ActiveLearningCreate(BaseModel):
    dataset_id: int
    image_id: int
    reason: str = Field(..., min_length=1, max_length=80)
    priority_score: float = Field(default=10.0, ge=0, le=100)
    suggested_class: Optional[str] = Field(default=None, max_length=100)
    source: str = Field(default="manual", max_length=80)
    model_version_id: Optional[int] = None
    evaluation_error_id: Optional[int] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class ActiveLearningStatusUpdate(BaseModel):
    status: ActiveLearningStatusEnum


class ActiveLearningItemResponse(BaseModel):
    id: int
    dataset_id: int
    image_id: int
    model_version_id: Optional[int]
    evaluation_error_id: Optional[int]
    reason: str
    priority_score: float
    status: str
    suggested_class: Optional[str]
    source: str
    details: Dict[str, Any] = Field(default_factory=dict)
    reviewer_email: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class EvaluationReportResponse(BaseModel):
    id: int
    model_version_id: int
    dataset_id: Optional[int]
    job_id: Optional[int]
    summary: Dict[str, Any]
    per_class_metrics: Dict[str, Any]
    slice_metrics: Dict[str, Any]
    speed_metrics: Dict[str, Any]
    confusion_matrix: Optional[Any]
    gate_result: str
    gate_reasons: List[str] = Field(default_factory=list)
    created_at: datetime

    class Config:
        from_attributes = True


class EvaluationErrorResponse(BaseModel):
    id: int
    report_id: int
    dataset_image_id: Optional[int]
    error_type: str
    class_name: Optional[str]
    confidence: Optional[float]
    priority_score: float
    bbox: Optional[Any]
    details: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

    class Config:
        from_attributes = True
