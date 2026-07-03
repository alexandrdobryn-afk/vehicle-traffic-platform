from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ModelTypeEnum(str, Enum):
    VEHICLE_DETECTOR = "vehicle_detector"
    PLATE_DETECTOR = "plate_detector"
    OCR = "ocr"
    COLOR_CLASSIFIER = "color_classifier"


class AnnotationTypeEnum(str, Enum):
    BBOX = "bbox"
    CLASSIFICATION = "classification"
    OCR = "ocr"


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
    APPROVED = "approved"
    DEPLOYED = "deployed"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


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
    created_at: datetime

    class Config:
        from_attributes = True


class DatasetSplitConfig(BaseModel):
    train_ratio: float = Field(default=0.7, ge=0.1, le=0.9)
    val_ratio: float = Field(default=0.2, ge=0.05, le=0.5)
    test_ratio: float = Field(default=0.1, ge=0.0, le=0.3)
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
    ocr_text: Optional[str]
    label: Optional[str]
    confidence: Optional[float]
    is_auto: bool
    is_verified: bool
    created_at: datetime

    class Config:
        from_attributes = True


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


# ─── Training Job ─────────────────────────────────────────────────

class TrainingJobCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    model_type: ModelTypeEnum
    architecture: str                   # yolo11n, yolo11s, mobilenetv3, etc.
    dataset_id: int
    hyperparams: HyperParams = HyperParams()
    augmentation: AugmentationConfig = AugmentationConfig()
    description: Optional[str] = None


class TrainingJobResponse(BaseModel):
    id: int
    name: str
    model_type: str
    architecture: str
    dataset_id: int
    status: str
    celery_task_id: Optional[str]
    hyperparams: Optional[Dict[str, Any]]
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
    plate_accuracy: Optional[float]
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
