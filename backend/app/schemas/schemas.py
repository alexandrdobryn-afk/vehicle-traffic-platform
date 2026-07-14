from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator
from typing import Optional, List, Any, Literal
from datetime import datetime
from enum import Enum
from urllib.parse import urlsplit

from app.defaults import default_source_pipeline_config


class AIMode(str, Enum):
    SPEED = "speed"
    BALANCED = "balanced"
    QUALITY = "quality"
    HYBRID = "hybrid"
    PRACTICAL = "practical"
    MAX_ACCURACY = "max_accuracy"
    EDGE_ONNX = "edge_onnx"


class CameraSourceType(str, Enum):
    RTSP = "rtsp"
    HLS = "hls"
    MJPEG = "mjpeg"
    JPEG = "jpeg"
    FILE = "file"
    USB = "usb"
    DRONE = "drone"


class TaskProfile(str, Enum):
    GENERIC_OBJECTS = "generic_objects"
    AERIAL_SMALL_OBJECTS = "aerial_small_objects"


def validate_pipeline_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate source-local recognition settings stored in the JSON column."""
    ranges = {
        "object_confidence_threshold": (0.05, 0.99),
        "track_missing_grace_frames": (1, 120),
        "min_track_frames_for_event": (1, 120),
        "min_track_duration_seconds": (0.0, 30.0),
        "min_box_width": (1, 4096),
        "min_box_height": (1, 4096),
        "min_box_area_ratio": (0.0, 1.0),
        "max_box_area_ratio": (0.0, 1.0),
        "max_box_aspect_ratio": (1.0, 50.0),
        "frame_skip": (0, 10),
    }
    for key, (minimum, maximum) in ranges.items():
        if key in config:
            try:
                numeric_value = float(config[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} must be numeric") from exc
            if not minimum <= numeric_value <= maximum:
                raise ValueError(f"{key} must be between {minimum} and {maximum}")
    target_classes = config.get("target_classes")
    if target_classes is not None and (
        not isinstance(target_classes, list)
        or any(not isinstance(item, str) or not item.strip() for item in target_classes)
    ):
        raise ValueError("target_classes must be a list of non-empty class names")
    aerial = config.get("aerial") or {}
    if not isinstance(aerial, dict):
        raise ValueError("aerial must be an object")
    if "tile_size" in aerial and not 320 <= int(aerial["tile_size"]) <= 2048:
        raise ValueError("aerial.tile_size must be between 320 and 2048")
    if "tile_overlap" in aerial and not 0 <= float(aerial["tile_overlap"]) <= 0.5:
        raise ValueError("aerial.tile_overlap must be between 0 and 0.5")
    modules = {
        "object_memory": {
            "max_gap_frames": (1, 600),
            "merge_threshold": (0.0, 1.0),
            "duplicate_iou": (0.0, 1.0),
            "duplicate_contained": (0.0, 1.0),
            "appearance_threshold": (0.0, 1.0),
        },
        "kalman_prediction": {"max_prediction_frames": (0, 120)},
        "classification": {"confidence_threshold": (0.0, 1.0), "interval_frames": (1, 300)},
        "segmentation": {"confidence_threshold": (0.0, 1.0), "interval_frames": (1, 300)},
        "ocr": {"confidence_threshold": (0.0, 1.0), "interval_frames": (1, 600)},
        "reid": {"similarity_threshold": (0.0, 1.0)},
        "super_resolution": {"min_object_size_px": (4, 512), "max_crops_per_frame": (1, 100)},
    }
    for module_name, module_ranges in modules.items():
        module = config.get(module_name) or {}
        if not isinstance(module, dict):
            raise ValueError(f"{module_name} must be an object")
        if "enabled" in module and not isinstance(module["enabled"], bool):
            raise ValueError(f"{module_name}.enabled must be boolean")
        for key, (minimum, maximum) in module_ranges.items():
            if key in module and not minimum <= float(module[key]) <= maximum:
                raise ValueError(f"{module_name}.{key} must be between {minimum} and {maximum}")
    geo = config.get("geo") or {}
    if not isinstance(geo, dict):
        raise ValueError("geo must be an object")
    if "enabled" in geo and not isinstance(geo["enabled"], bool):
        raise ValueError("geo.enabled must be boolean")
    if "coordinate_output" in geo and not isinstance(geo["coordinate_output"], bool):
        raise ValueError("geo.coordinate_output must be boolean")
    return config


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=255)
    slug: str = Field(..., pattern=r"^[a-z0-9][a-z0-9-]{1,98}[a-z0-9]$")
    description: Optional[str] = None
    task_profile: TaskProfile = TaskProfile.GENERIC_OBJECTS
    target_classes: List[str] = Field(default_factory=list)


class ProjectUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=2, max_length=255)
    description: Optional[str] = None
    task_profile: Optional[TaskProfile] = None
    target_classes: Optional[List[str]] = None
    status: Optional[Literal["active", "archived"]] = None


class ProjectResponse(BaseModel):
    id: int
    name: str
    slug: str
    description: Optional[str]
    task_profile: str
    target_classes: List[str]
    status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PipelineDefinitionCreate(BaseModel):
    project_id: int
    name: str = Field(..., min_length=2, max_length=255)
    description: Optional[str] = None
    task_profile: TaskProfile = TaskProfile.GENERIC_OBJECTS
    config: dict[str, Any] = Field(default_factory=dict)
    is_default: bool = False

    @field_validator("config")
    @classmethod
    def validate_config(cls, value):
        return validate_pipeline_config(value)


class PipelineDefinitionResponse(BaseModel):
    id: int
    project_id: int
    name: str
    description: Optional[str]
    version: int
    task_profile: str
    config: dict[str, Any]
    is_default: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class EvaluationRunCreate(BaseModel):
    project_id: Optional[int] = None
    name: str = Field(..., min_length=2, max_length=255)
    task_type: Literal[
        "detection",
        "tracking",
        "segmentation",
        "classification",
        "pipeline",
        "replay",
        "profiling",
    ]
    model_name: str = Field(..., min_length=1, max_length=255)
    dataset_ref: str = Field(..., min_length=1, max_length=500)
    config: dict[str, Any] = Field(default_factory=dict)


class EvaluationRunResponse(BaseModel):
    id: int
    project_id: int
    name: str
    task_type: str
    model_name: str
    dataset_ref: str
    status: str
    config: dict[str, Any]
    metrics: Optional[dict[str, Any]]
    error_message: Optional[str]
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


def validate_camera_source_url(source_type: CameraSourceType | str, url: str) -> None:
    """Validate the transport scheme without assuming a filename extension."""
    source_type = CameraSourceType(source_type)
    if source_type == CameraSourceType.FILE:
        raise ValueError("Recorded videos must be added through the video upload endpoint")
    if source_type == CameraSourceType.USB:
        if not url.lower().startswith("usb://") or not url[6:].isdigit():
            raise ValueError("USB sources use usb://<device-index>, for example usb://0")
        return
    parsed = urlsplit(url.strip())
    allowed_schemes = {
        CameraSourceType.RTSP: {"rtsp", "rtsps"},
        CameraSourceType.HLS: {"http", "https"},
        CameraSourceType.MJPEG: {"http", "https"},
        CameraSourceType.JPEG: {"http", "https"},
        CameraSourceType.DRONE: {"rtsp", "rtsps", "udp", "http", "https"},
    }
    if parsed.scheme.lower() not in allowed_schemes[source_type]:
        expected = "RTSP/RTSPS" if source_type == CameraSourceType.RTSP else "HTTP/HTTPS"
        raise ValueError(f"{source_type.value.upper()} sources require an {expected} URL")
    if not parsed.hostname:
        raise ValueError("Camera source URL must include a host")


class UserRole(str, Enum):
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"


# Auth
class LoginRequest(BaseModel):
    # Login identifies an existing account; domain deliverability validation
    # would incorrectly reject local/offline deployments such as bevp.local.
    email: str = Field(..., min_length=3, max_length=255)
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    email: str


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    role: UserRole = UserRole.VIEWER


class UserResponse(BaseModel):
    id: int
    email: str
    role: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


# Camera
class CameraCreate(BaseModel):
    project_id: Optional[int] = None
    pipeline_id: Optional[int] = None
    name: str = Field(..., min_length=1, max_length=255)
    rtsp_url: str = Field(..., min_length=5)
    source_type: CameraSourceType = CameraSourceType.RTSP
    snapshot_interval_seconds: float = Field(default=1.0, ge=0.25, le=300)
    location: Optional[str] = None
    ai_mode: AIMode = AIMode.BALANCED
    task_profile: TaskProfile = TaskProfile.AERIAL_SMALL_OBJECTS
    pipeline_mode: Literal["automatic", "manual"] = "automatic"
    pipeline_config: dict[str, Any] = Field(default_factory=default_source_pipeline_config)
    priority: int = Field(default=1, ge=1, le=10)
    max_fps: int = Field(default=25, ge=1, le=60)
    save_crops: bool = True
    anonymization: bool = False
    gemini_enabled: bool = False
    gemini_verify_predictions: bool = True
    gemini_collect_training: bool = True
    gemini_sample_interval_seconds: int = Field(default=30, ge=5, le=3600)
    gemini_max_candidates_per_run: int = Field(default=25, ge=1, le=500)

    @field_validator("pipeline_config")
    @classmethod
    def validate_pipeline(cls, value):
        return validate_pipeline_config(value)

    @model_validator(mode="after")
    def validate_source(self):
        validate_camera_source_url(self.source_type, self.rtsp_url)
        return self


class CameraUpdate(BaseModel):
    project_id: Optional[int] = None
    pipeline_id: Optional[int] = None
    name: Optional[str] = None
    rtsp_url: Optional[str] = Field(default=None, min_length=5)
    source_type: Optional[CameraSourceType] = None
    snapshot_interval_seconds: Optional[float] = Field(default=None, ge=0.25, le=300)
    location: Optional[str] = None
    ai_mode: Optional[AIMode] = None
    task_profile: Optional[TaskProfile] = None
    pipeline_mode: Optional[Literal["automatic", "manual"]] = None
    pipeline_config: Optional[dict[str, Any]] = None
    priority: Optional[int] = None
    max_fps: Optional[int] = None
    save_crops: Optional[bool] = None
    anonymization: Optional[bool] = None
    gemini_enabled: Optional[bool] = None
    gemini_verify_predictions: Optional[bool] = None
    gemini_collect_training: Optional[bool] = None
    gemini_sample_interval_seconds: Optional[int] = Field(default=None, ge=5, le=3600)
    gemini_max_candidates_per_run: Optional[int] = Field(default=None, ge=1, le=500)

    @field_validator("pipeline_config")
    @classmethod
    def validate_pipeline(cls, value):
        return validate_pipeline_config(value) if value is not None else value

    @model_validator(mode="after")
    def validate_source(self):
        if self.source_type is not None and self.rtsp_url is not None:
            validate_camera_source_url(self.source_type, self.rtsp_url)
        return self


class CameraResponse(BaseModel):
    id: int
    project_id: Optional[int] = None
    pipeline_id: Optional[int] = None
    name: str
    location: Optional[str]
    status: str
    source_type: str
    source_file_name: Optional[str] = None
    source_duration_seconds: Optional[float] = None
    source_fps: Optional[float] = None
    progress_percent: Optional[float] = None
    snapshot_interval_seconds: float
    ai_mode: str
    task_profile: str = "aerial_small_objects"
    pipeline_mode: str = "automatic"
    pipeline_config: Optional[Any] = None
    priority: int
    max_fps: int
    is_active: bool
    save_crops: bool
    anonymization: bool
    gemini_enabled: bool = False
    gemini_verify_predictions: bool = True
    gemini_collect_training: bool = True
    gemini_sample_interval_seconds: int = 30
    gemini_max_candidates_per_run: int = 25
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Tracks
class BBoxSchema(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class ObjectTrackResponse(BaseModel):
    id: int
    source_id: int
    processing_run_id: str
    track_id: int
    object_class: str
    confidence: float
    trajectory: List[List[float]]
    speed_pixels_per_second: float
    direction_degrees: Optional[float]
    state: str
    attributes: dict[str, Any]
    best_crop_path: Optional[str]
    last_bbox: List[int] = Field(default_factory=list)
    first_video_timestamp_seconds: Optional[float] = None
    last_video_timestamp_seconds: Optional[float] = None
    first_seen: datetime
    last_seen: datetime
    duration_seconds: float

    class Config:
        from_attributes = True


class ActiveTrackSchema(BaseModel):
    track_id: int
    camera_id: int
    object_class: str
    bbox: List[int]
    first_seen: str
    last_seen: str
    diagnostics: Optional[Any] = None


# Events
class EventResponse(BaseModel):
    id: int
    camera_id: int
    object_track_id: Optional[int] = None
    event_type: str
    payload_json: Optional[Any]
    frame_path: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


# WebSocket
class WSObjectSchema(BaseModel):
    track_id: int
    bbox: List[int]
    object_class: Optional[str] = None
    diagnostics: Optional[Any] = None
    trajectory: List[List[float]] = Field(default_factory=list)
    speed_pixels_per_second: float = 0.0
    direction_degrees: Optional[float] = None
    state: str = "active"


class WSFrameSchema(BaseModel):
    camera_id: int
    timestamp: str
    fps: float
    latency_ms: int
    frame_width: int
    frame_height: int
    source_frame_width: Optional[int] = None
    source_frame_height: Optional[int] = None
    processing_run_id: Optional[str] = None
    stage_counts: Optional[dict] = None
    objects: List[WSObjectSchema]


# Analytics
class AnalyticsSummaryResponse(BaseModel):
    total_objects_today: int
    total_events_today: int
    active_cameras: int
    avg_fps: float
    avg_latency_ms: float


class ObjectVolumePoint(BaseModel):
    timestamp: str
    count: int


class ClassDistribution(BaseModel):
    object_class: str
    count: int
    percentage: float


# Settings
class AppSettingsSchema(BaseModel):
    execution_provider: Literal["auto", "cpu", "cuda"] = "auto"
    runtime_fallback: Literal["fail_closed", "allow_cpu"] = "fail_closed"
    gpu_device_index: int = Field(default=0, ge=0, le=15)
    inference_precision: Literal["fp32"] = "fp32"


class GeminiSettingsUpdate(BaseModel):
    api_key: Optional[str] = Field(default=None, min_length=10, max_length=500)
    model: str = Field(default="gemini-2.5-flash", min_length=3, max_length=100)
    enabled: bool = True
    request_timeout_seconds: int = Field(default=45, ge=10, le=120)
    max_concurrent_requests: int = Field(default=1, ge=1, le=4)


class GeminiCandidateReview(BaseModel):
    status: Literal["approved", "rejected"]
    class_name: Optional[str] = Field(default=None, min_length=1, max_length=80)
    annotation_index: int = Field(default=0, ge=0, le=100)
    polygon: Optional[List[List[float]]] = None

    @field_validator("polygon")
    @classmethod
    def validate_polygon(cls, value):
        if value is None:
            return value
        if len(value) < 3:
            raise ValueError("A segmentation polygon requires at least three points")
        for point in value:
            if len(point) != 2 or any(coord < 0 or coord > 1 for coord in point):
                raise ValueError("Polygon coordinates must be normalized pairs between 0 and 1")
        return value
