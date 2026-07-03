from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator
from typing import Optional, List, Any, Literal
from datetime import datetime
from enum import Enum
from urllib.parse import urlsplit


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


def validate_pipeline_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate source-local recognition settings stored in the JSON column."""
    ranges = {
        "vehicle_confidence_threshold": (0.1, 0.99),
        "plate_confidence_threshold": (0.1, 0.99),
        "ocr_threshold": (0.1, 0.99),
        "ocr_voting_window": (3, 30),
        "track_missing_grace_frames": (1, 120),
        "minimum_plate_width": (20, 300),
        "minimum_plate_height": (10, 150),
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
    if config.get("plate_regex_profile", "AUTO") not in {"AUTO", "UA", "UK", "IN", "EU", "US"}:
        raise ValueError("Unsupported plate_regex_profile")
    if config.get("input_resolution", "1280x720") not in {"640x360", "1280x720", "1920x1080"}:
        raise ValueError("Unsupported input_resolution")
    return config


def validate_camera_source_url(source_type: CameraSourceType | str, url: str) -> None:
    """Validate the transport scheme without assuming a filename extension."""
    source_type = CameraSourceType(source_type)
    if source_type == CameraSourceType.FILE:
        raise ValueError("Recorded videos must be added through the video upload endpoint")
    parsed = urlsplit(url.strip())
    allowed_schemes = {
        CameraSourceType.RTSP: {"rtsp", "rtsps"},
        CameraSourceType.HLS: {"http", "https"},
        CameraSourceType.MJPEG: {"http", "https"},
        CameraSourceType.JPEG: {"http", "https"},
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
    # would incorrectly reject local/offline deployments such as vtp.local.
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
    name: str = Field(..., min_length=1, max_length=255)
    rtsp_url: str = Field(..., min_length=5)
    source_type: CameraSourceType = CameraSourceType.RTSP
    snapshot_interval_seconds: float = Field(default=1.0, ge=0.25, le=300)
    location: Optional[str] = None
    ai_mode: AIMode = AIMode.BALANCED
    pipeline_mode: Literal["automatic", "manual"] = "automatic"
    pipeline_config: dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=1, ge=1, le=10)
    max_fps: int = Field(default=25, ge=1, le=60)
    save_crops: bool = True
    anonymization: bool = False

    @field_validator("pipeline_config")
    @classmethod
    def validate_pipeline(cls, value):
        return validate_pipeline_config(value)

    @model_validator(mode="after")
    def validate_source(self):
        validate_camera_source_url(self.source_type, self.rtsp_url)
        return self


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    rtsp_url: Optional[str] = Field(default=None, min_length=5)
    source_type: Optional[CameraSourceType] = None
    snapshot_interval_seconds: Optional[float] = Field(default=None, ge=0.25, le=300)
    location: Optional[str] = None
    ai_mode: Optional[AIMode] = None
    pipeline_mode: Optional[Literal["automatic", "manual"]] = None
    pipeline_config: Optional[dict[str, Any]] = None
    priority: Optional[int] = None
    max_fps: Optional[int] = None
    save_crops: Optional[bool] = None
    anonymization: Optional[bool] = None

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
    pipeline_mode: str = "automatic"
    pipeline_config: Optional[Any] = None
    priority: int
    max_fps: int
    is_active: bool
    save_crops: bool
    anonymization: bool
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


class TrackResponse(BaseModel):
    id: int
    camera_id: int
    track_id: int
    vehicle_class: str
    final_plate: Optional[str]
    plate_status: str
    final_plate_confidence: float
    color: str
    color_confidence: float
    vehicle_make: str
    make_confidence: float
    first_seen: datetime
    last_seen: datetime
    duration_seconds: float
    best_vehicle_crop_path: Optional[str]
    best_plate_crop_path: Optional[str]
    processing_run_id: str
    recognition_diagnostics: Optional[Any] = None

    class Config:
        from_attributes = True


class ActiveTrackSchema(BaseModel):
    track_id: int
    camera_id: int
    vehicle_class: str
    bbox: List[int]
    color: str
    color_confidence: float
    vehicle_make: str
    make_confidence: float
    plate: Optional[str]
    plate_status: str
    plate_confidence: float
    first_seen: str
    last_seen: str
    recognition_diagnostics: Optional[Any] = None


# Events
class EventResponse(BaseModel):
    id: int
    camera_id: int
    vehicle_track_id: Optional[int]
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
    vehicle_class: str
    plate: Optional[str]
    plate_status: str
    plate_confidence: float
    color: str
    color_confidence: float
    recognition_diagnostics: Optional[Any] = None


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


# Watchlist
class WatchlistCreate(BaseModel):
    plate_number: str = Field(..., min_length=3, max_length=20)
    description: Optional[str] = None
    alert_channels: List[str] = ["frontend"]


class WatchlistResponse(BaseModel):
    id: int
    plate_number: str
    description: Optional[str]
    alert_channels: List[str]
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


# Analytics
class AnalyticsSummaryResponse(BaseModel):
    total_vehicles_today: int
    total_plates_recognized: int
    ocr_success_rate: float
    active_cameras: int
    avg_fps: float
    avg_latency_ms: float
    watchlist_matches_today: int


class TrafficVolumePoint(BaseModel):
    timestamp: str
    count: int


class ColorDistribution(BaseModel):
    color: str
    count: int
    percentage: float


# Settings
class AppSettingsSchema(BaseModel):
    execution_provider: Literal["auto", "cpu", "cuda"] = "auto"
    runtime_fallback: Literal["fail_closed", "allow_cpu"] = "fail_closed"
    gpu_device_index: int = Field(default=0, ge=0, le=15)
    inference_precision: Literal["fp32"] = "fp32"
    tracker_mode: Literal["bytetrack", "botsort", "tracktrack"] = "bytetrack"
    ocr_engine: Literal["lprnet", "easyocr", "paddleocr", "fastalpr"] = "easyocr"
    plate_regex_profile: Literal["AUTO", "UA", "UK", "IN", "EU", "US"] = "AUTO"
    vehicle_confidence_threshold: float = Field(default=0.45, ge=0.1, le=0.99)
    plate_confidence_threshold: float = Field(default=0.40, ge=0.1, le=0.99)
    ocr_threshold: float = Field(default=0.60, ge=0.1, le=0.99)
    frame_skip: int = Field(default=2, ge=1, le=10)
    input_resolution: Literal["640x360", "1280x720", "1920x1080"] = "1280x720"
    max_fps_per_camera: int = Field(default=25, ge=1, le=60)
    recorded_analysis_fps: int = Field(default=5, ge=1, le=30)
    ocr_voting_window: int = Field(default=10, ge=3, le=30)
    track_missing_grace_frames: int = Field(default=15, ge=1, le=120)
    minimum_plate_width: int = Field(default=60, ge=20, le=300)
    minimum_plate_height: int = Field(default=20, ge=10, le=150)
    save_crops: bool = True
    anonymization_mode: bool = False


class PlateCandidate(BaseModel):
    plate_text: str
    confidence: float
    regex_valid: bool
    regex_score: float
    image_quality_score: float
    frame_timestamp: datetime

    class Config:
        from_attributes = True
