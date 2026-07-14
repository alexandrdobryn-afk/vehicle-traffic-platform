import asyncio
import importlib.util
import json
import os
import secrets
import time
from pathlib import Path
from uuid import uuid4

import aiofiles
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks, File, Form, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc, delete
from typing import Optional, List
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import logging

from app.models.database import (
    Camera, ObjectTrack, Event,
    User, AppSettings, GeminiReviewCandidate, PipelineDefinition, Project, get_db
)
from app.schemas.schemas import (
    CameraCreate, CameraUpdate, CameraResponse,
    ObjectTrackResponse, EventResponse,
    AnalyticsSummaryResponse, LoginRequest, TokenResponse,
    UserCreate, UserResponse, AppSettingsSchema, GeminiSettingsUpdate, GeminiCandidateReview,
    AIMode, validate_camera_source_url, validate_pipeline_config,
)
from app.defaults import default_source_pipeline_config
from app.config import settings
from app.utils.auth import (
    hash_password, verify_password, create_access_token,
    encrypt_url, decrypt_url,
    get_current_user, require_admin, require_operator, require_viewer
)
from app.services.camera_manager import camera_manager
from app.services.video_capture_service import VideoCaptureService
from app.services.gemini_training_service import gemini_training_service

logger = logging.getLogger(__name__)

# === AUTH ========================================================
auth_router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@auth_router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == req.email))
    user = result.scalar_one_or_none()

    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")

    token = create_access_token({"sub": str(user.id), "role": user.role, "email": user.email})
    return TokenResponse(access_token=token, role=user.role, email=user.email)


@auth_router.post("/logout")
async def logout(user=Depends(get_current_user)):
    return {"message": "Logged out"}


@auth_router.get("/me", response_model=UserResponse)
async def me(user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.id == user["id"]))
    db_user = result.scalar_one_or_none()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


@auth_router.post("/users", response_model=UserResponse, dependencies=[Depends(require_admin)])
async def create_user(req: UserCreate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == req.email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(email=req.email, password_hash=hash_password(req.password), role=req.role)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


# === CAMERAS =====================================================
cameras_router = APIRouter(prefix="/api/v1/cameras", tags=["cameras"])
videos_router = APIRouter(prefix="/api/v1/videos", tags=["recorded videos"])

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
STREAM_TICKET_TTL_SECONDS = 120
_stream_tickets: dict[str, tuple[int, float]] = {}


async def require_stream_ticket(camera_id: int, ticket: str = Query(...)):
    now = time.monotonic()
    expired = [key for key, (_, expires_at) in _stream_tickets.items() if expires_at <= now]
    for key in expired:
        _stream_tickets.pop(key, None)
    ticket_data = _stream_tickets.get(ticket)
    if not ticket_data or ticket_data[0] != camera_id or ticket_data[1] <= now:
        raise HTTPException(status_code=401, detail="Invalid or expired stream ticket")
    return True


def _camera_source(cam: Camera) -> str:
    if cam.source_type == "file":
        if not cam.source_file_path:
            raise HTTPException(status_code=409, detail="Recorded video file is missing")
        return cam.source_file_path
    return decrypt_url(cam.rtsp_url_encrypted)


def _decorate_source_status(cam: Camera):
    capture = camera_manager.get_capture(cam.id)
    running = camera_manager.is_running(cam.id)
    cam.is_active = running
    if running:
        cam.status = "online"
    elif cam.source_type == "file" and cam.status in {"completed", "error"}:
        pass
    else:
        cam.status = "offline"
    stats = capture.get_stats() if capture else {}
    cam.progress_percent = stats.get("progress_percent")
    if cam.source_type == "file" and cam.status == "completed":
        cam.progress_percent = 100.0
    return cam


def _parse_pipeline_config(raw: str | None) -> dict:
    if not raw:
        return default_source_pipeline_config()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="Invalid pipeline_config JSON") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=422, detail="pipeline_config must be a JSON object")
    try:
        return validate_pipeline_config(parsed)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


SOURCE_LOCAL_SETTING_KEYS = {
    "object_confidence_threshold",
    "track_missing_grace_frames",
}


RUNTIME_SETTING_KEYS = {
    "execution_provider", "runtime_fallback", "gpu_device_index", "inference_precision",
}


async def _global_settings(db: AsyncSession) -> dict:
    defaults = {**default_source_pipeline_config(), **AppSettingsSchema().model_dump()}
    result = await db.execute(select(AppSettings).where(AppSettings.key == "global"))
    row = result.scalar_one_or_none()
    return {**defaults, **(row.value or {})} if row else defaults


def _pipeline_runtime_settings(cam: Camera, global_settings: Optional[dict] = None, pipeline: Optional[PipelineDefinition] = None) -> dict:
    """Combine source recognition choices with global compute policy only."""
    source_defaults = default_source_pipeline_config()
    runtime_defaults = AppSettingsSchema().model_dump()
    local_config = {
        key: value for key, value in (cam.pipeline_config or {}).items()
        if value not in (None, "")
    }
    config = {**source_defaults, **((pipeline.config if pipeline else {}) or {}), **local_config}
    runtime_settings = {
        key: config.get(key, source_defaults[key])
        for key in SOURCE_LOCAL_SETTING_KEYS
    }
    frame_skip = config.get("frame_skip")
    if isinstance(frame_skip, int) and 1 <= frame_skip <= 10:
        runtime_settings["frame_skip"] = frame_skip
    runtime_settings["pipeline_mode"] = "manual" if pipeline else (cam.pipeline_mode or "automatic")
    runtime_settings["pipeline_config"] = config
    runtime_settings["task_profile"] = "aerial_small_objects"
    runtime_settings["pipeline_definition"] = (
        {"id": pipeline.id, "name": pipeline.name, "version": pipeline.version}
        if pipeline else None
    )
    runtime_settings["save_crops"] = bool(cam.save_crops)
    runtime_settings["anonymization_mode"] = bool(cam.anonymization)
    runtime_settings["gemini_enabled"] = bool(cam.gemini_enabled)
    runtime_settings["gemini_verify_predictions"] = bool(cam.gemini_verify_predictions)
    runtime_settings["gemini_collect_training"] = bool(cam.gemini_collect_training)
    runtime_settings["gemini_sample_interval_seconds"] = int(cam.gemini_sample_interval_seconds or 30)
    runtime_settings["gemini_max_candidates_per_run"] = int(cam.gemini_max_candidates_per_run or 25)
    global_settings = global_settings or {**source_defaults, **runtime_defaults}
    for key in RUNTIME_SETTING_KEYS:
        runtime_settings[key] = global_settings.get(key, runtime_defaults[key])
    return runtime_settings


async def _resolved_pipeline_runtime_settings(cam: Camera, db: AsyncSession, global_settings: Optional[dict] = None) -> dict:
    return _pipeline_runtime_settings(cam, global_settings or await _global_settings(db), None)


async def _delete_source_records(cam: Camera, db: AsyncSession):
    candidates = (await db.execute(
        select(GeminiReviewCandidate).where(GeminiReviewCandidate.camera_id == cam.id)
    )).scalars().all()
    for candidate in candidates:
        for stored_path in [candidate.frame_path, candidate.mask_path]:
            if stored_path:
                Path(stored_path).unlink(missing_ok=True)
        for annotation in candidate.proposed_annotations or []:
            if annotation.get("mask_path"):
                Path(annotation["mask_path"]).unlink(missing_ok=True)
    await db.execute(delete(GeminiReviewCandidate).where(GeminiReviewCandidate.camera_id == cam.id))
    await db.execute(delete(Event).where(Event.camera_id == cam.id))
    await db.execute(delete(ObjectTrack).where(ObjectTrack.source_id == cam.id))
    await db.delete(cam)


def _remove_managed_video(file_path: Optional[str]):
    if not file_path:
        return
    root = Path(settings.VIDEOS_PATH).resolve()
    candidate = Path(file_path).resolve()
    if root not in candidate.parents:
        logger.error("Refusing to delete video outside managed storage: %s", candidate)
        return
    try:
        candidate.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not delete managed video %s: %s", candidate, exc)


@cameras_router.get("", response_model=List[CameraResponse])
async def list_cameras(db: AsyncSession = Depends(get_db), user=Depends(require_viewer)):
    result = await db.execute(
        select(Camera).where(Camera.source_type != "file").order_by(Camera.priority.desc())
    )
    cameras = result.scalars().all()
    # Update is_active status from manager
    for cam in cameras:
        _decorate_source_status(cam)
    return cameras


@videos_router.get("", response_model=List[CameraResponse])
async def list_recorded_videos(
    db: AsyncSession = Depends(get_db),
    user=Depends(require_viewer),
):
    result = await db.execute(
        select(Camera).where(Camera.source_type == "file").order_by(Camera.created_at.desc())
    )
    videos = result.scalars().all()
    for video in videos:
        _decorate_source_status(video)
    return videos


@videos_router.post("/upload", response_model=CameraResponse)
async def upload_recorded_video(
    file: UploadFile = File(...),
    name: str = Form(...),
    project_id: Optional[int] = Form(default=None),
    task_profile: str = Form(default="aerial_small_objects"),
    location: Optional[str] = Form(default=None),
    ai_mode: str = Form(default="balanced"),
    pipeline_mode: str = Form(default="automatic"),
    pipeline_config: str = Form(default="{}"),
    priority: int = Form(default=1),
    max_fps: int = Form(default=25),
    save_crops: bool = Form(default=True),
    anonymization: bool = Form(default=False),
    gemini_enabled: bool = Form(default=False),
    gemini_verify_predictions: bool = Form(default=True),
    gemini_collect_training: bool = Form(default=True),
    gemini_sample_interval_seconds: int = Form(default=30, ge=5, le=3600),
    gemini_max_candidates_per_run: int = Form(default=25, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
    user=Depends(require_operator),
):
    original_name = Path(file.filename or "").name
    extension = Path(original_name).suffix.lower()
    clean_name = name.strip()
    if not clean_name or len(clean_name) > 255:
        raise HTTPException(status_code=422, detail="Video name must contain 1 to 255 characters")
    if extension not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Supported video containers: MP4, MOV, MKV, AVI, WEBM",
        )
    if ai_mode not in {mode.value for mode in AIMode}:
        raise HTTPException(status_code=422, detail="Invalid AI mode")
    if task_profile != "aerial_small_objects":
        raise HTTPException(status_code=422, detail="Invalid task profile")
    if pipeline_mode not in {"automatic", "manual"}:
        raise HTTPException(status_code=422, detail="Invalid pipeline mode")
    parsed_pipeline_config = _parse_pipeline_config(pipeline_config)
    if not 1 <= priority <= 10 or not 1 <= max_fps <= 60:
        raise HTTPException(status_code=422, detail="Priority or FPS is outside the allowed range")
    if file.size is not None and file.size > settings.MAX_VIDEO_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Video exceeds the configured upload limit")

    videos_root = Path(settings.VIDEOS_PATH)
    videos_root.mkdir(parents=True, exist_ok=True)
    destination = videos_root / f"{uuid4().hex}{extension}"
    total_bytes = 0

    try:
        async with aiofiles.open(destination, "wb") as output:
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > settings.MAX_VIDEO_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Video exceeds the configured upload limit")
                await output.write(chunk)
        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="Uploaded video is empty")

        probe = await asyncio.to_thread(
            VideoCaptureService.test_connection,
            str(destination),
            "file",
        )
        if not probe.get("success"):
            raise HTTPException(
                status_code=415,
                detail=f"Video cannot be decoded: {probe.get('error', 'unknown decoder error')}",
            )

        video = Camera(
            project_id=project_id,
            name=clean_name,
            rtsp_url_encrypted=encrypt_url("managed-recorded-video"),
            source_type="file",
            source_file_path=str(destination),
            source_file_name=original_name,
            source_duration_seconds=probe.get("duration_seconds"),
            source_fps=probe.get("fps"),
            location=location.strip() if location else None,
            status="offline",
            ai_mode=ai_mode,
            task_profile=task_profile,
            pipeline_mode=pipeline_mode,
            pipeline_config=parsed_pipeline_config,
            priority=priority,
            max_fps=max_fps,
            save_crops=save_crops,
            anonymization=anonymization,
            gemini_enabled=gemini_enabled,
            gemini_verify_predictions=gemini_verify_predictions,
            gemini_collect_training=gemini_collect_training,
            gemini_sample_interval_seconds=gemini_sample_interval_seconds,
            gemini_max_candidates_per_run=gemini_max_candidates_per_run,
        )
        db.add(video)
        await db.commit()
        await db.refresh(video)
        return _decorate_source_status(video)
    except Exception:
        _remove_managed_video(str(destination))
        raise
    finally:
        await file.close()


@videos_router.post("/{video_id}/test")
async def test_recorded_video(
    video_id: int,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_operator),
):
    video = await _get_camera_or_404(video_id, db)
    if video.source_type != "file":
        raise HTTPException(status_code=404, detail="Recorded video not found")
    return await asyncio.to_thread(VideoCaptureService.test_connection, _camera_source(video), "file")


@videos_router.post("/{video_id}/start")
async def start_recorded_video(
    video_id: int,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_operator),
):
    video = await _get_camera_or_404(video_id, db)
    if video.source_type != "file":
        raise HTTPException(status_code=404, detail="Recorded video not found")
    if not Path(_camera_source(video)).is_file():
        video.status = "error"
        video.is_active = False
        await db.commit()
        raise HTTPException(status_code=409, detail="Recorded video file is missing")
    if video.gemini_enabled:
        gemini_config = await gemini_training_service.get_public_settings()
        if not gemini_config.get("configured") or not gemini_config.get("enabled"):
            raise HTTPException(status_code=409, detail="Configure and enable Gemini API before starting this source")

    await db.execute(delete(Event).where(Event.camera_id == video.id))
    await db.execute(delete(ObjectTrack).where(ObjectTrack.source_id == video.id))
    await db.flush()

    runtime_settings = await _resolved_pipeline_runtime_settings(video, db)
    success = await camera_manager.start_camera(
        camera_id=video.id,
        rtsp_url=_camera_source(video),
        ai_mode=video.ai_mode,
        max_fps=video.max_fps,
        runtime_settings=runtime_settings,
        source_type="file",
    )
    if success:
        video.is_active = True
        video.status = "online"
        await db.commit()
    else:
        video.is_active = False
        video.status = "error"
        await db.commit()
        reason = camera_manager.get_last_start_error(video.id) or "pipeline did not start"
        raise HTTPException(status_code=409, detail=f"Could not start video analysis: {reason}")
    return {"success": success, "video_id": video_id}


@videos_router.post("/{video_id}/stop")
async def stop_recorded_video(
    video_id: int,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_operator),
):
    video = await _get_camera_or_404(video_id, db)
    if video.source_type != "file":
        raise HTTPException(status_code=404, detail="Recorded video not found")
    success = await camera_manager.stop_camera(video_id)
    video.is_active = False
    video.status = "offline"
    await db.commit()
    return {"success": success, "video_id": video_id}


@videos_router.delete("/{video_id}")
async def delete_recorded_video(
    video_id: int,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_admin),
):
    video = await _get_camera_or_404(video_id, db)
    if video.source_type != "file":
        raise HTTPException(status_code=404, detail="Recorded video not found")
    file_path = video.source_file_path
    await camera_manager.stop_camera(video_id)
    await _delete_source_records(video, db)
    await db.commit()
    _remove_managed_video(file_path)
    camera_manager.delete_persisted_preview(video_id)
    return {"message": "Recorded video deleted"}


@cameras_router.post("", response_model=CameraResponse)
async def create_camera(
    req: CameraCreate,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_operator),
):
    if req.project_id and not await db.get(Project, req.project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    if req.pipeline_id:
        selected_pipeline = await db.get(PipelineDefinition, req.pipeline_id)
        if not selected_pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        if req.project_id != selected_pipeline.project_id:
            raise HTTPException(status_code=422, detail="Pipeline must belong to the selected project")
    cam = Camera(
        project_id=req.project_id,
        pipeline_id=req.pipeline_id,
        name=req.name,
        rtsp_url_encrypted=encrypt_url(req.rtsp_url),
        source_type=req.source_type.value,
        snapshot_interval_seconds=req.snapshot_interval_seconds,
        location=req.location,
        ai_mode=req.ai_mode,
        task_profile=req.task_profile.value,
        pipeline_mode=req.pipeline_mode,
        pipeline_config=req.pipeline_config,
        priority=req.priority,
        max_fps=req.max_fps,
        save_crops=req.save_crops,
        anonymization=req.anonymization,
        gemini_enabled=req.gemini_enabled,
        gemini_verify_predictions=req.gemini_verify_predictions,
        gemini_collect_training=req.gemini_collect_training,
        gemini_sample_interval_seconds=req.gemini_sample_interval_seconds,
        gemini_max_candidates_per_run=req.gemini_max_candidates_per_run,
    )
    db.add(cam)
    await db.commit()
    await db.refresh(cam)
    return cam


@cameras_router.get("/{camera_id}", response_model=CameraResponse)
async def get_camera(camera_id: int, db: AsyncSession = Depends(get_db), user=Depends(require_viewer)):
    cam = await _get_camera_or_404(camera_id, db)
    return _decorate_source_status(cam)


@cameras_router.patch("/{camera_id}", response_model=CameraResponse)
async def update_camera(
    camera_id: int,
    req: CameraUpdate,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_operator),
):
    cam = await _get_camera_or_404(camera_id, db)
    changes = req.model_dump(exclude_none=True)
    effective_project_id = changes.get("project_id", cam.project_id)
    effective_pipeline_id = changes.get("pipeline_id", cam.pipeline_id)
    if effective_pipeline_id:
        selected_pipeline = await db.get(PipelineDefinition, effective_pipeline_id)
        if not selected_pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        if selected_pipeline.project_id != effective_project_id:
            raise HTTPException(status_code=422, detail="Pipeline must belong to the selected project")
    current_source_type = cam.source_type or "rtsp"
    effective_source_type = changes.get("source_type", current_source_type)
    if hasattr(effective_source_type, "value"):
        effective_source_type = effective_source_type.value
    new_url = changes.get("rtsp_url")
    if current_source_type == "file" or effective_source_type == "file":
        if effective_source_type != current_source_type or new_url:
            raise HTTPException(status_code=422, detail="Recorded video files must be managed in the Videos section")
    else:
        try:
            validate_camera_source_url(
                effective_source_type,
                new_url or decrypt_url(cam.rtsp_url_encrypted),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    if "source_type" in changes and effective_source_type != current_source_type and not new_url:
        raise HTTPException(status_code=422, detail="Changing source type requires a new source URL")

    was_running = camera_manager.is_running(camera_id)
    for field, val in changes.items():
        if field == "rtsp_url":
            cam.rtsp_url_encrypted = encrypt_url(val)
        elif field == "source_type":
            cam.source_type = val.value if hasattr(val, "value") else val
        else:
            setattr(cam, field, val)
    await db.commit()
    await db.refresh(cam)
    if was_running:
        runtime_settings = await _resolved_pipeline_runtime_settings(cam, db)
        await camera_manager.restart_camera(
            camera_id=camera_id,
            rtsp_url=_camera_source(cam),
            ai_mode=cam.ai_mode,
            max_fps=cam.max_fps,
            runtime_settings=runtime_settings,
            source_type=cam.source_type,
            snapshot_interval_seconds=cam.snapshot_interval_seconds,
        )
    return cam


@cameras_router.delete("/{camera_id}")
async def delete_camera(camera_id: int, db: AsyncSession = Depends(get_db), user=Depends(require_admin)):
    cam = await _get_camera_or_404(camera_id, db)
    file_path = cam.source_file_path
    await camera_manager.stop_camera(camera_id)
    await _delete_source_records(cam, db)
    await db.commit()
    if cam.source_type == "file":
        _remove_managed_video(file_path)
        camera_manager.delete_persisted_preview(camera_id)
    return {"message": "Camera deleted"}


@cameras_router.post("/{camera_id}/test")
async def test_camera(camera_id: int, db: AsyncSession = Depends(get_db), user=Depends(require_operator)):
    cam = await _get_camera_or_404(camera_id, db)
    url = _camera_source(cam)
    result = await asyncio.to_thread(
        lambda: __import__("app.services.video_capture_service", fromlist=["VideoCaptureService"])
        .VideoCaptureService.test_connection(url, cam.source_type)
    )
    return result


@cameras_router.post("/{camera_id}/start")
async def start_camera(camera_id: int, db: AsyncSession = Depends(get_db), user=Depends(require_operator)):
    cam = await _get_camera_or_404(camera_id, db)
    url = _camera_source(cam)
    if cam.gemini_enabled:
        gemini_config = await gemini_training_service.get_public_settings()
        if not gemini_config.get("configured") or not gemini_config.get("enabled"):
            raise HTTPException(status_code=409, detail="Configure and enable Gemini API before starting this source")
    runtime_settings = await _resolved_pipeline_runtime_settings(cam, db)
    success = await camera_manager.start_camera(
        camera_id=camera_id,
        rtsp_url=url,
        ai_mode=cam.ai_mode,
        max_fps=cam.max_fps,
        runtime_settings=runtime_settings,
        source_type=cam.source_type,
        snapshot_interval_seconds=cam.snapshot_interval_seconds,
    )
    if success:
        cam.is_active = True
        cam.status = "online"
        await db.commit()
    return {"success": success, "camera_id": camera_id}


@cameras_router.post("/{camera_id}/stop")
async def stop_camera(camera_id: int, db: AsyncSession = Depends(get_db), user=Depends(require_operator)):
    cam = await _get_camera_or_404(camera_id, db)
    success = await camera_manager.stop_camera(camera_id)
    cam.is_active = False
    cam.status = "offline"
    await db.commit()
    return {"success": success, "camera_id": camera_id}


# === STREAMS (MJPEG) =============================================
streams_router = APIRouter(prefix="/api/v1/stream", tags=["streams"])


@streams_router.post("/{camera_id}/ticket")
async def create_stream_ticket(
    camera_id: int,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_viewer),
):
    if not camera_manager.get_capture(camera_id) and camera_manager.get_latest_frame(camera_id) is None:
        cam = await _get_camera_or_404(camera_id, db)
        restored = False
        if cam.source_type == "file" and cam.source_file_path:
            restored = await asyncio.to_thread(
                camera_manager.restore_file_preview,
                camera_id,
                cam.source_file_path,
            )
        if not restored:
            raise HTTPException(status_code=404, detail="Source has no preview")
    ticket = secrets.token_urlsafe(24)
    _stream_tickets[ticket] = (camera_id, time.monotonic() + STREAM_TICKET_TTL_SECONDS)
    return {"ticket": ticket, "expires_in": STREAM_TICKET_TTL_SECONDS}


@streams_router.get("/{camera_id}")
async def mjpeg_stream(camera_id: int, authorized=Depends(require_stream_ticket)):
    if not camera_manager.get_capture(camera_id) and camera_manager.get_latest_frame(camera_id) is None:
        raise HTTPException(status_code=404, detail="Camera not running")

    return StreamingResponse(
        camera_manager.mjpeg_generator(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@streams_router.get("/{camera_id}/snapshot")
async def preview_snapshot(camera_id: int, authorized=Depends(require_stream_ticket)):
    encoded = camera_manager.get_preview_jpeg(camera_id)
    if encoded is None:
        raise HTTPException(status_code=404, detail="Source has no preview")
    return Response(
        content=encoded,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


# === TRACKS ======================================================
tracks_router = APIRouter(prefix="/api/v1/tracks", tags=["tracks"])


@tracks_router.get("", response_model=List[ObjectTrackResponse])
async def list_tracks(
    source_id: Optional[int] = None,
    object_class: Optional[str] = None,
    limit: int = Query(50, le=500),
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_viewer),
):
    q = select(ObjectTrack).order_by(desc(ObjectTrack.last_seen))
    if source_id:
        q = q.where(ObjectTrack.source_id == source_id)
    if object_class:
        q = q.where(ObjectTrack.object_class == object_class)
    q = q.limit(limit).offset(offset)
    result = await db.execute(q)
    return result.scalars().all()


@tracks_router.get("/active")
async def active_tracks(camera_id: Optional[int] = None, user=Depends(require_viewer)):
    """Return currently tracked objects from in-memory state."""
    if camera_id:
        pipeline = camera_manager.get_pipeline(camera_id)
        if pipeline:
            tracks = await pipeline.track_state.get_all_active()
            return [t.to_dict() for t in tracks]
        return []
    # All cameras
    all_tracks = []
    for stats in camera_manager.get_all_stats():
        cam_id = stats.get("camera_id")
        if cam_id:
            pipeline = camera_manager.get_pipeline(cam_id)
            if pipeline:
                tracks = await pipeline.track_state.get_all_active()
                all_tracks.extend([t.to_dict() for t in tracks])
    return all_tracks


@tracks_router.get("/{track_id}", response_model=ObjectTrackResponse)
async def get_track(track_id: int, db: AsyncSession = Depends(get_db), user=Depends(require_viewer)):
    result = await db.execute(select(ObjectTrack).where(ObjectTrack.id == track_id))
    track = result.scalar_one_or_none()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    return track


# === EVENTS ======================================================
events_router = APIRouter(prefix="/api/v1/events", tags=["events"])


@events_router.get("", response_model=List[EventResponse])
async def list_events(
    camera_id: Optional[int] = None,
    event_type: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    alert_only: bool = False,
    limit: int = Query(100, le=1000),
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_viewer),
):
    q = select(Event).order_by(desc(Event.created_at))
    if camera_id:
        q = q.where(Event.camera_id == camera_id)
    if event_type:
        q = q.where(Event.event_type == event_type)
    if alert_only:
        q = q.where(Event.event_type == "object_alert")
    if date_from:
        q = q.where(Event.created_at >= date_from)
    if date_to:
        q = q.where(Event.created_at <= date_to)
    q = q.limit(limit).offset(offset)
    result = await db.execute(q)
    return result.scalars().all()


@events_router.get("/export")
async def export_events(
    format: str = Query("csv", regex="^(csv|excel|json)$"),
    camera_id: Optional[int] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_viewer),
):
    import pandas as pd
    from io import BytesIO

    q = select(Event).order_by(desc(Event.created_at)).limit(10000)
    if camera_id:
        q = q.where(Event.camera_id == camera_id)
    if date_from:
        q = q.where(Event.created_at >= date_from)
    if date_to:
        q = q.where(Event.created_at <= date_to)

    result = await db.execute(q)
    events = result.scalars().all()

    rows = [
        {
            "id": e.id,
            "camera_id": e.camera_id,
            "event_type": e.event_type,
            "created_at": e.created_at.isoformat() if e.created_at else "",
            "object_class": (e.payload_json or {}).get("object_class", ""),
            "track_id": (e.payload_json or {}).get("track_id", ""),
            "state": (e.payload_json or {}).get("state", ""),
        }
        for e in events
    ]
    df = pd.DataFrame(rows)

    if format == "csv":
        output = df.to_csv(index=False).encode()
        media_type = "text/csv"
        filename = "events.csv"
    elif format == "excel":
        buf = BytesIO()
        df.to_excel(buf, index=False)
        output = buf.getvalue()
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        filename = "events.xlsx"
    else:
        output = df.to_json(orient="records").encode()
        media_type = "application/json"
        filename = "events.json"

    from fastapi.responses import Response
    return Response(
        content=output,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# === WATCHLIST ====================================================
# === ANALYTICS ===================================================
analytics_router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


@analytics_router.get("/summary")
async def analytics_summary(db: AsyncSession = Depends(get_db), user=Depends(require_viewer)):
    # "Today" is a user-facing local-day metric, not a UTC-day metric. Around
    # midnight in Kyiv the previous implementation displayed zero despite
    # tracks having just been persisted.
    local_tz = ZoneInfo("Europe/Kyiv")
    today_start = datetime.now(local_tz).replace(
        hour=0, minute=0, second=0, microsecond=0
    ).astimezone(timezone.utc)

    object_total_today = await db.scalar(
        select(func.count(ObjectTrack.id)).where(ObjectTrack.created_at >= today_start)
    )
    total_today = object_total_today or 0
    event_total_today = await db.scalar(
        select(func.count(Event.id)).where(Event.created_at >= today_start)
    )
    stats = camera_manager.get_all_stats()
    avg_fps = sum(s.get("fps", 0) for s in stats) / len(stats) if stats else 0.0
    avg_latency = (
        sum(s.get("inference_latency_ms", 0) for s in stats) / len(stats)
        if stats else 0.0
    )
    active_cams = sum(1 for s in stats if s.get("is_running"))

    return {
        "total_objects_today": total_today,
        "total_events_today": event_total_today or 0,
        "active_cameras": active_cams,
        "avg_fps": round(avg_fps, 1),
        "avg_latency_ms": round(avg_latency, 1),
    }


@analytics_router.get("/object-volume")
async def object_volume(
    hours: int = Query(24, le=168),
    source_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_viewer),
):
    from_time = datetime.now(timezone.utc) - timedelta(hours=hours)
    q = select(
        func.date_trunc("hour", ObjectTrack.created_at).label("hour"),
        func.count(ObjectTrack.id).label("count")
    ).where(ObjectTrack.created_at >= from_time).group_by("hour").order_by("hour")
    if source_id:
        q = q.where(ObjectTrack.source_id == source_id)
    result = await db.execute(q)
    rows = result.fetchall()
    return [{"timestamp": str(r.hour), "count": r.count} for r in rows]


@analytics_router.get("/classes")
async def class_distribution(db: AsyncSession = Depends(get_db), user=Depends(require_viewer)):
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    result = await db.execute(
        select(ObjectTrack.object_class, func.count(ObjectTrack.id).label("count"))
        .where(ObjectTrack.created_at >= today_start)
        .group_by(ObjectTrack.object_class)
        .order_by(desc("count"))
    )
    rows = result.fetchall()
    total = sum(r.count for r in rows)
    return [
        {"object_class": r.object_class, "count": r.count, "percentage": round(r.count / total * 100, 1) if total else 0}
        for r in rows
    ]


@analytics_router.get("/performance")
async def performance(user=Depends(require_viewer)):
    return camera_manager.get_all_stats()


# === SETTINGS ====================================================
settings_router = APIRouter(prefix="/api/v1/settings", tags=["settings"])


@settings_router.get("")
async def get_settings(db: AsyncSession = Depends(get_db), user=Depends(require_viewer)):
    return await _global_settings(db)


@settings_router.get("/runtime-status")
async def runtime_status(db: AsyncSession = Depends(get_db), user=Depends(require_viewer)):
    from app.services.runtime_service import resolve_execution_policy

    config = await _global_settings(db)
    return await asyncio.to_thread(
        resolve_execution_policy,
        config["execution_provider"],
        config["gpu_device_index"],
        config["runtime_fallback"],
    )


@settings_router.get("/gemini")
async def get_gemini_settings(user=Depends(require_viewer)):
    return await gemini_training_service.get_public_settings()


@settings_router.put("/gemini")
async def update_gemini_settings(
    req: GeminiSettingsUpdate,
    user=Depends(require_admin),
):
    return await gemini_training_service.save_settings(req.model_dump(exclude_none=True))


@settings_router.post("/gemini/test")
async def test_gemini_settings(user=Depends(require_admin)):
    result = await gemini_training_service.test_connection()
    if not result.get("success"):
        raise HTTPException(status_code=409, detail=result.get("error", "Gemini connection failed"))
    return result


# Gemini-assisted review queue. Candidate images stay behind normal API auth;
# they are not exposed through a public browser URL.
gemini_router = APIRouter(prefix="/api/v1/gemini", tags=["gemini-training"])


def _candidate_response(candidate: GeminiReviewCandidate) -> dict:
    return {
        "id": candidate.id,
        "camera_id": candidate.camera_id,
        "processing_run_id": candidate.processing_run_id,
        "track_id": candidate.track_id,
        "status": candidate.status,
        "selection_reason": candidate.selection_reason,
        "frame_width": candidate.frame_width,
        "frame_height": candidate.frame_height,
        "target_model_type": candidate.target_model_type,
        "local_predictions": candidate.local_predictions or [],
        "gemini_verification": candidate.gemini_verification,
        "proposed_annotations": candidate.proposed_annotations or [],
        "provider_model": candidate.provider_model,
        "usage_metadata": candidate.usage_metadata,
        "error_message": candidate.error_message,
        "reviewed_by": candidate.reviewed_by,
        "reviewed_at": candidate.reviewed_at,
        "created_at": candidate.created_at,
        "image_url": f"/api/v1/gemini/candidates/{candidate.id}/image",
        "mask_url": f"/api/v1/gemini/candidates/{candidate.id}/mask" if candidate.mask_path else None,
    }


@gemini_router.get("/candidates")
async def list_gemini_candidates(
    status: Optional[str] = None,
    camera_id: Optional[int] = None,
    limit: int = Query(default=100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
    user=Depends(require_viewer),
):
    query = select(GeminiReviewCandidate)
    if status:
        query = query.where(GeminiReviewCandidate.status == status)
    if camera_id is not None:
        query = query.where(GeminiReviewCandidate.camera_id == camera_id)
    result = await db.execute(query.order_by(GeminiReviewCandidate.created_at.desc()).limit(limit))
    return [_candidate_response(candidate) for candidate in result.scalars().all()]


@gemini_router.get("/candidates/{candidate_id}")
async def get_gemini_candidate(
    candidate_id: int,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_viewer),
):
    candidate = await db.get(GeminiReviewCandidate, candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Gemini review candidate not found")
    return _candidate_response(candidate)


def _candidate_file(candidate: GeminiReviewCandidate, kind: str) -> FileResponse:
    path = candidate.frame_path if kind == "image" else candidate.mask_path
    if not path or not Path(path).is_file():
        raise HTTPException(status_code=404, detail=f"Candidate {kind} is missing")
    return FileResponse(path, media_type="image/jpeg" if kind == "image" else "image/png")


@gemini_router.get("/candidates/{candidate_id}/image")
async def get_gemini_candidate_image(
    candidate_id: int,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_viewer),
):
    candidate = await db.get(GeminiReviewCandidate, candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Gemini review candidate not found")
    return _candidate_file(candidate, "image")


@gemini_router.get("/candidates/{candidate_id}/mask")
async def get_gemini_candidate_mask(
    candidate_id: int,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_viewer),
):
    candidate = await db.get(GeminiReviewCandidate, candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Gemini review candidate not found")
    return _candidate_file(candidate, "mask")


@gemini_router.patch("/candidates/{candidate_id}/review")
async def review_gemini_candidate(
    candidate_id: int,
    req: GeminiCandidateReview,
    user=Depends(require_operator),
):
    try:
        candidate = await gemini_training_service.review_candidate(
            candidate_id, req.model_dump(exclude_none=True), user["email"]
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _candidate_response(candidate)


@gemini_router.post("/candidates/{candidate_id}/retry")
async def retry_gemini_candidate(candidate_id: int, user=Depends(require_operator)):
    try:
        await gemini_training_service.retry_candidate(candidate_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"message": "Candidate queued for Gemini retry"}


@settings_router.patch("")
async def update_settings(
    req: AppSettingsSchema,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_admin),
):
    from app.services.runtime_service import resolve_execution_policy

    runtime = await asyncio.to_thread(
        resolve_execution_policy,
        req.execution_provider, req.gpu_device_index, req.runtime_fallback
    )
    if not runtime["available"]:
        raise HTTPException(
            status_code=409,
            detail="Requested CUDA device is unavailable. Select Auto/CPU or allow CPU fallback.",
        )
    result = await db.execute(select(AppSettings).where(AppSettings.key == "global"))
    row = result.scalar_one_or_none()
    if row:
        row.value = req.model_dump()
    else:
        row = AppSettings(key="global", value=req.model_dump())
        db.add(row)
    await db.commit()

    # Apply settings immediately to active cameras.
    active = await db.execute(select(Camera).where(Camera.is_active == True))
    restarted = 0
    for camera in active.scalars().all():
        url = _camera_source(camera)
        if await camera_manager.restart_camera(
            camera_id=camera.id,
            rtsp_url=url,
            ai_mode=camera.ai_mode,
            max_fps=camera.max_fps,
            runtime_settings=await _resolved_pipeline_runtime_settings(camera, db, req.model_dump()),
            source_type=camera.source_type,
            snapshot_interval_seconds=camera.snapshot_interval_seconds,
        ):
            restarted += 1
    return {
        "message": "Settings updated",
        "restarted_cameras": restarted,
        "runtime": runtime,
    }


# === SYSTEM HEALTH ================================================
health_router = APIRouter(prefix="/api/v1/health", tags=["health"])


@health_router.get("/live")
async def liveness_check():
    return {"status": "ok"}


@health_router.get("/ready")
async def readiness_check(response: Response, db: AsyncSession = Depends(get_db)):
    import redis.asyncio as aioredis
    from app.config import settings as cfg

    database_ok = False
    redis_ok = False

    try:
        await db.execute(select(func.count(User.id)))
        database_ok = True
    except Exception:
        pass

    try:
        r = aioredis.from_url(cfg.REDIS_URL)
        await r.ping()
        await r.close()
        redis_ok = True
    except Exception:
        pass

    ready = database_ok and redis_ok
    if not ready:
        response.status_code = 503

    return {
        "status": "ok" if ready else "degraded",
        "database": "ok" if database_ok else "error",
        "redis": "ok" if redis_ok else "error",
    }


@health_router.get("")
async def health_check():
    from app.models.model_registry import model_registry
    detector = model_registry.get_object_detector("balanced")
    segmenter = model_registry.get_object_segmenter()
    classifier = model_registry.get_object_classifier()
    required_ready = detector.available
    return {
        "status": "ok" if required_ready else "degraded",
        "version": "1.0.0",
        "models": {
            "object_detector": {"name": detector.name, "available": detector.available},
            "object_segmenter": {"name": segmenter.name, "available": segmenter.available},
            "object_classifier": {"name": classifier.name, "available": classifier.available},
        },
    }


@health_router.get("/full")
async def full_health(db: AsyncSession = Depends(get_db), user=Depends(require_viewer)):
    import redis.asyncio as aioredis
    from app.config import settings as cfg

    db_ok = False
    redis_ok = False

    try:
        await db.execute(select(func.count(User.id)))
        db_ok = True
    except Exception:
        pass

    try:
        r = aioredis.from_url(cfg.REDIS_URL)
        await r.ping()
        await r.close()
        redis_ok = True
    except Exception:
        pass

    return {
        "status": "ok" if db_ok and redis_ok else "degraded",
        "database": "ok" if db_ok else "error",
        "redis": "ok" if redis_ok else "error",
        "cameras": camera_manager.get_all_stats(),
        "total_ws_connections": __import__(
            "app.services.websocket_service", fromlist=["websocket_manager"]
        ).websocket_manager.get_total_connections(),
    }


@health_router.get("/ai-modes")
async def ai_mode_profiles(user=Depends(require_viewer)):
    """Return the model artifacts currently resolved for every AI mode."""
    from app.models.model_registry import (
        OBJECT_CLASSIFIER_PRODUCTION, OBJECT_DETECTOR_PRODUCTION,
        OBJECT_DETECTOR_RFDETR_MEDIUM, OBJECT_DETECTOR_RFDETR_NANO,
        OBJECT_DETECTOR_RTDETR, OBJECT_DETECTOR_YOLO11N,
        OBJECT_DETECTOR_YOLO11S, OBJECT_SEGMENTER_PRODUCTION, model_registry,
    )

    frame_skip_defaults = {
        "speed": settings.FRAME_SKIP_SPEED,
        "balanced": settings.FRAME_SKIP_BALANCED,
        "quality": settings.FRAME_SKIP_QUALITY,
        "hybrid": settings.FRAME_SKIP_BALANCED,
        "practical": settings.FRAME_SKIP_BALANCED,
        "max_accuracy": settings.FRAME_SKIP_QUALITY,
        "edge_onnx": settings.FRAME_SKIP_SPEED,
    }
    tracker_defaults = {
        "speed": "bytetrack",
        "balanced": "bytetrack",
        "quality": "botsort",
        "hybrid": "bytetrack",
        "practical": "ocsort",
        "max_accuracy": "botsort",
        "edge_onnx": "bytetrack",
    }
    tracker_availability = {
        "bytetrack": True,
        "botsort": True,
        "ocsort": importlib.util.find_spec("ocsort") is not None,
        "deepsort": importlib.util.find_spec("deep_sort_realtime") is not None,
        # The current backend maps strongsort to the BoT-SORT adapter without
        # external ReID. Keep it visible but do not mark it as true StrongSORT.
        "strongsort": False,
    }

    modes = []
    for mode in AIMode:
        mode_name = mode.value
        detector = model_registry.get_object_detector(mode_name)
        models = [
            _model_profile("detection", detector),
            {
                "role": "tracking",
                "name": tracker_defaults[mode_name],
                "format": "algorithm",
                "available": tracker_availability.get(tracker_defaults[mode_name], False),
            },
            {"role": "prediction", "name": "constant_velocity_kalman", "format": "algorithm", "available": True},
            _model_profile("classification", OBJECT_CLASSIFIER_PRODUCTION),
            _model_profile("segmentation", OBJECT_SEGMENTER_PRODUCTION),
        ]
        if mode_name == "hybrid":
            models.insert(1, _model_profile("detection", OBJECT_DETECTOR_RFDETR_MEDIUM))
        modes.append({
            "id": mode_name,
            "frame_skip_default": frame_skip_defaults[mode_name],
            "tracker_default": tracker_defaults[mode_name],
            "models": models,
            "features": {
                "object_memory": True,
                "identity_stitching": True,
                "confidence_engine": True,
                "tiled_inference": True,
                "onnx_runtime": mode_name == "edge_onnx",
                "tensorrt_preferred": mode_name in {"speed", "edge_onnx"},
            },
        })

    return {
        "modes": modes,
        "manual_options": {
            "object_detectors": [
                _model_profile("detection", model)
                for model in [
                    OBJECT_DETECTOR_PRODUCTION,
                    OBJECT_DETECTOR_YOLO11N,
                    OBJECT_DETECTOR_YOLO11S,
                    OBJECT_DETECTOR_RFDETR_NANO,
                    OBJECT_DETECTOR_RFDETR_MEDIUM,
                    OBJECT_DETECTOR_RTDETR,
                ]
            ],
            "verifier_detectors": [
                _model_profile("detection", model)
                for model in [
                    OBJECT_DETECTOR_RFDETR_NANO,
                    OBJECT_DETECTOR_RFDETR_MEDIUM,
                ]
            ],
            "trackers": [
                {"name": "bytetrack", "format": "algorithm", "available": tracker_availability["bytetrack"]},
                {"name": "botsort", "format": "algorithm", "available": tracker_availability["botsort"]},
                {"name": "ocsort", "format": "optional algorithm", "available": tracker_availability["ocsort"]},
                {"name": "deepsort", "format": "optional algorithm", "available": tracker_availability["deepsort"]},
                {"name": "strongsort", "format": "optional algorithm", "available": tracker_availability["strongsort"]},
            ],
        },
        "settings_override_note": (
            "Detection thresholds, aerial tiling, target classes, tracking and optional modules are source-local."
        ),
    }


# === HELPERS ======================================================
async def _get_camera_or_404(camera_id: int, db: AsyncSession) -> Camera:
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    cam = result.scalar_one_or_none()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    return cam


def _model_profile(role: str, model) -> dict:
    return {
        "role": role,
        "name": model.name,
        "format": model.format,
        "available": model.available,
    }


import asyncio
