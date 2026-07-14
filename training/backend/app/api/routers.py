import os
import json
import asyncio
from pathlib import Path
from typing import Optional, List
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, update, text

from app.models.database import (
    Dataset, DatasetImage, DatasetVideo, Annotation,
    TrainingJob, TrainingMetrics, ModelVersion, DeployLog,
    EvaluationReport,
    get_db, JobStatus,
)
from app.schemas.schemas import (
    DatasetCreate, DatasetUpdate, DatasetResponse, DatasetImageResponse,
    DatasetSplitConfig, AnnotationCreate, AnnotationResponse,
    TrainingJobCreate, TrainingJobResponse, TrainingMetricsResponse,
    ModelVersionResponse, DeployApproval, RollbackRequest,
    VideoExtractConfig, AutoAnnotateConfig, GeminiCandidateImport,
    FrameStatusUpdate, ActiveLearningCreate, ActiveLearningStatusUpdate,
    ActiveLearningItemResponse, EvaluationReportResponse, EvaluationErrorResponse,
)
from app.config import settings
from app.services.dataset_service import dataset_service
from app.services.annotation_service import annotation_service
from app.services.model_registry_service import model_registry_service
from app.services.gpu_service import gpu_service
from app.services.lifecycle_service import lifecycle_service
from app.utils.auth import get_current_user, require_operator, require_admin

import logging
logger = logging.getLogger(__name__)

MODEL_TYPE_DIRS = {
    "object_detector": "object_detector",
    "object_segmenter": "object_segmenter",
    "object_classifier": "object_classifier",
}


def _base_model_display_name(item: dict) -> str:
    label = item.get("label") or item.get("name") or item.get("id")
    version = item.get("version")
    return f"{label} {version}".strip() if version else str(label)

# ═══ DATASETS ════════════════════════════════════════════════════
datasets_router = APIRouter(prefix="/api/v1/training/datasets", tags=["training-datasets"])


@datasets_router.get("", response_model=List[DatasetResponse])
async def list_datasets(
    model_type: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    return await dataset_service.list_datasets(db, model_type)


@datasets_router.post("", response_model=DatasetResponse)
async def create_dataset(
    data: DatasetCreate,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_operator),
):
    return await dataset_service.create_dataset(db, data, user["email"])


@datasets_router.post("/import-gemini")
async def import_gemini_candidates(
    data: GeminiCandidateImport,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_operator),
):
    """Import human-approved real frames and segmentation masks into Training."""
    model_type = data.target_model_type.value
    if not model_type.endswith("_segmenter"):
        raise HTTPException(400, "Gemini mask candidates require a segmenter dataset")

    dataset = await dataset_service.get_dataset(db, data.dataset_id) if data.dataset_id else None
    if data.dataset_id and dataset is None:
        raise HTTPException(404, "Dataset not found")
    if dataset is None:
        if not data.dataset_name:
            raise HTTPException(422, "dataset_name is required when dataset_id is omitted")
        from app.schemas.schemas import DatasetCreate, AnnotationTypeEnum, ModelTypeEnum
        dataset = await dataset_service.create_dataset(
            db,
            DatasetCreate(
                name=data.dataset_name,
                description="Human-approved Gemini-assisted masks from real platform frames",
                model_type=ModelTypeEnum(model_type),
                annotation_type=AnnotationTypeEnum.SEGMENTATION,
                classes=["object"],
                tags=["gemini-assisted", "human-verified", "real-frames"],
            ),
            user["email"],
        )
    if dataset.model_type != model_type or dataset.annotation_type != "segmentation":
        raise HTTPException(400, "Target dataset is not compatible with the selected segmenter")

    query = text("""
        SELECT id, frame_path, frame_width, frame_height, proposed_annotations,
               provider_model, processing_run_id, camera_id, track_id
        FROM gemini_review_candidates
        WHERE id = ANY(:ids) AND status = 'approved' AND target_model_type != 'verification_only'
        ORDER BY id
    """)
    rows = (await db.execute(query, {"ids": data.candidate_ids})).mappings().all()
    imported = 0
    annotations_created = 0
    allowed_classes = set(dataset.classes or [])
    source_root = Path(settings.SOURCE_STORAGE_PATH)

    for row in rows:
        source_path = Path(row["frame_path"])
        try:
            relative = source_path.relative_to("/app/storage")
        except ValueError:
            raise HTTPException(409, f"Candidate {row['id']} has an unmanaged source path")
        mounted_path = source_root / relative
        if not mounted_path.is_file():
            raise HTTPException(409, f"Candidate frame {row['id']} is not available to Training API")
        image = await dataset_service.save_uploaded_image(
            db, dataset.id, f"gemini_candidate_{row['id']}{mounted_path.suffix or '.jpg'}", mounted_path.read_bytes()
        )
        proposed = row["proposed_annotations"] or []
        for annotation in proposed:
            class_name = annotation.get("class_name")
            polygon = annotation.get("polygon") or []
            if class_name not in allowed_classes or len(polygon) < 3:
                continue
            xs = [float(point[0]) for point in polygon]
            ys = [float(point[1]) for point in polygon]
            x0, x1 = max(0.0, min(xs)), min(1.0, max(xs))
            y0, y1 = max(0.0, min(ys)), min(1.0, max(ys))
            ann = Annotation(
                image_id=image.id,
                annotation_type="segmentation",
                class_name=class_name,
                class_id=list(dataset.classes).index(class_name),
                x_center=(x0 + x1) / 2,
                y_center=(y0 + y1) / 2,
                bbox_width=x1 - x0,
                bbox_height=y1 - y0,
                polygon=polygon,
                confidence=annotation.get("confidence"),
                is_auto=True,
                is_verified=True,
                provenance={
                    "source": "gemini_review_candidate",
                    "candidate_id": row["id"],
                    "provider_model": row["provider_model"],
                    "camera_id": row["camera_id"],
                    "processing_run_id": row["processing_run_id"],
                    "track_id": row["track_id"],
                    "reviewed_by": user["email"],
                },
            )
            db.add(ann)
            annotations_created += 1
        image.is_annotated = True
        imported += 1

    await db.execute(
        update(Dataset)
        .where(Dataset.id == dataset.id)
        .values(annotation_count=Dataset.annotation_count + annotations_created)
    )
    await db.commit()
    return {
        "dataset_id": dataset.id,
        "imported_candidates": imported,
        "annotations_created": annotations_created,
        "skipped_candidates": len(data.candidate_ids) - imported,
    }


@datasets_router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: int,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    ds = await dataset_service.get_dataset(db, dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")
    return ds


@datasets_router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: int,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_admin),
):
    ok = await dataset_service.delete_dataset(db, dataset_id)
    if not ok:
        raise HTTPException(404)
    return {"message": "Deleted"}


@datasets_router.get("/{dataset_id}/stats")
async def dataset_stats(
    dataset_id: int,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    return await dataset_service.get_stats(db, dataset_id)


@datasets_router.post("/{dataset_id}/freeze", response_model=DatasetResponse)
async def freeze_dataset_version(
    dataset_id: int,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_operator),
):
    try:
        return await dataset_service.freeze_dataset(db, dataset_id)
    except ValueError as exc:
        raise HTTPException(404 if str(exc) == "Dataset not found" else 409, str(exc)) from exc


@datasets_router.post("/{dataset_id}/versions", response_model=DatasetResponse)
async def create_dataset_version(
    dataset_id: int,
    version: str = Query(..., min_length=1, max_length=20),
    db: AsyncSession = Depends(get_db),
    user=Depends(require_operator),
):
    try:
        return await dataset_service.create_version(db, dataset_id, version, user["email"])
    except ValueError as exc:
        raise HTTPException(404 if str(exc) == "Dataset not found" else 409, str(exc)) from exc


# ─── Images ───────────────────────────────────────────────────────
@datasets_router.post("/{dataset_id}/images")
async def upload_images(
    dataset_id: int,
    files: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
    user=Depends(require_operator),
):
    ALLOWED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    saved = []
    for f in files:
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in ALLOWED:
            continue
        data = await f.read()
        img = await dataset_service.save_uploaded_image(db, dataset_id, f.filename, data)
        saved.append({"id": img.id, "filename": img.filename})
    return {"uploaded": len(saved), "images": saved}


@datasets_router.get("/{dataset_id}/images", response_model=List[DatasetImageResponse])
async def list_images(
    dataset_id: int,
    split: Optional[str] = None,
    annotated: Optional[bool] = None,
    frame_status: Optional[str] = None,
    limit: int = Query(50, le=500),
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    q = select(DatasetImage).where(DatasetImage.dataset_id == dataset_id)
    if split:
        q = q.where(DatasetImage.split == split)
    if annotated is not None:
        q = q.where(DatasetImage.is_annotated == annotated)
    if frame_status:
        q = q.where(DatasetImage.frame_status == frame_status)
    q = q.order_by(DatasetImage.created_at).limit(limit).offset(offset)
    result = await db.execute(q)
    return result.scalars().all()


@datasets_router.get("/{dataset_id}/images/{image_id}/file")
async def get_image_file(
    dataset_id: int,
    image_id: int,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    result = await db.execute(
        select(DatasetImage).where(
            DatasetImage.id == image_id,
            DatasetImage.dataset_id == dataset_id,
        )
    )
    img = result.scalar_one_or_none()
    if not img or not os.path.exists(img.file_path):
        raise HTTPException(404)
    return FileResponse(img.file_path)


@datasets_router.patch("/{dataset_id}/images/{image_id}/status", response_model=DatasetImageResponse)
async def update_image_status(
    dataset_id: int,
    image_id: int,
    data: FrameStatusUpdate,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_operator),
):
    image = await db.get(DatasetImage, image_id)
    if not image or image.dataset_id != dataset_id:
        raise HTTPException(404, "Dataset image not found")
    try:
        return await lifecycle_service.update_frame_status(
            db,
            image_id=image_id,
            status=data.status.value,
            review_reason=data.review_reason,
            review_priority=data.review_priority,
            scene_tags=data.scene_tags,
            quality_tags=data.quality_tags,
        )
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


# ─── Videos ───────────────────────────────────────────────────────
@datasets_router.post("/{dataset_id}/videos")
async def upload_video(
    dataset_id: int,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    user=Depends(require_operator),
):
    ALLOWED = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"Unsupported video format: {ext}")
    data = await file.read()
    video = await dataset_service.save_uploaded_video(db, dataset_id, file.filename, data)
    return {"id": video.id, "filename": video.filename, "duration": video.duration_seconds}


@datasets_router.post("/{dataset_id}/videos/{video_id}/extract")
async def extract_frames(
    dataset_id: int,
    video_id: int,
    config: VideoExtractConfig,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_operator),
):
    dataset = await dataset_service.get_dataset(db, dataset_id)
    if not dataset:
        raise HTTPException(404, "Dataset not found")
    try:
        dataset_service.require_mutable(dataset)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    video = await db.get(DatasetVideo, video_id)
    if not video or video.dataset_id != dataset_id:
        raise HTTPException(404, "Dataset video not found")
    video.status = "queued"
    await db.commit()
    from app.celery_app import celery_app
    try:
        task = celery_app.send_task(
            "app.workers.video_worker.extract_frames_task",
            kwargs={
                "dataset_id": dataset_id,
                "video_id": video_id,
                "config": config.model_dump(),
            },
            queue="cpu",
        )
    except Exception:
        video.status = "uploaded"
        await db.commit()
        raise
    return {"task_id": task.id, "message": "Frame extraction queued"}


# ─── Split ────────────────────────────────────────────────────────
@datasets_router.post("/{dataset_id}/split")
async def apply_split(
    dataset_id: int,
    config: DatasetSplitConfig,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_operator),
):
    counts = await dataset_service.apply_split(db, dataset_id, config)
    return {"message": "Split applied", "counts": counts}


# ─── Export ───────────────────────────────────────────────────────
@datasets_router.post("/{dataset_id}/export/yolo")
async def export_yolo(
    dataset_id: int,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_operator),
):
    path = await dataset_service.export_yolo(db, dataset_id)
    return {"message": "Exported", "path": path}


# ─── Auto-annotate ────────────────────────────────────────────────
@datasets_router.post("/{dataset_id}/auto-annotate")
async def auto_annotate(
    dataset_id: int,
    config: AutoAnnotateConfig,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_operator),
):
    return await annotation_service.auto_annotate(db, dataset_id, config)


# ═══ ANNOTATIONS ═════════════════════════════════════════════════
annotations_router = APIRouter(prefix="/api/v1/training/annotations", tags=["training-annotations"])


@annotations_router.get("/image/{image_id}", response_model=List[AnnotationResponse])
async def get_annotations(
    image_id: int,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    return await annotation_service.get_image_annotations(db, image_id)


@annotations_router.post("/image/{image_id}", response_model=AnnotationResponse)
async def create_annotation(
    image_id: int,
    data: AnnotationCreate,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_operator),
):
    return await annotation_service.create_annotation(db, image_id, data)


@annotations_router.post("/image/{image_id}/bulk")
async def create_annotations_bulk(
    image_id: int,
    data: List[AnnotationCreate],
    db: AsyncSession = Depends(get_db),
    user=Depends(require_operator),
):
    anns = await annotation_service.create_annotations_bulk(db, image_id, data)
    return {"created": len(anns)}


@annotations_router.put("/{annotation_id}", response_model=AnnotationResponse)
async def update_annotation(
    annotation_id: int,
    data: AnnotationCreate,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_operator),
):
    ann = await annotation_service.update_annotation(db, annotation_id, data)
    if not ann:
        raise HTTPException(404)
    return ann


@annotations_router.delete("/{annotation_id}")
async def delete_annotation(
    annotation_id: int,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_operator),
):
    ok = await annotation_service.delete_annotation(db, annotation_id)
    if not ok:
        raise HTTPException(404)
    return {"message": "Deleted"}


@annotations_router.delete("/image/{image_id}/all")
async def clear_annotations(
    image_id: int,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_operator),
):
    count = await annotation_service.delete_image_annotations(db, image_id)
    return {"deleted": count}


@annotations_router.post("/image/{source_id}/copy-to/{target_id}")
async def copy_annotations(
    source_id: int,
    target_id: int,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_operator),
):
    count = await annotation_service.copy_annotations(db, source_id, target_id)
    return {"copied": count}


@annotations_router.get("/dataset/{dataset_id}/stats")
async def annotation_stats(
    dataset_id: int,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    return await annotation_service.get_annotation_stats(db, dataset_id)


# ═══ TRAINING JOBS ═══════════════════════════════════════════════
jobs_router = APIRouter(prefix="/api/v1/training/jobs", tags=["training-jobs"])


@jobs_router.get("", response_model=List[TrainingJobResponse])
async def list_jobs(
    status: Optional[str] = None,
    model_type: Optional[str] = None,
    limit: int = Query(50, le=200),
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    q = select(TrainingJob).order_by(desc(TrainingJob.created_at)).limit(limit)
    if status:
        q = q.where(TrainingJob.status == status)
    if model_type:
        q = q.where(TrainingJob.model_type == model_type)
    result = await db.execute(q)
    return result.scalars().all()


@jobs_router.post("", response_model=TrainingJobResponse)
async def create_job(
    data: TrainingJobCreate,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_operator),
):
    # Verify dataset exists
    ds = await dataset_service.get_dataset(db, data.dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")
    training_mode = data.training_mode.value
    if training_mode != "baseline_inference" and (not ds.is_frozen or not ds.content_hash):
        raise HTTPException(409, "Freeze the dataset version before starting a reproducible training job")
    model_type = data.model_type.value
    if str(ds.model_type) != model_type:
        raise HTTPException(400, "Dataset model type does not match the training job")

    allowed_architectures = {
        key: {item["id"] for item in values}
        for key, values in ARCHITECTURES.items()
    }
    if data.architecture not in allowed_architectures.get(model_type, set()):
        raise HTTPException(400, f"Unsupported architecture for {model_type}: {data.architecture}")
    base_model = await resolve_base_model(
        db,
        model_type=model_type,
        architecture=data.architecture,
        base_model_id=data.base_model_id,
    )

    # Check if split exists
    if training_mode != "baseline_inference":
        result = await db.execute(
            select(DatasetImage).where(
                DatasetImage.dataset_id == data.dataset_id,
                DatasetImage.split.isnot(None),
            ).limit(1)
        )
        if not result.scalar_one_or_none():
            raise HTTPException(400, "Dataset has no split. Apply train/val/test split first.")

    hyperparams = data.hyperparams.model_dump()
    hyperparams["base_model"] = base_model

    job = TrainingJob(
        name=data.name,
        model_type=model_type,
        architecture=data.architecture,
        dataset_id=data.dataset_id,
        training_mode=training_mode,
        hyperparams=hyperparams,
        augmentation_config=data.augmentation.model_dump(),
        tile_config=data.tile_config.model_dump(),
        evaluation_policy=data.evaluation_policy.model_dump(),
        total_epochs=data.hyperparams.epochs,
        status=JobStatus.QUEUED,
        author_email=user["email"],
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Enqueue Celery task
    from app.celery_app import celery_app
    task = celery_app.send_task(
        "app.workers.training_worker.run_training",
        kwargs={"job_id": job.id},
        queue="gpu",
    )
    job.celery_task_id = task.id
    await db.commit()

    logger.info(f"Training job {job.id} queued (task={task.id})")
    return job


@jobs_router.get("/{job_id}", response_model=TrainingJobResponse)
async def get_job(
    job_id: int,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    result = await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(404)
    return job


@jobs_router.post("/{job_id}/cancel")
async def cancel_job(
    job_id: int,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_operator),
):
    result = await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(404)

    if job.celery_task_id:
        from app.celery_app import celery_app
        celery_app.control.revoke(job.celery_task_id, terminate=True, signal="SIGTERM")

    job.status = JobStatus.CANCELLED
    job.finished_at = datetime.now(timezone.utc)
    await db.commit()
    return {"message": "Cancelled"}


@jobs_router.get("/{job_id}/metrics", response_model=List[TrainingMetricsResponse])
async def job_metrics(
    job_id: int,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    result = await db.execute(
        select(TrainingMetrics)
        .where(TrainingMetrics.job_id == job_id)
        .order_by(TrainingMetrics.epoch)
    )
    return result.scalars().all()


@jobs_router.get("/{job_id}/progress")
async def job_progress(
    job_id: int,
    user=Depends(get_current_user),
):
    """Get latest progress from Redis (fast path for polling)."""
    import redis as redis_lib
    from app.config import settings
    r = redis_lib.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    key = f"training:progress:{job_id}"
    data = r.get(key)
    if data:
        return json.loads(data)
    return {"job_id": job_id, "message": "No progress data yet"}


# ═══ MODEL REGISTRY ══════════════════════════════════════════════
registry_router = APIRouter(prefix="/api/v1/training/registry", tags=["training-registry"])


@registry_router.get("", response_model=List[ModelVersionResponse])
async def list_models(
    model_type: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    return await model_registry_service.list_versions(db, model_type)


@registry_router.get("/{mv_id}", response_model=ModelVersionResponse)
async def get_model(
    mv_id: int,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    mv = await model_registry_service.get_version(db, mv_id)
    if not mv:
        raise HTTPException(404)
    return mv


@registry_router.post("/{mv_id}/approve")
async def approve_deploy(
    mv_id: int,
    approval: DeployApproval,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_admin),
):
    return await model_registry_service.approve_and_deploy(db, mv_id, approval, user["email"])


@registry_router.post("/{mv_id}/reject")
async def reject_model(
    mv_id: int,
    comment: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_admin),
):
    return await model_registry_service.reject(db, mv_id, comment, user["email"])


@registry_router.post("/rollback")
async def rollback(
    request: RollbackRequest,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_admin),
):
    return await model_registry_service.rollback(db, request, user["email"])


@registry_router.post("/{mv_id}/export/{format}")
async def trigger_export(
    mv_id: int,
    format: str,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_operator),
):
    if format not in ("onnx", "tensorrt"):
        raise HTTPException(400, "Format must be 'onnx' or 'tensorrt'")
    return await model_registry_service.trigger_export(db, mv_id, format)


@registry_router.post("/{mv_id}/validate")
async def trigger_validation(
    mv_id: int,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_operator),
):
    return await model_registry_service.trigger_validation(db, mv_id)


@registry_router.get("/{mv_id}/deploy-logs")
async def deploy_logs(
    mv_id: int,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    logs = await model_registry_service.get_deploy_logs(db, mv_id)
    return [
        {
            "id": l.id, "action": l.action, "actor_email": l.actor_email,
            "comment": l.comment, "created_at": l.created_at.isoformat(),
        }
        for l in logs
    ]


# ═══ GPU / SYSTEM ════════════════════════════════════════════════
gpu_router = APIRouter(prefix="/api/v1/training/gpu", tags=["training-gpu"])


@gpu_router.get("")
async def gpu_info(user=Depends(get_current_user)):
    gpus, system = await asyncio.to_thread(
        lambda: (gpu_service.get_gpu_info(), gpu_service.get_system_stats())
    )
    return {
        "gpus": [g.model_dump() for g in gpus],
        "system": system,
    }


# ═══ ARCHITECTURES (for UI dropdown) ════════════════════════════
arch_router = APIRouter(prefix="/api/v1/training/architectures", tags=["training-arch"])


active_learning_router = APIRouter(prefix="/api/v1/training/active-learning", tags=["training-active-learning"])


@active_learning_router.get("", response_model=List[ActiveLearningItemResponse])
async def list_active_learning(
    dataset_id: Optional[int] = None,
    status: Optional[str] = None,
    reason: Optional[str] = None,
    limit: int = Query(100, le=500),
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    return await lifecycle_service.list_active_learning_items(
        db, dataset_id=dataset_id, status=status, reason=reason, limit=limit
    )


@active_learning_router.post("", response_model=ActiveLearningItemResponse)
async def create_active_learning_item(
    data: ActiveLearningCreate,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_operator),
):
    try:
        return await lifecycle_service.create_active_learning_item(
            db,
            dataset_id=data.dataset_id,
            image_id=data.image_id,
            reason=data.reason,
            priority_score=data.priority_score,
            suggested_class=data.suggested_class,
            source=data.source,
            model_version_id=data.model_version_id,
            evaluation_error_id=data.evaluation_error_id,
            details=data.details,
        )
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@active_learning_router.patch("/{item_id}/status", response_model=ActiveLearningItemResponse)
async def update_active_learning_status(
    item_id: int,
    data: ActiveLearningStatusUpdate,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_operator),
):
    try:
        return await lifecycle_service.update_active_learning_status(
            db, item_id, data.status.value, user["email"]
        )
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


evaluation_router = APIRouter(prefix="/api/v1/training/evaluation", tags=["training-evaluation"])


@evaluation_router.get("/reports", response_model=List[EvaluationReportResponse])
async def list_evaluation_reports(
    model_version_id: Optional[int] = None,
    limit: int = Query(100, le=500),
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    return await lifecycle_service.list_evaluation_reports(
        db, model_version_id=model_version_id, limit=limit
    )


@evaluation_router.get("/reports/{report_id}", response_model=EvaluationReportResponse)
async def get_evaluation_report(
    report_id: int,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    report = await db.get(EvaluationReport, report_id)
    if not report:
        raise HTTPException(404, "Evaluation report not found")
    return report


@evaluation_router.get("/errors", response_model=List[EvaluationErrorResponse])
async def list_evaluation_errors(
    report_id: Optional[int] = None,
    error_type: Optional[str] = None,
    limit: int = Query(200, le=1000),
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    return await lifecycle_service.list_evaluation_errors(
        db, report_id=report_id, error_type=error_type, limit=limit
    )

ARCHITECTURES = {
    "object_detector": [
        {"id": "yolo11n", "name": "YOLO11 Nano", "desc": "Fast generic detector baseline", "params": "2.6M"},
        {"id": "yolo11s", "name": "YOLO11 Small", "desc": "Balanced aerial detector", "params": "9.4M"},
        {"id": "yolo11m", "name": "YOLO11 Medium", "desc": "Higher-capacity aerial detector", "params": "20.1M"},
        {"id": "yolov8n", "name": "YOLOv8 Nano", "desc": "Stable fallback", "params": "3.2M"},
        {"id": "yolov8s", "name": "YOLOv8 Small", "desc": "Stable balanced fallback", "params": "11.2M"},
    ],
    "object_segmenter": [
        {"id": "yolo11n-seg", "name": "YOLO11 Nano Seg", "desc": "Fast generic instance segmentation", "params": "2.9M"},
        {"id": "yolo11s-seg", "name": "YOLO11 Small Seg", "desc": "Balanced generic mask quality", "params": "10.1M"},
    ],
    "object_classifier": [
        {"id": "mobilenetv3", "name": "MobileNetV3 Small", "desc": "Fast generic classifier", "params": "2.5M"},
        {"id": "efficientnet", "name": "EfficientNet-B0", "desc": "Accuracy-oriented classifier", "params": "5.3M"},
        {"id": "resnet18", "name": "ResNet-18", "desc": "Stable generic baseline", "params": "11.7M"},
    ],
}


async def list_base_models(db: AsyncSession, model_type: str) -> list[dict]:
    archs = ARCHITECTURES.get(model_type)
    if not archs:
        raise HTTPException(404, f"Unknown model type: {model_type}")

    candidates: list[dict] = []
    result = await db.execute(
        select(ModelVersion)
        .where(ModelVersion.model_type == model_type)
        .order_by(desc(ModelVersion.is_production), desc(ModelVersion.created_at))
        .limit(50)
    )
    for mv in result.scalars().all():
        if not mv.weights_path:
            continue
        path = Path(mv.weights_path)
        candidates.append({
            "id": f"registry:{mv.id}",
            "label": mv.name,
            "source": "registry",
            "architecture": mv.architecture,
            "version": mv.version,
            "model_version_id": mv.id,
            "path": mv.weights_path,
            "available": path.is_file(),
            "is_production": bool(mv.is_production),
            "metrics": mv.metrics or {},
            "training": (mv.artifact_metadata or {}).get("training", {}),
        })

    type_dir = MODEL_TYPE_DIRS.get(model_type, model_type)
    inference_dir = Path(settings.INFERENCE_MODELS_PATH) / type_dir
    for arch in archs:
        if model_type == "object_classifier":
            candidates.append({
                "id": f"torchvision:{arch['id']}",
                "label": f"{arch['name']} torchvision pretrained",
                "source": "torchvision",
                "architecture": arch["id"],
                "version": "pretrained",
                "path": None,
                "available": True,
                "is_production": False,
                "metrics": {},
                "training": {},
            })
            continue

        local_path = inference_dir / f"{arch['id']}.pt"
        if local_path.is_file():
            candidates.append({
                "id": f"installed:{arch['id']}",
                "label": f"{arch['name']} installed weights",
                "source": "installed",
                "architecture": arch["id"],
                "version": "local",
                "path": str(local_path),
                "available": True,
                "is_production": False,
                "metrics": {},
                "training": {},
            })
        candidates.append({
            "id": f"ultralytics:{arch['id']}",
            "label": f"{arch['name']} Ultralytics pretrained",
            "source": "ultralytics",
            "architecture": arch["id"],
            "version": "pretrained",
            "path": f"{arch['id']}.pt",
            "available": True,
            "is_production": False,
            "metrics": {},
            "training": {},
        })

    seen = set()
    unique = []
    for item in candidates:
        if item["id"] in seen:
            continue
        seen.add(item["id"])
        item["label"] = _base_model_display_name(item)
        unique.append(item)
    return unique


async def resolve_base_model(
    db: AsyncSession,
    *,
    model_type: str,
    architecture: str,
    base_model_id: Optional[str],
) -> dict:
    candidates = await list_base_models(db, model_type)
    requested = base_model_id or f"ultralytics:{architecture}"
    base_model = next((item for item in candidates if item["id"] == requested), None)
    if base_model is None:
        raise HTTPException(400, f"Unknown base model for {model_type}: {requested}")
    if base_model["architecture"] != architecture:
        raise HTTPException(
            400,
            f"Base model {requested} uses {base_model['architecture']}, not {architecture}",
        )
    if base_model["source"] == "registry" and not base_model["available"]:
        raise HTTPException(409, f"Selected registry weights are not available: {base_model['path']}")
    return base_model


@arch_router.get("/{model_type}")
async def get_architectures(model_type: str, user=Depends(get_current_user)):
    archs = ARCHITECTURES.get(model_type)
    if not archs:
        raise HTTPException(404, f"Unknown model type: {model_type}")
    return archs


@arch_router.get("/{model_type}/base-models")
async def get_base_models(
    model_type: str,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    return await list_base_models(db, model_type)
