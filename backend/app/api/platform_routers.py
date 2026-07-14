from typing import Any, Literal
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import EvaluationRun, ObjectTrack, PipelineDefinition, Project, get_db
from app.schemas.schemas import (
    EvaluationRunCreate, EvaluationRunResponse,
    PipelineDefinitionCreate, PipelineDefinitionResponse,
    ObjectTrackResponse, ProjectCreate, ProjectResponse, ProjectUpdate,
)
from app.services.evaluation_service import evaluation_service
from app.services.module_capability_service import module_capability_service
from app.utils.auth import require_admin, require_operator, require_viewer


projects_router = APIRouter(prefix="/api/v1/projects", tags=["projects"])
pipelines_router = APIRouter(prefix="/api/v1/pipelines", tags=["pipelines"])
evaluation_router = APIRouter(prefix="/api/v1/evaluations", tags=["evaluations"])
objects_router = APIRouter(prefix="/api/v1/objects", tags=["objects"])
modules_router = APIRouter(prefix="/api/v1/modules", tags=["modules"])


@modules_router.get("")
async def list_platform_modules():
    return module_capability_service.summary()


@objects_router.get("", response_model=list[ObjectTrackResponse])
async def list_object_tracks(
    source_id: int | None = Query(default=None),
    object_class: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
    user=Depends(require_viewer),
):
    query = select(ObjectTrack).order_by(ObjectTrack.last_seen.desc()).limit(limit)
    if source_id is not None:
        query = query.where(ObjectTrack.source_id == source_id)
    if object_class:
        query = query.where(ObjectTrack.object_class == object_class)
    result = await db.execute(query)
    return result.scalars().all()


@projects_router.get("", response_model=list[ProjectResponse])
async def list_projects(db: AsyncSession = Depends(get_db), user=Depends(require_viewer)):
    result = await db.execute(select(Project).order_by(Project.created_at.desc()))
    return result.scalars().all()


@projects_router.post("", response_model=ProjectResponse)
async def create_project(req: ProjectCreate, db: AsyncSession = Depends(get_db), user=Depends(require_operator)):
    existing = await db.scalar(select(Project).where(Project.slug == req.slug))
    if existing:
        raise HTTPException(409, "Project slug already exists")
    project = Project(**req.model_dump(mode="json"))
    db.add(project)
    await db.commit()
    await db.refresh(project)
    return project


@projects_router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(project_id: int, req: ProjectUpdate, db: AsyncSession = Depends(get_db), user=Depends(require_operator)):
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    for key, value in req.model_dump(exclude_none=True, mode="json").items():
        setattr(project, key, value)
    await db.commit()
    await db.refresh(project)
    return project


@pipelines_router.get("", response_model=list[PipelineDefinitionResponse])
async def list_pipelines(project_id: int | None = Query(default=None), db: AsyncSession = Depends(get_db), user=Depends(require_viewer)):
    query = select(PipelineDefinition).order_by(PipelineDefinition.updated_at.desc())
    if project_id is not None:
        query = query.where(PipelineDefinition.project_id == project_id)
    result = await db.execute(query)
    return result.scalars().all()


@pipelines_router.post("", response_model=PipelineDefinitionResponse)
async def create_pipeline(req: PipelineDefinitionCreate, db: AsyncSession = Depends(get_db), user=Depends(require_operator)):
    if not await db.get(Project, req.project_id):
        raise HTTPException(404, "Project not found")
    if req.is_default:
        result = await db.execute(select(PipelineDefinition).where(PipelineDefinition.project_id == req.project_id))
        for existing in result.scalars():
            existing.is_default = False
    latest_version = await db.scalar(select(func.max(PipelineDefinition.version)).where(
        PipelineDefinition.project_id == req.project_id,
        PipelineDefinition.name == req.name,
    ))
    pipeline = PipelineDefinition(**req.model_dump(mode="json"), version=int(latest_version or 0) + 1)
    db.add(pipeline)
    await db.commit()
    await db.refresh(pipeline)
    return pipeline


@evaluation_router.get("", response_model=list[EvaluationRunResponse])
async def list_evaluations(project_id: int | None = Query(default=None), db: AsyncSession = Depends(get_db), user=Depends(require_viewer)):
    query = select(EvaluationRun).order_by(EvaluationRun.created_at.desc())
    if project_id is not None:
        query = query.where(EvaluationRun.project_id == project_id)
    result = await db.execute(query)
    return result.scalars().all()


@evaluation_router.post("", response_model=EvaluationRunResponse)
async def create_evaluation(req: EvaluationRunCreate, db: AsyncSession = Depends(get_db), user=Depends(require_operator)):
    payload = req.model_dump()
    project_id = payload.get("project_id")
    if project_id is None:
        project_id = await _ensure_default_experiment_project(db)
    elif not await db.get(Project, project_id):
        raise HTTPException(404, "Project not found")
    payload["project_id"] = project_id
    run = EvaluationRun(**payload)
    db.add(run)
    await db.commit()
    await db.refresh(run)
    return run


class DetectionEvaluationPayload(BaseModel):
    frames: list[dict[str, Any]]
    iou_threshold: float = Field(default=0.5, ge=0.1, le=0.95)


@evaluation_router.post("/compute/detection")
async def compute_detection(payload: DetectionEvaluationPayload, user=Depends(require_operator)):
    return evaluation_service.evaluate_detection(payload.frames, payload.iou_threshold)


class TrackingEvaluationPayload(BaseModel):
    ground_truth_detections: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    id_switches: int = Field(ge=0)
    id_true_positives: int = Field(default=0, ge=0)
    id_false_positives: int = Field(default=0, ge=0)
    id_false_negatives: int = Field(default=0, ge=0)


@evaluation_router.post("/compute/tracking")
async def compute_tracking(payload: TrackingEvaluationPayload, user=Depends(require_operator)):
    return evaluation_service.evaluate_tracking(payload.model_dump())


class EvaluationExecutePayload(BaseModel):
    data: dict[str, Any]
    runtime: dict[str, float] = Field(default_factory=dict)


@evaluation_router.post("/{run_id}/execute", response_model=EvaluationRunResponse)
async def execute_evaluation(run_id: int, payload: EvaluationExecutePayload, db: AsyncSession = Depends(get_db), user=Depends(require_operator)):
    run = await db.get(EvaluationRun, run_id)
    if not run:
        raise HTTPException(404, "Evaluation run not found")
    run.status = "running"
    run.started_at = datetime.now(timezone.utc)
    run.error_message = None
    await db.flush()
    try:
        if run.task_type == "detection":
            metrics = evaluation_service.evaluate_detection_suite(list(payload.data.get("frames") or []))
        elif run.task_type == "tracking":
            metrics = evaluation_service.evaluate_tracking(payload.data)
        elif run.task_type == "classification":
            metrics = evaluation_service.evaluate_classification(list(payload.data.get("items") or []))
        elif run.task_type == "segmentation":
            metrics = evaluation_service.evaluate_segmentation(list(payload.data.get("items") or []))
        elif run.task_type in {"pipeline", "replay", "profiling"}:
            metrics = _operational_metrics(payload.data)
        else:
            raise ValueError(f"Unsupported task type: {run.task_type}")
        metrics["runtime"] = {
            key: round(float(value), 4)
            for key, value in payload.runtime.items()
            if key in {"fps", "latency_ms", "peak_memory_mb", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms"}
        }
        run.metrics = metrics
        run.status = "completed"
    except (TypeError, ValueError, KeyError) as exc:
        run.status = "failed"
        run.error_message = str(exc)
    run.finished_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(run)
    return run


async def _ensure_default_experiment_project(db: AsyncSession) -> int:
    project = await db.scalar(select(Project).where(Project.slug == "default-experiments"))
    if project:
        return project.id
    project = Project(
        name="Default Experiments",
        slug="default-experiments",
        description="Default project for saved experiment and profiling runs.",
        task_profile="aerial_small_objects",
        target_classes=[],
    )
    db.add(project)
    await db.flush()
    return project.id


def _operational_metrics(data: dict[str, Any]) -> dict[str, Any]:
    runtime = data.get("runtime") or {}
    stage_counts = data.get("stage_counts") or {}
    stage_latency_ms = data.get("stage_latency_ms") or {}
    warnings = list(data.get("warnings") or [])
    if not data.get("has_ground_truth", False):
        warnings.append("quality_metrics_unavailable_without_ground_truth")
    quality_metrics_available = bool(data.get("quality_metrics_available", False))
    return {
        "metric_source": "runtime",
        "has_ground_truth": bool(data.get("has_ground_truth", False)),
        "quality_metrics_available": quality_metrics_available,
        "frames_processed": int(stage_counts.get("frames_processed") or runtime.get("frames_processed") or 0),
        "objects_detected": int(stage_counts.get("objects_detected") or 0),
        "tracks_merged": int(stage_counts.get("object_memory_merged_tracks") or 0),
        "duplicates_suppressed": int(stage_counts.get("object_memory_suppressed_duplicates") or 0),
        "segments_detected": int(stage_counts.get("segments_detected") or 0),
        "objects_classified": int(stage_counts.get("objects_classified") or 0),
        "stage_latency_ms": stage_latency_ms,
        "stage_counts": stage_counts,
        "requested_pipeline": data.get("requested_pipeline") or {},
        "resolved_pipeline": data.get("resolved_pipeline") or {},
        "warnings": sorted(set(warnings)),
    }
