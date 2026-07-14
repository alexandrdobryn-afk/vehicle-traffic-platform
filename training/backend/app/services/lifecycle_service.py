from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import (
    ActiveLearningItem,
    DatasetImage,
    EvaluationError,
    EvaluationReport,
    ModelVersion,
    TrainingJob,
)


class LifecycleService:
    async def update_frame_status(
        self,
        db: AsyncSession,
        image_id: int,
        status: str,
        review_reason: Optional[str] = None,
        review_priority: Optional[float] = None,
        scene_tags: Optional[list[str]] = None,
        quality_tags: Optional[list[str]] = None,
    ) -> DatasetImage:
        image = await db.get(DatasetImage, image_id)
        if not image:
            raise ValueError("Dataset image not found")
        image.frame_status = status
        if review_reason is not None:
            image.review_reason = review_reason
        if review_priority is not None:
            image.review_priority = review_priority
        if scene_tags is not None:
            image.scene_tags = scene_tags
        if quality_tags is not None:
            image.quality_tags = quality_tags
        await db.commit()
        await db.refresh(image)
        return image

    async def create_active_learning_item(
        self,
        db: AsyncSession,
        *,
        dataset_id: int,
        image_id: int,
        reason: str,
        priority_score: float = 10.0,
        suggested_class: Optional[str] = None,
        source: str = "manual",
        model_version_id: Optional[int] = None,
        evaluation_error_id: Optional[int] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> ActiveLearningItem:
        image = await db.get(DatasetImage, image_id)
        if not image or image.dataset_id != dataset_id:
            raise ValueError("Dataset image not found")

        item = ActiveLearningItem(
            dataset_id=dataset_id,
            image_id=image_id,
            model_version_id=model_version_id,
            evaluation_error_id=evaluation_error_id,
            reason=reason,
            priority_score=priority_score,
            suggested_class=suggested_class,
            source=source,
            details=details or {},
        )
        db.add(item)

        image.frame_status = "needs_review" if reason != "hard_negative" else "hard_negative"
        image.review_reason = reason
        image.review_priority = max(float(image.review_priority or 0), float(priority_score))
        await db.commit()
        await db.refresh(item)
        return item

    async def list_active_learning_items(
        self,
        db: AsyncSession,
        *,
        dataset_id: Optional[int] = None,
        status: Optional[str] = None,
        reason: Optional[str] = None,
        limit: int = 100,
    ) -> list[ActiveLearningItem]:
        query = select(ActiveLearningItem).order_by(
            ActiveLearningItem.priority_score.desc(),
            ActiveLearningItem.created_at.desc(),
        ).limit(limit)
        if dataset_id is not None:
            query = query.where(ActiveLearningItem.dataset_id == dataset_id)
        if status:
            query = query.where(ActiveLearningItem.status == status)
        if reason:
            query = query.where(ActiveLearningItem.reason == reason)
        result = await db.execute(query)
        return list(result.scalars().all())

    async def update_active_learning_status(
        self,
        db: AsyncSession,
        item_id: int,
        status: str,
        reviewer_email: Optional[str],
    ) -> ActiveLearningItem:
        item = await db.get(ActiveLearningItem, item_id)
        if not item:
            raise ValueError("Active learning item not found")
        item.status = status
        item.reviewer_email = reviewer_email
        image = await db.get(DatasetImage, item.image_id)
        if image and status in {"annotated", "approved"}:
            image.frame_status = "reviewed" if status == "annotated" else "approved"
        elif image and status == "rejected":
            image.frame_status = "rejected"
        await db.commit()
        await db.refresh(item)
        return item

    async def list_evaluation_reports(
        self,
        db: AsyncSession,
        *,
        model_version_id: Optional[int] = None,
        limit: int = 100,
    ) -> list[EvaluationReport]:
        query = select(EvaluationReport).order_by(EvaluationReport.created_at.desc()).limit(limit)
        if model_version_id is not None:
            query = query.where(EvaluationReport.model_version_id == model_version_id)
        result = await db.execute(query)
        return list(result.scalars().all())

    async def list_evaluation_errors(
        self,
        db: AsyncSession,
        *,
        report_id: Optional[int] = None,
        error_type: Optional[str] = None,
        limit: int = 200,
    ) -> list[EvaluationError]:
        query = select(EvaluationError).order_by(
            EvaluationError.priority_score.desc(),
            EvaluationError.created_at.desc(),
        ).limit(limit)
        if report_id is not None:
            query = query.where(EvaluationError.report_id == report_id)
        if error_type:
            query = query.where(EvaluationError.error_type == error_type)
        result = await db.execute(query)
        return list(result.scalars().all())

    def evaluate_gate(
        self,
        metrics: dict[str, Any],
        benchmark: dict[str, Any],
        policy: Optional[dict[str, Any]] = None,
    ) -> tuple[str, list[str]]:
        policy = policy or {}
        reasons: list[str] = []
        blockers: list[str] = []

        def number(name: str, default: float = 0.0) -> float:
            value = metrics.get(name, default)
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        if benchmark.get("is_first"):
            reasons.append("first production candidate: no previous model baseline")
        else:
            if benchmark.get("is_improvement") is not True:
                blockers.append("primary metric did not improve against production")

            recall_delta = benchmark.get("recall_delta")
            if recall_delta is not None and float(recall_delta) < -float(policy.get("max_recall_small_drop", 0.0)):
                blockers.append("recall regressed beyond policy")

            map_delta = benchmark.get("map50_95_delta")
            if map_delta is not None and float(map_delta) < -float(policy.get("max_old_holdout_drop", 0.02)):
                blockers.append("holdout mAP50-95 regressed beyond policy")

        ap_small_delta = metrics.get("ap_small_delta")
        if ap_small_delta is not None and float(ap_small_delta) < float(policy.get("min_ap_small_delta", 0.05)):
            blockers.append("AP-small improvement is below policy")

        recall_small_delta = metrics.get("recall_small_delta")
        if recall_small_delta is not None and float(recall_small_delta) < -float(policy.get("max_recall_small_drop", 0.0)):
            blockers.append("Recall-small dropped below policy")

        min_recall_small = policy.get("min_recall_small")
        if min_recall_small is not None and metrics.get("recall_small") is not None:
            if float(metrics.get("recall_small") or 0.0) < float(min_recall_small):
                blockers.append("Recall-small is below policy")

        fp_ratio = metrics.get("fp_per_frame_increase_ratio")
        if fp_ratio is not None and float(fp_ratio) > float(policy.get("max_fp_per_frame_increase_ratio", 0.10)):
            blockers.append("false positives per frame increased beyond policy")

        max_fp_per_frame = policy.get("max_fp_per_frame")
        if max_fp_per_frame is not None and metrics.get("fp_per_frame") is not None:
            if float(metrics.get("fp_per_frame") or 0.0) > float(max_fp_per_frame):
                blockers.append("false positives per frame are above policy")

        min_fps = float(policy.get("min_fps", 0.0) or 0.0)
        if min_fps and number("fps") < min_fps:
            blockers.append("FPS is below policy")

        max_latency = policy.get("max_latency_p95_ms")
        if max_latency is not None and number("latency_p95_ms") > float(max_latency):
            blockers.append("P95 latency is above policy")

        if blockers:
            return "candidate", blockers
        return "approved", reasons or ["all configured quality gates passed"]

    def build_report_for_model(self, db, mv: ModelVersion, val_metrics: dict[str, Any], benchmark: dict[str, Any]) -> EvaluationReport:
        job = db.query(TrainingJob).filter(TrainingJob.id == mv.job_id).first() if mv.job_id else None
        policy = (job.evaluation_policy or {}) if job else {}
        gate_result, gate_reasons = self.evaluate_gate(val_metrics, benchmark, policy)

        summary = {
            "map50": val_metrics.get("map50"),
            "map50_95": val_metrics.get("map50_95"),
            "precision": val_metrics.get("precision"),
            "recall": val_metrics.get("recall"),
            "accuracy": val_metrics.get("accuracy"),
            "fitness": val_metrics.get("fitness"),
            "samples": val_metrics.get("samples"),
            "ap_small": val_metrics.get("ap_small"),
            "recall_small": val_metrics.get("recall_small"),
            "fp_per_frame": val_metrics.get("fp_per_frame"),
            "fn_per_frame": val_metrics.get("fn_per_frame"),
            "false_positives": val_metrics.get("false_positives"),
            "false_negatives": val_metrics.get("false_negatives"),
            "wrong_class": val_metrics.get("wrong_class"),
        }
        speed_metrics = {
            key: val_metrics.get(key)
            for key in ("fps", "latency_p50_ms", "latency_p95_ms", "gpu_memory_mb", "model_size_mb")
            if key in val_metrics
        }
        report = EvaluationReport(
            model_version_id=mv.id,
            dataset_id=mv.dataset_id,
            job_id=mv.job_id,
            summary={k: v for k, v in summary.items() if v is not None},
            per_class_metrics=val_metrics.get("per_class", {}),
            slice_metrics=val_metrics.get("slices", {}),
            speed_metrics=speed_metrics,
            confusion_matrix=val_metrics.get("confusion_matrix"),
            gate_result=gate_result,
            gate_reasons=gate_reasons,
        )
        db.add(report)
        db.flush()

        for error in val_metrics.get("errors", []) or []:
            evaluation_error = EvaluationError(
                report_id=report.id,
                dataset_image_id=error.get("dataset_image_id"),
                error_type=error.get("error_type", "needs_annotation"),
                class_name=error.get("class_name"),
                confidence=error.get("confidence"),
                priority_score=float(error.get("priority_score", 0.0) or 0.0),
                bbox=error.get("bbox"),
                details=error.get("details", {}),
            )
            db.add(evaluation_error)
            db.flush()

            if error.get("dataset_image_id") and mv.dataset_id:
                db.add(ActiveLearningItem(
                    dataset_id=mv.dataset_id,
                    image_id=error["dataset_image_id"],
                    model_version_id=mv.id,
                    evaluation_error_id=evaluation_error.id,
                    reason=error.get("error_type", "needs_annotation"),
                    priority_score=float(error.get("priority_score", 0.0) or 0.0),
                    suggested_class=error.get("class_name"),
                    source="evaluation",
                    details={
                        "report_id": report.id,
                        "bbox": error.get("bbox"),
                        **(error.get("details", {}) or {}),
                    },
                ))
                image = db.query(DatasetImage).filter(DatasetImage.id == error["dataset_image_id"]).first()
                if image:
                    image.frame_status = "needs_review"
                    image.review_reason = error.get("error_type", "needs_annotation")
                    image.review_priority = max(
                        float(image.review_priority or 0),
                        float(error.get("priority_score", 0.0) or 0.0),
                    )

        mv.gate_result = gate_result
        mv.gate_reasons = gate_reasons
        mv.evaluation_report_id = report.id
        mv.deploy_status = "candidate" if gate_result in {"approved", "candidate"} else "rejected"
        mv.validation_passed = gate_result in {"approved", "candidate"}
        return report


lifecycle_service = LifecycleService()
