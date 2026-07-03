import logging
from typing import List, Optional
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, update

from app.models.database import Annotation, DatasetImage, Dataset, ModelVersion
from app.schemas.schemas import AnnotationCreate, AutoAnnotateConfig
from app.config import settings

logger = logging.getLogger(__name__)


class AnnotationService:

    async def get_image_annotations(
        self, db: AsyncSession, image_id: int
    ) -> List[Annotation]:
        result = await db.execute(
            select(Annotation).where(Annotation.image_id == image_id)
        )
        return result.scalars().all()

    async def create_annotation(
        self, db: AsyncSession, image_id: int, data: AnnotationCreate
    ) -> Annotation:
        ann = Annotation(
            image_id=image_id,
            annotation_type=data.annotation_type,
            class_name=data.class_name,
            class_id=data.class_id,
            x_center=data.x_center,
            y_center=data.y_center,
            bbox_width=data.bbox_width,
            bbox_height=data.bbox_height,
            ocr_text=data.ocr_text,
            label=data.label,
            confidence=data.confidence,
            is_auto=data.is_auto,
            **({"is_verified": data.is_verified} if data.is_verified is not None else {}),
        )
        db.add(ann)

        # Mark image as annotated
        await db.execute(
            update(DatasetImage)
            .where(DatasetImage.id == image_id)
            .values(is_annotated=True)
        )
        await db.commit()
        await db.refresh(ann)
        return ann

    async def create_annotations_bulk(
        self, db: AsyncSession, image_id: int, annotations: List[AnnotationCreate]
    ) -> List[Annotation]:
        results = []
        for data in annotations:
            ann = Annotation(
                image_id=image_id,
                annotation_type=data.annotation_type,
                class_name=data.class_name,
                class_id=data.class_id,
                x_center=data.x_center,
                y_center=data.y_center,
                bbox_width=data.bbox_width,
                bbox_height=data.bbox_height,
                ocr_text=data.ocr_text,
                label=data.label,
                confidence=data.confidence,
                is_auto=data.is_auto,
                **({"is_verified": data.is_verified} if data.is_verified is not None else {}),
            )
            db.add(ann)
            results.append(ann)

        await db.execute(
            update(DatasetImage)
            .where(DatasetImage.id == image_id)
            .values(is_annotated=True)
        )
        await db.commit()
        return results

    async def update_annotation(
        self, db: AsyncSession, annotation_id: int, data: AnnotationCreate
    ) -> Optional[Annotation]:
        result = await db.execute(
            select(Annotation).where(Annotation.id == annotation_id)
        )
        ann = result.scalar_one_or_none()
        if not ann:
            return None

        for field, val in data.model_dump(exclude_none=True).items():
            setattr(ann, field, val)
        await db.commit()
        await db.refresh(ann)
        return ann

    async def delete_annotation(self, db: AsyncSession, annotation_id: int) -> bool:
        result = await db.execute(
            select(Annotation).where(Annotation.id == annotation_id)
        )
        ann = result.scalar_one_or_none()
        if not ann:
            return False
        await db.delete(ann)
        await db.commit()
        return True

    async def delete_image_annotations(self, db: AsyncSession, image_id: int) -> int:
        result = await db.execute(
            select(Annotation).where(Annotation.image_id == image_id)
        )
        anns = result.scalars().all()
        count = len(anns)
        for a in anns:
            await db.delete(a)
        await db.execute(
            update(DatasetImage)
            .where(DatasetImage.id == image_id)
            .values(is_annotated=False)
        )
        await db.commit()
        return count

    async def copy_annotations(
        self,
        db: AsyncSession,
        source_image_id: int,
        target_image_id: int,
    ) -> int:
        """Copy all annotations from one image to another."""
        source_anns = await self.get_image_annotations(db, source_image_id)
        for ann in source_anns:
            new_ann = Annotation(
                image_id=target_image_id,
                annotation_type=ann.annotation_type,
                class_name=ann.class_name,
                class_id=ann.class_id,
                x_center=ann.x_center,
                y_center=ann.y_center,
                bbox_width=ann.bbox_width,
                bbox_height=ann.bbox_height,
                ocr_text=ann.ocr_text,
                label=ann.label,
            )
            db.add(new_ann)
        if source_anns:
            await db.execute(
                update(DatasetImage)
                .where(DatasetImage.id == target_image_id)
                .values(is_annotated=True)
            )
        await db.commit()
        return len(source_anns)

    async def auto_annotate(
        self,
        db: AsyncSession,
        dataset_id: int,
        config: AutoAnnotateConfig,
    ) -> dict:
        """
        Run auto-annotation using existing production model.
        Queues a Celery task that runs inference and saves annotations.
        """
        from app.celery_app import celery_app

        dataset = await db.get(Dataset, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        if dataset.model_type not in ("vehicle_detector", "plate_detector"):
            raise HTTPException(status_code=400, detail="Auto-annotation supports detector datasets only")

        if config.model_version_id:
            model_version = await db.get(ModelVersion, config.model_version_id)
            if not model_version:
                raise HTTPException(status_code=404, detail="Model version not found")
            if model_version.model_type != dataset.model_type:
                raise HTTPException(status_code=400, detail="Model type does not match dataset type")

        # Get images to annotate
        q = select(DatasetImage).where(DatasetImage.dataset_id == dataset_id)
        if config.image_ids:
            q = q.where(DatasetImage.id.in_(config.image_ids))
        if not config.overwrite_existing:
            q = q.where(DatasetImage.is_annotated == False)

        result = await db.execute(q)
        images = result.scalars().all()

        if not images:
            return {"queued": 0, "message": "No images to annotate"}

        image_ids = [img.id for img in images]

        task = celery_app.send_task(
            "app.workers.training_worker.auto_annotate_task",
            kwargs={
                "dataset_id": dataset_id,
                "model_type": dataset.model_type,
                "image_ids": image_ids,
                "model_version_id": config.model_version_id,
                "confidence_threshold": config.confidence_threshold,
                "overwrite": config.overwrite_existing,
            },
            queue="gpu",
        )

        return {
            "queued": len(image_ids),
            "task_id": task.id,
            "message": f"Auto-annotation queued for {len(image_ids)} images",
        }

    async def get_annotation_stats(
        self, db: AsyncSession, dataset_id: int
    ) -> dict:
        from sqlalchemy import func as sqlfunc
        result = await db.execute(
            select(
                Annotation.class_name,
                sqlfunc.count(Annotation.id).label("count"),
            )
            .join(DatasetImage)
            .where(DatasetImage.dataset_id == dataset_id)
            .group_by(Annotation.class_name)
        )
        rows = result.fetchall()

        ann_img_count = await db.scalar(
            select(sqlfunc.count(DatasetImage.id)).where(
                DatasetImage.dataset_id == dataset_id,
                DatasetImage.is_annotated == True,
            )
        )
        total_img = await db.scalar(
            select(sqlfunc.count(DatasetImage.id)).where(
                DatasetImage.dataset_id == dataset_id
            )
        )

        return {
            "annotated_images": ann_img_count or 0,
            "total_images": total_img or 0,
            "annotation_rate": (
                round(ann_img_count / total_img * 100, 1)
                if total_img else 0
            ),
            "class_distribution": {r.class_name or "unknown": r.count for r in rows},
        }


annotation_service = AnnotationService()
