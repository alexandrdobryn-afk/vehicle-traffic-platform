import os
import shutil
import json
import random
import asyncio
import hashlib
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timezone
import logging
import cv2
import numpy as np

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, func

from app.models.database import Annotation, Dataset, DatasetImage, DatasetVideo, DatasetSplit
from app.schemas.schemas import DatasetCreate, DatasetSplitConfig, VideoExtractConfig
from app.config import settings

logger = logging.getLogger(__name__)


class DatasetService:
    """Manages dataset lifecycle: creation, uploads, splits, video extraction."""

    def __init__(self):
        self.datasets_root = Path(settings.DATASETS_PATH)
        self.uploads_root = Path(settings.UPLOADS_PATH)
        self.datasets_root.mkdir(parents=True, exist_ok=True)
        self.uploads_root.mkdir(parents=True, exist_ok=True)

    # ─── Dataset CRUD ─────────────────────────────────────────────

    async def create_dataset(
        self, db: AsyncSession, data: DatasetCreate, author_email: str
    ) -> Dataset:
        dataset = Dataset(
            name=data.name,
            description=data.description,
            model_type=data.model_type,
            annotation_type=data.annotation_type,
            classes=data.classes,
            tags=data.tags,
            author_email=author_email,
        )
        db.add(dataset)
        await db.commit()
        await db.refresh(dataset)

        # Create storage directory
        ds_path = self.datasets_root / str(dataset.id)
        (ds_path / "images").mkdir(parents=True, exist_ok=True)
        (ds_path / "labels").mkdir(parents=True, exist_ok=True)
        (ds_path / "videos").mkdir(parents=True, exist_ok=True)

        dataset.storage_path = str(ds_path)
        dataset.status = "ready"
        await db.commit()
        await db.refresh(dataset)
        return dataset

    async def get_dataset(self, db: AsyncSession, dataset_id: int) -> Optional[Dataset]:
        result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
        return result.scalar_one_or_none()

    @staticmethod
    def require_mutable(dataset: Dataset) -> None:
        if dataset.is_frozen:
            raise ValueError(
                "Dataset version is frozen. Create a new version before changing files, annotations or splits."
            )

    async def freeze_dataset(self, db: AsyncSession, dataset_id: int) -> Dataset:
        dataset = await self.get_dataset(db, dataset_id)
        if not dataset:
            raise ValueError("Dataset not found")
        if dataset.is_frozen and dataset.content_hash:
            return dataset

        active_video_jobs = await db.scalar(select(func.count(DatasetVideo.id)).where(
            DatasetVideo.dataset_id == dataset_id,
            DatasetVideo.status.in_(["queued", "extracting"]),
        ))
        if active_video_jobs:
            raise ValueError("Wait for queued video extraction jobs before freezing this dataset")

        digest = hashlib.sha256()
        digest.update(json.dumps({
            "model_type": str(dataset.model_type),
            "annotation_type": str(dataset.annotation_type),
            "classes": dataset.classes or [],
        }, sort_keys=True, separators=(",", ":")).encode())
        result = await db.execute(
            select(DatasetImage).where(DatasetImage.dataset_id == dataset_id).order_by(DatasetImage.id)
        )
        images = result.scalars().all()
        videos = (await db.execute(
            select(DatasetVideo).where(DatasetVideo.dataset_id == dataset_id).order_by(DatasetVideo.id)
        )).scalars().all()
        for video in videos:
            digest.update(f"video:{video.filename}".encode())
            path = Path(video.file_path)
            if path.is_file():
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
        for image in images:
            path = Path(image.file_path)
            digest.update(f"{image.filename}:{image.split or ''}".encode())
            if path.is_file():
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
            annotations = (await db.execute(
                select(Annotation).where(Annotation.image_id == image.id).order_by(Annotation.id)
            )).scalars().all()
            for annotation in annotations:
                digest.update(json.dumps({
                    "type": annotation.annotation_type, "class": annotation.class_name,
                    "class_id": annotation.class_id, "bbox": [annotation.x_center, annotation.y_center, annotation.bbox_width, annotation.bbox_height],
                    "polygon": annotation.polygon, "ocr": annotation.ocr_text, "label": annotation.label,
                }, sort_keys=True, separators=(",", ":")).encode())
                if annotation.mask_path and Path(annotation.mask_path).is_file():
                    with Path(annotation.mask_path).open("rb") as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(chunk)

        dataset.content_hash = digest.hexdigest()
        dataset.is_frozen = True
        dataset.frozen_at = datetime.now(timezone.utc)
        dataset.status = "frozen"
        dataset.lineage = {
            **(dataset.lineage or {}),
            "frozen_image_count": len(images),
            "frozen_video_count": len(videos),
            "hash_algorithm": "sha256",
        }
        await db.commit()
        await db.refresh(dataset)
        return dataset

    async def create_version(
        self,
        db: AsyncSession,
        dataset_id: int,
        version: str,
        author_email: str,
    ) -> Dataset:
        source = await self.get_dataset(db, dataset_id)
        if not source:
            raise ValueError("Dataset not found")
        if not source.is_frozen or not source.content_hash:
            raise ValueError("Freeze the source dataset before creating its next version")

        target = Dataset(
            name=source.name, description=source.description,
            model_type=source.model_type, annotation_type=source.annotation_type,
            classes=source.classes or [], tags=source.tags or [], status="draft",
            version=version, parent_dataset_id=source.id, author_email=author_email,
            lineage={"parent_dataset_id": source.id, "parent_content_hash": source.content_hash},
        )
        db.add(target)
        await db.flush()
        target_root = self.datasets_root / str(target.id)
        for directory in ("images", "labels", "videos", "masks"):
            (target_root / directory).mkdir(parents=True, exist_ok=True)
        target.storage_path = str(target_root)

        video_map: dict[int, DatasetVideo] = {}
        source_videos = (await db.execute(
            select(DatasetVideo).where(DatasetVideo.dataset_id == source.id).order_by(DatasetVideo.id)
        )).scalars().all()
        for video in source_videos:
            destination = target_root / "videos" / video.filename
            if Path(video.file_path).is_file():
                shutil.copy2(video.file_path, destination)
            clone_video = DatasetVideo(
                dataset_id=target.id, filename=video.filename, file_path=str(destination),
                file_size=video.file_size, fps=video.fps, total_frames=video.total_frames,
                duration_seconds=video.duration_seconds, extracted_frames=video.extracted_frames,
                status=video.status,
            )
            db.add(clone_video)
            await db.flush()
            video_map[video.id] = clone_video

        image_map: dict[int, DatasetImage] = {}
        source_images = (await db.execute(
            select(DatasetImage).where(DatasetImage.dataset_id == source.id).order_by(DatasetImage.id)
        )).scalars().all()
        for image in source_images:
            destination = target_root / "images" / image.filename
            if Path(image.file_path).is_file():
                shutil.copy2(image.file_path, destination)
            clone = DatasetImage(
                dataset_id=target.id, filename=image.filename,
                original_filename=image.original_filename, file_path=str(destination),
                file_size=image.file_size, width=image.width, height=image.height,
                source=image.source, source_frame_idx=image.source_frame_idx,
                source_video_id=(video_map[image.source_video_id].id if image.source_video_id in video_map else None),
                split=image.split, is_annotated=image.is_annotated,
            )
            db.add(clone)
            await db.flush()
            image_map[image.id] = clone

        annotations = (await db.execute(
            select(Annotation).join(DatasetImage).where(DatasetImage.dataset_id == source.id)
        )).scalars().all()
        for annotation in annotations:
            cloned_mask_path = None
            if annotation.mask_path and Path(annotation.mask_path).is_file():
                mask_name = f"annotation_{annotation.id}_{Path(annotation.mask_path).name}"
                mask_destination = target_root / "masks" / mask_name
                shutil.copy2(annotation.mask_path, mask_destination)
                cloned_mask_path = str(mask_destination)
            db.add(Annotation(
                image_id=image_map[annotation.image_id].id,
                annotation_type=annotation.annotation_type, class_name=annotation.class_name,
                class_id=annotation.class_id, x_center=annotation.x_center, y_center=annotation.y_center,
                bbox_width=annotation.bbox_width, bbox_height=annotation.bbox_height,
                polygon=annotation.polygon, mask_path=cloned_mask_path, provenance=annotation.provenance,
                ocr_text=annotation.ocr_text, label=annotation.label, confidence=annotation.confidence,
                is_auto=annotation.is_auto, is_verified=annotation.is_verified,
            ))
        splits = (await db.execute(select(DatasetSplit).where(DatasetSplit.dataset_id == source.id))).scalars().all()
        for split in splits:
            db.add(DatasetSplit(
                dataset_id=target.id, split_name=split.split_name,
                image_count=split.image_count, ratio=split.ratio, seed=split.seed,
            ))
        target.image_count = len(source_images)
        target.annotation_count = len(annotations)
        target.video_count = len(source_videos)
        await db.commit()
        await db.refresh(target)
        return target

    async def list_datasets(
        self, db: AsyncSession, model_type: Optional[str] = None
    ) -> List[Dataset]:
        q = select(Dataset).order_by(Dataset.created_at.desc())
        if model_type:
            q = q.where(Dataset.model_type == model_type)
        result = await db.execute(q)
        return result.scalars().all()

    async def delete_dataset(self, db: AsyncSession, dataset_id: int) -> bool:
        ds = await self.get_dataset(db, dataset_id)
        if not ds:
            return False
        if ds.storage_path and os.path.exists(ds.storage_path):
            shutil.rmtree(ds.storage_path, ignore_errors=True)
        await db.delete(ds)
        await db.commit()
        return True

    # ─── Image Upload ─────────────────────────────────────────────

    async def save_uploaded_image(
        self,
        db: AsyncSession,
        dataset_id: int,
        filename: str,
        file_bytes: bytes,
    ) -> DatasetImage:
        ds = await self.get_dataset(db, dataset_id)
        if not ds:
            raise ValueError(f"Dataset {dataset_id} not found")
        self.require_mutable(ds)

        images_dir = Path(ds.storage_path) / "images"
        images_dir.mkdir(exist_ok=True)

        # Sanitize filename and avoid collisions
        safe_name = Path(filename).name
        dest = images_dir / safe_name
        counter = 1
        while dest.exists():
            stem = Path(safe_name).stem
            suffix = Path(safe_name).suffix
            dest = images_dir / f"{stem}_{counter}{suffix}"
            counter += 1

        dest.write_bytes(file_bytes)

        # Get image dimensions
        w, h = self._get_image_dimensions(dest)

        img = DatasetImage(
            dataset_id=dataset_id,
            filename=dest.name,
            original_filename=filename,
            file_path=str(dest),
            file_size=len(file_bytes),
            width=w,
            height=h,
            source="upload",
        )
        db.add(img)

        # Update dataset counter
        await db.execute(
            update(Dataset)
            .where(Dataset.id == dataset_id)
            .values(image_count=Dataset.image_count + 1)
        )
        await db.commit()
        await db.refresh(img)
        return img

    # ─── Video Upload & Frame Extraction ──────────────────────────

    async def save_uploaded_video(
        self,
        db: AsyncSession,
        dataset_id: int,
        filename: str,
        file_bytes: bytes,
    ) -> DatasetVideo:
        ds = await self.get_dataset(db, dataset_id)
        if not ds:
            raise ValueError(f"Dataset {dataset_id} not found")
        self.require_mutable(ds)

        videos_dir = Path(ds.storage_path) / "videos"
        videos_dir.mkdir(exist_ok=True)
        dest = videos_dir / Path(filename).name
        dest.write_bytes(file_bytes)

        # Get video metadata
        cap = cv2.VideoCapture(str(dest))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        video = DatasetVideo(
            dataset_id=dataset_id,
            filename=dest.name,
            file_path=str(dest),
            file_size=len(file_bytes),
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration,
            status="uploaded",
        )
        db.add(video)
        await db.execute(
            update(Dataset)
            .where(Dataset.id == dataset_id)
            .values(video_count=Dataset.video_count + 1)
        )
        await db.commit()
        await db.refresh(video)
        return video

    async def extract_frames(
        self,
        db: AsyncSession,
        dataset_id: int,
        video_id: int,
        config: VideoExtractConfig,
    ) -> int:
        """Extract frames from video — runs in background thread."""
        ds = await self.get_dataset(db, dataset_id)
        result = await db.execute(
            select(DatasetVideo).where(DatasetVideo.id == video_id)
        )
        video = result.scalar_one_or_none()
        if not ds or not video:
            return 0
        self.require_mutable(ds)
        if video.dataset_id != dataset_id:
            raise ValueError("Video does not belong to this dataset")

        images_dir = Path(ds.storage_path) / "images"
        images_dir.mkdir(exist_ok=True)

        await db.execute(
            update(DatasetVideo)
            .where(DatasetVideo.id == video_id)
            .values(status="extracting")
        )
        await db.commit()

        # Run extraction in thread to not block event loop
        count = await asyncio.to_thread(
            self._extract_frames_sync,
            video.file_path,
            str(images_dir),
            video_id,
            config,
        )

        # Save extracted images to DB
        frame_files = sorted(images_dir.glob(f"frame_v{video_id}_*.jpg"))
        for ff in frame_files:
            w, h = self._get_image_dimensions(ff)
            img = DatasetImage(
                dataset_id=dataset_id,
                filename=ff.name,
                file_path=str(ff),
                file_size=ff.stat().st_size,
                width=w,
                height=h,
                source="video_frame",
                source_video_id=video_id,
                frame_status="unlabeled",
            )
            db.add(img)

        await db.execute(
            update(DatasetVideo)
            .where(DatasetVideo.id == video_id)
            .values(status="done", extracted_frames=count)
        )
        await db.execute(
            update(Dataset)
            .where(Dataset.id == dataset_id)
            .values(image_count=Dataset.image_count + count)
        )
        await db.commit()
        logger.info(f"Extracted {count} frames from video {video_id}")
        return count

    def _extract_frames_sync(
        self,
        video_path: str,
        output_dir: str,
        video_id: int,
        config: VideoExtractConfig,
    ) -> int:
        cap = cv2.VideoCapture(video_path)
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if config.interval_seconds is not None:
            frame_interval = max(1, int(native_fps * config.interval_seconds))
        else:
            frame_interval = max(1, int(native_fps / config.fps))
        start_frame = int(config.start_time * native_fps)
        end_frame = int(config.end_time * native_fps) if config.end_time else total

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        extracted = 0
        frame_idx = start_frame

        while cap.isOpened() and frame_idx < end_frame:
            if config.max_frames and extracted >= config.max_frames:
                break

            ret, frame = cap.read()
            if not ret:
                break

            if (frame_idx - start_frame) % frame_interval == 0:
                out_path = os.path.join(output_dir, f"frame_v{video_id}_{frame_idx:06d}.jpg")
                cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, config.quality])
                extracted += 1

            frame_idx += 1

        cap.release()
        return extracted

    # ─── Dataset Split ────────────────────────────────────────────

    async def apply_split(
        self,
        db: AsyncSession,
        dataset_id: int,
        config: DatasetSplitConfig,
    ) -> Dict:
        """Split dataset images into train/val/test."""
        dataset = await self.get_dataset(db, dataset_id)
        if not dataset:
            raise ValueError("Dataset not found")
        self.require_mutable(dataset)
        result = await db.execute(
            select(DatasetImage).where(DatasetImage.dataset_id == dataset_id)
        )
        images = result.scalars().all()

        if not images:
            return {"train": 0, "val": 0, "test": 0}

        total = len(images)
        rng = random.Random(config.seed)

        # Sequential frames from one video must stay in one split; otherwise
        # validation sees near-duplicates of training images and metrics lie.
        groups: Dict[str, List[DatasetImage]] = {}
        for image in images:
            key = (
                f"video:{image.source_video_id}"
                if image.source_video_id is not None
                else f"image:{image.id}"
            )
            groups.setdefault(key, []).append(image)
        group_items = list(groups.values())
        rng.shuffle(group_items)

        target_train = total * config.train_ratio
        target_val = total * config.val_ratio
        counts = {"train": 0, "val": 0, "test": 0}
        splits: Dict[int, str] = {}
        for group in group_items:
            if counts["train"] < target_train:
                split = "train"
            elif counts["val"] < target_val:
                split = "val"
            else:
                split = "test"
            for image in group:
                splits[image.id] = split
            counts[split] += len(group)

        # Bulk update
        for img in images:
            img.split = splits[img.id]
            if img.is_annotated and img.frame_status in ("reviewed", "approved", "training_ready"):
                img.frame_status = "training_ready"
        dataset.status = "ready_for_training" if any(img.is_annotated for img in images) else dataset.status
        await db.commit()

        # Update/create split records
        for split_name, count in counts.items():
            result = await db.execute(
                select(DatasetSplit).where(
                    DatasetSplit.dataset_id == dataset_id,
                    DatasetSplit.split_name == split_name,
                )
            )
            existing = result.scalar_one_or_none()
            if existing:
                existing.image_count = count
            else:
                db.add(DatasetSplit(
                    dataset_id=dataset_id,
                    split_name=split_name,
                    image_count=count,
                    ratio=getattr(config, f"{split_name}_ratio"),
                    seed=config.seed,
                ))
        await db.commit()
        return counts

    # ─── Export to YOLO format ────────────────────────────────────

    async def export_yolo(self, db: AsyncSession, dataset_id: int) -> str:
        """Export dataset to YOLO format for training."""
        ds = await self.get_dataset(db, dataset_id)
        if not ds:
            raise ValueError("Dataset not found")

        export_dir = Path(settings.EXPORTS_PATH) / f"dataset_{dataset_id}_yolo"
        for split in ["train", "val", "test"]:
            (export_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (export_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        result = await db.execute(
            select(DatasetImage).where(
                DatasetImage.dataset_id == dataset_id,
                DatasetImage.split.isnot(None),
            )
        )
        images = result.scalars().all()

        for img in images:
            split = img.split or "train"
            # Copy image
            dest_img = export_dir / split / "images" / img.filename
            if os.path.exists(img.file_path):
                shutil.copy2(img.file_path, dest_img)

            # Write label file
            ann_result = await db.execute(
                select(Annotation := __import__(
                    "app.models.database", fromlist=["Annotation"]
                ).Annotation).where(
                    Annotation.image_id == img.id,
                    Annotation.annotation_type == "bbox",
                )
            )
            annotations = ann_result.scalars().all()
            label_file = export_dir / split / "labels" / (Path(img.filename).stem + ".txt")
            with open(label_file, "w") as f:
                for ann in annotations:
                    cls_id = ann.class_id or 0
                    f.write(
                        f"{cls_id} {ann.x_center:.6f} {ann.y_center:.6f} "
                        f"{ann.bbox_width:.6f} {ann.bbox_height:.6f}\n"
                    )

        # Write data.yaml
        yaml_content = {
            "path": str(export_dir),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": len(ds.classes),
            "names": ds.classes,
        }
        import yaml
        with open(export_dir / "data.yaml", "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        logger.info(f"Dataset {dataset_id} exported to YOLO format: {export_dir}")
        return str(export_dir)

    # ─── Helpers ──────────────────────────────────────────────────

    def _get_image_dimensions(self, path) -> Tuple[Optional[int], Optional[int]]:
        try:
            img = cv2.imread(str(path))
            if img is not None:
                h, w = img.shape[:2]
                return w, h
        except Exception:
            pass
        return None, None

    async def get_stats(self, db: AsyncSession, dataset_id: int) -> dict:
        ds = await self.get_dataset(db, dataset_id)
        if not ds:
            return {}
        ann_count = await db.scalar(
            select(func.count()).select_from(
                __import__("app.models.database", fromlist=["Annotation"]).Annotation
            ).join(DatasetImage).where(DatasetImage.dataset_id == dataset_id)
        )
        return {
            "image_count": ds.image_count,
            "video_count": ds.video_count,
            "annotation_count": ann_count or 0,
            "classes": ds.classes,
            "frame_status_counts": {
                row[0] or "unlabeled": row[1]
                for row in (
                    await db.execute(
                        select(DatasetImage.frame_status, func.count(DatasetImage.id))
                        .where(DatasetImage.dataset_id == dataset_id)
                        .group_by(DatasetImage.frame_status)
                    )
                ).all()
            },
        }


from typing import Dict
dataset_service = DatasetService()
