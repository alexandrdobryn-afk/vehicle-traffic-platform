import os
import shutil
import json
import random
import asyncio
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import logging
import cv2
import numpy as np

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, func

from app.models.database import Dataset, DatasetImage, DatasetVideo, DatasetSplit
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
        return dataset

    async def get_dataset(self, db: AsyncSession, dataset_id: int) -> Optional[Dataset]:
        result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
        return result.scalar_one_or_none()

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
        }


from typing import Dict
dataset_service = DatasetService()
