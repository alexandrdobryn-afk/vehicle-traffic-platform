"""
Training Worker — runs inside the GPU container.
All heavy computation happens here, never in the FastAPI process.
"""
import os
import json
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import redis
from celery import Task

from app.celery_app import celery_app
from app.config import settings
from app.utils.geometry import (
    annotation_to_abs_box,
    clip_box_to_tile,
    tile_origins,
    tile_stride,
)

logger = logging.getLogger(__name__)

# Redis client for realtime progress push
_redis = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

PROGRESS_KEY = "training:progress:{job_id}"
PROGRESS_TTL = 3600 * 24  # 24h


def push_progress(job_id: int, data: dict):
    """Push training progress to Redis for WebSocket pickup."""
    key = PROGRESS_KEY.format(job_id=job_id)
    _redis.setex(key, PROGRESS_TTL, json.dumps(data))
    # Also publish to channel for live WebSocket
    _redis.publish(f"training:job:{job_id}", json.dumps(data))


def get_db_session():
    """Sync DB session for Celery workers."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    engine = create_engine(settings.DATABASE_SYNC_URL)
    Session = sessionmaker(bind=engine)
    return Session()


# в”Ђв”Ђв”Ђ Main Training Task в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

@celery_app.task(
    bind=True,
    name="app.workers.training_worker.run_training",
    max_retries=0,
)
def run_training(self: Task, job_id: int):
    """
    Main training task. Dispatches to the correct trainer
    based on model_type.
    """
    db = get_db_session()
    try:
        from app.models.database import TrainingJob, JobStatus
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if not job:
            logger.error(f"Job {job_id} not found")
            return

        # Update status
        job.status = JobStatus.PREPARING
        job.celery_task_id = self.request.id
        job.started_at = datetime.now(timezone.utc)
        db.commit()

        push_progress(job_id, {
            "job_id": job_id,
            "status": "preparing",
            "message": "Preparing training environment...",
            "progress_pct": 0,
        })

        params = job.hyperparams or {}
        model_type = job.model_type

        if job.training_mode == "baseline_inference":
            _run_baseline_inference(job, db, params)
        elif model_type in ("object_detector", "object_segmenter"):
            _train_yolo(job, db, params)
        elif model_type == "object_classifier":
            _train_object_classifier(job, db, params)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    except Exception as e:
        logger.error(f"Training job {job_id} failed: {e}", exc_info=True)
        from app.models.database import TrainingJob, JobStatus
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if job:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.finished_at = datetime.now(timezone.utc)
            db.commit()
        push_progress(job_id, {
            "job_id": job_id,
            "status": "failed",
            "message": str(e),
            "progress_pct": 0,
        })
        raise
    finally:
        db.close()


# в”Ђв”Ђв”Ђ Baseline inference в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

def _run_baseline_inference(job, db, params: dict):
    """Run a ready detector over dataset frames and store reviewable predictions."""
    from ultralytics import YOLO
    from app.models.database import Annotation, DatasetImage, JobStatus, TrainingJob

    job_id = job.id
    confidence_threshold = float(params.get("confidence_threshold", 0.35))
    model_dir = Path(settings.INFERENCE_MODELS_PATH) / "object_detector"
    candidates = [
        model_dir / "production.pt",
        model_dir / "yolo11s.pt",
        model_dir / "yolo11n.pt",
    ]
    model_path = next((path for path in candidates if path.is_file()), None)
    if model_path is None:
        raise FileNotFoundError("No baseline object detector is installed")

    model = YOLO(str(model_path))
    images = db.query(DatasetImage).filter(DatasetImage.dataset_id == job.dataset_id).all()
    total = len(images)
    predictions = 0
    reviewed = 0

    job_rec = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    job_rec.status = JobStatus.TRAINING
    job_rec.total_epochs = 1
    db.commit()

    push_progress(job_id, {
        "job_id": job_id,
        "status": "training",
        "message": f"Running baseline inference over {total} frames...",
        "progress_pct": 5,
    })

    for index, image in enumerate(images, start=1):
        if not os.path.exists(image.file_path):
            continue
        db.query(Annotation).filter(
            Annotation.image_id == image.id,
            Annotation.is_auto == True,
            Annotation.is_verified == False,
        ).delete()
        results = model(image.file_path, conf=confidence_threshold, verbose=False)
        image_predictions = 0
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = str(result.names.get(cls_id, "unknown")).lower()
                cx, cy, w, h = box.xywhn[0].tolist()
                db.add(Annotation(
                    image_id=image.id,
                    annotation_type="bbox",
                    class_name=cls_name,
                    class_id=cls_id,
                    x_center=float(cx),
                    y_center=float(cy),
                    bbox_width=float(w),
                    bbox_height=float(h),
                    confidence=float(box.conf[0]),
                    is_auto=True,
                    is_verified=False,
                    provenance={
                        "source": "baseline_inference",
                        "model_path": str(model_path),
                        "training_job_id": job_id,
                    },
                ))
                image_predictions += 1

        predictions += image_predictions
        image.is_annotated = image_predictions > 0
        image.frame_status = "auto_labeled" if image_predictions > 0 else "needs_review"
        image.review_reason = "baseline_prediction" if image_predictions > 0 else "baseline_no_detection"
        image.review_priority = 30.0 if image_predictions > 0 else 50.0
        reviewed += 1

        if index % 10 == 0 or index == total:
            progress = 5 + (index / max(total, 1)) * 90
            job_rec.current_epoch = 1
            job_rec.progress_pct = progress
            db.commit()
            push_progress(job_id, {
                "job_id": job_id,
                "status": "training",
                "progress_pct": round(progress, 1),
                "latest_metrics": {"predictions": predictions, "frames_processed": reviewed},
                "message": f"Baseline inference: {index}/{total} frames",
            })

    _finalize_job(db, job_id, {
        "mode": "baseline_inference",
        "frames_processed": reviewed,
        "predictions": predictions,
        "confidence_threshold": confidence_threshold,
    })


# в”Ђв”Ђв”Ђ YOLO Training в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

def _model_type_dir(model_type: str) -> str:
    return {
        "object_detector": "object_detector",
        "object_segmenter": "object_segmenter",
        "object_classifier": "object_classifier",
    }.get(model_type, model_type)


def _resolve_yolo_model_path(job, arch: str, pretrained: bool, db) -> str:
    base_model = (job.hyperparams or {}).get("base_model") or {}
    source = base_model.get("source")
    path = base_model.get("path")

    if source == "registry":
        from app.models.database import ModelVersion
        model_version_id = base_model.get("model_version_id")
        mv = db.query(ModelVersion).filter(ModelVersion.id == model_version_id).first()
        if not mv or not mv.weights_path or not Path(mv.weights_path).is_file():
            raise FileNotFoundError(f"Selected registry base model is unavailable: {model_version_id}")
        return mv.weights_path

    if source == "installed" and path:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Selected installed base model is unavailable: {path}")
        return str(path)

    if source == "ultralytics" and path:
        return str(path)

    if pretrained:
        model_dir = Path(settings.INFERENCE_MODELS_PATH) / _model_type_dir(job.model_type)
        local_model = model_dir / f"{arch}.pt"
        return str(local_model) if local_model.exists() else f"{arch}.pt"
    return f"{arch}.yaml"


def _train_yolo(job, db, params: dict):
    from ultralytics import YOLO
    from app.models.database import TrainingJob, TrainingMetrics, JobStatus
    from app.services.dataset_service import DatasetService
    import yaml

    job_id = job.id
    arch = job.architecture      # yolo11n, yolo11s, yolov8n, etc.
    epochs = params.get("epochs", 100)
    batch = params.get("batch_size", 16)
    tile_config = job.tile_config or {}
    img_size = params.get("img_size", 640)
    use_tiled_export = job.training_mode == "tiled_training" and tile_config.get("enabled", True)
    if use_tiled_export:
        img_size = int(tile_config.get("tile_size", img_size))
    elif job.training_mode == "hard_negative_training":
        tile_config = {
            **tile_config,
            "include_empty_tiles": True,
            "max_empty_tile_ratio": max(float(tile_config.get("max_empty_tile_ratio", 0.25) or 0.25), 0.75),
        }
    lr = params.get("learning_rate", 0.01)
    device = params.get("device", "auto")
    pretrained = params.get("pretrained", True)

    # Export dataset to YOLO format
    push_progress(job_id, {
        "job_id": job_id, "status": "preparing",
        "message": "Exporting dataset to YOLO format...", "progress_pct": 2,
    })

    dataset = db.query(
        __import__("app.models.database", fromlist=["Dataset"]).Dataset
    ).filter_by(id=job.dataset_id).first()

    if use_tiled_export:
        tile_size_name = int(tile_config.get("tile_size", img_size))
        overlap_name = str(tile_config.get("overlap", 0.2)).replace(".", "p")
        export_dir = Path(settings.EXPORTS_PATH) / f"dataset_{job.dataset_id}_yolo_tiled_{tile_size_name}_{overlap_name}"
    elif job.training_mode == "hard_negative_training":
        export_dir = Path(settings.EXPORTS_PATH) / f"dataset_{job.dataset_id}_yolo_hard_negative"
    else:
        export_dir = Path(settings.EXPORTS_PATH) / f"dataset_{job.dataset_id}_yolo"
    data_yaml = str(export_dir / "data.yaml")

    if not os.path.exists(data_yaml):
        # Synchronous export
        _export_dataset_sync(
            db,
            job.dataset_id,
            export_dir,
            dataset,
            tile_config=tile_config if use_tiled_export else None,
            training_mode=job.training_mode,
        )

    model_path = _resolve_yolo_model_path(job, arch, pretrained, db)
    logger.info("Loading YOLO model for job %s from %s", job_id, model_path)
    model = YOLO(model_path)

    # Output directory
    run_dir = Path(settings.LOGS_PATH) / f"job_{job_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    job.status = JobStatus.TRAINING
    job.total_epochs = epochs
    job.log_path = str(run_dir)
    db.commit()

    push_progress(job_id, {
        "job_id": job_id, "status": "training",
        "message": f"Training {arch} for {epochs} epochs...", "progress_pct": 5,
    })

    # Custom callback to push metrics
    epoch_results = []

    def on_epoch_end(trainer):
        ep = trainer.epoch + 1
        metrics = trainer.metrics or {}
        loss_items = trainer.loss_items

        train_loss = float(sum(loss_items)) if loss_items is not None else None
        is_segmenter = job.model_type.endswith("_segmenter")
        metric_suffix = "M" if is_segmenter else "B"
        val_loss = metrics.get("val/seg_loss" if is_segmenter else "val/box_loss")
        precision = metrics.get(f"metrics/precision({metric_suffix})", 0)
        recall = metrics.get(f"metrics/recall({metric_suffix})", 0)
        map50 = metrics.get(f"metrics/mAP50({metric_suffix})", 0)
        map50_95 = metrics.get(f"metrics/mAP50-95({metric_suffix})", 0)
        lr_val = trainer.optimizer.param_groups[0]["lr"] if trainer.optimizer else None

        gpu_mem, gpu_util = _get_gpu_stats()

        progress_pct = 5 + (ep / epochs) * 90
        eta = int((epochs - ep) * (time.time() - job.started_at.timestamp()) / ep) if ep > 0 else None

        # Save to DB
        m = TrainingMetrics(
            job_id=job_id,
            epoch=ep,
            train_loss=train_loss,
            val_loss=float(val_loss) if val_loss else None,
            precision=float(precision),
            recall=float(recall),
            map50=float(map50),
            map50_95=float(map50_95),
            lr=float(lr_val) if lr_val else None,
            gpu_memory_mb=gpu_mem,
            gpu_utilization=gpu_util,
        )
        db.add(m)

        # Update job progress
        job_update = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if job_update:
            job_update.current_epoch = ep
            job_update.progress_pct = progress_pct
            job_update.eta_seconds = eta
            if map50 > (job_update.best_metrics or {}).get("map50", 0):
                job_update.best_metrics = {
                    "map50": float(map50), "map50_95": float(map50_95),
                    "precision": float(precision), "recall": float(recall),
                    "epoch": ep,
                }
        db.commit()

        push_progress(job_id, {
            "job_id": job_id,
            "status": "training",
            "current_epoch": ep,
            "total_epochs": epochs,
            "progress_pct": round(progress_pct, 1),
            "eta_seconds": eta,
            "latest_metrics": {
                "train_loss": train_loss,
                "val_loss": float(val_loss) if val_loss else None,
                "precision": float(precision),
                "recall": float(recall),
                "map50": float(map50),
                "map50_95": float(map50_95),
            },
            "gpu_utilization": gpu_util,
            "gpu_memory_mb": gpu_mem,
            "message": f"Epoch {ep}/{epochs} — mAP50: {map50:.4f}",
        })
        epoch_results.append({"epoch": ep, "map50": float(map50)})

    model.add_callback("on_train_epoch_end", on_epoch_end)

    # Augmentation config
    aug = job.augmentation_config or {}
    augmentation_enabled = aug.get("enabled", True)
    train_kwargs = dict(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=img_size,
        lr0=lr,
        device=device if device != "auto" else (0 if _has_gpu() else "cpu"),
        project=str(run_dir),
        name="train",
        exist_ok=True,
        pretrained=pretrained,
        optimizer=params.get("optimizer", "SGD"),
        warmup_epochs=params.get("warmup_epochs", 3),
        patience=params.get("patience", 50),
        workers=params.get("workers", 4),
        half=params.get("half", False),
        # Augmentation
        degrees=aug.get("rotation", 10) if augmentation_enabled else 0.0,
        scale=(aug.get("scale_max", 1.2) - 1.0) if augmentation_enabled else 0.0,
        flipud=0.0,
        fliplr=aug.get("horizontal_flip", 0.5) if augmentation_enabled else 0.0,
        mosaic=aug.get("mosaic", 0.5) if augmentation_enabled else 0.0,
        mixup=aug.get("mixup", 0.1) if augmentation_enabled else 0.0,
        hsv_h=aug.get("hue", 0.05) if augmentation_enabled else 0.0,
        hsv_s=aug.get("saturation", 0.2) if augmentation_enabled else 0.0,
        hsv_v=aug.get("brightness", 0.2) if augmentation_enabled else 0.0,
    )
    if job.training_mode == "head_finetune":
        train_kwargs["freeze"] = int(params.get("freeze_backbone_layers", 10))
    elif job.training_mode == "semi_supervised":
        train_kwargs["close_mosaic"] = max(1, int(epochs * 0.2))

    results = model.train(**train_kwargs)

    # Save best weights to model registry
    best_weights = str(run_dir / "train" / "weights" / "best.pt")
    _register_trained_model(db, job, best_weights, results)


# в”Ђв”Ђв”Ђ Color Classifier Training в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

def _train_object_classifier(job, db, params: dict):
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    from torch.utils.data import DataLoader, Dataset as TorchDataset
    from PIL import Image
    from app.models.database import TrainingJob, TrainingMetrics, JobStatus

    job_id = job.id
    epochs = params.get("epochs", 50)
    batch = params.get("batch_size", 32)
    lr = params.get("learning_rate", 1e-4)
    arch = job.architecture   # mobilenetv3, efficientnet, resnet18

    dataset = db.query(
        __import__("app.models.database", fromlist=["Dataset"]).Dataset
    ).filter_by(id=job.dataset_id).first()
    if not dataset:
        raise ValueError("Dataset not found")
    classes = dataset.classes or []
    n_classes = len(classes)
    if n_classes < 2:
        raise ValueError("Classification requires at least two classes")

    run_dir = Path(settings.LOGS_PATH) / f"job_{job_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    if arch == "mobilenetv3":
        model = models.mobilenet_v3_small(pretrained=True)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, n_classes)
    elif arch == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    else:
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)

    if job.training_mode == "head_finetune":
        for param in model.parameters():
            param.requires_grad = False
        if arch == "resnet18":
            for param in model.fc.parameters():
                param.requires_grad = True
        else:
            for param in model.classifier.parameters():
                param.requires_grad = True

    model = model.to(device)

    # Dataset
    class ClassificationDS(TorchDataset):
        def __init__(self, records, transform):
            self.records = records
            self.transform = transform
            self.class_to_idx = {c: i for i, c in enumerate(classes)}

        def __len__(self): return len(self.records)

        def __getitem__(self, idx):
            img_rec, label_str = self.records[idx]
            img = Image.open(img_rec.file_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, self.class_to_idx[label_str]

    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    from app.models.database import Annotation, DatasetImage
    train_imgs = db.query(DatasetImage).filter_by(
        dataset_id=job.dataset_id, split="train"
    ).all()
    val_imgs = db.query(DatasetImage).filter_by(
        dataset_id=job.dataset_id, split="val"
    ).all()

    def labeled_records(images):
        records = []
        for image in images:
            annotation = (
                db.query(Annotation)
                .filter(
                    Annotation.image_id == image.id,
                    Annotation.annotation_type == "classification",
                    Annotation.is_verified == True,
                )
                .first()
            )
            if annotation and annotation.label in classes and os.path.exists(image.file_path):
                records.append((image, annotation.label))
        return records

    train_records = labeled_records(train_imgs)
    val_records = labeled_records(val_imgs)
    if not train_records or not val_records:
        raise ValueError(
            "Classification training requires verified labels in train and val splits"
        )

    train_loader = DataLoader(ClassificationDS(train_records, train_tf), batch_size=batch, shuffle=True, num_workers=2)
    val_loader = DataLoader(ClassificationDS(val_records, val_tf), batch_size=batch, shuffle=False, num_workers=2)

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    class_counts = [sum(label == class_name for _, label in train_records) for class_name in classes]
    class_weights = torch.tensor(
        [len(train_records) / (n_classes * max(count, 1)) for count in class_counts],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    job_rec = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    job_rec.status = JobStatus.TRAINING
    job_rec.total_epochs = epochs
    db.commit()

    best_acc = 0.0
    best_path = str(run_dir / "best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                _, pred = out.max(1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

        acc = correct / total if total else 0
        avg_loss = train_loss / len(train_loader)
        gpu_mem, gpu_util = _get_gpu_stats()
        progress_pct = 5 + (epoch / epochs) * 90

        m = TrainingMetrics(
            job_id=job_id, epoch=epoch,
            train_loss=avg_loss, accuracy=acc,
            gpu_memory_mb=gpu_mem, gpu_utilization=gpu_util,
            lr=scheduler.get_last_lr()[0],
        )
        db.add(m)

        j = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        j.current_epoch = epoch
        j.progress_pct = progress_pct
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_path)
            j.best_metrics = {"accuracy": acc, "epoch": epoch}
        db.commit()

        push_progress(job_id, {
            "job_id": job_id, "status": "training",
            "current_epoch": epoch, "total_epochs": epochs,
            "progress_pct": round(progress_pct, 1),
            "latest_metrics": {"train_loss": avg_loss, "accuracy": acc},
            "gpu_utilization": gpu_util, "gpu_memory_mb": gpu_mem,
            "message": f"Epoch {epoch}/{epochs} — accuracy: {acc:.4f}",
        })

    # Export ONNX
    _export_to_onnx_classification(model, best_path, run_dir, device, arch)
    (run_dir / "labels.json").write_text(json.dumps({"classes": classes}, indent=2), encoding="utf-8")
    _register_trained_model(db, job, best_path, {"accuracy": best_acc})

@celery_app.task(name="app.workers.training_worker.auto_annotate_task")
def auto_annotate_task(
    dataset_id: int,
    model_type: str,
    image_ids: list,
    model_version_id: Optional[int],
    confidence_threshold: float,
    overwrite: bool,
):
    """Run inference on images and save auto-annotations to DB."""
    from ultralytics import YOLO
    from app.models.database import DatasetImage, Annotation, ModelVersion
    import cv2

    db = get_db_session()
    try:
        # Load model
        if model_version_id:
            mv = db.query(ModelVersion).filter(ModelVersion.id == model_version_id).first()
            model_path = mv.weights_path if mv else None
        else:
            candidates = {
                "object_detector": [
                    Path(settings.INFERENCE_MODELS_PATH) / "object_detector" / "production.pt",
                    Path(settings.INFERENCE_MODELS_PATH) / "object_detector" / "yolo11s.pt",
                    Path(settings.INFERENCE_MODELS_PATH) / "object_detector" / "yolo11n.pt",
                ],
            }.get(model_type, [])
            model_path = next((str(path) for path in candidates if path.is_file()), None)

        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"No usable {model_type} model for auto-annotation")

        model = YOLO(model_path)
        annotated = 0

        for img_id in image_ids:
            img_rec = db.query(DatasetImage).filter(DatasetImage.id == img_id).first()
            if not img_rec or not os.path.exists(img_rec.file_path):
                continue

            if overwrite:
                db.query(Annotation).filter(Annotation.image_id == img_id).delete()

            results = model(img_rec.file_path, conf=confidence_threshold, verbose=False)
            image_annotation_count = 0
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = result.names.get(cls_id, "unknown")
                    cx, cy, w, h = box.xywhn[0].tolist()
                    conf = float(box.conf[0])
                    ann = Annotation(
                        image_id=img_id,
                        annotation_type="bbox",
                        class_name=cls_name,
                        class_id=cls_id,
                        x_center=cx, y_center=cy,
                        bbox_width=w, bbox_height=h,
                        confidence=conf,
                        is_auto=True,
                        is_verified=False,
                    )
                    db.add(ann)
                    annotated += 1
                    image_annotation_count += 1

            img_rec.is_annotated = image_annotation_count > 0
            img_rec.frame_status = "auto_labeled" if image_annotation_count > 0 else "unlabeled"
            img_rec.review_reason = "auto_annotation" if image_annotation_count > 0 else "no_predictions"
            img_rec.review_priority = 25.0 if image_annotation_count > 0 else 5.0
            db.commit()

        logger.info(f"Auto-annotated {annotated} boxes across {len(image_ids)} images")
        return {"annotated": annotated}
    finally:
        db.close()


# в”Ђв”Ђв”Ђ Helpers в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

def _get_gpu_stats():
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            g = gpus[0]
            return g.memoryUsed, g.load * 100
    except Exception:
        pass
    return None, None


def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _finalize_job(db, job_id: int, metrics: dict):
    from app.models.database import TrainingJob, JobStatus
    j = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if j:
        j.status = JobStatus.COMPLETED
        j.finished_at = datetime.now(timezone.utc)
        j.progress_pct = 100.0
        j.final_metrics = metrics
        db.commit()
    push_progress(job_id, {
        "job_id": job_id, "status": "completed",
        "progress_pct": 100, "message": "Training complete!",
    })


def _register_trained_model(db, job, weights_path: str, results):
    """Save trained model to model registry."""
    from app.models.database import TrainingJob, ModelVersion, JobStatus, DeployStatus
    import hashlib

    weights = Path(weights_path)
    if not weights.is_file():
        raise FileNotFoundError(f"Training completed without weights: {weights}")

    onnx_candidates = [
        weights.parent / "lprnet.onnx",
        weights.parent / f"{job.architecture}.onnx",
        weights.parent.parent / f"{job.architecture}.onnx",
    ]
    onnx_path = next((str(path) for path in onnx_candidates if path.is_file()), None)

    def artifact_info(path: str):
        artifact = Path(path)
        digest = hashlib.sha256()
        with artifact.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return {
            "path": str(artifact),
            "sha256": digest.hexdigest(),
            "size_bytes": artifact.stat().st_size,
        }

    job_id = job.id

    # Determine version number
    existing = db.query(ModelVersion).filter_by(model_type=job.model_type).count()
    version_num = existing + 1
    version_str = f"v{version_num}.0"
    job_record = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    epochs_completed = int(
        (job_record.current_epoch if job_record else 0)
        or (job.hyperparams or {}).get("epochs")
        or job.total_epochs
        or 0
    )
    artifact_dir = (
        Path(settings.TRAINED_MODELS_PATH)
        / job.model_type
        / f"{job.architecture}_{version_str}_job_{job_id}"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    saved_weights = artifact_dir / f"epoch_{epochs_completed}_best{weights.suffix}"
    shutil.copy2(weights, saved_weights)
    weights_path = str(saved_weights)

    if onnx_path:
        source_onnx = Path(onnx_path)
        saved_onnx = artifact_dir / source_onnx.name
        shutil.copy2(source_onnx, saved_onnx)
        onnx_path = str(saved_onnx)

    base_model = (job.hyperparams or {}).get("base_model") or {}
    parent_epochs = 0
    parent_model_id = base_model.get("model_version_id")
    if parent_model_id:
        parent = db.query(ModelVersion).filter(ModelVersion.id == parent_model_id).first()
        parent_training = (parent.artifact_metadata or {}).get("training", {}) if parent else {}
        parent_epochs = int(
            parent_training.get("cumulative_epochs")
            or parent_training.get("epochs_completed")
            or 0
        )
    cumulative_epochs = parent_epochs + epochs_completed

    # Get metrics
    if isinstance(results, dict):
        metrics = results
    else:
        try:
            metric_suffix = "M" if job.model_type.endswith("_segmenter") else "B"
            metrics = {
                "map50": float(results.results_dict.get(f"metrics/mAP50({metric_suffix})", 0)),
                "map50_95": float(results.results_dict.get(f"metrics/mAP50-95({metric_suffix})", 0)),
                "precision": float(results.results_dict.get(f"metrics/precision({metric_suffix})", 0)),
                "recall": float(results.results_dict.get(f"metrics/recall({metric_suffix})", 0)),
                "metric_type": "mask" if metric_suffix == "M" else "box",
            }
        except Exception:
            metrics = {}

    metrics = {**metrics, "epoch": epochs_completed, "cumulative_epochs": cumulative_epochs}

    mv = ModelVersion(
        name=f"{job.name} - {version_str}",
        model_type=job.model_type,
        architecture=job.architecture,
        version=version_str,
        version_number=version_num,
        job_id=job_id,
        dataset_id=job.dataset_id,
        dataset_name=job.dataset.name if job.dataset else None,
        weights_path=weights_path,
        onnx_path=onnx_path,
        metrics=metrics,
        hyperparams=job.hyperparams,
        artifact_metadata={
            "source": "training_job",
            "training_job_id": job_id,
            "dataset_id": job.dataset_id,
            "classes": (job.dataset.classes or []) if job.dataset else [],
            "base_model": base_model,
            "training": {
                "mode": job.training_mode,
                "epochs_completed": epochs_completed,
                "parent_epochs": parent_epochs,
                "cumulative_epochs": cumulative_epochs,
                "backbone": {
                    "trainable": job.training_mode != "head_finetune",
                    "freeze_layers": (job.hyperparams or {}).get("freeze_backbone_layers", 10)
                    if job.training_mode == "head_finetune"
                    else 0,
                },
                "head": {
                    "trainable": True,
                    "architecture": job.architecture,
                },
                "tile_config": job.tile_config or {},
                "evaluation_policy": job.evaluation_policy or {},
            },
            "license": (
                "AGPL-3.0-or-Ultralytics-Enterprise"
                if job.model_type in ("object_detector", "object_segmenter")
                else "project-training-output"
            ),
            "weights": artifact_info(weights_path),
            "onnx": artifact_info(onnx_path) if onnx_path else None,
        },
        deploy_status=DeployStatus.PENDING,
        author_email=job.author_email,
    )
    db.add(mv)
    db.commit()
    db.refresh(mv)

    # Update job
    j = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    j.output_model_id = mv.id
    if (job.evaluation_policy or {}).get("auto_validate_after_training", True):
        try:
            celery_app.send_task(
                "app.workers.validation_worker.run_validation",
                kwargs={"model_version_id": mv.id},
                queue="gpu",
            )
            metrics = {**metrics, "validation_queued": True}
        except Exception as exc:
            logger.warning("Could not queue validation for model %s: %s", mv.id, exc)
            metrics = {**metrics, "validation_queued": False, "validation_queue_error": str(exc)}
    _finalize_job(db, job_id, metrics)

    logger.info(f"Model registered: {mv.name} (id={mv.id})")
    return mv


def _export_to_onnx_classification(model, pt_path: str, run_dir: Path, device, arch: str):
    """Export PyTorch classification model to ONNX."""
    import torch
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)
    onnx_path = str(run_dir / f"{arch}.onnx")
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=11,
    )
    logger.info(f"ONNX exported: {onnx_path}")
    return onnx_path


def _export_dataset_sync(db, dataset_id, export_dir, dataset, tile_config: Optional[dict] = None, training_mode: str = "full_finetune"):
    """Synchronous YOLO dataset export for use inside Celery worker."""
    import shutil
    from PIL import Image
    from app.models.database import DatasetImage, Annotation

    for split in ["train", "val", "test"]:
        (export_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (export_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    images = db.query(DatasetImage).filter(
        DatasetImage.dataset_id == dataset_id,
        DatasetImage.split.isnot(None),
    ).all()

    use_tiling = bool(tile_config and tile_config.get("enabled", True) and str(dataset.annotation_type) == "bbox")
    tile_size = int((tile_config or {}).get("tile_size", 1024))
    overlap = float((tile_config or {}).get("overlap", 0.2))
    min_visibility = float((tile_config or {}).get("min_visibility", 0.2))
    include_empty_tiles = bool((tile_config or {}).get("include_empty_tiles", False))
    max_empty_tile_ratio = float((tile_config or {}).get("max_empty_tile_ratio", 0.25))
    manifest = {
        "dataset_id": dataset_id,
        "tiled": use_tiling,
        "tile_config": tile_config or {},
        "images": 0,
        "tiles": 0,
        "labels": 0,
        "empty_tiles": 0,
    }

    for img in images:
        split = img.split or "train"
        if not os.path.exists(img.file_path):
            continue

        annotation_type = "segmentation" if str(dataset.annotation_type) == "segmentation" else "bbox"
        anns = db.query(Annotation).filter(
            Annotation.image_id == img.id,
            Annotation.annotation_type == annotation_type,
            Annotation.is_verified == True,
        ).all()
        if training_mode == "hard_negative_training" and img.frame_status == "hard_negative":
            anns = []
        manifest["images"] += 1

        if not use_tiling:
            dest = export_dir / split / "images" / img.filename
            shutil.copy2(img.file_path, dest)
            label_file = export_dir / split / "labels" / (Path(img.filename).stem + ".txt")
            with open(label_file, "w") as f:
                for ann in anns:
                    if annotation_type == "segmentation":
                        polygon = ann.polygon or []
                        if len(polygon) < 3:
                            continue
                        coordinates = " ".join(
                            f"{float(point[0]):.6f} {float(point[1]):.6f}" for point in polygon
                        )
                        f.write(f"{ann.class_id or 0} {coordinates}\n")
                        manifest["labels"] += 1
                    else:
                        f.write(
                            f"{ann.class_id or 0} {ann.x_center:.6f} {ann.y_center:.6f} "
                            f"{ann.bbox_width:.6f} {ann.bbox_height:.6f}\n"
                        )
                        manifest["labels"] += 1
            continue

        with Image.open(img.file_path) as source_image:
            source_image = source_image.convert("RGB")
            width, height = source_image.size
            stride = tile_stride(tile_size, overlap)
            image_tile_origins = tile_origins(width, height, tile_size, stride)
            empty_tiles_for_image = 0
            empty_tile_ratio = max(0.0, min(max_empty_tile_ratio, 1.0))
            max_empty_tiles_for_image = int(len(image_tile_origins) * empty_tile_ratio)
            if include_empty_tiles and empty_tile_ratio > 0 and max_empty_tiles_for_image == 0:
                max_empty_tiles_for_image = 1
            abs_boxes = [annotation_to_abs_box(ann, width, height) for ann in anns]
            abs_boxes = [box for box in abs_boxes if box is not None]

            for tile_index, (left, top, right, bottom) in enumerate(image_tile_origins):
                tile_labels = []
                for ann, box in abs_boxes:
                    clipped = clip_box_to_tile(box, (left, top, right, bottom), min_visibility)
                    if clipped is None:
                        continue
                    x1, y1, x2, y2 = clipped
                    tw = right - left
                    th = bottom - top
                    cx = ((x1 + x2) / 2 - left) / tw
                    cy = ((y1 + y2) / 2 - top) / th
                    bw = (x2 - x1) / tw
                    bh = (y2 - y1) / th
                    tile_labels.append(
                        f"{ann.class_id or 0} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                    )

                if not tile_labels:
                    if not include_empty_tiles or empty_tiles_for_image >= max_empty_tiles_for_image:
                        continue
                    empty_tiles_for_image += 1

                tile_name = f"{Path(img.filename).stem}_tile_{tile_index:05d}_{left}_{top}.jpg"
                tile_path = export_dir / split / "images" / tile_name
                label_file = export_dir / split / "labels" / f"{Path(tile_name).stem}.txt"
                source_image.crop((left, top, right, bottom)).save(tile_path, quality=95)
                label_file.write_text("\n".join(tile_labels) + ("\n" if tile_labels else ""), encoding="utf-8")
                manifest["tiles"] += 1
                manifest["labels"] += len(tile_labels)
                if not tile_labels:
                    manifest["empty_tiles"] += 1

    import yaml
    with open(export_dir / "data.yaml", "w") as f:
        yaml.dump({
            "path": str(export_dir),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": len(dataset.classes or []),
            "names": dataset.classes or [],
        }, f)
    (export_dir / "tile_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
