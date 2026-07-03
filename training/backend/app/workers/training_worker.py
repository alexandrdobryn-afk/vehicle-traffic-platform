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


# ─── Main Training Task ───────────────────────────────────────────

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

        # Dispatch to correct trainer
        if model_type in ("vehicle_detector", "plate_detector"):
            _train_yolo(job, db, params)
        elif model_type == "color_classifier":
            _train_color_classifier(job, db, params)
        elif model_type == "ocr":
            _train_ocr(job, db, params)
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


# ─── YOLO Training ────────────────────────────────────────────────

def _train_yolo(job, db, params: dict):
    from ultralytics import YOLO
    from app.models.database import TrainingJob, TrainingMetrics, JobStatus
    from app.services.dataset_service import DatasetService
    import yaml

    job_id = job.id
    arch = job.architecture      # yolo11n, yolo11s, yolov8n, etc.
    epochs = params.get("epochs", 100)
    batch = params.get("batch_size", 16)
    img_size = params.get("img_size", 640)
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

    export_dir = Path(settings.EXPORTS_PATH) / f"dataset_{job.dataset_id}_yolo"
    data_yaml = str(export_dir / "data.yaml")

    if not os.path.exists(data_yaml):
        # Synchronous export
        _export_dataset_sync(db, job.dataset_id, export_dir, dataset)

    # Load model
    model_path = f"{arch}.pt" if pretrained else f"{arch}.yaml"
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
        val_loss = metrics.get("val/box_loss")
        precision = metrics.get("metrics/precision(B)", 0)
        recall = metrics.get("metrics/recall(B)", 0)
        map50 = metrics.get("metrics/mAP50(B)", 0)
        map50_95 = metrics.get("metrics/mAP50-95(B)", 0)
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
    results = model.train(
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
        degrees=aug.get("rotation", 10),
        scale=aug.get("scale_max", 1.2) - 1.0,
        flipud=0.0,
        fliplr=aug.get("horizontal_flip", 0.5),
        mosaic=aug.get("mosaic", 0.5),
        mixup=aug.get("mixup", 0.1),
        hsv_h=aug.get("hue", 0.05),
        hsv_s=aug.get("saturation", 0.2),
        hsv_v=aug.get("brightness", 0.2),
        blur=aug.get("blur", 0.1),
    )

    # Save best weights to model registry
    best_weights = str(run_dir / "train" / "weights" / "best.pt")
    _register_trained_model(db, job, best_weights, results)


# ─── Color Classifier Training ────────────────────────────────────

def _train_color_classifier(job, db, params: dict):
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
        raise ValueError("Color classification requires at least two classes")

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

    model = model.to(device)

    # Dataset
    class ColorDS(TorchDataset):
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
            "Color training requires verified classification labels in train and val splits"
        )

    train_loader = DataLoader(ColorDS(train_records, train_tf), batch_size=batch, shuffle=True, num_workers=2)
    val_loader = DataLoader(ColorDS(val_records, val_tf), batch_size=batch, shuffle=False, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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
    _register_trained_model(db, job, best_path, {"accuracy": best_acc})


# ─── OCR Training ─────────────────────────────────────────────────

def _train_ocr(job, db, params: dict):
    """Train a supported OCR architecture without simulated success paths."""
    if job.architecture != "lprnet":
        raise ValueError(f"Unsupported OCR training architecture: {job.architecture}")
    _train_lprnet(job, db, params)


def _train_lprnet(job, db, params: dict):
    """Train LPRNet with CTC loss on verified OCR annotations."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset as TorchDataset
    from PIL import Image
    from app.models.database import (
        Annotation,
        DatasetImage,
        TrainingJob,
        JobStatus,
        TrainingMetrics,
    )

    job_id = job.id
    epochs = params.get("epochs", 100)
    batch_size = params.get("batch_size", 16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(settings.LOGS_PATH) / f"job_{job_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    chars_table = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    char_to_idx = {char: index for index, char in enumerate(chars_table)}
    blank_idx = len(chars_table)
    max_len = 12

    class LPRNet(nn.Module):
        def __init__(self, num_chars: int, sequence_len: int = 12):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.Conv2d(256, num_chars + 1, 1),
            )
            self.pool = nn.AdaptiveAvgPool2d((1, sequence_len))

        def forward(self, x):
            x = self.backbone(x)
            x = self.pool(x)
            x = x.squeeze(2)           # (B, C, T)
            x = x.permute(2, 0, 1)    # (T, B, C)
            return x

    cyrillic_map = str.maketrans(
        {"А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "І": "I",
         "К": "K", "М": "M", "О": "O", "Р": "P", "Т": "T", "Х": "X"}
    )

    def normalize(text: str) -> str:
        value = (text or "").upper().translate(cyrillic_map)
        return "".join(char for char in value if char in char_to_idx)

    def collect_records(split: str):
        records = []
        images = db.query(DatasetImage).filter_by(dataset_id=job.dataset_id, split=split).all()
        for image in images:
            annotation = (
                db.query(Annotation)
                .filter(
                    Annotation.image_id == image.id,
                    Annotation.annotation_type == "ocr",
                    Annotation.is_verified == True,
                )
                .first()
            )
            text = normalize(annotation.ocr_text if annotation else "")
            if os.path.exists(image.file_path) and 1 <= len(text) <= max_len:
                records.append((image.file_path, text))
        return records

    train_records = collect_records("train")
    val_records = collect_records("val")
    if not train_records or not val_records:
        raise ValueError(
            "LPRNet requires verified OCR annotations in both train and val splits"
        )

    class PlateDataset(TorchDataset):
        def __init__(self, records):
            self.records = records

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            path, text = self.records[index]
            image = Image.open(path).convert("RGB").resize((128, 32))
            array = np.asarray(image, dtype=np.float32) / 255.0
            array = (array - 0.5) / 0.5
            tensor = torch.from_numpy(array.transpose(2, 0, 1))
            target = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)
            return tensor, target, text

    def collate(batch):
        images, targets, texts = zip(*batch)
        lengths = torch.tensor([len(target) for target in targets], dtype=torch.long)
        return torch.stack(images), torch.cat(targets), lengths, list(texts)

    train_loader = DataLoader(
        PlateDataset(train_records), batch_size=batch_size, shuffle=True,
        num_workers=params.get("workers", 2), collate_fn=collate,
    )
    val_loader = DataLoader(
        PlateDataset(val_records), batch_size=batch_size, shuffle=False,
        num_workers=params.get("workers", 2), collate_fn=collate,
    )

    model = LPRNet(num_chars=len(chars_table), sequence_len=max_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.get("learning_rate", 1e-3))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5)
    ctc_loss = nn.CTCLoss(blank=blank_idx, zero_infinity=True)

    best_path = str(run_dir / "lprnet_best.pt")

    def decode(logits):
        predictions = logits.argmax(dim=2).transpose(0, 1).cpu().tolist()
        decoded = []
        for sequence in predictions:
            previous = -1
            chars = []
            for token in sequence:
                if token != previous and token != blank_idx:
                    chars.append(chars_table[token])
                previous = token
            decoded.append("".join(chars))
        return decoded

    def edit_distance(left: str, right: str) -> int:
        row = list(range(len(right) + 1))
        for i, left_char in enumerate(left, 1):
            next_row = [i]
            for j, right_char in enumerate(right, 1):
                next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (left_char != right_char)))
            row = next_row
        return row[-1]

    job_rec = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    job_rec.status = JobStatus.TRAINING
    job_rec.total_epochs = epochs
    db.commit()

    best_plate_accuracy = -1.0
    best_char_accuracy = 0.0
    epochs_without_improvement = 0
    patience = params.get("patience", 20)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images, targets, target_lengths, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            optimizer.zero_grad()
            logits = model(images)
            input_lengths = torch.full(
                (images.size(0),), logits.size(0), dtype=torch.long, device=device
            )
            loss = ctc_loss(logits.log_softmax(2), targets, input_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        exact = char_total = char_errors = sample_total = 0
        with torch.no_grad():
            for images, _, _, texts in val_loader:
                predictions = decode(model(images.to(device)))
                for predicted, expected in zip(predictions, texts):
                    exact += int(predicted == expected)
                    char_total += max(len(expected), 1)
                    char_errors += edit_distance(predicted, expected)
                    sample_total += 1

        plate_accuracy = exact / sample_total
        char_accuracy = max(0.0, 1.0 - char_errors / char_total)
        train_loss = total_loss / len(train_loader)
        scheduler.step(plate_accuracy)

        if plate_accuracy > best_plate_accuracy:
            best_plate_accuracy = plate_accuracy
            best_char_accuracy = char_accuracy
            torch.save(model.state_dict(), best_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        gpu_mem, gpu_util = _get_gpu_stats()
        progress = 5 + (epoch / epochs) * 90
        db.add(TrainingMetrics(
            job_id=job_id, epoch=epoch, train_loss=train_loss,
            char_accuracy=char_accuracy, plate_accuracy=plate_accuracy,
            gpu_memory_mb=gpu_mem, gpu_utilization=gpu_util,
            lr=optimizer.param_groups[0]["lr"],
        ))
        push_progress(job_id, {
            "job_id": job_id, "status": "training",
            "current_epoch": epoch, "total_epochs": epochs,
            "progress_pct": round(progress, 1),
            "latest_metrics": {
                "train_loss": train_loss,
                "char_accuracy": char_accuracy,
                "plate_accuracy": plate_accuracy,
            },
            "message": f"LPRNet epoch {epoch}/{epochs} — plate accuracy: {plate_accuracy:.4f}",
            "gpu_utilization": gpu_util, "gpu_memory_mb": gpu_mem,
        })
        j = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        j.current_epoch = epoch
        j.progress_pct = progress
        j.best_metrics = {
            "char_accuracy": best_char_accuracy,
            "plate_accuracy": best_plate_accuracy,
            "epoch": epoch,
        }
        db.commit()
        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    dummy = torch.randn(1, 3, 32, 128).to(device)
    onnx_path = str(run_dir / "lprnet.onnx")
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {1: "batch"}},
        opset_version=17,
    )
    _register_trained_model(
        db, job, best_path,
        {"char_accuracy": best_char_accuracy, "plate_accuracy": best_plate_accuracy},
    )


# ─── Auto Annotation Task ─────────────────────────────────────────

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
                "vehicle_detector": [
                    Path(settings.INFERENCE_MODELS_PATH) / "vehicle_detector" / "production.pt",
                    Path(settings.INFERENCE_MODELS_PATH) / "vehicle_detector" / "yolo11s.pt",
                    Path(settings.INFERENCE_MODELS_PATH) / "vehicle_detector" / "yolo11n.pt",
                ],
                "plate_detector": [
                    Path(settings.INFERENCE_MODELS_PATH) / "plate_detector" / "production.pt",
                    Path(settings.INFERENCE_MODELS_PATH) / "plate_detector" / "yolov8n_plate.pt",
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

            img_rec.is_annotated = True
            db.commit()

        logger.info(f"Auto-annotated {annotated} boxes across {len(image_ids)} images")
        return {"annotated": annotated}
    finally:
        db.close()


# ─── Helpers ──────────────────────────────────────────────────────

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

    # Get metrics
    if isinstance(results, dict):
        metrics = results
    else:
        try:
            metrics = {
                "map50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
                "map50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
                "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
                "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
            }
        except Exception:
            metrics = {}

    mv = ModelVersion(
        name=f"{job.model_type}_{job.architecture}_{version_str}",
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
            "license": (
                "AGPL-3.0-or-Ultralytics-Enterprise"
                if job.model_type in ("vehicle_detector", "plate_detector")
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


def _export_dataset_sync(db, dataset_id, export_dir, dataset):
    """Synchronous YOLO dataset export for use inside Celery worker."""
    import shutil
    from app.models.database import DatasetImage, Annotation

    for split in ["train", "val", "test"]:
        (export_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (export_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    images = db.query(DatasetImage).filter(
        DatasetImage.dataset_id == dataset_id,
        DatasetImage.split.isnot(None),
    ).all()

    for img in images:
        split = img.split or "train"
        dest = export_dir / split / "images" / img.filename
        if os.path.exists(img.file_path):
            shutil.copy2(img.file_path, dest)

        anns = db.query(Annotation).filter(
            Annotation.image_id == img.id,
            Annotation.annotation_type == "bbox",
        ).all()
        label_file = export_dir / split / "labels" / (Path(img.filename).stem + ".txt")
        with open(label_file, "w") as f:
            for ann in anns:
                f.write(
                    f"{ann.class_id or 0} {ann.x_center:.6f} {ann.y_center:.6f} "
                    f"{ann.bbox_width:.6f} {ann.bbox_height:.6f}\n"
                )

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
