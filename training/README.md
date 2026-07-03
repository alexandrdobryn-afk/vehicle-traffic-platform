# AI Training Platform Module

Integrated AI model training and lifecycle management for the Vehicle Traffic Platform.

---

## Overview

This module adds a complete ML training workflow to the VTP:

```
Upload Data → Annotate → Configure Training → Monitor Live →
Validate → Benchmark vs Production → Manual Approve → Deploy → Rollback
```

---

## Architecture

```
training/
├── backend/
│   ├── app/
│   │   ├── main.py              FastAPI + WebSocket for live progress
│   │   ├── config.py            Settings (port 8001, Redis DB 1)
│   │   ├── celery_app.py        Celery config (gpu + cpu queues)
│   │   ├── api/routers.py       All REST endpoints
│   │   ├── models/database.py   DB models (tr_* tables)
│   │   ├── schemas/schemas.py   Pydantic schemas
│   │   ├── services/
│   │   │   ├── dataset_service.py       Upload, split, export
│   │   │   ├── annotation_service.py    CRUD + auto-annotate
│   │   │   ├── model_registry_service.py Deploy, rollback
│   │   │   └── gpu_service.py           GPU monitoring
│   │   └── workers/
│   │       ├── training_worker.py       YOLO, MobileNetV3, LPRNet
│   │       ├── export_worker.py         ONNX, TensorRT export
│   │       ├── validation_worker.py     Metrics + benchmark
│   │       └── video_worker.py          Frame extraction
│   ├── alembic/                 DB migrations (002_training)
│   ├── Dockerfile               API container
│   └── requirements.txt
└── Dockerfile.worker            GPU worker (CUDA base image)
```

---

## Integration with Inference Backend

These two modules share:
- **Same PostgreSQL** — different table prefix (`tr_*` for training)
- **Same Redis** — different DB index (0=inference, 1=training)
- **Same JWT secret** — same login works for both APIs
- **Shared model directory** — `backend/models/` mounted in both containers

When a model is approved and deployed:
1. Training worker copies weights → `backend/models/{type}/`
2. Inference backend auto-picks them up on next camera restart
3. No inference downtime during model swap

---

## API Endpoints

### Datasets
```
GET    /api/v1/training/datasets
POST   /api/v1/training/datasets
GET    /api/v1/training/datasets/{id}
DELETE /api/v1/training/datasets/{id}
GET    /api/v1/training/datasets/{id}/stats
POST   /api/v1/training/datasets/{id}/images        (multipart upload)
GET    /api/v1/training/datasets/{id}/images
GET    /api/v1/training/datasets/{id}/images/{img_id}/file
POST   /api/v1/training/datasets/{id}/videos        (multipart upload)
POST   /api/v1/training/datasets/{id}/videos/{vid_id}/extract
POST   /api/v1/training/datasets/{id}/split
POST   /api/v1/training/datasets/{id}/export/yolo
POST   /api/v1/training/datasets/{id}/auto-annotate
```

### Annotations
```
GET    /api/v1/training/annotations/image/{image_id}
POST   /api/v1/training/annotations/image/{image_id}
POST   /api/v1/training/annotations/image/{image_id}/bulk
PUT    /api/v1/training/annotations/{id}
DELETE /api/v1/training/annotations/{id}
DELETE /api/v1/training/annotations/image/{image_id}/all
POST   /api/v1/training/annotations/image/{src}/copy-to/{dst}
GET    /api/v1/training/annotations/dataset/{id}/stats
```

### Training Jobs
```
GET    /api/v1/training/jobs
POST   /api/v1/training/jobs
GET    /api/v1/training/jobs/{id}
POST   /api/v1/training/jobs/{id}/cancel
GET    /api/v1/training/jobs/{id}/metrics
GET    /api/v1/training/jobs/{id}/progress      (Redis fast-path)
```

### Model Registry
```
GET    /api/v1/training/registry
GET    /api/v1/training/registry/{id}
POST   /api/v1/training/registry/{id}/approve   (admin only)
POST   /api/v1/training/registry/{id}/reject    (admin only)
POST   /api/v1/training/registry/rollback       (admin only)
POST   /api/v1/training/registry/{id}/export/{format}
POST   /api/v1/training/registry/{id}/validate
GET    /api/v1/training/registry/{id}/deploy-logs
```

### GPU & System
```
GET    /api/v1/training/gpu
GET    /api/v1/training/architectures/{model_type}
GET    /api/v1/training/health
```

### WebSocket (live training progress)
```
WS /ws/training/{job_id}?token=JWT
```

---

## Supported Training Targets

| Model Type | Architecture | Framework | Notes |
|---|---|---|---|
| vehicle_detector | yolo11n/s/m, yolov8n/s | Ultralytics | COCO pretrained, fine-tune |
| plate_detector | yolo11n, yolov8n | Ultralytics | Plate-specific fine-tune |
| color_classifier | mobilenetv3, efficientnet, resnet18 | PyTorch | UFPR-VCR dataset |
| ocr | paddleocr | PaddlePaddle | Fine-tune on UA plates |
| ocr | lprnet | PyTorch/ONNX | Lightweight, edge deploy |

---

## Workflow Example

```bash
# 1. Create dataset
POST /api/v1/training/datasets
{ "name": "UA Vehicles", "model_type": "vehicle_detector",
  "annotation_type": "bbox", "classes": ["car","truck","bus"] }

# 2. Upload images
POST /api/v1/training/datasets/1/images
(multipart: files[])

# 3. Annotate via UI (Canvas annotator at /training/annotate/1)
# OR auto-annotate using existing production model:
POST /api/v1/training/datasets/1/auto-annotate
{ "confidence_threshold": 0.5 }

# 4. Apply split
POST /api/v1/training/datasets/1/split
{ "train_ratio": 0.7, "val_ratio": 0.2, "test_ratio": 0.1, "seed": 42 }

# 5. Create training job
POST /api/v1/training/jobs
{ "name": "YOLO11n v2", "model_type": "vehicle_detector",
  "architecture": "yolo11n", "dataset_id": 1,
  "hyperparams": { "epochs": 100, "batch_size": 16 } }

# 6. Monitor via WebSocket: ws://localhost:8001/ws/training/1?token=JWT
# Or poll: GET /api/v1/training/jobs/1/progress

# 7. After completion, validate:
POST /api/v1/training/registry/1/validate

# 8. Export to ONNX/TRT:
POST /api/v1/training/registry/1/export/onnx
POST /api/v1/training/registry/1/export/tensorrt

# 9. Deploy (admin only):
POST /api/v1/training/registry/1/approve
{ "comment": "Better mAP50 than v1", "run_auto_tests": true }

# 10. If issues arise, rollback:
POST /api/v1/training/registry/rollback
{ "model_version_id": 1, "comment": "Rolling back to v1" }
```

---

## Running Celery Workers

```bash
# GPU worker (training, export, validation)
celery -A app.celery_app worker --queues=gpu --concurrency=1 --loglevel=info

# CPU worker (video extraction)
celery -A app.celery_app worker --queues=cpu --concurrency=4 --loglevel=info

# Monitor tasks (Flower UI at :5555)
make flower
```

---

## Annotation Format

Bounding boxes stored in YOLO normalized format:
```
class_id  x_center  y_center  width  height
# All values 0.0–1.0 relative to image dimensions
```

Exported YOLO dataset structure:
```
export/dataset_1_yolo/
├── train/images/ + labels/
├── val/images/ + labels/
├── test/images/ + labels/
└── data.yaml
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Worker not picking up jobs | Check `redis://redis:6379/1` connection |
| Training fails immediately | Check GPU availability: `GET /api/v1/training/gpu` |
| YOLO export fails | Ensure weights file exists; check logs |
| Auto-annotate returns 0 | No production model deployed yet |
| WebSocket disconnects | JWT token expired; refresh and reconnect |
| TRT export fails | CUDA required; CPU-only machines skip TRT |
