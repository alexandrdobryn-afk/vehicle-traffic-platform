# 🚗 Vehicle Traffic Platform

Professional AI-powered vehicle traffic analysis platform with real-time detection, license plate recognition, color classification, and a modern web dashboard.

---

## 📋 Overview

| Feature | Details |
|---|---|
| Vehicle Detection | YOLO11 / YOLO26; optional RF-DETR |
| Tracking | ByteTrack / BoT-SORT with identity stitching |
| Plate Detection | YOLOv8 / YOLO11 / OpenImageModels ONNX |
| Plate OCR | PaddleOCR / EasyOCR / FastPlateOCR / optional LPRNet + temporal voting |
| Best Evidence | Track-level best vehicle and plate crop selection |
| Color & Brand | HSV/KMeans fallback, optional classifiers, multi-frame voting |
| Regex Validation | UA / EU / US plate formats |
| Sources | RTSP, HLS, MJPEG, JPEG snapshots and uploaded video |
| Video Transport | MJPEG preview + WebSocket metadata |
| Auth | JWT, role-based (admin/operator/viewer) |
| Alerts | Frontend / Webhook / Telegram |
| AI Modes | Four core presets, three experimental presets, and manual pipeline selection |
| Compute Runtime | Automatic / CPU / NVIDIA CUDA, FP32 quality policy |
| Storage | PostgreSQL + Redis + local crops |

---

## 🏗 Architecture

```
RTSP Camera
    ↓
VideoCaptureService  (separate thread, auto-reconnect)
    ↓
FrameQualityFilter   (blur, brightness, contrast check)
    ↓
VehicleDetectionService  (YOLO11 / YOLO26 / RF-DETR when installed)
    ↓
TrackingService      (ByteTrack / OC-SORT / BoT-SORT)
    ↓
PlateDetectionService → OCRService → Temporal Voting
    ↓
ColorRecognitionService → Multi-frame Voting
    ↓
TrackStateService    (in-memory + Redis TTL)
    ↓
EventService         (async queue → PostgreSQL)
    ↓
WebSocket            (metadata only)
MJPEG stream         (annotated video)
    ↓
Next.js Dashboard
```

---

## 🛠 Tech Stack

**Backend:** Python 3.11, FastAPI, OpenCV, PyTorch, Ultralytics YOLO11, PaddleOCR, EasyOCR, ByteTrack, SQLAlchemy, PostgreSQL, Redis, JWT

**Frontend:** Next.js 15, React 18, TypeScript, Tailwind CSS, Recharts, WebSocket

**Infrastructure:** Docker Compose, Nginx, Fernet encryption

---

## ⚡ Quick Start

### Prerequisites

- Docker + Docker Compose
- NVIDIA GPU (optional but recommended)
- NVIDIA Container Toolkit (for GPU support)

### 1. Clone and configure

```bash
git clone <repo>
cd vehicle-traffic-platform
cp .env.example .env
```

Edit `.env`:
```env
SECRET_KEY=your-super-secret-key-min-32-chars
FERNET_KEY=  # Optional; derived stably from SECRET_KEY when empty
```

Generate Fernet key:
```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# Add output to FERNET_KEY= in .env
```

### 2. Download base models

```bash
python scripts/download_models.py all
python scripts/download_models.py --verify-only
```

### 3. Start platform

```bash
# Development (no GPU)
docker compose up --build

# Production with Nginx
docker compose --profile production up --build

# With NVIDIA GPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

### 4. Access

| Service | URL |
|---|---|
| Dashboard | http://localhost:3000 |
| API Docs | http://localhost:8000/docs |
| Health | http://localhost:8000/api/v1/health |

Development credentials come from `INITIAL_ADMIN_EMAIL` and `INITIAL_ADMIN_PASSWORD` in `.env`. Production rejects the example password and secret.

---

## 📷 Adding a Camera

1. Open Dashboard → Cameras
2. Click **Add Camera**
3. Select RTSP, HLS, MJPEG or JPEG snapshot and enter the source URL
4. Select AI Mode (Balanced recommended)
5. Click **Save** → **Start**

The system auto-restarts cameras on platform restart.

---

## 🤖 AI Modes

The registry resolves every preset against models that are actually installed. The UI shows the effective detector, tracker, plate detector and OCR engine; preset names are not guarantees that an unavailable model loaded successfully.

| Mode | Preferred Pipeline | Use Case |
|---|---|---|
| **Speed** | Lightweight detector + ByteTrack + fast OCR fallback | Throughput and edge baseline |
| **Balanced** | YOLO production path + ByteTrack + temporal OCR | Default production starting point |
| **Quality** | Best installed detector + BoT-SORT + full OCR | Difficult footage and forensic review |
| **Hybrid** | Balanced first pass + optional RF-DETR re-check | Variable traffic and uncertain scenes |
| **Practical NextGen** | YOLO26 + TrackTrack target + PaddleOCR | Experimental A/B testing |
| **Maximum Accuracy** | RF-DETR target + TrackTrack + PaddleOCR | Heavy experimental comparison |
| **ONNX Edge** | ONNX/TensorRT target + ByteTrack + FastPlateOCR | Edge and optimized-runtime testing |

### Hardware Requirements

| Runtime | Requirement | Notes |
|---|---|---|
| CPU | Any supported x86-64 installation | Compatible but slower for multi-camera inference |
| NVIDIA CUDA | NVIDIA GPU, driver and Container Toolkit | Recommended; verified by the runtime status screen |
| TensorRT | Engine built for the target GPU and compatible TensorRT runtime | Provider availability alone does not mean an engine exists |

---

## 🧠 Models

### Directory Structure

```
backend/models/
├── vehicle_detector/
│   ├── yolo11n.pt / yolo11s.pt
│   ├── yolo26n.pt / yolo26s.pt
│   └── rf-detr-nano.pth / rf-detr-medium.pth
├── plate_detector/
│   ├── yolov8n_plate.pt
│   ├── yolo11n_plate.pt
│   └── openimagemodels_yolov9t_384.onnx
├── brand_classifier/      ← optional local artifact
├── color_classifier/      ← optional local artifact
└── ocr/                   ← optional local artifact
```

Model binaries are intentionally excluded from Git. Their URLs, revisions, licenses and checksums are recorded in `backend/models/manifest.json` and `manifest.lock.json`.

### Getting Plate Detector

**Download the approved prototype model pack:**
```bash
python scripts/download_models.py all
python scripts/download_models.py --verify-only
```

**Option 2 — Train your own (recommended for production):**
```bash
# 1. Collect plate images and annotate with CVAT / Roboflow
# 2. Train:
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='plate_dataset.yaml', epochs=100, imgsz=640)
"
# 3. Export:
model.export(format='engine', half=True, device=0)
```

### Getting Color Classifier

**Option 1 — HSV fallback (no model needed):**
The system automatically uses HSV+KMeans if no ONNX model is found. This is a heuristic fallback; measure it on the client holdout set.

**Option 2 — Train MobileNetV3:**
```bash
# Datasets: UFPR-VCR, Vehicle Color Recognition Dataset
# See docs/TRAINING.md for full training pipeline
python scripts/train_color_classifier.py
```

### Exporting to TensorRT (for GPU speedup)

```python
from ultralytics import YOLO
model = YOLO("backend/models/vehicle_detector/yolo11n.pt")
model.export(format="engine", half=True, device=0)
# Build on the deployment GPU. TensorRT engines are machine-specific.
```

---

## 🔌 API Reference

### Auth
```
POST /api/v1/auth/login       { email, password } → token
GET  /api/v1/auth/me          → current user
```

### Cameras
```
GET    /api/v1/cameras
POST   /api/v1/cameras        { name, rtsp_url, ai_mode, ... }
PATCH  /api/v1/cameras/{id}
DELETE /api/v1/cameras/{id}
POST   /api/v1/cameras/{id}/start
POST   /api/v1/cameras/{id}/stop
POST   /api/v1/cameras/{id}/test
```

### Tracks & Events
```
GET /api/v1/tracks            ?plate=&color=&camera_id=
GET /api/v1/tracks/active
GET /api/v1/events            ?event_type=&date_from=&date_to=
GET /api/v1/events/export     ?format=csv|excel|json
```

### Video Stream
```
GET /api/v1/stream/{camera_id}    → MJPEG stream (annotated video)
WS  /ws/live/{camera_id}?token=   → WebSocket metadata
```

### WebSocket Message Format
```json
{
  "camera_id": 1,
  "timestamp": "2026-06-25T10:30:00",
  "fps": 24.5,
  "latency_ms": 87,
  "frame_width": 1920,
  "frame_height": 1080,
  "objects": [
    {
      "track_id": 17,
      "bbox": [430, 220, 810, 560],
      "vehicle_class": "car",
      "plate": "AA1234BB",
      "plate_status": "verified",
      "plate_confidence": 0.91,
      "color": "blue",
      "color_confidence": 0.84
    }
  ]
}
```

---

## 🚨 Watchlist & Alerts

1. Go to **Watchlist** → Add plate number
2. Select alert channels: frontend / webhook / telegram
3. When detected, alert fires in real-time

**Telegram setup:**
```env
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

**Webhook setup:**
```env
WEBHOOK_URL=https://your-endpoint.com/alert
```
Payload: `{ type, plate, camera_id, description, payload }`

---

## 🗄 Database Schema

Key tables: `cameras`, `vehicle_tracks`, `plate_candidates`, `events`, `users`, `watchlist`, `system_logs`, `app_settings`

All RTSP credentials stored encrypted (Fernet AES-128-CBC).

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| Camera won't connect | Check RTSP URL format. Test with VLC first |
| Low OCR accuracy | Check plate crop size in logs. Min 60×20px |
| High latency | Reduce max_fps, enable frame_skip, use Speed mode |
| GPU not used | Install `nvidia-container-toolkit`, check `nvidia-smi` |
| Redis connection failed | System continues with in-memory only — non-critical |
| PaddleOCR slow on CPU | Use LPRNet in Speed mode, or add GPU |
| RF-DETR not loading | RF-DETR is optional; the registry selects the best installed YOLO artifact |

---

## 🔒 Security Notes

- Default admin password **must** be changed in production
- RTSP URLs encrypted at rest with Fernet (AES-128)
- All API endpoints require JWT token
- WebSocket requires token via query param
- Role-based access: `admin > operator > viewer`
- Change `SECRET_KEY` in `.env` before production deploy

---

## 📊 Target Performance

| Metric | Value |
|---|---|
| Throughput and latency | Measure on the target CPU/GPU and real camera streams |
| Vehicle detection precision | Measure detector mAP on client holdout |
| OCR accuracy | Measure exact-match and character accuracy on client holdout |
| Color recognition accuracy | Measure macro-F1 on client holdout |
| ID stability | Measure IDF1 on representative camera sequences |

Accuracy depends on camera angle, resolution, lighting, and bitrate. Use temporal voting and regex validation to compensate for difficult conditions.

The current repository is a working development build, not a production certification. Validate online cameras, retention policy, alert delivery, model licenses and accuracy on a frozen labeled dataset before deployment.

---

## 📜 Licensing

No project-level license has been selected yet. Third-party model licenses are listed in `backend/models/manifest.json`; in particular, Ultralytics artifacts require compliance with their AGPL-3.0 or Enterprise terms. Choose an explicit repository license before presenting this project as open source.

---

## 📁 Project Structure

```
vehicle-traffic-platform/
├── backend/
│   ├── app/
│   │   ├── main.py              FastAPI app + lifespan
│   │   ├── config.py            All settings
│   │   ├── api/routers.py       All REST endpoints
│   │   ├── models/
│   │   │   ├── database.py      SQLAlchemy models
│   │   │   └── model_registry.py  AI model config
│   │   ├── services/
│   │   │   ├── inference_service.py     Main pipeline
│   │   │   ├── vehicle_detection_service.py
│   │   │   ├── tracking_service.py
│   │   │   ├── plate_detection_service.py
│   │   │   ├── ocr_service.py
│   │   │   ├── color_recognition_service.py
│   │   │   ├── frame_quality_service.py
│   │   │   ├── track_state_service.py
│   │   │   ├── video_capture_service.py
│   │   │   ├── camera_manager.py
│   │   │   ├── event_service.py
│   │   │   └── websocket_service.py
│   │   ├── schemas/schemas.py   Pydantic schemas
│   │   └── utils/auth.py        JWT, encryption
│   ├── models/                  AI model weights
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/app/
│   │   ├── dashboard/page.tsx
│   │   ├── cameras/page.tsx
│   │   ├── live/page.tsx
│   │   ├── events/page.tsx
│   │   ├── vehicles/page.tsx
│   │   ├── analytics/page.tsx
│   │   ├── watchlist/page.tsx
│   │   ├── settings/page.tsx
│   │   ├── health/page.tsx
│   │   └── login/page.tsx
│   ├── src/hooks/
│   │   ├── useAuth.ts
│   │   └── useWebSocket.ts
│   ├── src/lib/api.ts
│   ├── src/types/index.ts
│   └── Dockerfile
├── nginx/nginx.conf
├── scripts/download_models.py
├── docker-compose.yml
├── .env.example
└── README.md
```
