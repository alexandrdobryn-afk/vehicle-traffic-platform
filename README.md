# Bird's-Eye Vision Platform

BEVP is a Computer Vision platform for detecting, classifying, segmenting and tracking small objects in aerial video. It accepts recorded drone footage and live camera streams, preserves high-resolution detail through tiled inference and supports the full dataset → annotation → training → deployment workflow.

This repository is the aerial-analysis product. The previous road-monitoring application is a separate project and is not a runtime profile of BEVP.

## What is implemented

| Area | Capability |
|---|---|
| Sources | Video files, drone streams, RTSP, HLS, MJPEG, JPEG snapshots and USB cameras |
| Detection | YOLO, RT-DETR and RF-DETR adapters behind one object-detector interface |
| Small objects | Source-resolution inference, overlapping tiles, global NMS and optional enhancement |
| Tracking | ByteTrack, BoT-SORT, OC-SORT, DeepSORT and StrongSORT-compatible adapters with stable IDs and trajectories |
| Prediction | Separate constant-velocity Kalman module for smoothing and short detection gaps |
| Classification | Optional project-trained ONNX object classifier |
| Segmentation | Optional project-trained instance-segmentation model with polygon output |
| Training | Aerial object detection, classification and segmentation datasets/jobs |
| Review loop | Frame statuses, manual review actions, hard-negative capture and active-learning queue |
| Evaluation | Validation reports, generated FP/FN/class-confusion error items, model comparison fields and decision gate metadata |
| Runtime | Python/OpenCV/PyTorch/ONNX, optional CUDA and a C++ ONNX reference runtime |
| UI | Dashboard, cameras, videos, live view, events, analytics, settings and training |

## Normal workflow

1. Open **Cameras** to connect a drone/live source, or **Videos** to upload a recorded flight.
2. Choose **Speed**, **Balanced**, **Quality** or **Manual**.
3. Configure the source-local detection threshold, target classes, tile size and overlap.
4. Enable or disable Kalman prediction, object classification and instance segmentation.
5. Start processing and inspect live tracks, events and analytics.
6. For custom classes, run baseline inference, review predictions, freeze a dataset version, train a model, validate it, inspect the decision gate and promote only an approved artifact into `backend/models/object_detector`, `object_classifier` or `object_segmenter`.
7. Use **Active Learning** to move low-confidence, missed, rejected and hard-negative frames into the next dataset version.

Training jobs expose the same knobs used by the backend: full fine-tune, head-only, frozen-backbone, tiled training and baseline inference; tile size, overlap, object visibility and empty-tile ratio; small-object recall gates, FP/frame limits, latency limits and validation-error thresholds. These settings are stored with the job and used by the worker instead of being UI-only labels.

The default source profile is `aerial_small_objects` with Balanced mode, 1024-pixel tiles, 20% overlap, Kalman prediction enabled, classification enabled when its model exists, and segmentation disabled until explicitly requested.

## Model directories

```text
backend/models/
  object_detector/
    production.pt
    yolo11n.pt
    yolo11s.pt
    rf-detr-nano.pth
    rf-detr-medium.pth
    rtdetr-l.pt
  object_classifier/
    production.onnx
    labels.json
  object_segmenter/
    production.pt
```

BEVP does not silently substitute an unrelated model. If an optional or manually selected artifact is missing, the API reports it as unavailable.

## Start on Windows

Requirements: WSL2, Docker Engine inside the selected WSL distribution, a current Windows NVIDIA driver with WSL CUDA support, and NVIDIA Container Toolkit in WSL.

Normal NVIDIA launch:

```powershell
Set-Location "C:\Users\Admin\Desktop\CV drone\drone-vision-platform"; .\start.cmd -Build -Foreground
```

The launcher treats CUDA as required by default. It selects the GPU compose overlay only when Docker reports the NVIDIA runtime, then verifies that Docker attached NVIDIA `DeviceRequests` to the backend. Heavy CUDA smoke probes are opt-in because a wedged WSL/Docker GPU runtime can hang container creation itself. If WSL/Docker appears wedged, normal startup stops with a diagnostic and leaves WSL recovery to the explicit repair command.

GPU access is intentionally limited to compute services: the inference backend and training worker. The training API is a control plane and the frontend can start even if the training subsystem is degraded.

Strict CUDA container validation, for diagnostics only:

```powershell
.\start.cmd -Build -Foreground -VerifyCudaContainers
```

Explicit CPU emergency launch:

```powershell
Set-Location "C:\Users\Admin\Desktop\CV drone\drone-vision-platform"; .\start.cmd -Cpu -Build -Foreground
```

The launcher defaults to foreground mode. Keep that terminal open while testing. In this Windows + WSL setup, detached `docker compose up -d` can appear healthy and then be stopped when the WSL session is torn down. Use `-Detached` only when WSL is known to stay alive independently.

If WSL returns `Wsl/EnumerateDistros/Service/E_ACCESSDENIED`, or if CUDA disappears after reboot, run from Administrator PowerShell:

```powershell
Set-Location "C:\Users\Admin\Desktop\CV drone\drone-vision-platform"
powershell -ExecutionPolicy Bypass -File .\scripts\repair_wsl_docker.ps1 -CheckCuda
```

The CUDA repair path is successful only after a real Docker GPU smoke test passes. It should be an occasional repair command, not the normal launch path.

See [WSL_DOCKER_STABILITY.md](docs/WSL_DOCKER_STABILITY.md).

## Addresses

| Service | Address |
|---|---|
| Web UI | http://localhost:3100 |
| Inference API | http://localhost:8100/docs |
| Training API | http://localhost:8101/docs |
| Flower (`monitoring` profile) | http://localhost:5565/flower |

Development credentials come from `INITIAL_ADMIN_EMAIL` and `INITIAL_ADMIN_PASSWORD` in `.env`.

## Main APIs

- `/api/v1/cameras` — live/drone source configuration and control
- `/api/v1/videos` — recorded flight upload and processing
- `/api/v1/objects` — persisted object tracks and trajectories
- `/api/v1/events` — source and object lifecycle events
- `/api/v1/analytics/performance` — runtime speed and latency
- `/api/v1/health/ai-modes` — resolved object models for every mode
- `/api/v1/training/*` on port 8101 — datasets, annotation, jobs and model registry
- `/api/v1/training/active-learning` — review queue for useful frames and hard negatives
- `/api/v1/training/evaluation/*` — validation reports, error items and gate output
- `/ws/live/{source_id}` — live frames and object metadata

## Validation

```powershell
Set-Location frontend; npm run build
Set-Location ..; python -m compileall -q backend/app training/backend/app
```

For release-quality validation, use a frozen labeled aerial holdout and record per-class precision/recall/mAP, AP-small/Recall-small, FP/frame, FN/frame, IDF1/MOTA/HOTA, P50/P95 latency, FPS and memory on the target hardware. The current control plane stores evaluation reports and gate decisions; final production acceptance still depends on real holdout data.

## Repository layout

```text
backend/             inference API and live runtime
frontend/            Next.js/Tailwind operator UI
training/backend/    dataset, annotation, training and registry API
runtime/cpp/         C++ ONNX edge reference runtime
simulation/unity/    Unity integration contract
scripts/             launch and model utilities
docs/                engineering status and roadmap
```

## Production boundary

This is an engineering platform, not a certified safety system. Production approval still requires target-hardware benchmarks, long-running source tests, retention/backup rules, secret management and explicit acceptance thresholds.
