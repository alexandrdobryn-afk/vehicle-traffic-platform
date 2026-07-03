# Release readiness

Updated: 2026-07-03.

## Verified in the current development environment

- Docker Compose starts the frontend, inference backend, PostgreSQL, Redis and training services.
- The inference backend runs on CPU and on NVIDIA CUDA; the runtime status screen reports the effective provider and hardware.
- Uploaded video processing, annotated preview, rerun and end-of-file lifecycle work end to end.
- Vehicle tracking, plate detection, temporal OCR voting, regional syntax validation, best-frame evidence selection, color voting and conservative vehicle-brand recognition are wired into the runtime pipeline.
- The UI exposes four core AI presets, three experimental presets and a manual pipeline selector with compatibility warnings.
- Camera and video recognition thresholds are source-local. Compute provider selection is global.
- The interface supports Ukrainian and English.
- Backend recognition tests and the Next.js production build pass in the Docker-backed development workflow.

## Requires target data or deployment infrastructure

- Long-running RTSP/HLS/MJPEG/JPEG soak tests on the actual camera models, including day, night, rain, glare and network interruption scenarios.
- A frozen labeled evaluation set for detector recall/precision, plate exact match, character accuracy, color macro-F1, brand accuracy and tracking IDF1.
- Performance benchmarks for each preset on every supported CPU/GPU target. Local FPS figures must not be generalized to other hardware.
- TensorRT engines built and validated on each target NVIDIA GPU and compatible TensorRT runtime.
- Retention limits, backup/restore, monitoring, alert-delivery retries and production secret management.
- A project-level license and a documented decision on third-party model and dataset licenses, including Ultralytics terms.

## Release classification

This repository is a functional development baseline suitable for controlled testing. It is not yet a production-certified traffic enforcement or safety system.
