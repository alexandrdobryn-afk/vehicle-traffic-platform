# BEVP Full Module Completion Plan

This is the execution plan for making the platform cover the full aerial computer-vision stack:
video/image input, small-object processing, detection, segmentation, classification, OCR, tracking,
Kalman prediction, ReID, geo, training, evaluation, production export, C++ runtime, edge, API,
VLM assistance and simulation.

## Definition of Done

Every module is considered complete only when all items are true:

- API exposes module status, configuration and test result.
- UI shows the module in the operator workflow and does not hide failures.
- Runtime either executes the real adapter or clearly reports missing dependencies/artifacts.
- Tests cover at least one successful path and one missing-dependency path.
- Documentation explains required model files, dependencies and launch command.

## Phase 1: Stabilize and Make Capabilities Visible

- Add `/api/v1/modules` capability map for all 22 modules.
- Add `/modules` UI page with status, coverage, dependencies and next steps.
- Keep CUDA-first launch as the normal runtime and fail closed when CUDA is requested but unavailable.
- Add Docker cache cleanup workflow and document safe commands.

## Phase 2: Complete Core Runtime Controls

- Expose source-local controls for detector, tracker, tiling, enhancement, segmentation, classification, OCR and Kalman settings.
- Persist all controls in `pipeline_config`.
- Show resolved pipeline from live source stats.
- Add runtime validation before starting a source.

## Phase 3: Fill Missing Runtime Modules

- Restore generic OCR runtime with PaddleOCR and EasyOCR adapters.
- Add Geo telemetry schema for GPS, altitude, heading, camera angle and object coordinate output.
- Add ReID model role with OSNet/TorchReID/FastReID adapters behind availability checks.
- Add image enhancement chain: denoise, deblur placeholder adapter, white balance and stabilization hooks.
- Add super-resolution adapters for Real-ESRGAN/SwinIR/BasicSR behind explicit GPU budget controls.

## Phase 4: Expand Models

- Add detector plugin adapters for EfficientDet, Faster R-CNN, RetinaNet, FCOS, CenterNet and DINO.
- Add Grounding DINO as a text-prompt detector, not as a replacement for production detection.
- Add SAM, MobileSAM and FastSAM as annotation/refinement modules.
- Add SegFormer and Mask R-CNN only after segmentation dataset/export path is stable.

## Phase 5: Training and Dataset Completeness

- Add COCO import/export beside YOLO export.
- Add segmentation mask export verification.
- Add hyperparameter search jobs.
- Add RF-DETR fine-tuning path.
- Add distributed training strategy fields and guardrails.
- Add dataset diff and quality-gate UI.

## Phase 6: Evaluation and Model Governance

- Add benchmark runs for fixed datasets.
- Add HOTA and TrackEval-compatible tracking evaluation.
- Add runtime metric timeline: FPS, latency, GPU, CPU, RAM.
- Block model approval when required benchmark gates fail.
- Compare model versions in UI with metrics, artifacts and deployment history.

## Phase 7: Production, Edge and Simulation

- Add OpenVINO export.
- Complete C++ runtime with OpenCV, ONNX Runtime, TensorRT and CUDA execution paths.
- Add Jetson/industrial-PC edge profiles.
- Add Unity simulation ingestion endpoint and synthetic dataset provenance.
- Add gRPC API after REST/WebSocket contracts are stable.

## Current First Deliverable

The first deliverable is a truthful capability map and UI page. It prevents the project from claiming support for
libraries that are not installed or model artifacts that are missing, while giving us a concrete checklist for closing
the remaining modules.
