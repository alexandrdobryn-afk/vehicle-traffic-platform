# BEVP implementation roadmap

## Implemented baseline

- [x] Recorded aerial video and live camera/stream ingestion
- [x] Source-local Speed, Balanced, Quality and Manual modes
- [x] High-resolution overlapping-tile inference with global NMS
- [x] Generic object detector registry for YOLO, RT-DETR and RF-DETR artifacts
- [x] ByteTrack, BoT-SORT, OC-SORT, DeepSORT and StrongSORT-compatible tracking layer
- [x] Separate Kalman prediction and trajectory smoothing module
- [x] Optional object classification and instance segmentation modules
- [x] Object tracks, trajectories, events and performance analytics
- [x] Dataset import, annotation, frozen versions, training jobs and registry
- [x] Frame lifecycle statuses for unlabeled, auto-labeled, review, approved and hard-negative states
- [x] Active Learning queue API and operator page for useful review frames
- [x] Evaluation report API/page with decision gate result and stored error items
- [x] Baseline inference mode for creating reviewable auto-labels before training
- [x] Physical tiled YOLO export for small-object training with tile manifest
- [x] Validation mining for false positives, false negatives and class-confusion items
- [x] Automatic Active Learning item creation from validation errors
- [x] UI controls for tile visibility, empty tile ratio, small-object gates and error thresholds
- [x] Human-reviewed Gemini-assisted mask candidates
- [x] Python runtime, C++ ONNX reference runtime and Unity integration contract
- [x] UA/EN operator interface on ports 3100/8100/8101/5565

## Required validation before production

- [ ] Install and fingerprint the approved aerial detector artifacts
- [ ] Fine-tune on representative drone heights, angles, blur, weather and occlusions
- [ ] Benchmark every mode on one frozen labeled aerial holdout
- [ ] Record per-class precision, recall and mAP50-95
- [ ] Record AP-small, Recall-small, FP/frame and FN/frame from real labeled holdouts
- [ ] Review generated false-positive, false-negative and class-confusion queues against real data
- [ ] Record MOTA, IDF1 and HOTA on representative flight sequences
- [ ] Record FPS, P50/P95 latency and peak memory on every target device
- [ ] Validate ONNX/TensorRT exports on the deployment GPU
- [ ] Run long-duration multi-stream and recovery tests

The UI exposes source-based aerial workflows and training lifecycle pages. The evaluation, tiled export and active-learning control plane is functional, but release claims still require real aerial holdout evidence and target-hardware benchmarking.
