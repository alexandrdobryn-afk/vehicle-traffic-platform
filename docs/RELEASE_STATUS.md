# Release status

BEVP currently provides a development baseline for aerial small-object analysis.

Available:

- live/drone streams and recorded-video processing;
- source-resolution tiled object detection;
- configurable object detector and tracker selection;
- Kalman prediction, optional classification and optional segmentation;
- persistent object tracks, trajectories, events and runtime performance;
- dataset annotation, immutable dataset versions, training and model registry;
- frame review statuses, hard-negative capture and Active Learning queue;
- validation report storage, generated FP/FN/class-confusion error items, decision gate metadata and Evaluation UI;
- baseline inference mode for reviewable auto-label generation;
- physical tiled YOLO training export with tile manifest and original-frame lineage;
- CPU/CUDA launch policy and C++ ONNX reference runtime;
- Ukrainian and English operator UI.

Not yet release evidence:

- target-domain detector accuracy;
- AP-small/Recall-small, FP/frame and FN/frame acceptance evidence on a frozen aerial holdout;
- HOTA automation;
- native TensorRT validation on the deployment GPU;
- production soak, retention, backup and security acceptance.

No unrelated detector is substituted when an aerial model is missing. Availability is reported from the real artifact registry.
