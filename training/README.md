# BEVP training service

The training service manages aerial object datasets, human-reviewed annotations, immutable dataset versions, training jobs, validation, export and model promotion.

Supported model types:

- `object_detector` — bounding-box detection with YOLO-family trainers;
- `object_segmenter` — instance segmentation with YOLO segmentation trainers;
- `object_classifier` — crop-level classification with lightweight image classifiers.

Recommended flow:

1. Create a dataset for one measurable aerial task.
2. Import images or extract frames from whole drone videos.
3. Annotate bounding boxes, masks or image classes.
4. Split by entire flight/video to avoid frame leakage.
5. Freeze the dataset version and record its SHA-256 fingerprint.
6. Start training from that frozen version.
7. Validate on the frozen holdout and export the selected artifact.
8. Promote it into the matching inference directory only after acceptance.

The training API runs on `http://localhost:8101/docs`. Celery workers execute jobs and Redis carries progress events. Dataset mutations are rejected after a version is frozen; create a child version for further edits.
