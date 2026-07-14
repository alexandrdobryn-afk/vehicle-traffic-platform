"""Shared defaults that must stay identical across API and runtime entry points."""


def default_source_pipeline_config() -> dict:
    return {
        "object_confidence_threshold": 0.55,
        "object_detector": "yolo11s_object",
        "verifier_detector": "rfdetr_medium_object",
        "tracker": "bytetrack",
        "target_classes": [],
        "track_missing_grace_frames": 6,
        "min_track_frames_for_event": 4,
        "min_track_duration_seconds": 0.0,
        "min_box_width": 14,
        "min_box_height": 14,
        "min_box_area_ratio": 0.00008,
        "max_box_area_ratio": 0.25,
        "max_box_aspect_ratio": 6.0,
        "frame_skip": 0,
        "aerial": {
            "enabled": True,
            "tile_size": 1024,
            "tile_overlap": 0.2,
            "nms_iou": 0.35,
            "enhance": False,
        },
        "object_memory": {
            "enabled": True,
            "max_gap_frames": 90,
            "merge_threshold": 0.48,
            "duplicate_iou": 0.45,
            "duplicate_contained": 0.78,
            "appearance_threshold": 0.68,
        },
        "kalman_prediction": {
            "enabled": True,
            "smoothing": True,
            "max_prediction_frames": 2,
        },
        "classification": {
            "enabled": True,
            "confidence_threshold": 0.35,
            "interval_frames": 15,
        },
        "segmentation": {
            "enabled": False,
            "confidence_threshold": 0.35,
            "interval_frames": 5,
        },
        "ocr": {
            "enabled": False,
            "engine": "auto",
            "confidence_threshold": 0.35,
            "interval_frames": 30,
        },
        "reid": {
            "enabled": False,
            "model": "hsv_histogram_v1",
            "similarity_threshold": 0.72,
            "store_vector": False,
        },
        "geo": {
            "enabled": False,
            "telemetry_source": "metadata",
            "coordinate_output": False,
        },
        "super_resolution": {
            "enabled": False,
            "engine": "auto",
            "min_object_size_px": 32,
            "max_crops_per_frame": 8,
        },
    }
