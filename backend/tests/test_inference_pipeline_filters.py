from collections import Counter
from types import SimpleNamespace

import numpy as np

from app.services.inference_service import InferencePipeline
from app.services.object_detection_service import Detection


def test_empty_target_classes_keep_class_agnostic_detection():
    assert InferencePipeline._normalize_target_classes([]) == []
    assert InferencePipeline._normalize_target_classes(["*"]) == []


def test_geometry_filter_rejects_fragments_and_keeps_vehicle_like_boxes():
    pipeline = SimpleNamespace(
        detection_filters={
            "min_box_width": 14,
            "min_box_height": 14,
            "min_box_area_ratio": 0.00008,
            "max_box_area_ratio": 0.25,
            "max_box_aspect_ratio": 6.0,
        },
        stage_counts=Counter(),
    )
    detections = [
        Detection([10, 10, 18, 18], 0.9, "car", 2),
        Detection([10, 20, 210, 30], 0.9, "car", 2),
        Detection([100, 100, 180, 145], 0.9, "car", 2),
    ]

    kept = InferencePipeline._filter_detections(pipeline, detections, np.zeros((1080, 1920, 3)).shape)

    assert kept == [detections[2]]
    assert pipeline.stage_counts["detections_rejected_geometry"] == 2
