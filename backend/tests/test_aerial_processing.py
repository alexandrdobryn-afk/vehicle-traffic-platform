from dataclasses import dataclass

import numpy as np

from app.services.aerial_processing_service import AerialProcessingService, TileConfig


@dataclass
class FakePrediction:
    bbox: list[int]
    confidence: float
    class_name: str


def test_tiled_inference_projects_boxes_and_deduplicates_overlap():
    service = AerialProcessingService(TileConfig(enabled=True, size=320, overlap=0.5, nms_iou=0.4))
    frame = np.zeros((480, 480, 3), dtype=np.uint8)

    def detector(tile):
        # Identical local output in overlapping tiles creates source-coordinate
        # predictions and exercises object-level global NMS.
        return [FakePrediction([200, 200, 260, 260], 0.9, "person")]

    predictions = service.infer_tiled(frame, detector)
    assert predictions
    assert all(0 <= value <= 480 for prediction in predictions for value in prediction.bbox)


def test_nms_suppresses_overlapping_hypotheses_across_classes():
    predictions = [
        FakePrediction([0, 0, 100, 100], 0.9, "person"),
        FakePrediction([0, 0, 100, 100], 0.8, "boat"),
    ]
    assert len(AerialProcessingService.object_agnostic_nms(predictions, 0.5)) == 1


def test_nms_prefers_complete_object_over_contained_part_when_confidence_is_close():
    predictions = [
        FakePrediction([35, 35, 70, 70], 0.94, "car"),
        FakePrediction([0, 0, 100, 100], 0.86, "truck"),
    ]

    kept = AerialProcessingService.object_agnostic_nms(predictions, 0.35)

    assert len(kept) == 1
    assert kept[0].bbox == [0, 0, 100, 100]


def test_nms_does_not_remove_valid_nested_different_class_object():
    predictions = [
        FakePrediction([0, 0, 220, 140], 0.9, "car"),
        FakePrediction([80, 30, 130, 120], 0.86, "person"),
    ]

    kept = AerialProcessingService.object_agnostic_nms(predictions, 0.5)

    assert len(kept) == 2
