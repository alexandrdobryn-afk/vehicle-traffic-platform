import unittest

from app.schemas.schemas import (
    ActiveLearningCreate,
    AnnotationCreate,
    DatasetCreate,
    EvaluationPolicy,
    FrameStatusUpdate,
    TrainingJobCreate,
)
from app.utils.geometry import abs_box_to_norm, bbox_iou, clip_box_to_tile, tile_origins, tile_stride


class SegmentationSchemaTest(unittest.TestCase):
    def test_segmenter_dataset_and_polygon_are_supported(self):
        dataset = DatasetCreate(
            name="segmentation-test",
            model_type="object_segmenter",
            annotation_type="segmentation",
            classes=["building"],
        )
        annotation = AnnotationCreate(
            annotation_type="segmentation",
            class_name="building",
            class_id=0,
            polygon=[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
            is_auto=True,
            is_verified=True,
            provenance={"source": "gemini_review_candidate"},
        )
        self.assertEqual(dataset.model_type.value, "object_segmenter")
        self.assertEqual(annotation.annotation_type.value, "segmentation")
        self.assertEqual(len(annotation.polygon), 4)

    def test_lifecycle_training_contracts_are_supported(self):
        status = FrameStatusUpdate(status="needs_review", review_reason="low_confidence", review_priority=50)
        queue_item = ActiveLearningCreate(
            dataset_id=1,
            image_id=2,
            reason="small_object_miss",
            priority_score=75,
            source="evaluation_report",
        )
        job = TrainingJobCreate(
            name="tiled-detector",
            model_type="object_detector",
            architecture="yolo11s",
            dataset_id=1,
            training_mode="tiled_training",
            evaluation_policy=EvaluationPolicy(
                min_ap_small_delta=0.03,
                min_fps=10,
                min_recall_small=0.8,
                max_fp_per_frame=1.5,
                error_iou_threshold=0.45,
            ),
        )
        self.assertEqual(status.status.value, "needs_review")
        self.assertEqual(queue_item.reason, "small_object_miss")
        self.assertEqual(job.training_mode.value, "tiled_training")
        self.assertEqual(job.evaluation_policy.min_fps, 10)
        self.assertEqual(job.evaluation_policy.min_recall_small, 0.8)
        self.assertEqual(job.evaluation_policy.max_fp_per_frame, 1.5)

    def test_tiled_training_geometry_clips_visible_objects(self):
        stride = tile_stride(tile_size=100, overlap=0.25)
        self.assertEqual(stride, 75)

        origins = tile_origins(width=220, height=160, tile_size=100, stride=stride)
        self.assertIn((120, 60, 220, 160), origins)

        box = (80, 20, 140, 80)
        self.assertEqual(clip_box_to_tile(box, (75, 0, 175, 100), 0.2), box)
        self.assertIsNone(clip_box_to_tile(box, (130, 0, 220, 100), 0.2))

    def test_validation_geometry_helpers_match_and_normalize_boxes(self):
        self.assertAlmostEqual(bbox_iou([0, 0, 100, 100], [50, 50, 150, 150]), 1 / 7)
        normalized = abs_box_to_norm([25, 25, 75, 75], width=100, height=200)
        self.assertEqual(normalized["x_center"], 0.5)
        self.assertEqual(normalized["y_center"], 0.25)
        self.assertEqual(normalized["width"], 0.5)
        self.assertEqual(normalized["height"], 0.25)


if __name__ == "__main__":
    unittest.main()
