import unittest

from app.schemas.schemas import AnnotationCreate, DatasetCreate


class SegmentationSchemaTest(unittest.TestCase):
    def test_segmenter_dataset_and_polygon_are_supported(self):
        dataset = DatasetCreate(
            name="segmentation-test",
            model_type="vehicle_segmenter",
            annotation_type="segmentation",
            classes=["car"],
        )
        annotation = AnnotationCreate(
            annotation_type="segmentation",
            class_name="car",
            class_id=0,
            polygon=[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
            is_auto=True,
            is_verified=True,
            provenance={"source": "gemini_review_candidate"},
        )
        self.assertEqual(dataset.model_type.value, "vehicle_segmenter")
        self.assertEqual(annotation.annotation_type.value, "segmentation")
        self.assertEqual(len(annotation.polygon), 4)


if __name__ == "__main__":
    unittest.main()
