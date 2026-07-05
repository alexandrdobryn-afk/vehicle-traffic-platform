import base64
import os
import unittest

import cv2
import numpy as np

from app.services.gemini_training_service import gemini_training_service


class GeminiTrainingServiceTest(unittest.TestCase):
    def test_probability_mask_becomes_normalized_polygon(self):
        mask = np.full((4, 4), 255, dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", mask)
        self.assertTrue(ok)
        objects = [{
            "label": "car",
            "confidence": 0.9,
            "box_2d": [100, 100, 900, 900],
            "mask": base64.b64encode(encoded).decode("ascii"),
        }]
        annotations, mask_path = gemini_training_service._normalize_annotations(
            987654321, 100, 100, objects
        )
        try:
            self.assertEqual(len(annotations), 1)
            self.assertGreaterEqual(len(annotations[0]["polygon"]), 3)
            self.assertTrue(os.path.isfile(mask_path))
            for point in annotations[0]["polygon"]:
                self.assertTrue(all(0 <= value <= 1 for value in point))
        finally:
            if mask_path and os.path.exists(mask_path):
                os.remove(mask_path)


if __name__ == "__main__":
    unittest.main()
