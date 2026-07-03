import unittest
from types import SimpleNamespace

import numpy as np

from app.services.tracking_service import TrackingService


class _FakeUltralyticsTracker:
    def update(self, results, frame):
        assert results.xyxy.shape == (1, 4)
        assert results.xywh.shape == (1, 4)
        # x1, y1, x2, y2, track_id, score, class_id, detection_index
        return np.asarray([[10, 20, 110, 80, 7, 0.91, 2, 0]], dtype=np.float32)


class TrackingServiceTests(unittest.TestCase):
    def test_current_ultralytics_output_is_mapped(self):
        service = TrackingService.__new__(TrackingService)
        service._tracker = _FakeUltralyticsTracker()
        service._simple_tracker = None
        detection = SimpleNamespace(
            bbox=[10, 20, 110, 80], confidence=0.91, class_name="car", class_id=2
        )

        tracks = service._update_ultralytics_tracker(
            [detection], np.zeros((100, 160, 3), dtype=np.uint8)
        )

        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track_id, 7)
        self.assertEqual(tracks[0].class_name, "car")
        self.assertEqual(tracks[0].bbox, [10, 20, 110, 80])


if __name__ == "__main__":
    unittest.main()
