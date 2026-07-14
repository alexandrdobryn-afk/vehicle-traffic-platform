import unittest
from types import SimpleNamespace

import numpy as np

from app.services.tracking_service import TrackedObject, TrackingService


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
            bbox=[10, 20, 110, 80], confidence=0.91, class_name="target", class_id=2
        )

        tracks = service._update_ultralytics_tracker(
            [detection], np.zeros((100, 160, 3), dtype=np.uint8)
        )

        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track_id, 7)
        self.assertEqual(tracks[0].class_name, "target")
        self.assertEqual(tracks[0].bbox, [10, 20, 110, 80])

    def test_stabilizer_stitches_growing_vehicle_after_raw_id_reset(self):
        service = TrackingService.__new__(TrackingService)
        service._identity_map = {}
        service._identity_state = {}
        service._next_identity_id = 1
        service._identity_frame = 0

        frame = np.full((2160, 3840, 3), 210, dtype=np.uint8)

        first = service._stabilize_track_ids([
            TrackedObject(1, [2614, 766, 3013, 1098], 0.91, "truck")
        ], frame)
        second = service._stabilize_track_ids([
            TrackedObject(15, [2320, 1366, 2918, 2160], 0.86, "car")
        ], frame)

        self.assertEqual(first[0].track_id, second[0].track_id)

    def test_stabilizer_keeps_separate_simultaneous_vehicle_tracks(self):
        service = TrackingService.__new__(TrackingService)
        service._identity_map = {}
        service._identity_state = {}
        service._next_identity_id = 1
        service._identity_frame = 0

        frame = np.full((1080, 1920, 3), 180, dtype=np.uint8)
        tracks = service._stabilize_track_ids([
            TrackedObject(1, [100, 100, 260, 220], 0.9, "car"),
            TrackedObject(2, [1000, 100, 1160, 220], 0.88, "truck"),
        ], frame)

        self.assertEqual({track.track_id for track in tracks}, {1, 2})


if __name__ == "__main__":
    unittest.main()
