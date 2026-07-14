import unittest

from app.services.track_state_service import TrackStateService


class ObjectRegistryFieldTests(unittest.IsolatedAsyncioTestCase):
    async def test_track_keeps_video_position_and_latest_bbox(self):
        service = TrackStateService(camera_id=4, processing_run_id="run-1")

        track, is_new = await service.upsert_track(
            track_id=9,
            object_class="target",
            bbox=[10, 20, 50, 80],
            detection_conf=0.81,
            now="2026-07-12T10:00:00+00:00",
            video_timestamp_seconds=12.5,
        )

        self.assertTrue(is_new)
        self.assertEqual(track.first_video_timestamp_seconds, 12.5)
        self.assertEqual(track.last_video_timestamp_seconds, 12.5)
        self.assertEqual(track.to_dict()["last_bbox"], [10, 20, 50, 80])

        track, is_new = await service.upsert_track(
            track_id=9,
            object_class="target",
            bbox=[12, 22, 52, 82],
            detection_conf=0.88,
            now="2026-07-12T10:00:01+00:00",
            video_timestamp_seconds=13.0,
        )

        self.assertFalse(is_new)
        self.assertEqual(track.last_video_timestamp_seconds, 13.0)
        self.assertEqual(track.to_dict()["last_bbox"], [12, 22, 52, 82])


if __name__ == "__main__":
    unittest.main()
