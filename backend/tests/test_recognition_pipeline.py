import unittest

import cv2
import numpy as np

from app.services.color_recognition_service import ColorRecognitionService
from app.services.brand_recognition_service import BrandRecognitionService
from app.services.inference_service import InferencePipeline
from app.services.ocr_service import OCRCandidate, OCRResult, OCRService
from app.services.track_state_service import TrackStateService
from app.services.tracking_service import TrackedObject, TrackingService


class DualScalePipelineTests(unittest.TestCase):
    def test_analysis_bbox_maps_back_to_source_pixels(self):
        source = np.zeros((200, 400, 3), dtype=np.uint8)
        source[50:150, 100:200] = (10, 20, 30)
        analysis = np.zeros((100, 200, 3), dtype=np.uint8)

        crop = InferencePipeline._crop_source_vehicle(
            analysis, source, [50, 25, 100, 75], padding=0
        )

        self.assertEqual(crop.shape, (100, 100, 3))
        self.assertTrue(np.all(crop == (10, 20, 30)))

    def test_complete_sharp_vehicle_crop_beats_clipped_close_fragment(self):
        rng = np.random.default_rng(7)
        complete = rng.integers(0, 255, (70, 110, 3), dtype=np.uint8)
        clipped = np.full((90, 160, 3), 100, dtype=np.uint8)
        frame = np.zeros((100, 200, 3), dtype=np.uint8)

        complete_score = InferencePipeline._score_vehicle_crop(
            complete, frame, [45, 15, 155, 85], 0.82, object()
        )
        clipped_score = InferencePipeline._score_vehicle_crop(
            clipped, frame, [0, 10, 160, 100], 0.92, None
        )

        self.assertGreater(complete_score, clipped_score)

    def test_larger_clear_plate_replaces_early_small_crop(self):
        rng = np.random.default_rng(11)
        early = rng.integers(35, 220, (14, 56, 3), dtype=np.uint8)
        closer = cv2.resize(early, (168, 42), interpolation=cv2.INTER_CUBIC)

        early_score = InferencePipeline._score_plate_crop(early, 0.82)
        closer_score = InferencePipeline._score_plate_crop(closer, 0.78)

        self.assertGreater(closer_score, early_score + 0.01)

    def test_large_blurred_plate_does_not_beat_readable_crop(self):
        rng = np.random.default_rng(17)
        readable = rng.integers(20, 235, (28, 112, 3), dtype=np.uint8)
        blurred = np.full((50, 180, 3), 128, dtype=np.uint8)

        readable_score = InferencePipeline._score_plate_crop(readable, 0.80)
        blurred_score = InferencePipeline._score_plate_crop(blurred, 0.90)

        self.assertGreater(readable_score, blurred_score)

    def test_plate_sampling_interval_one_processes_every_frame(self):
        self.assertTrue(all(
            InferencePipeline._is_interval_due(frame, 1)
            for frame in range(1, 6)
        ))
        self.assertEqual(
            [frame for frame in range(1, 12) if InferencePipeline._is_interval_due(frame, 5)],
            [1, 6, 11],
        )


class OverlayLabelTests(unittest.TestCase):
    def test_searching_label_uses_ascii_bars_without_question_marks(self):
        label = InferencePipeline._build_overlay_label({
            "track_id": 87,
            "plate": None,
            "plate_status": "searching",
            "plate_confidence": 0.0,
            "color": "gray",
        })

        self.assertEqual(label, "#87 | SEARCHING | GRAY")
        self.assertTrue(label.isascii())
        self.assertNotIn("?", label)

    def test_verified_label_is_compact(self):
        label = InferencePipeline._build_overlay_label({
            "track_id": 92,
            "plate": "SC56DYP",
            "plate_status": "verified",
            "plate_confidence": 0.74,
            "color": "red",
        })

        self.assertEqual(label, "#92 | SC56DYP | 74% | RED")


class TrackingIdentityTests(unittest.TestCase):
    def test_same_raw_id_cannot_merge_motorcycle_and_car(self):
        service = TrackingService.__new__(TrackingService)
        service._identity_map = {}
        service._identity_state = {}
        service._next_identity_id = 1
        service._identity_frame = 0
        motorcycle = TrackedObject(4, [0, 0, 20, 20], 0.8, "motorcycle")
        car = TrackedObject(4, [0, 0, 80, 60], 0.9, "car")

        motorcycle_id = service._stabilize_track_ids([motorcycle])[0].track_id
        car_id = service._stabilize_track_ids([car])[0].track_id

        self.assertNotEqual(motorcycle_id, car_id)

    def test_overlapping_car_fragment_with_new_raw_id_is_stitched(self):
        service = TrackingService.__new__(TrackingService)
        service._identity_map = {}
        service._identity_state = {}
        service._next_identity_id = 1
        service._identity_frame = 0
        first = TrackedObject(4, [20, 10, 140, 90], 0.85, "car")
        continuation = TrackedObject(9, [25, 12, 145, 92], 0.82, "truck")

        first_id = service._stabilize_track_ids([first])[0].track_id
        continuation_id = service._stabilize_track_ids([continuation])[0].track_id

        self.assertEqual(first_id, continuation_id)


class OCRVotingTests(unittest.TestCase):
    def setUp(self):
        self.service = OCRService.__new__(OCRService)
        self.service.voting_window = 10
        self.service.regex_profile = "AUTO"

    def test_one_character_jitter_is_clustered(self):
        candidates = [OCRCandidate(
            text="GX15OCJ",
            raw_text="GX15OCJ",
            confidence=0.55,
            regex_valid=True,
            regex_score=1.0,
            image_quality_score=0.8,
            count=1,
        )]
        result = OCRResult(
            text="GX15OCT",
            raw_text="GX15OCT",
            confidence=0.70,
            regex_valid=True,
            regex_score=1.0,
        )

        updated = self.service.update_voting(candidates, result, 0.9)

        self.assertEqual(len(updated), 1)
        self.assertEqual(updated[0].count, 2)
        self.assertEqual(updated[0].text, "GX15OCT")

    def test_short_wide_plate_gets_large_sharpened_variants(self):
        crop = np.full((18, 79, 3), 160, dtype=np.uint8)
        variants = self.service._prepare_variants(crop)

        self.assertGreaterEqual(len(variants), 5)
        self.assertEqual(variants[0].shape[1], 420)
        self.assertGreaterEqual(variants[0].shape[0], 64)

    def test_character_consensus_can_replace_high_confidence_single_frame_error(self):
        candidates = [OCRCandidate(
            text="KA02HK1826", raw_text="K402HK1826", confidence=0.37,
            regex_valid=True, regex_score=1.0, image_quality_score=0.25,
        )]
        better_frame = OCRResult(
            text="KA02MN1826", raw_text="KA02MN1826", confidence=0.30,
            regex_valid=True, regex_score=1.0,
        )

        self.service.update_voting(candidates, better_frame, 0.90)
        updated = self.service.update_voting(candidates, better_frame, 0.90)

        self.assertEqual(updated[0].text, "KA02MN1826")


class ColorConfidenceTests(unittest.TestCase):
    def test_unanimous_weak_fallback_does_not_become_confident_blue(self):
        service = ColorRecognitionService.__new__(ColorRecognitionService)
        color, confidence = service.resolve_color([("blue", 0.20)] * 10)
        self.assertEqual(color, "unknown")
        self.assertLess(confidence, 0.32)

    def test_scene_blue_cast_is_not_reported_as_vehicle_blue(self):
        service = ColorRecognitionService(None)
        scene = np.full((180, 320, 3), (120, 90, 70), dtype=np.uint8)
        crop = np.full((100, 180, 3), (125, 95, 75), dtype=np.uint8)
        service.update_scene_context(scene)

        color, _ = service.recognize(crop)

        self.assertIn(color, {"black", "gray", "silver", "white", "unknown"})

    def test_glossy_black_body_survives_bright_reflections(self):
        service = ColorRecognitionService(None)
        hsv = np.full((100, 180, 3), (0, 12, 135), dtype=np.uint8)
        hsv[:35, :, 2] = 70
        hsv[80:, :, 2] = 225
        crop = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        color, confidence = service._classify_hsv(crop)

        self.assertEqual(color, "black")
        self.assertGreaterEqual(confidence, 0.36)

    def test_bright_silver_body_is_not_relabelled_black(self):
        service = ColorRecognitionService(None)
        crop = np.full((100, 180, 3), (178, 180, 182), dtype=np.uint8)

        color, _ = service._classify_hsv(crop)

        self.assertEqual(color, "silver")

    def test_black_vehicle_wins_over_blue_windshield_and_red_lamps(self):
        service = ColorRecognitionService(None)
        crop = np.full((220, 140, 3), (28, 28, 30), dtype=np.uint8)
        crop[65:135, 28:112] = (135, 75, 25)
        crop[165:205, 8:30] = (20, 20, 220)
        crop[165:205, 110:132] = (20, 20, 220)

        color, confidence = service.recognize(crop)

        self.assertEqual(color, "black")
        self.assertGreaterEqual(confidence, 0.36)

    def test_white_vehicle_wins_over_dark_glass(self):
        service = ColorRecognitionService(None)
        crop = np.full((220, 140, 3), (225, 225, 225), dtype=np.uint8)
        crop[65:135, 28:112] = (55, 45, 35)

        color, confidence = service.recognize(crop)

        self.assertEqual(color, "white")
        self.assertGreaterEqual(confidence, 0.36)

    def test_dark_blue_paint_is_not_relabelled_black(self):
        service = ColorRecognitionService(None)
        crop = np.full((220, 140, 3), (125, 65, 25), dtype=np.uint8)
        crop[65:135, 28:112] = (70, 50, 35)

        color, confidence = service.recognize(crop)

        self.assertEqual(color, "blue")
        self.assertGreaterEqual(confidence, 0.36)

    def test_red_body_wins_over_one_large_blue_reflection_cluster(self):
        service = ColorRecognitionService(None)
        crop = np.full((140, 240, 3), (120, 10, 210), dtype=np.uint8)
        crop[55:105, 145:205] = (210, 90, 25)

        color, confidence = service.recognize(crop)

        self.assertEqual(color, "red")
        self.assertGreaterEqual(confidence, 0.40)

    def test_track_consensus_prefers_high_quality_close_blue_samples(self):
        service = ColorRecognitionService.__new__(ColorRecognitionService)
        history = [
            {"color": "white", "confidence": 0.55, "quality_score": 0.25, "hex_code": "#E8EEF2"}
            for _ in range(3)
        ] + [
            {"color": "blue", "confidence": 0.72, "quality_score": 0.90, "hex_code": "#315D8C"}
            for _ in range(7)
        ]

        result = service.resolve_color_details(history)

        self.assertEqual(result["color"], "blue")
        self.assertEqual(result["support_count"], 7)
        self.assertEqual(result["sample_count"], 10)
        self.assertGreater(result["distribution"]["blue"], result["distribution"]["white"])
        self.assertRegex(result["hex_code"], r"^#[0-9A-F]{6}$")

    def test_frame_observation_contains_measured_html_color(self):
        service = ColorRecognitionService(None)
        crop = np.full((140, 240, 3), (120, 10, 210), dtype=np.uint8)

        result = service.recognize_details(crop)

        self.assertEqual(result["color"], "red")
        self.assertRegex(result["hex_code"], r"^#[0-9A-F]{6}$")
        self.assertGreater(result["quality_score"], 0.0)


class BrandRecognitionTests(unittest.TestCase):
    def test_repeated_clear_brand_is_resolved(self):
        history = [
            {"candidates": [{"brand": "Toyota", "confidence": 0.72}]},
            {"candidates": [{"brand": "Toyota", "confidence": 0.68}]},
            {"candidates": [{"brand": "Toyota", "confidence": 0.75}]},
        ]

        result = BrandRecognitionService.resolve_brand(history)

        self.assertEqual(result["brand"], "Toyota")
        self.assertEqual(result["support_count"], 3)

    def test_one_clear_logo_can_be_shown_as_provisional(self):
        result = BrandRecognitionService.resolve_brand([{
            "candidates": [
                {"brand": "Honda", "confidence": 0.55},
                {"brand": "Toyota", "confidence": 0.22},
            ]
        }])

        self.assertEqual(result["brand"], "Honda")
        self.assertTrue(result["provisional"])

    def test_close_competing_logos_remain_unknown(self):
        history = [
            {"candidates": [
                {"brand": "Honda", "confidence": 0.48},
                {"brand": "Toyota", "confidence": 0.45},
            ]}
            for _ in range(4)
        ]

        result = BrandRecognitionService.resolve_brand(history)

        self.assertEqual(result["brand"], "unknown")
        self.assertEqual(result["candidate_brand"], "Honda")

    def test_dataset_spelling_is_normalized_for_ui(self):
        self.assertEqual(BrandRecognitionService._display_name("wolksvogen"), "Volkswagen")
        self.assertEqual(BrandRecognitionService._display_name("pegout"), "Peugeot")


class TrackGraceTests(unittest.IsolatedAsyncioTestCase):
    async def test_short_detector_gap_does_not_close_track(self):
        state = TrackStateService(
            camera_id=7,
            processing_run_id="run-a",
            max_missing_frames=2,
        )
        await state.upsert_track(1, "car", [1, 2, 3, 4], 0.9, "2026-06-30T00:00:00+00:00")

        self.assertEqual(await state.remove_stale_tracks([], 0), [])
        self.assertEqual(await state.remove_stale_tracks([], 0), [])
        removed = await state.remove_stale_tracks([], 0)

        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0].processing_run_id, "run-a")

    async def test_vehicle_class_uses_multi_frame_confidence_voting(self):
        state = TrackStateService(camera_id=8, processing_run_id="run-class")
        await state.upsert_track(1, "truck", [1, 2, 30, 40], 0.35, "2026-07-01T00:00:00+00:00")
        for second in range(1, 4):
            track, _ = await state.upsert_track(
                1, "car", [1, 2, 30, 40], 0.85, f"2026-07-01T00:00:0{second}+00:00"
            )

        self.assertEqual(track.vehicle_class, "car")


if __name__ == "__main__":
    unittest.main()
