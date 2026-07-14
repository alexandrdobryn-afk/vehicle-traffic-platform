import cv2
import numpy as np
import asyncio
import time
import os
import uuid
from collections import Counter
from typing import Optional, List, Dict
from datetime import datetime, timezone
import logging

from app.config import settings, AIMode
from app.services.object_detection_service import ObjectDetectionService, Detection
from app.services.tracking_service import TrackingService, TrackedObject
from app.services.frame_quality_service import FrameQualityService
from app.services.track_state_service import TrackStateService, ActiveTrack
from app.models.model_registry import model_registry
from app.services.runtime_service import resolve_execution_policy
from app.services.kalman_prediction_service import KalmanPredictionService
from app.services.classification_service import ClassificationService
from app.services.segmentation_service import SegmentationService
from app.services.ocr_service import OCRService
from app.services.reid_service import ReIdentificationService
from app.services.super_resolution_service import SuperResolutionService
from app.services.geo_projection_service import GeoProjectionService
from app.services.object_memory_service import ObjectMemoryService

logger = logging.getLogger(__name__)

_COMPUTE_CONCURRENCY = max(1, int(os.environ.get("AI_INFERENCE_CONCURRENCY", "1")))
_COMPUTE_SEMAPHORE = asyncio.Semaphore(_COMPUTE_CONCURRENCY)


async def _run_compute(callback, *args, **kwargs):
    """Keep model inference off the API event loop without oversubscribing it."""
    async with _COMPUTE_SEMAPHORE:
        return await asyncio.to_thread(callback, *args, **kwargs)


class InferencePipeline:
    """
    Orchestrates the full AI pipeline per camera:
    Frame -> Detect -> Track -> Kalman -> Classify/Segment -> State
    Supports Speed / Balanced / Quality / Hybrid modes.
    """

    def __init__(
        self,
        camera_id: int,
        ai_mode: str = "balanced",
        redis_client=None,
        db_session=None,
        runtime_settings: Optional[dict] = None,
    ):
        self.camera_id = camera_id
        self.ai_mode = ai_mode
        self.redis = redis_client
        self.db = db_session
        self.runtime_settings = runtime_settings or {}
        self.pipeline_mode = self.runtime_settings.get("pipeline_mode", "automatic")
        self.pipeline_config = self.runtime_settings.get("pipeline_config") or {}
        self.task_profile = "aerial_small_objects"
        configured_classes = self.runtime_settings.get("target_classes")
        self.target_classes = self._normalize_target_classes(
            configured_classes if isinstance(configured_classes, list)
            else self.pipeline_config.get("target_classes", [])
        )
        raw_aerial = self.runtime_settings.get("aerial") or self.pipeline_config.get("aerial") or {}
        self.aerial_config = {
            "enabled": self.task_profile == "aerial_small_objects" or bool(raw_aerial.get("enabled", False)),
            "size": raw_aerial.get("tile_size", raw_aerial.get("size", 1024)),
            "overlap": raw_aerial.get("tile_overlap", raw_aerial.get("overlap", 0.2)),
            "nms_iou": raw_aerial.get("nms_iou", 0.5),
            "scale": raw_aerial.get("scale", 1.0),
            "enhance": raw_aerial.get("enhance", False),
        }
        self.runtime_policy = resolve_execution_policy(
            self.runtime_settings.get("execution_provider", "auto"),
            int(self.runtime_settings.get("gpu_device_index", 0)),
            self.runtime_settings.get("runtime_fallback", "fail_closed"),
        )
        if not self.runtime_policy["available"]:
            raise RuntimeError("Requested CUDA runtime is unavailable and fallback is disabled")
        self.execution_device = self.runtime_policy["device"]
        self.processing_run_id = uuid.uuid4().hex
        self.stage_counts = Counter()
        self.last_stage_latency_ms: Dict[str, float] = {}
        self.source_resolution = None

        self.frame_count = 0
        self.fps_counter = 0
        self.fps = 0.0
        self.last_latency_ms = 0
        self._fps_time = time.time()
        self._running = False

        # Frame skip settings
        default_frame_skip = {
            "speed": settings.FRAME_SKIP_SPEED,
            "balanced": settings.FRAME_SKIP_BALANCED,
            "quality": settings.FRAME_SKIP_QUALITY,
            "hybrid": settings.FRAME_SKIP_BALANCED,
            "practical": settings.FRAME_SKIP_BALANCED,
            "max_accuracy": settings.FRAME_SKIP_QUALITY,
            "edge_onnx": settings.FRAME_SKIP_SPEED,
        }.get(ai_mode, 2)
        self.frame_skip = max(1, int(self.runtime_settings.get("frame_skip", default_frame_skip)))
        self.object_confidence = float(self.runtime_settings.get(
            "object_confidence_threshold", 0.35
        ))
        self.min_track_frames_for_event = max(1, int(self.runtime_settings.get(
            "min_track_frames_for_event",
            self.pipeline_config.get("min_track_frames_for_event", 4),
        )))
        self.min_track_duration_seconds = max(0.0, float(self.runtime_settings.get(
            "min_track_duration_seconds",
            self.pipeline_config.get("min_track_duration_seconds", 0.25),
        )))
        self.detection_filters = {
            "min_box_width": max(1, int(self.runtime_settings.get(
                "min_box_width", self.pipeline_config.get("min_box_width", 14)
            ))),
            "min_box_height": max(1, int(self.runtime_settings.get(
                "min_box_height", self.pipeline_config.get("min_box_height", 14)
            ))),
            "min_box_area_ratio": max(0.0, float(self.runtime_settings.get(
                "min_box_area_ratio", self.pipeline_config.get("min_box_area_ratio", 0.00008)
            ))),
            "max_box_area_ratio": min(1.0, float(self.runtime_settings.get(
                "max_box_area_ratio", self.pipeline_config.get("max_box_area_ratio", 0.25)
            ))),
            "max_box_aspect_ratio": max(1.0, float(self.runtime_settings.get(
                "max_box_aspect_ratio", self.pipeline_config.get("max_box_aspect_ratio", 6.0)
            ))),
        }
        self.save_crops = bool(self.runtime_settings.get("save_crops", True))
        self.anonymization_mode = bool(self.runtime_settings.get("anonymization_mode", False))
        self.classification_config = self.runtime_settings.get("classification") or self.pipeline_config.get("classification") or {}
        self.segmentation_config = self.runtime_settings.get("segmentation") or self.pipeline_config.get("segmentation") or {}
        self.kalman_config = self.runtime_settings.get("kalman_prediction") or self.pipeline_config.get("kalman_prediction") or {}
        self.ocr_config = self.runtime_settings.get("ocr") or self.pipeline_config.get("ocr") or {}
        self.reid_config = self.runtime_settings.get("reid") or self.pipeline_config.get("reid") or {}
        self.object_memory_config = self.runtime_settings.get("object_memory") or self.pipeline_config.get("object_memory") or {}
        self.geo_config = self.runtime_settings.get("geo") or self.pipeline_config.get("geo") or {}
        self.super_resolution_config = self.runtime_settings.get("super_resolution") or self.pipeline_config.get("super_resolution") or {}
        self.telemetry_config = self.runtime_settings.get("telemetry") or self.pipeline_config.get("telemetry") or {}
        self.classification_interval_frames = max(1, int(self.classification_config.get("interval_frames", 15)))
        self.segmentation_interval_frames = max(1, int(self.segmentation_config.get("interval_frames", 5)))
        self.ocr_interval_frames = max(1, int(self.ocr_config.get("interval_frames", 30)))
        resolution = str(self.runtime_settings.get("input_resolution", "")).lower().split("x")
        self.input_resolution = None
        if len(resolution) == 2 and all(part.isdigit() for part in resolution):
            self.input_resolution = (int(resolution[0]), int(resolution[1]))
        # Tiling must receive the untouched high-resolution frame. A global
        # resize before tiling destroys the small-object detail it is meant to preserve.
        if self.aerial_config["enabled"]:
            self.input_resolution = None

        self._init_services()

    def _init_services(self):
        """Initialize all AI services based on mode."""
        logger.info(f"[Pipeline cam={self.camera_id}] Initializing {self.ai_mode} mode")

        manual = self.pipeline_config if self.pipeline_mode == "manual" else {}
        object_model = (
            model_registry.get_named_object_detector(
                manual.get("object_detector"),
                self.ai_mode,
            )
            if manual.get("object_detector")
            else model_registry.get_object_detector(self.ai_mode)
        )
        if self.ai_mode == "hybrid" and object_model.model_type != "yolo":
            raise RuntimeError(
                "Dual verification mode requires a YOLO detector as the first pass"
            )

        # Choose tracker
        default_tracker = {
            "speed": "bytetrack",
            "balanced": "bytetrack",
            "quality": "botsort",
            "hybrid": "bytetrack",
            "practical": "ocsort",
            "max_accuracy": "botsort",
            "edge_onnx": "bytetrack",
        }.get(self.ai_mode, "bytetrack")
        tracker_mode = manual.get("tracker") or self.runtime_settings.get("tracker_mode", default_tracker)
        object_memory_enabled = bool(self.object_memory_config.get("enabled", True))

        self.detector = ObjectDetectionService(
            ai_mode=AIMode(self.ai_mode),
            model_config=object_model,
            execution_device=self.execution_device,
            accepted_classes=self.target_classes,
            aerial_config=self.aerial_config,
        )
        self.rfdetr_detector = None
        if self.ai_mode == "hybrid":
            try:
                verifier_name = manual.get("verifier_detector") or "rfdetr_medium_object"
                rfdetr_model = model_registry.get_named_object_detector(
                    verifier_name, self.ai_mode
                )
                if rfdetr_model.model_type != "rfdetr":
                    raise RuntimeError("Dual verification requires RF-DETR as verifier")
                self.rfdetr_detector = ObjectDetectionService(
                    ai_mode=AIMode.QUALITY,
                    model_config=rfdetr_model,
                    execution_device=self.execution_device,
                    accepted_classes=self.target_classes,
                    aerial_config=self.aerial_config,
                )
            except Exception as exc:
                logger.warning("Hybrid RF-DETR re-check disabled: %s", exc)
        self.tracker = TrackingService(
            tracker_mode=tracker_mode,
            ai_mode=self.ai_mode,
            identity_stitching=False,
        )
        reid_enabled = bool(self.reid_config.get("enabled", False))
        reid_model_name = str(self.reid_config.get("model", "hsv_histogram_v1"))
        if reid_model_name == "auto":
            reid_model_name = "hsv_histogram_v1"
        self.re_identifier = ReIdentificationService(
            similarity_threshold=float(self.reid_config.get("similarity_threshold", 0.72)),
            model_name=reid_model_name,
        ) if reid_enabled else None
        self.object_memory = ObjectMemoryService(
            max_gap_frames=int(self.object_memory_config.get("max_gap_frames", 90)),
            merge_threshold=float(self.object_memory_config.get("merge_threshold", 0.48)),
            duplicate_iou=float(self.object_memory_config.get("duplicate_iou", 0.45)),
            duplicate_contained=float(self.object_memory_config.get("duplicate_contained", 0.78)),
            appearance_threshold=float(self.object_memory_config.get("appearance_threshold", 0.68)),
            embedding_enabled=reid_enabled,
            embedding_model=reid_model_name,
            store_embedding_vector=bool(self.reid_config.get("store_vector", False)),
        ) if object_memory_enabled else None
        kalman_enabled = bool(self.kalman_config.get("enabled", self.task_profile == "aerial_small_objects"))
        max_prediction_frames = max(0, min(2, int(self.kalman_config.get("max_prediction_frames", 2))))
        self.kalman_predictor = KalmanPredictionService(
            max_prediction_frames=max_prediction_frames,
            smoothing=bool(self.kalman_config.get("smoothing", True)),
        ) if kalman_enabled else None
        classifier_model = model_registry.get_object_classifier()
        classifier_enabled = bool(self.classification_config.get("enabled", False))
        self.classifier = ClassificationService(
            classifier_model,
            execution_device=self.execution_device,
            threshold=float(self.classification_config.get("confidence_threshold", 0.35)),
        ) if classifier_enabled else None
        segmenter_model = model_registry.get_object_segmenter()
        segmentation_enabled = bool(self.segmentation_config.get("enabled", False))
        self.segmenter = SegmentationService(
            segmenter_model,
            execution_device=self.execution_device,
            threshold=float(self.segmentation_config.get("confidence_threshold", 0.35)),
        ) if segmentation_enabled else None
        ocr_enabled = bool(self.ocr_config.get("enabled", False))
        ocr_model = model_registry.get_text_ocr_engine(self.ai_mode) if ocr_enabled else None
        self.ocr = OCRService(
            ocr_model,
            execution_device=self.execution_device,
            threshold=float(self.ocr_config.get("confidence_threshold", 0.35)),
        ) if ocr_enabled and ocr_model is not None else None
        geo_enabled = bool(self.geo_config.get("enabled", False))
        self.geo_projector = GeoProjectionService(
            telemetry=self.telemetry_config,
            coordinate_output=bool(self.geo_config.get("coordinate_output", False)),
        ) if geo_enabled else None
        super_resolution_enabled = bool(self.super_resolution_config.get("enabled", False))
        self.super_resolver = SuperResolutionService(
            engine=str(self.super_resolution_config.get("engine", "auto")),
            min_object_size_px=int(self.super_resolution_config.get("min_object_size_px", 32)),
            max_crops_per_frame=int(self.super_resolution_config.get("max_crops_per_frame", 8)),
        ) if super_resolution_enabled else None
        self.quality_service = FrameQualityService()
        self.track_state = TrackStateService(
            camera_id=self.camera_id,
            redis_client=self.redis,
            processing_run_id=self.processing_run_id,
            max_missing_frames=max(1, min(6, int(self.runtime_settings.get("track_missing_grace_frames", 6)))),
        )
        self.resolved_pipeline = {
            "pipeline_mode": self.pipeline_mode,
            "ai_mode": self.ai_mode,
            "task_profile": self.task_profile,
            "target_classes": self.target_classes,
            "runtime": {
                key: self.runtime_policy[key]
                for key in (
                    "requested", "resolved", "device", "fallback_policy",
                    "fallback_reason", "gpu_device_index", "precision",
                )
            },
            "object_detector": {"name": object_model.name, "format": object_model.format},
            "escalation_detector": (
                {"name": self.rfdetr_detector.model_config.name, "format": self.rfdetr_detector.model_config.format}
                if self.rfdetr_detector else None
            ),
            "tracker": {"name": self.tracker.resolved_mode, "format": "algorithm"},
            "object_memory": {
                "name": "canonical_object_memory",
                "format": "algorithm",
                "enabled": object_memory_enabled,
                "available": object_memory_enabled,
                "embedding_matching": reid_enabled,
            },
            "kalman_prediction": ({"name": "constant_velocity_kalman", "format": "algorithm"} if self.kalman_predictor else None),
            "classification": ({"name": classifier_model.name, "format": classifier_model.format, "available": self.classifier.available} if self.classifier else None),
            "segmentation": ({"name": segmenter_model.name, "format": segmenter_model.format, "available": self.segmenter.available} if self.segmenter else None),
            "ocr": {
                "name": ocr_model.name if ocr_model else self.ocr_config.get("engine", "auto"),
                "format": ocr_model.format if ocr_model else "optional_adapter",
                "enabled": ocr_enabled,
                "available": bool(self.ocr and self.ocr.available),
            },
            "reid": {
                "name": reid_model_name,
                "format": "algorithm",
                "enabled": reid_enabled,
                "available": bool(self.re_identifier and self.re_identifier.available),
                "role": "embedding_matching",
                "store_vector": bool(self.reid_config.get("store_vector", False)),
            },
            "geo": {
                "name": self.geo_config.get("telemetry_source", "metadata"),
                "format": "metadata",
                "enabled": geo_enabled,
                "available": bool(self.geo_projector and self.geo_projector.available),
                "coordinate_output": bool(self.geo_config.get("coordinate_output", False)),
            },
            "super_resolution": {
                "name": "opencv_lanczos",
                "format": "algorithm",
                "enabled": super_resolution_enabled,
                "available": bool(self.super_resolver and self.super_resolver.available),
            },
            "features": {
                "object_memory": object_memory_enabled,
                "embedding_matching": reid_enabled,
                "identity_stitching": object_memory_enabled,
                "confidence_engine": True,
                "tiled_inference": self.aerial_config["enabled"],
                "preserve_source_resolution": self.aerial_config["enabled"],
            },
        }

        logger.info(f"[Pipeline cam={self.camera_id}] Ready")

    async def process_frame(
        self,
        frame: np.ndarray,
        source_frame: Optional[np.ndarray] = None,
        source_metadata: Optional[dict] = None,
    ) -> Optional[dict]:
        """
        Process one frame through the full pipeline.
        Returns metadata dict for WebSocket broadcast or None if skipped.
        """
        self.frame_count += 1
        t_start = time.time()
        source_frame = source_frame if source_frame is not None else frame
        source_metadata = source_metadata or {}
        video_timestamp_seconds = source_metadata.get("video_timestamp_seconds")
        self.source_resolution = (source_frame.shape[1], source_frame.shape[0])

        # Frame skip
        if self.frame_count % self.frame_skip != 0:
            self.stage_counts["frames_skipped"] += 1
            return None
        self.stage_counts["frames_processed"] += 1

        # Frame quality check
        quality = self.quality_service.check_frame(frame)
        if not quality.is_good and self.ai_mode != "quality":
            self.stage_counts["frames_rejected_quality"] += 1
            return None

        now = datetime.now(timezone.utc).isoformat()
        stage_latency = {
            "object_detection": 0.0,
            "tracking": 0.0,
            "embedding": 0.0,
            "object_memory": 0.0,
            "redis": 0.0,
            "kalman_prediction": 0.0,
            "classification": 0.0,
            "segmentation": 0.0,
            "ocr": 0.0,
            "reid": 0.0,
            "geo": 0.0,
            "super_resolution": 0.0,
        }

        # в”Ђв”Ђ 1. Object detection в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
        use_rfdetr = self.detector.rfdetr_model is not None and self.ai_mode in {"quality", "max_accuracy"}
        stage_started = time.perf_counter()
        detections: List[Detection] = await _run_compute(
            self.detector.detect,
            frame,
            confidence_threshold=self.object_confidence,
            use_rfdetr=use_rfdetr,
        )
        detections = self._filter_detections(detections, frame.shape)
        stage_latency["object_detection"] += (time.perf_counter() - stage_started) * 1000
        self.stage_counts["objects_detected"] += len(detections)

        # Hybrid: re-check with RF-DETR if low confidence
        if self.ai_mode == "hybrid" and self.rfdetr_detector is not None:
            scheduled_check = self.frame_count % 5 == 0
            uncertainty_check = bool(detections) and self.detector.should_use_rfdetr(frame, detections)
            if scheduled_check or uncertainty_check:
                stage_started = time.perf_counter()
                detections = await _run_compute(
                    self.rfdetr_detector.detect,
                    frame,
                    confidence_threshold=self.object_confidence,
                    use_rfdetr=True,
                )
                detections = self._filter_detections(detections, frame.shape)
                stage_latency["object_detection"] += (time.perf_counter() - stage_started) * 1000
                self.stage_counts["rfdetr_escalations"] += 1

        # в”Ђв”Ђ 2. Tracking в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
        stage_started = time.perf_counter()
        tracked_objects: List[TrackedObject] = await _run_compute(
            self.tracker.update, detections, frame
        )
        stage_latency["tracking"] += (time.perf_counter() - stage_started) * 1000
        if self.object_memory is not None and self.re_identifier is not None and self.re_identifier.available:
            stage_started = time.perf_counter()
            embedded = 0
            for tracked in tracked_objects:
                object_crop = self._crop_source_object(frame, frame, tracked.bbox, padding=4)
                embedding = await _run_compute(self.re_identifier.embed, object_crop)
                vector = embedding.get("vector")
                if vector:
                    tracked.appearance_signature = vector
                    embedded += 1
            self.stage_counts["objects_embedded"] += embedded
            stage_latency["embedding"] += (time.perf_counter() - stage_started) * 1000
        if self.object_memory is not None:
            stage_started = time.perf_counter()
            tracked_objects = self.object_memory.update(tracked_objects, frame)
            object_memory_stats = self.object_memory.last_stats
            if object_memory_stats.get("merged_tracks"):
                self.stage_counts["object_memory_merged_tracks"] += object_memory_stats["merged_tracks"]
            if object_memory_stats.get("same_frame_duplicates"):
                self.stage_counts["object_memory_suppressed_duplicates"] += object_memory_stats["same_frame_duplicates"]
            stage_latency["object_memory"] += (time.perf_counter() - stage_started) * 1000
        if self.kalman_predictor is not None:
            stage_started = time.perf_counter()
            tracked_objects = self.kalman_predictor.update(tracked_objects, frame.shape)
            before_dedupe = len(tracked_objects)
            tracked_objects = self._suppress_duplicate_tracks(tracked_objects)
            suppressed = before_dedupe - len(tracked_objects)
            if suppressed:
                self.stage_counts["tracks_suppressed_duplicate"] += suppressed
            stage_latency["kalman_prediction"] += (time.perf_counter() - stage_started) * 1000
            self.stage_counts["kalman_predicted_objects"] += sum(1 for item in tracked_objects if item.predicted)
        active_ids = [t.track_id for t in tracked_objects]

        segments = []
        if self.segmenter is not None and self.segmenter.available and self.frame_count % self.segmentation_interval_frames == 0:
            stage_started = time.perf_counter()
            segments = await _run_compute(self.segmenter.segment, frame)
            stage_latency["segmentation"] += (time.perf_counter() - stage_started) * 1000
            self.stage_counts["segments_detected"] += len(segments)
        if self.super_resolver is not None:
            self.super_resolver.begin_frame()

        # в”Ђв”Ђ 3. Per-track processing в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
        ws_objects = []
        redis_tracks = []
        for tracked in tracked_objects:
            tid = tracked.track_id
            track, is_new = await self.track_state.upsert_track(
                track_id=tid,
                object_class=tracked.class_name,
                bbox=tracked.bbox,
                detection_conf=tracked.confidence,
                now=now,
                predicted=tracked.predicted,
                motion_vector=tracked.motion_vector,
                video_timestamp_seconds=video_timestamp_seconds,
            )
            if self.object_memory is not None:
                memory_card = self.object_memory.card_summary(tid)
                if memory_card:
                    await self.track_state.update_attributes(tid, object_memory=memory_card)

            # Detect/track on the analysis frame, but preserve source pixels for
            # stored evidence.
            object_crop = self._crop_source_object(
                frame, source_frame, tracked.bbox, padding=10
            )
            analysis_crop = object_crop

            if tracked.predicted:
                updated_track = await self.track_state.get_track(tid)
                if updated_track:
                    redis_tracks.append(updated_track)
                    ws_objects.append(updated_track.to_ws_dict())
                continue

            if self.super_resolver is not None and self.super_resolver.available:
                stage_started = time.perf_counter()
                analysis_crop, super_resolution = await _run_compute(
                    self.super_resolver.enhance,
                    object_crop,
                )
                stage_latency["super_resolution"] += (time.perf_counter() - stage_started) * 1000
                await self.track_state.update_attributes(tid, super_resolution=super_resolution)
                self.stage_counts["super_resolution_enhanced"] += int(super_resolution["status"] == "enhanced")

            if (
                not tracked.predicted and self.classifier is not None and self.classifier.available
                and self._is_interval_due(track.frame_count, self.classification_interval_frames)
            ):
                stage_started = time.perf_counter()
                classification = await _run_compute(self.classifier.classify, analysis_crop)
                stage_latency["classification"] += (time.perf_counter() - stage_started) * 1000
                await self.track_state.update_attributes(tid, classification=classification)
                self.stage_counts["objects_classified"] += int(classification["status"] == "classified")

            if (
                self.ocr is not None and self.ocr.available
                and self._is_interval_due(track.frame_count, self.ocr_interval_frames)
            ):
                stage_started = time.perf_counter()
                ocr = await _run_compute(self.ocr.recognize, analysis_crop)
                stage_latency["ocr"] += (time.perf_counter() - stage_started) * 1000
                await self.track_state.update_attributes(tid, ocr=ocr)
                self.stage_counts["ocr_recognized"] += int(ocr["status"] == "recognized")

            if self.re_identifier is not None and self.re_identifier.available:
                stage_started = time.perf_counter()
                reid = await _run_compute(self.re_identifier.update, tid, analysis_crop)
                stage_latency["reid"] += (time.perf_counter() - stage_started) * 1000
                await self.track_state.update_attributes(tid, reid=reid)
                self.stage_counts["reid_candidate_matches"] += int(reid["status"] == "candidate_match")

            if self.geo_projector is not None and self.geo_projector.available:
                stage_started = time.perf_counter()
                geo = self.geo_projector.project(tracked.bbox, frame.shape, source_frame.shape)
                stage_latency["geo"] += (time.perf_counter() - stage_started) * 1000
                await self.track_state.update_attributes(tid, geo=geo)
                self.stage_counts["geo_projected_objects"] += 1

            # Save the clearest, most complete object crop. Detector confidence
            # alone can prefer a clipped frame over a better training sample.
            object_crop_score = self._score_object_crop(
                object_crop, frame, tracked.bbox, tracked.confidence
            )
            if (
                self._is_confirmed_track(track)
                and settings.STORAGE_PATH
                and object_crop.size
                and (not track.best_crop_path or object_crop_score > track.best_crop_score)
            ):
                stored_crop = object_crop.copy()
                await self._save_object_crop(tid, stored_crop, track)
                track.best_crop_score = object_crop_score

            # A confirmed object is the product record. Persist it after its
            # evidence crop is available; lifecycle labels stay internal.
            if (
                not tracked.predicted
                and not track.events_fired.get("object_entered")
                and self._is_confirmed_track(track)
            ):
                await self._fire_event("object_entered", track, frame)
                await self.track_state.mark_event_fired(tid, "object_entered")

            # Sync to Redis
            updated_track = await self.track_state.get_track(tid)
            if updated_track:
                redis_tracks.append(updated_track)
                ws_objects.append(updated_track.to_ws_dict())
                if updated_track.frame_count % 30 == 0:
                    await self._fire_event("track_checkpoint", updated_track, None)

        stage_started = time.perf_counter()
        await self.track_state.sync_many_to_redis(redis_tracks)
        stage_latency["redis"] += (time.perf_counter() - stage_started) * 1000

        # в”Ђв”Ђ 4. Remove stale tracks в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
        stale = await self.track_state.remove_stale_tracks(active_ids, time.time())
        for stale_track in stale:
            if stale_track.events_fired.get("object_entered"):
                await self._fire_event("object_left", stale_track, None)
            await self.track_state.remove_from_redis(stale_track)

        # в”Ђв”Ђ 5. FPS calculation в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
        self.fps_counter += 1
        elapsed = time.time() - self._fps_time
        if elapsed >= 1.0:
            self.fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self._fps_time = time.time()

        latency_ms = int((time.time() - t_start) * 1000)
        self.last_latency_ms = latency_ms
        self.last_stage_latency_ms = {
            key: round(value, 1) for key, value in stage_latency.items()
        }

        return {
            "camera_id": self.camera_id,
            "timestamp": now,
            "fps": round(self.fps, 1),
            "latency_ms": latency_ms,
            "stage_latency_ms": self.last_stage_latency_ms,
            "frame_width": frame.shape[1],
            "frame_height": frame.shape[0],
            "source_frame_width": source_frame.shape[1],
            "source_frame_height": source_frame.shape[0],
            "processing_run_id": self.processing_run_id,
            "stage_counts": dict(self.stage_counts),
            "resolved_pipeline": self.resolved_pipeline,
            "objects": ws_objects,
            "segments": segments,
        }

    @staticmethod
    def _normalize_target_classes(classes: Optional[List[str]]) -> List[str]:
        normalized = [str(item).strip().lower() for item in (classes or []) if str(item).strip()]
        if any(item in {"*", "all", "all_classes"} for item in normalized):
            return []
        return normalized

    def _filter_detections(self, detections: List[Detection], frame_shape: tuple) -> List[Detection]:
        frame_h, frame_w = frame_shape[:2]
        frame_area = max(1, frame_w * frame_h)
        kept: List[Detection] = []
        rejected = 0
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            if width < self.detection_filters["min_box_width"] or height < self.detection_filters["min_box_height"]:
                rejected += 1
                continue
            area_ratio = (width * height) / frame_area
            if area_ratio < self.detection_filters["min_box_area_ratio"]:
                rejected += 1
                continue
            if area_ratio > self.detection_filters["max_box_area_ratio"]:
                rejected += 1
                continue
            aspect = max(width / max(height, 1), height / max(width, 1))
            if aspect > self.detection_filters["max_box_aspect_ratio"]:
                rejected += 1
                continue
            kept.append(detection)
        if rejected:
            self.stage_counts["detections_rejected_geometry"] += rejected
        return kept

    def _suppress_duplicate_tracks(self, tracks: List[TrackedObject]) -> List[TrackedObject]:
        """Keep the detector as source of truth and hide code-generated duplicates."""
        observed: List[TrackedObject] = []
        predicted: List[TrackedObject] = []
        for track in tracks:
            (predicted if track.predicted else observed).append(track)

        kept_observed: List[TrackedObject] = []
        for track in sorted(observed, key=self._track_rank, reverse=True):
            if any(self._duplicate_observed_track(track, kept) for kept in kept_observed):
                continue
            kept_observed.append(track)

        kept_predicted: List[TrackedObject] = []
        for track in sorted(predicted, key=self._track_rank, reverse=True):
            if any(self._same_physical_track(track, kept) for kept in kept_observed):
                continue
            if any(self._same_physical_track(track, kept) for kept in kept_predicted):
                continue
            kept_predicted.append(track)

        return sorted(kept_observed + kept_predicted, key=lambda item: item.track_id)

    @classmethod
    def _duplicate_observed_track(cls, left: TrackedObject, right: TrackedObject) -> bool:
        if cls._class_group(left.class_name) != cls._class_group(right.class_name):
            return False
        return cls._bbox_iou(left.bbox, right.bbox) >= 0.45 or cls._contained_overlap(left.bbox, right.bbox) >= 0.78

    @classmethod
    def _same_physical_track(cls, left: TrackedObject, right: TrackedObject) -> bool:
        if cls._class_group(left.class_name) != cls._class_group(right.class_name):
            return False
        iou = cls._bbox_iou(left.bbox, right.bbox)
        contained = cls._contained_overlap(left.bbox, right.bbox)
        center_score = cls._center_continuity_score(left.bbox, right.bbox)
        return iou >= 0.18 or contained >= 0.70 or center_score >= 0.72

    @staticmethod
    def _class_group(class_name: str) -> str:
        normalized = (class_name or "unknown").strip().lower() or "unknown"
        vehicle_like = {"bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"}
        return "vehicle_like" if normalized in vehicle_like else normalized

    @classmethod
    def _track_rank(cls, track: TrackedObject) -> tuple:
        return (not track.predicted, float(track.confidence), cls._bbox_area(track.bbox))

    @staticmethod
    def _bbox_area(box: List[int]) -> float:
        return float(max(0, box[2] - box[0]) * max(0, box[3] - box[1]))

    @classmethod
    def _bbox_iou(cls, left: List[int], right: List[int]) -> float:
        x1, y1 = max(left[0], right[0]), max(left[1], right[1])
        x2, y2 = min(left[2], right[2]), min(left[3], right[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = cls._bbox_area(left) + cls._bbox_area(right) - intersection
        return float(intersection / union) if union > 0 else 0.0

    @classmethod
    def _contained_overlap(cls, left: List[int], right: List[int]) -> float:
        x1, y1 = max(left[0], right[0]), max(left[1], right[1])
        x2, y2 = min(left[2], right[2]), min(left[3], right[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        return float(intersection / max(1.0, min(cls._bbox_area(left), cls._bbox_area(right))))

    @staticmethod
    def _center_continuity_score(left: List[int], right: List[int]) -> float:
        left_w, left_h = max(1, left[2] - left[0]), max(1, left[3] - left[1])
        right_w, right_h = max(1, right[2] - right[0]), max(1, right[3] - right[1])
        left_center = ((left[0] + left[2]) / 2, (left[1] + left[3]) / 2)
        right_center = ((right[0] + right[2]) / 2, (right[1] + right[3]) / 2)
        distance = float(np.hypot(left_center[0] - right_center[0], left_center[1] - right_center[1]))
        size_gate = max(50.0, 0.55 * max(left_w, left_h, right_w, right_h))
        if distance > size_gate:
            return 0.0
        area_ratio = (left_w * left_h) / max(1, right_w * right_h)
        if area_ratio < 0.18 or area_ratio > 5.5:
            return 0.0
        return 1.0 - (distance / size_gate)

    def _is_confirmed_track(self, track: ActiveTrack) -> bool:
        return (
            track.frame_count >= self.min_track_frames_for_event
            and track.duration_seconds >= self.min_track_duration_seconds
        )

    @staticmethod
    def _crop_source_object(
        analysis_frame: np.ndarray,
        source_frame: np.ndarray,
        bbox: List[int],
        padding: int = 10,
    ) -> np.ndarray:
        """Map an analysis-space bbox back to the untouched source frame."""
        analysis_h, analysis_w = analysis_frame.shape[:2]
        source_h, source_w = source_frame.shape[:2]
        scale_x = source_w / max(analysis_w, 1)
        scale_y = source_h / max(analysis_h, 1)
        x1, y1, x2, y2 = bbox
        sx1 = max(0, int((x1 - padding) * scale_x))
        sy1 = max(0, int((y1 - padding) * scale_y))
        sx2 = min(source_w, int((x2 + padding) * scale_x))
        sy2 = min(source_h, int((y2 + padding) * scale_y))
        return source_frame[sy1:sy2, sx1:sx2].copy()

    @staticmethod
    def _score_object_crop(
        crop: np.ndarray,
        analysis_frame: np.ndarray,
        bbox: List[int],
        detection_confidence: float,
    ) -> float:
        """Score size, sharpness and completeness instead of confidence alone."""
        if crop is None or crop.size == 0:
            return 0.0
        crop_h, crop_w = crop.shape[:2]
        frame_h, frame_w = analysis_frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bbox_area_ratio = max(0, x2 - x1) * max(0, y2 - y1) / max(frame_w * frame_h, 1)
        size_score = min(1.0, np.sqrt(bbox_area_ratio / 0.16))

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness_score = min(1.0, float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 350.0)
        edge_hits = sum((x1 <= 3, y1 <= 3, x2 >= frame_w - 3, y2 >= frame_h - 3))
        completeness_score = max(0.15, 1.0 - edge_hits * 0.28)
        aspect = crop_w / max(crop_h, 1)
        aspect_score = 1.0 if 0.75 <= aspect <= 3.5 else 0.55
        return float(np.clip(
            detection_confidence * 0.28
            + size_score * 0.24
            + sharpness_score * 0.22
            + completeness_score * 0.21
            + aspect_score * 0.05,
            0.0,
            1.0,
        ))

    @staticmethod
    def _is_interval_due(frame_count: int, interval: int) -> bool:
        """Sample frame 1 and then every configured interval, including interval=1."""
        return (max(1, int(frame_count)) - 1) % max(1, int(interval)) == 0


    @staticmethod
    def _build_overlay_label(obj: dict) -> str:
        """Build a compact label using characters supported by OpenCV."""
        track_id = obj.get("track_id", "-")

        object_class = obj.get("object_class")
        parts = [f"#{track_id}", str(object_class).upper()] if object_class else [f"#{track_id}"]
        if obj.get("predicted"):
            parts.append("PREDICTED")
            return " | ".join(parts)
        return " | ".join(parts)

    def draw_overlay(self, frame: np.ndarray, metadata: Optional[dict]) -> np.ndarray:
        """Draw readable, low-noise bounding boxes and labels on a frame."""
        if not metadata or (not metadata.get("objects") and not metadata.get("segments")):
            return frame

        overlay = frame.copy()
        frame_h, frame_w = overlay.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.46 if frame_w <= 1280 else 0.50
        accent = (80, 220, 120)

        for segment in metadata.get("segments", []):
            polygon = np.asarray(segment.get("polygon") or [], dtype=np.int32)
            if len(polygon) < 3:
                continue
            mask_layer = overlay.copy()
            cv2.fillPoly(mask_layer, [polygon], (70, 170, 255))
            cv2.addWeighted(mask_layer, 0.22, overlay, 0.78, 0, overlay)
            cv2.polylines(overlay, [polygon], True, (70, 170, 255), 2)

        for obj in metadata.get("objects", []):
            bbox = obj.get("bbox") or []
            if len(bbox) < 4:
                continue

            x1, y1, x2, y2 = [int(value) for value in bbox]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), accent, 2)

            label = self._build_overlay_label(obj)
            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, font_scale, 1
            )
            pad_x, pad_y, accent_w = 7, 4, 3
            box_w = min(frame_w, text_w + pad_x * 2 + accent_w)
            box_h = text_h + baseline + pad_y * 2
            label_left = max(0, min(x1, frame_w - box_w))
            label_top = y1 - box_h - 3
            if label_top < 0:
                label_top = min(max(0, y1 + 3), max(0, frame_h - box_h))
            label_right = min(frame_w, label_left + box_w)
            label_bottom = min(frame_h, label_top + box_h)

            roi = overlay[label_top:label_bottom, label_left:label_right]
            if not roi.size:
                continue
            dark = np.full_like(roi, (10, 16, 27))
            cv2.addWeighted(dark, 0.90, roi, 0.10, 0, roi)
            cv2.rectangle(
                overlay,
                (label_left, label_top),
                (min(label_left + accent_w, label_right - 1), label_bottom - 1),
                accent,
                -1,
            )
            cv2.putText(
                overlay,
                label,
                (label_left + accent_w + pad_x, label_top + pad_y + text_h),
                font,
                font_scale,
                (245, 247, 250),
                1,
                cv2.LINE_AA,
            )

        fps_label = (
            f"FPS {float(metadata.get('fps', 0)):.1f} | "
            f"LAT {float(metadata.get('latency_ms', 0)):.0f} ms | "
            f"OBJECTS {len(metadata['objects'])}"
        )
        (stats_w, stats_h), stats_base = cv2.getTextSize(fps_label, font, 0.52, 1)
        stats_right = min(frame_w, 10 + stats_w + 18)
        stats_bottom = min(frame_h, 10 + stats_h + stats_base + 10)
        stats_roi = overlay[10:stats_bottom, 10:stats_right]
        if stats_roi.size:
            stats_dark = np.full_like(stats_roi, (10, 16, 27))
            cv2.addWeighted(stats_dark, 0.88, stats_roi, 0.12, 0, stats_roi)
            cv2.rectangle(overlay, (10, 10), (13, stats_bottom - 1), (80, 220, 120), -1)
            cv2.putText(
                overlay,
                fps_label,
                (20, 10 + stats_h + 4),
                font,
                0.52,
                (110, 245, 150),
                1,
                cv2.LINE_AA,
            )

        return overlay

    async def finalize(self):
        """Close remaining tracks when a finite recording reaches EOF."""
        remaining = await self.track_state.remove_all_tracks()
        for track in remaining:
            if track.events_fired.get("object_entered"):
                await self._fire_event(
                    "object_left",
                    track,
                    None,
                )
            await self.track_state.remove_from_redis(track)

    async def _save_object_crop(self, track_id: int, crop: np.ndarray, track: ActiveTrack):
        try:
            path = os.path.join(
                settings.CROPS_PATH,
                f"object_{self.camera_id}_{self.processing_run_id}_{track_id}.jpg",
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)
            await asyncio.to_thread(
                cv2.imwrite, path, crop, [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
            await self.track_state.update_crops(track_id, crop_path=path)
        except Exception as e:
            logger.debug(f"Save object crop error: {e}")


    async def _fire_event(self, event_type: str, track: ActiveTrack, frame: Optional[np.ndarray]):
        """Queue event for database write (non-blocking)."""
        from app.services.event_service import event_queue
        try:
            frame_path = None
            if frame is not None and settings.STORAGE_PATH:
                frame_path = os.path.join(
                    settings.FRAMES_PATH,
                    f"event_{self.camera_id}_{self.processing_run_id}_{track.track_id}_{event_type}.jpg"
                )
                os.makedirs(os.path.dirname(frame_path), exist_ok=True)
                await asyncio.to_thread(
                    cv2.imwrite, frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 75]
                )

            payload = {
                "camera_id": self.camera_id,
                "track_id": track.track_id,
                "object_class": track.object_class,
                "detection_confidence": round(track.best_detection_conf, 3),
                "trajectory": track.trajectory,
                "speed_pixels_per_second": round(track.speed_pixels_per_second, 3),
                "direction_degrees": track.direction_degrees,
                "state": track.state,
                "predicted": track.predicted,
                "processing_run_id": self.processing_run_id,
                "first_seen": track.first_seen,
                "last_seen": track.last_seen,
                "duration_seconds": track.duration_seconds,
                "attributes": track.attributes,
                "best_crop_path": track.best_crop_path,
                "last_bbox": track.bbox,
                "first_video_timestamp_seconds": track.first_video_timestamp_seconds,
                "last_video_timestamp_seconds": track.last_video_timestamp_seconds,
            }
            await event_queue.put({
                "camera_id": self.camera_id,
                "event_type": event_type,
                "payload_json": payload,
                "frame_path": frame_path,
            })
        except Exception as e:
            logger.debug(f"Event fire error: {e}")

    def get_stats(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "ai_mode": self.ai_mode,
            "fps": self.fps,
            "inference_latency_ms": self.last_latency_ms,
            "frame_count": self.frame_count,
            "active_tracks": self.track_state.get_active_count(),
            "processing_run_id": self.processing_run_id,
            "analysis_resolution": list(self.input_resolution) if self.input_resolution else None,
            "source_resolution": list(self.source_resolution) if self.source_resolution else None,
            "stage_counts": dict(self.stage_counts),
            "stage_latency_ms": self.last_stage_latency_ms,
            "resolved_pipeline": self.resolved_pipeline,
        }
