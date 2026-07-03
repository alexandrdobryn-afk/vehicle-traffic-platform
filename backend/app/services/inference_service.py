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
from app.services.vehicle_detection_service import VehicleDetectionService, Detection
from app.services.plate_detection_service import PlateDetectionService
from app.services.ocr_service import OCRService, OCRCandidate
from app.services.color_recognition_service import ColorRecognitionService
from app.services.brand_recognition_service import BrandRecognitionService
from app.services.tracking_service import TrackingService, TrackedObject
from app.services.frame_quality_service import FrameQualityService
from app.services.track_state_service import TrackStateService, ActiveTrack
from app.models.model_registry import model_registry
from app.services.runtime_service import resolve_execution_policy

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
    Frame → Detect → Track → Crop → Plate → OCR → Color → State
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
        self.vehicle_confidence = float(self.runtime_settings.get(
            "vehicle_confidence_threshold", settings.VEHICLE_CONFIDENCE_THRESHOLD
        ))
        self.save_crops = bool(self.runtime_settings.get("save_crops", True))
        self.anonymization_mode = bool(self.runtime_settings.get("anonymization_mode", False))
        self.color_interval_frames = max(1, int(self.runtime_settings.get("color_interval_frames", 5)))
        self.brand_interval_frames = max(1, int(self.runtime_settings.get("brand_interval_frames", 30)))
        self.plate_interval_frames = max(1, int(self.runtime_settings.get("plate_interval_frames", 5)))
        resolution = str(self.runtime_settings.get("input_resolution", "")).lower().split("x")
        self.input_resolution = None
        if len(resolution) == 2 and all(part.isdigit() for part in resolution):
            self.input_resolution = (int(resolution[0]), int(resolution[1]))

        self._init_services()

    def _init_services(self):
        """Initialize all AI services based on mode."""
        logger.info(f"[Pipeline cam={self.camera_id}] Initializing {self.ai_mode} mode")

        manual = self.pipeline_config if self.pipeline_mode == "manual" else {}
        vehicle_model = (
            model_registry.get_named_vehicle_detector(
                manual.get("vehicle_detector"),
                self.ai_mode,
            )
            if manual.get("vehicle_detector")
            else model_registry.get_vehicle_detector(self.ai_mode)
        )
        plate_model = (
            model_registry.get_named_plate_detector(
                manual.get("plate_detector"),
                self.ai_mode,
            )
            if manual.get("plate_detector")
            else model_registry.get_plate_detector(self.ai_mode)
        )
        ocr_engine = manual.get("ocr_engine") or self.runtime_settings.get("ocr_engine")
        ocr_model = (
            model_registry.get_named_ocr_engine(ocr_engine, self.ai_mode)
            if ocr_engine else model_registry.get_ocr_engine(self.ai_mode)
        )
        color_model = model_registry.get_color_classifier()
        brand_model = model_registry.get_brand_classifier()

        # Choose tracker
        default_tracker = {
            "speed": "bytetrack",
            "balanced": "bytetrack",
            "quality": "botsort",
            "hybrid": "bytetrack",
            "practical": "tracktrack",
            "max_accuracy": "tracktrack",
            "edge_onnx": "bytetrack",
        }.get(self.ai_mode, "bytetrack")
        tracker_mode = manual.get("tracker") or self.runtime_settings.get("tracker_mode", default_tracker)

        self.detector = VehicleDetectionService(
            ai_mode=AIMode(self.ai_mode),
            model_config=vehicle_model,
            execution_device=self.execution_device,
        )
        self.rfdetr_detector = None
        if self.ai_mode == "hybrid":
            rfdetr_model = model_registry.get_named_vehicle_detector(
                "rfdetr_medium_vehicle", self.ai_mode
            )
            self.rfdetr_detector = VehicleDetectionService(
                ai_mode=AIMode.QUALITY,
                model_config=rfdetr_model,
                execution_device=self.execution_device,
            )
        self.tracker = TrackingService(tracker_mode=tracker_mode, ai_mode=self.ai_mode)
        self.plate_detector = PlateDetectionService(
            model_config=plate_model,
            execution_device=self.execution_device,
            confidence_threshold=float(self.runtime_settings.get(
                "plate_confidence_threshold", settings.PLATE_CONFIDENCE_THRESHOLD
            )),
        )
        self.ocr = OCRService(
            engine=ocr_model.model_type,
            model_config=ocr_model,
            regex_profile=self.runtime_settings.get("plate_regex_profile", "AUTO"),
            confidence_threshold=float(self.runtime_settings.get(
                "ocr_threshold", settings.OCR_CONFIDENCE_THRESHOLD
            )),
            voting_window=int(self.runtime_settings.get(
                "ocr_voting_window", settings.OCR_VOTING_WINDOW
            )),
            execution_device=self.execution_device,
        )
        self.color_service = ColorRecognitionService(
            model_config=color_model, execution_device=self.execution_device
        )
        self.brand_service = BrandRecognitionService(
            model_config=brand_model, execution_device=self.execution_device
        )
        self.quality_service = FrameQualityService(
            min_plate_width=int(self.runtime_settings.get(
                "minimum_plate_width", settings.MIN_PLATE_WIDTH
            )),
            min_plate_height=int(self.runtime_settings.get(
                "minimum_plate_height", settings.MIN_PLATE_HEIGHT
            )),
        )
        self.track_state = TrackStateService(
            camera_id=self.camera_id,
            redis_client=self.redis,
            processing_run_id=self.processing_run_id,
            max_missing_frames=int(self.runtime_settings.get("track_missing_grace_frames", 15)),
        )

        # Track-level OCR candidate pools
        self._ocr_candidates: Dict[int, List[OCRCandidate]] = {}
        self._color_histories: Dict[int, list] = {}
        self._brand_histories: Dict[int, list] = {}
        self.resolved_pipeline = {
            "pipeline_mode": self.pipeline_mode,
            "ai_mode": self.ai_mode,
            "runtime": {
                key: self.runtime_policy[key]
                for key in (
                    "requested", "resolved", "device", "fallback_policy",
                    "fallback_reason", "gpu_device_index", "precision",
                )
            },
            "vehicle_detector": {"name": vehicle_model.name, "format": vehicle_model.format},
            "escalation_detector": (
                {"name": "rfdetr_medium_vehicle", "format": "auto"}
                if self.rfdetr_detector else None
            ),
            "tracker": {"name": self.tracker.resolved_mode, "format": "algorithm"},
            "plate_detector": {"name": plate_model.name, "format": plate_model.format},
            "ocr": {"name": self.ocr.resolved_engine, "format": ocr_model.format},
            "color": {"name": color_model.name if color_model.available else "HSV/KMeans fallback", "format": color_model.format if color_model.available else "algorithm"},
            "brand": {"name": brand_model.name if brand_model.available else "unavailable", "format": brand_model.format if brand_model.available else "optional"},
            "features": {
                "temporal_voting": True,
                "vehicle_reid": False,
                "identity_stitching": True,
                "confidence_engine": True,
            },
        }

        logger.info(f"[Pipeline cam={self.camera_id}] Ready")

    async def process_frame(
        self,
        frame: np.ndarray,
        source_frame: Optional[np.ndarray] = None,
    ) -> Optional[dict]:
        """
        Process one frame through the full pipeline.
        Returns metadata dict for WebSocket broadcast or None if skipped.
        """
        self.frame_count += 1
        t_start = time.time()
        source_frame = source_frame if source_frame is not None else frame
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
            "vehicle_detection": 0.0,
            "tracking": 0.0,
            "color": 0.0,
            "brand": 0.0,
            "plate_detection": 0.0,
            "ocr": 0.0,
            "redis": 0.0,
        }

        # ── 1. Vehicle Detection ───────────────────────────────────
        use_rfdetr = self.detector.rfdetr_model is not None and self.ai_mode in {"quality", "max_accuracy"}
        stage_started = time.perf_counter()
        detections: List[Detection] = await _run_compute(
            self.detector.detect,
            frame,
            confidence_threshold=self.vehicle_confidence,
            use_rfdetr=use_rfdetr,
        )
        stage_latency["vehicle_detection"] += (time.perf_counter() - stage_started) * 1000
        self.stage_counts["vehicles_detected"] += len(detections)

        # Hybrid: re-check with RF-DETR if low confidence
        if self.ai_mode == "hybrid" and self.rfdetr_detector is not None:
            scheduled_check = self.frame_count % 5 == 0
            uncertainty_check = bool(detections) and self.detector.should_use_rfdetr(frame, detections)
            if scheduled_check or uncertainty_check:
                stage_started = time.perf_counter()
                detections = await _run_compute(
                    self.rfdetr_detector.detect,
                    frame,
                    confidence_threshold=self.vehicle_confidence,
                    use_rfdetr=True,
                )
                stage_latency["vehicle_detection"] += (time.perf_counter() - stage_started) * 1000
                self.stage_counts["rfdetr_escalations"] += 1

        # ── 2. Tracking ───────────────────────────────────────────
        stage_started = time.perf_counter()
        tracked_objects: List[TrackedObject] = await _run_compute(
            self.tracker.update, detections, frame
        )
        stage_latency["tracking"] += (time.perf_counter() - stage_started) * 1000
        active_ids = [t.track_id for t in tracked_objects]

        # ── 3. Per-track processing ───────────────────────────────
        ws_objects = []
        redis_tracks = []
        await _run_compute(self.color_service.update_scene_context, source_frame)

        for tracked in tracked_objects:
            tid = tracked.track_id
            track, is_new = await self.track_state.upsert_track(
                track_id=tid,
                vehicle_class=tracked.class_name,
                bbox=tracked.bbox,
                detection_conf=tracked.confidence,
                now=now,
            )

            # Fire "vehicle_entered" event on first appearance
            if is_new:
                await self._fire_event("vehicle_entered", track, frame)

            # Detect/track on the analysis frame, but preserve source pixels for
            # color, plate detection, OCR and stored evidence.
            vehicle_crop = self._crop_source_vehicle(
                frame, source_frame, tracked.bbox, padding=10
            )

            # ── Color recognition (every N frames) ────────────────
            if track.frame_count % self.color_interval_frames == 1:
                stage_started = time.perf_counter()
                color_observation = await _run_compute(
                    self.color_service.recognize_details, vehicle_crop
                )
                stage_latency["color"] += (time.perf_counter() - stage_started) * 1000
                if tid not in self._color_histories:
                    self._color_histories[tid] = []
                self._color_histories[tid] = self.color_service.update_color_voting(
                    self._color_histories[tid],
                    color_observation["color"],
                    color_observation["confidence"],
                    hex_code=color_observation["hex_code"],
                    quality_score=color_observation["quality_score"],
                    frame_index=track.frame_count,
                )
                color_result = self.color_service.resolve_color_details(
                    self._color_histories[tid]
                )
                final_color = color_result["color"]
                final_color_conf = color_result["confidence"]
                await self.track_state.update_color(tid, final_color, final_color_conf, self._color_histories[tid])
                await self.track_state.update_diagnostics(
                    tid,
                    color={
                        "status": "recognized" if final_color != "unknown" else "uncertain",
                        "value": final_color,
                        "confidence": round(final_color_conf, 3),
                        "engine": "onnx" if self.color_service.model_session is not None else "hsv_fallback",
                        "html_hex": color_result["hex_code"],
                        "candidate_value": color_result["candidate_color"],
                        "sample_count": color_result["sample_count"],
                        "support_count": color_result["support_count"],
                        "distribution": color_result["distribution"],
                        "provisional": color_result["provisional"],
                    },
                )
                self.stage_counts[
                    "colors_recognized" if final_color != "unknown" else "colors_uncertain"
                ] += 1

            # Brand recognition is deliberately slower and conservative. It
            # only emits a make after a visible logo agrees across frames.
            if (
                self.brand_service.model is not None
                and track.vehicle_class != "motorcycle"
                and (track.frame_count + tid * 7) % self.brand_interval_frames == 1
            ):
                stage_started = time.perf_counter()
                observation = await _run_compute(
                    self.brand_service.recognize_details, vehicle_crop
                )
                stage_latency["brand"] += (time.perf_counter() - stage_started) * 1000
                history = self._brand_histories.setdefault(tid, [])
                self._brand_histories[tid] = self.brand_service.update_voting(history, observation)
                brand_result = self.brand_service.resolve_brand(self._brand_histories[tid])
                await self.track_state.update_brand(
                    tid,
                    brand_result["brand"],
                    brand_result["confidence"],
                    self._brand_histories[tid],
                )
                await self.track_state.update_diagnostics(
                    tid,
                    brand={
                        "status": "recognized" if brand_result["brand"] != "unknown" else "uncertain",
                        "value": brand_result["brand"],
                        "candidate_value": brand_result.get("candidate_brand"),
                        "confidence": round(brand_result["confidence"], 3),
                        "engine": "brandeye_logo_detector",
                        "support_count": brand_result["support_count"],
                        "sample_count": brand_result["sample_count"],
                        "distribution": brand_result["distribution"],
                        "provisional": brand_result.get("provisional", True),
                    },
                )
                self.stage_counts[
                    "brands_recognized" if brand_result["brand"] != "unknown" else "brands_uncertain"
                ] += 1

            # ── Plate detection + OCR ──────────────────────────────
            # Keep sampling plate crops for the full lifetime of the track.
            # OCR may finish early, while a much clearer crop often appears
            # later as the vehicle approaches the camera.
            plate_sample_due = self._is_interval_due(
                track.frame_count, self.plate_interval_frames
            )
            should_detect_plate = plate_sample_due
            should_run_ocr = plate_sample_due and track.plate_status != "verified"
            stage_started = time.perf_counter()
            plate_detections = (
                await _run_compute(self.plate_detector.detect, vehicle_crop)
                if should_detect_plate else []
            )
            if should_detect_plate:
                stage_latency["plate_detection"] += (time.perf_counter() - stage_started) * 1000
            best_plate_det = self.plate_detector.get_best_plate(plate_detections)

            if not should_detect_plate:
                best_plate_det = None
            elif best_plate_det and best_plate_det.crop is not None:
                self.stage_counts["plates_detected"] += 1
                plate_quality = self.quality_service.check_plate_crop(best_plate_det.crop)
                plate_h, plate_w = best_plate_det.crop.shape[:2]
                crop_score = self._score_plate_crop(
                    best_plate_det.crop,
                    float(best_plate_det.confidence),
                )
                # Avoid rewriting near-identical frames, but replace the
                # evidence whenever quality improves materially.
                improvement_margin = max(0.01, track.best_plate_crop_score * 0.015)
                is_best_plate_crop = (
                    track.best_plate_crop_score <= 0.0
                    or crop_score > track.best_plate_crop_score + improvement_margin
                )
                crop_saved = True
                if (
                    self.save_crops and not self.anonymization_mode
                    and settings.STORAGE_PATH
                    and is_best_plate_crop
                ):
                    crop_saved = bool(
                        await self._save_plate_crop(tid, best_plate_det.crop, track)
                    )
                if is_best_plate_crop and crop_saved:
                    track.best_plate_crop_score = crop_score
                    best_crop_diagnostics = dict(
                        track.recognition_diagnostics.get("plate") or {}
                    )
                    best_crop_diagnostics.update({
                        "best_crop_score": round(crop_score, 3),
                        "best_crop_quality_score": round(plate_quality.overall_score, 3),
                        "best_crop_size": [plate_w, plate_h],
                        "best_crop_detector_confidence": round(best_plate_det.confidence, 3),
                    })
                    if track.plate_status == "verified":
                        best_crop_diagnostics.update({
                            "crop_size": [plate_w, plate_h],
                            "quality_score": round(plate_quality.overall_score, 3),
                            "detector_confidence": round(best_plate_det.confidence, 3),
                        })
                    await self.track_state.update_diagnostics(
                        tid, plate=best_crop_diagnostics
                    )

                # Small and stacked motorcycle plates are often below the
                # conservative quality gate, but upscaling can still yield
                # useful OCR. Only reject crops that are truly unusable.
                recoverable_small_crop = plate_w >= 20 and plate_h >= 12
                ocr_eligible = plate_quality.is_good or recoverable_small_crop
                effective_quality = max(0.25, plate_quality.overall_score)

                if should_run_ocr and ocr_eligible:
                    self.stage_counts["plates_quality_passed"] += 1
                    stage_started = time.perf_counter()
                    ocr_result = await _run_compute(
                        self.ocr.run_ocr,
                        best_plate_det.crop,
                        quality_score=effective_quality,
                    )
                    stage_latency["ocr"] += (time.perf_counter() - stage_started) * 1000

                    if ocr_result and ocr_result.text:
                        self.stage_counts["ocr_text_results"] += 1
                        if tid not in self._ocr_candidates:
                            self._ocr_candidates[tid] = []

                        self._ocr_candidates[tid] = self.ocr.update_voting(
                            self._ocr_candidates[tid],
                            ocr_result,
                            quality_score=effective_quality,
                        )

                        voting_result = self.ocr.resolve_plate(
                            self._ocr_candidates[tid],
                            total_frames=track.frame_count,
                        )

                        old_status = track.plate_status
                        await self.track_state.update_plate(
                            tid,
                            plate=voting_result.final_plate,
                            plate_status=voting_result.status,
                            plate_confidence=voting_result.confidence,
                            ocr_candidates=self._ocr_candidates[tid],
                        )
                        await self.track_state.update_diagnostics(
                            tid,
                            plate={**(track.recognition_diagnostics.get("plate") or {}),
                                "status": voting_result.status,
                                "reason": "regex_valid" if ocr_result.regex_valid else "regex_pending",
                                "detector_confidence": round(best_plate_det.confidence, 3),
                                "ocr_confidence": round(ocr_result.confidence, 3),
                                "quality_score": round(plate_quality.overall_score, 3),
                                "crop_size": [plate_w, plate_h],
                                "profile": self.ocr.regex_profile,
                                "upscaled": not plate_quality.is_good,
                            },
                        )

                        # Fire event when plate gets verified
                        if old_status != "verified" and voting_result.status == "verified":
                            await self._fire_event("plate_verified", track, frame)

                    elif not self._ocr_candidates.get(tid) and (
                        is_best_plate_crop or not track.recognition_diagnostics.get("plate")
                    ):
                        self.stage_counts["ocr_no_text"] += 1
                        await self.track_state.update_diagnostics(
                            tid,
                            plate={**(track.recognition_diagnostics.get("plate") or {}),
                                "status": "searching",
                                "reason": "ocr_no_text",
                                "detector_confidence": round(best_plate_det.confidence, 3),
                                "quality_score": round(plate_quality.overall_score, 3),
                                "crop_size": [plate_w, plate_h],
                                "profile": self.ocr.regex_profile,
                                "upscaled": not plate_quality.is_good,
                            },
                        )
                elif should_run_ocr:
                    reason = "plate_too_small" if "too small" in plate_quality.reason else "plate_quality_rejected"
                    self.stage_counts[reason] += 1
                    if not self._ocr_candidates.get(tid) and (
                        is_best_plate_crop or not track.recognition_diagnostics.get("plate")
                    ):
                        await self.track_state.update_diagnostics(
                            tid,
                            plate={**(track.recognition_diagnostics.get("plate") or {}),
                                "status": "searching",
                                "reason": reason,
                                "detail": plate_quality.reason,
                                "detector_confidence": round(best_plate_det.confidence, 3),
                                "crop_size": [plate_w, plate_h],
                                "profile": self.ocr.regex_profile,
                            },
                        )
            elif (
                should_run_ocr
                and not self._ocr_candidates.get(tid)
                and track.best_plate_crop_score <= 0.0
            ):
                self.stage_counts["plate_not_found"] += 1
                await self.track_state.update_diagnostics(
                    tid,
                    plate={
                        "status": "searching",
                        "reason": "plate_not_found",
                        "profile": self.ocr.regex_profile,
                    },
                )

            # Save the clearest, most complete vehicle view. Detector
            # confidence alone often prefers a late frame where only a door,
            # lamp or grille remains inside the image.
            vehicle_crop_score = self._score_vehicle_crop(
                vehicle_crop, frame, tracked.bbox, tracked.confidence, best_plate_det
            )
            if (
                self.save_crops and settings.STORAGE_PATH
                and vehicle_crop_score > track.best_vehicle_crop_score
            ):
                stored_crop = vehicle_crop.copy()
                if self.anonymization_mode and best_plate_det:
                    px1, py1, px2, py2 = best_plate_det.bbox
                    region = stored_crop[py1:py2, px1:px2]
                    if region.size:
                        stored_crop[py1:py2, px1:px2] = cv2.GaussianBlur(region, (31, 31), 0)
                await self._save_vehicle_crop(tid, stored_crop, track)
                track.best_vehicle_crop_score = vehicle_crop_score

            # Sync to Redis
            updated_track = await self.track_state.get_track(tid)
            if updated_track:
                redis_tracks.append(updated_track)
                ws_objects.append(updated_track.to_ws_dict())

        stage_started = time.perf_counter()
        await self.track_state.sync_many_to_redis(redis_tracks)
        stage_latency["redis"] += (time.perf_counter() - stage_started) * 1000

        # ── 4. Remove stale tracks ────────────────────────────────
        stale = await self.track_state.remove_stale_tracks(active_ids, time.time())
        for stale_track in stale:
            await self._fire_event("vehicle_left", stale_track, None)
            await self.track_state.remove_from_redis(stale_track)
            # Clean up voting state
            self._ocr_candidates.pop(stale_track.track_id, None)
            self._color_histories.pop(stale_track.track_id, None)
            self._brand_histories.pop(stale_track.track_id, None)

        # ── 5. FPS calculation ────────────────────────────────────
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
        }

    @staticmethod
    def _crop_source_vehicle(
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
    def _score_vehicle_crop(
        crop: np.ndarray,
        analysis_frame: np.ndarray,
        bbox: List[int],
        detection_confidence: float,
        plate_detection=None,
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
        plate_bonus = 1.0 if plate_detection is not None else 0.0
        return float(np.clip(
            detection_confidence * 0.22
            + size_score * 0.22
            + sharpness_score * 0.20
            + completeness_score * 0.24
            + aspect_score * 0.05
            + plate_bonus * 0.07,
            0.0,
            1.0,
        ))

    @staticmethod
    def _is_interval_due(frame_count: int, interval: int) -> bool:
        """Sample frame 1 and then every configured interval, including interval=1."""
        return (max(1, int(frame_count)) - 1) % max(1, int(interval)) == 0

    @staticmethod
    def _score_plate_crop(crop: np.ndarray, detection_confidence: float) -> float:
        """Rank plate evidence by readable detail, not by first detection time."""
        if crop is None or crop.size == 0:
            return 0.0

        height, width = crop.shape[:2]
        gray = (
            cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if len(crop.shape) == 3 else crop
        )
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast = float(gray.std())
        brightness = float(gray.mean())
        aspect = width / max(height, 1)

        # Width and character height are the strongest predictors that the
        # stored crop will remain useful when enlarged in the UI.
        resolution_score = float(np.sqrt(
            min(1.0, width / 180.0) * min(1.0, height / 50.0)
        ))
        sharpness_score = min(1.0, np.log1p(max(0.0, blur)) / np.log1p(700.0))
        contrast_score = min(1.0, contrast / 60.0)
        brightness_score = max(0.0, 1.0 - abs(brightness - 128.0) / 128.0)
        aspect_score = 1.0 if 1.5 <= aspect <= 8.0 else 0.35

        return float(np.clip(
            resolution_score * 0.35
            + sharpness_score * 0.25
            + contrast_score * 0.10
            + brightness_score * 0.08
            + aspect_score * 0.07
            + float(np.clip(detection_confidence, 0.0, 1.0)) * 0.15,
            0.0,
            1.0,
        ))

    @staticmethod
    def _build_overlay_label(obj: dict) -> str:
        """Build a compact label using characters supported by OpenCV."""
        track_id = obj.get("track_id", "-")
        plate = obj.get("plate")
        plate_status = obj.get("plate_status", "searching")
        plate_confidence = float(obj.get("plate_confidence") or 0.0)
        color = str(obj.get("color") or "unknown").upper()

        parts = [f"#{track_id}"]
        if plate:
            parts.append(str(plate) if plate_status == "verified" else f"~{plate}")
            if plate_confidence > 0:
                parts.append(f"{plate_confidence:.0%}")
        else:
            parts.append("SEARCHING")
        if color != "UNKNOWN":
            parts.append(color)
        return " | ".join(parts)

    def draw_overlay(self, frame: np.ndarray, metadata: Optional[dict]) -> np.ndarray:
        """Draw readable, low-noise bounding boxes and labels on a frame."""
        if not metadata or not metadata.get("objects"):
            return frame

        overlay = frame.copy()
        frame_h, frame_w = overlay.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.46 if frame_w <= 1280 else 0.50
        color_map = {
            "red": (0, 0, 220),
            "blue": (220, 100, 0),
            "green": (0, 180, 0),
            "yellow": (0, 220, 220),
            "orange": (0, 140, 255),
            "white": (230, 230, 230),
            "black": (40, 40, 40),
            "gray": (140, 140, 140),
            "silver": (192, 192, 192),
            "brown": (42, 42, 165),
            "beige": (170, 200, 210),
            "unknown": (100, 100, 100),
        }

        for obj in metadata["objects"]:
            bbox = obj.get("bbox") or []
            if len(bbox) < 4:
                continue

            x1, y1, x2, y2 = [int(value) for value in bbox]
            color = obj.get("color", "unknown")
            accent = color_map.get(color, (100, 100, 100))
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
            f"VEHICLES {len(metadata['objects'])}"
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
            await self._fire_event("vehicle_left", track, None)
            await self.track_state.remove_from_redis(track)
        self._ocr_candidates.clear()
        self._color_histories.clear()
        self._brand_histories.clear()

    async def _save_vehicle_crop(self, track_id: int, crop: np.ndarray, track: ActiveTrack):
        try:
            path = os.path.join(
                settings.CROPS_PATH,
                f"vehicle_{self.camera_id}_{self.processing_run_id}_{track_id}.jpg",
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)
            await asyncio.to_thread(
                cv2.imwrite, path, crop, [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
            await self.track_state.update_crops(track_id, vehicle_crop_path=path)
        except Exception as e:
            logger.debug(f"Save vehicle crop error: {e}")

    async def _save_plate_crop(
        self, track_id: int, crop: np.ndarray, track: ActiveTrack
    ) -> Optional[str]:
        try:
            path = os.path.join(
                settings.CROPS_PATH,
                f"plate_{self.camera_id}_{self.processing_run_id}_{track_id}.jpg",
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)
            written = await asyncio.to_thread(
                cv2.imwrite, path, crop, [cv2.IMWRITE_JPEG_QUALITY, 90]
            )
            if not written:
                return None
            await self.track_state.update_crops(track_id, plate_crop_path=path)
            return path
        except Exception as e:
            logger.debug(f"Save plate crop error: {e}")
            return None

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

            await event_queue.put({
                "camera_id": self.camera_id,
                "vehicle_track_id": track.db_track_id,
                "event_type": event_type,
                "payload_json": track.to_dict(),
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
