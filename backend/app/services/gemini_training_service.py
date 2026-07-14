import asyncio
import base64
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

import cv2
import httpx
import numpy as np
from sqlalchemy import select

from app.config import settings
from app.models.database import AppSettings, GeminiReviewCandidate
from app.utils.auth import decrypt_url, encrypt_url

logger = logging.getLogger(__name__)


class GeminiTrainingService:
    """Selects bounded real keyframes and turns Gemini masks into review candidates."""

    def __init__(self):
        self._session_factory: Optional[Callable] = None
        self._last_capture: dict[int, float] = {}
        self._run_counts: dict[tuple[int, str], int] = {}
        self._seen_tracks: set[tuple[int, str, int]] = set()
        self._tasks: set[asyncio.Task] = set()
        self._semaphore = asyncio.Semaphore(1)

    def set_session_factory(self, session_factory: Callable):
        self._session_factory = session_factory

    async def shutdown(self):
        tasks = list(self._tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def get_public_settings(self) -> dict[str, Any]:
        config = await self._load_config(include_key=False)
        return {
            "configured": bool(config.get("configured")),
            "enabled": bool(config.get("enabled", False)),
            "model": config.get("model", "gemini-2.5-flash"),
            "request_timeout_seconds": int(config.get("request_timeout_seconds", 45)),
            "max_concurrent_requests": int(config.get("max_concurrent_requests", 1)),
            "api_key_masked": "********" if config.get("configured") else None,
        }

    async def save_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._session_factory is None:
            raise RuntimeError("Gemini service is not initialized")
        async with self._session_factory() as db:
            result = await db.execute(select(AppSettings).where(AppSettings.key == "gemini"))
            row = result.scalar_one_or_none()
            existing = dict(row.value or {}) if row else {}
            api_key = payload.pop("api_key", None)
            if api_key:
                existing["api_key_encrypted"] = encrypt_url(api_key.strip())
            existing.update(payload)
            existing["configured"] = bool(existing.get("api_key_encrypted"))
            if row:
                row.value = existing
            else:
                db.add(AppSettings(key="gemini", value=existing))
            await db.commit()
        concurrency = int(existing.get("max_concurrent_requests", 1))
        self._semaphore = asyncio.Semaphore(max(1, min(concurrency, 4)))
        return await self.get_public_settings()

    async def test_connection(self) -> dict[str, Any]:
        config = await self._load_config(include_key=True)
        if not config.get("api_key"):
            return {"success": False, "error": "Gemini API key is not configured"}
        model = config.get("model", "gemini-2.5-flash")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.get(url, headers={"x-goog-api-key": config["api_key"]})
                response.raise_for_status()
            return {"success": True, "model": model}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def consider_frame(
        self,
        camera_id: int,
        source_frame: np.ndarray,
        metadata: dict[str, Any],
        runtime_settings: dict[str, Any],
    ) -> None:
        if not runtime_settings.get("gemini_enabled"):
            return
        if not (runtime_settings.get("gemini_verify_predictions") or runtime_settings.get("gemini_collect_training")):
            return
        objects = metadata.get("objects") or []
        if not objects:
            return
        run_id = str(metadata.get("processing_run_id") or "unknown")
        max_candidates = int(runtime_settings.get("gemini_max_candidates_per_run", 25))
        run_key = (camera_id, run_id)
        if self._run_counts.get(run_key, 0) >= max_candidates:
            return
        interval = int(runtime_settings.get("gemini_sample_interval_seconds", 30))
        now = time.monotonic()
        if now - self._last_capture.get(camera_id, 0.0) < interval:
            return
        provider = await self._load_config(include_key=False)
        if not provider.get("configured") or not provider.get("enabled"):
            return

        unseen = [obj for obj in objects if (camera_id, run_id, int(obj.get("track_id", -1))) not in self._seen_tracks]
        selected = min(unseen or objects, key=lambda obj: float(obj.get("detection_confidence", 1.0)))
        track_id = int(selected.get("track_id", -1))
        track_key = (camera_id, run_id, track_id)
        if track_key in self._seen_tracks:
            return
        self._seen_tracks.add(track_key)
        self._last_capture[camera_id] = now
        self._run_counts[run_key] = self._run_counts.get(run_key, 0) + 1

        frame_copy = source_frame.copy()
        task = asyncio.create_task(
            self._create_candidate(camera_id, frame_copy, metadata, selected, runtime_settings),
            name=f"gemini-candidate-{camera_id}-{run_id}-{track_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def retry_candidate(self, candidate_id: int) -> None:
        candidate = await self.get_candidate(candidate_id)
        frame = cv2.imread(candidate.frame_path)
        if frame is None:
            raise FileNotFoundError("Candidate frame is missing")
        candidate.status = "processing"
        candidate.error_message = None
        await self._commit(candidate)
        task = asyncio.create_task(
            self._request_and_store(
                candidate.id,
                frame,
                candidate.local_predictions or [],
                verify_predictions=True,
                collect_training=candidate.target_model_type != "verification_only",
            ),
            name=f"gemini-retry-{candidate.id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def get_candidate(self, candidate_id: int) -> GeminiReviewCandidate:
        async with self._session_factory() as db:
            result = await db.execute(select(GeminiReviewCandidate).where(GeminiReviewCandidate.id == candidate_id))
            candidate = result.scalar_one_or_none()
            if candidate is None:
                raise LookupError("Gemini review candidate not found")
            db.expunge(candidate)
            return candidate

    async def review_candidate(
        self,
        candidate_id: int,
        review: dict[str, Any],
        reviewer: str,
    ) -> GeminiReviewCandidate:
        async with self._session_factory() as db:
            result = await db.execute(select(GeminiReviewCandidate).where(GeminiReviewCandidate.id == candidate_id))
            candidate = result.scalar_one_or_none()
            if candidate is None:
                raise LookupError("Gemini review candidate not found")
            annotations = list(candidate.proposed_annotations or [])
            index = int(review.get("annotation_index", 0))
            if annotations and index < len(annotations):
                annotation = dict(annotations[index])
                if review.get("class_name"):
                    annotation["class_name"] = review["class_name"]
                if review.get("polygon"):
                    annotation["polygon"] = review["polygon"]
                    annotation["human_edited"] = True
                annotations[index] = annotation
            candidate.proposed_annotations = annotations
            candidate.status = review["status"]
            candidate.reviewed_by = reviewer
            candidate.reviewed_at = datetime.now(timezone.utc)
            await db.commit()
            await db.refresh(candidate)
            db.expunge(candidate)
            return candidate

    async def _create_candidate(self, camera_id, frame, metadata, selected, runtime_settings):
        root = Path(settings.STORAGE_PATH) / "gemini_candidates"
        root.mkdir(parents=True, exist_ok=True)
        filename = f"{camera_id}_{metadata.get('processing_run_id', 'run')}_{uuid4().hex}.jpg"
        frame_path = root / filename
        if not await asyncio.to_thread(cv2.imwrite, str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            logger.error("Could not store Gemini candidate frame for camera %s", camera_id)
            return

        source_h, source_w = frame.shape[:2]
        analysis_w = max(1, int(metadata.get("frame_width") or source_w))
        analysis_h = max(1, int(metadata.get("frame_height") or source_h))
        local_predictions = []
        for obj in metadata.get("objects") or []:
            bbox = obj.get("bbox") or []
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            local_predictions.append({
                "track_id": obj.get("track_id"),
                "class_name": obj.get("object_class"),
                "confidence": obj.get("detection_confidence"),
                "box_2d": [
                    round(y1 / analysis_h * 1000), round(x1 / analysis_w * 1000),
                    round(y2 / analysis_h * 1000), round(x2 / analysis_w * 1000),
                ],
            })

        async with self._session_factory() as db:
            candidate = GeminiReviewCandidate(
                camera_id=camera_id,
                processing_run_id=str(metadata.get("processing_run_id") or "unknown"),
                track_id=selected.get("track_id"),
                status="processing",
                selection_reason="new_representative_track",
                frame_path=str(frame_path),
                frame_width=source_w,
                frame_height=source_h,
                target_model_type=(
                    "multi_segmenter"
                    if runtime_settings.get("gemini_collect_training")
                    else "verification_only"
                ),
                local_predictions=local_predictions,
            )
            db.add(candidate)
            await db.commit()
            await db.refresh(candidate)
            candidate_id = candidate.id
        await self._request_and_store(
            candidate_id,
            frame,
            local_predictions,
            verify_predictions=bool(runtime_settings.get("gemini_verify_predictions")),
            collect_training=bool(runtime_settings.get("gemini_collect_training")),
        )

    async def _request_and_store(
        self,
        candidate_id: int,
        frame: np.ndarray,
        local_predictions: list[dict],
        verify_predictions: bool = True,
        collect_training: bool = True,
    ):
        config = await self._load_config(include_key=True)
        if not config.get("enabled") or not config.get("api_key"):
            await self._mark_error(candidate_id, "Gemini is disabled or the API key is not configured")
            return
        model = config.get("model", "gemini-2.5-flash")
        timeout = int(config.get("request_timeout_seconds", 45))
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            await self._mark_error(candidate_id, "Could not encode candidate frame")
            return
        task_instruction = []
        if verify_predictions:
            task_instruction.append("Verify the supplied local aerial object predictions and report structured disagreements.")
        else:
            task_instruction.append("Do not score the local predictions; set verification confidence to 0 and summarize that verification was disabled.")
        if collect_training:
            task_instruction.append(
                "Segment every visible target object. Return a tight box_2d using "
                "[ymin,xmin,ymax,xmax] normalized 0..1000 and a base64 PNG probability mask scoped to that box."
            )
        else:
            task_instruction.append("Training capture is disabled, so return an empty objects array and no masks.")
        prompt = " ".join(task_instruction) + (
            " Use the local prediction labels when they are plausible, otherwise use concise aerial object labels. "
            "Do not invent occluded pixels. "
            "Local predictions: " + json.dumps(local_predictions, ensure_ascii=True)
        )
        schema = {
            "type": "object",
            "properties": {
                "verification": {
                    "type": "object",
                    "properties": {
                        "local_predictions_correct": {"type": "boolean"},
                        "summary": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["local_predictions_correct", "summary", "confidence"],
                },
                "objects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "box_2d": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
                            "mask": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": ["label", "box_2d", "mask", "confidence"],
                    },
                },
            },
            "required": ["verification", "objects"],
        }
        request = {
            "contents": [{"role": "user", "parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(encoded).decode("ascii")}},
            ]}],
            "generationConfig": {"responseMimeType": "application/json", "responseJsonSchema": schema},
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        try:
            async with self._semaphore:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(url, headers={"x-goog-api-key": config["api_key"]}, json=request)
                    response.raise_for_status()
                    payload = response.json()
            parts = payload.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            text = next((part.get("text") for part in parts if part.get("text")), None)
            if not text:
                raise ValueError("Gemini returned no structured candidate")
            result = json.loads(text)
            annotations, mask_path = await asyncio.to_thread(
                self._normalize_annotations, candidate_id, frame.shape[1], frame.shape[0], result.get("objects") or []
            )
            async with self._session_factory() as db:
                row = await db.get(GeminiReviewCandidate, candidate_id)
                row.status = "ready"
                row.gemini_verification = result.get("verification")
                row.proposed_annotations = annotations
                row.mask_path = mask_path
                row.raw_response = {"finish_reason": payload.get("candidates", [{}])[0].get("finishReason")}
                row.provider_model = payload.get("modelVersion") or model
                row.response_id = payload.get("responseId")
                row.usage_metadata = payload.get("usageMetadata")
                row.error_message = None
                await db.commit()
        except Exception as exc:
            logger.warning("Gemini candidate %s failed: %s", candidate_id, exc)
            await self._mark_error(candidate_id, str(exc)[:2000])

    def _normalize_annotations(self, candidate_id: int, width: int, height: int, objects: list[dict]):
        root = Path(settings.STORAGE_PATH) / "gemini_candidates" / "masks"
        root.mkdir(parents=True, exist_ok=True)
        annotations = []
        first_mask = None
        for index, obj in enumerate(objects):
            box = obj.get("box_2d") or []
            if len(box) != 4:
                continue
            y0, x0, y1, x1 = [max(0, min(1000, int(value))) for value in box]
            px0, px1 = round(x0 / 1000 * width), round(x1 / 1000 * width)
            py0, py1 = round(y0 / 1000 * height), round(y1 / 1000 * height)
            if px1 <= px0 or py1 <= py0:
                continue
            mask_value = str(obj.get("mask") or "")
            if "," in mask_value:
                mask_value = mask_value.split(",", 1)[1]
            raw_mask = cv2.imdecode(np.frombuffer(base64.b64decode(mask_value), np.uint8), cv2.IMREAD_GRAYSCALE)
            if raw_mask is None:
                continue
            resized = cv2.resize(raw_mask, (px1 - px0, py1 - py0), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((height, width), dtype=np.uint8)
            canvas[py0:py1, px0:px1] = resized
            binary = np.where(canvas >= 127, 255, 0).astype(np.uint8)
            mask_path = root / f"candidate_{candidate_id}_{index}.png"
            cv2.imwrite(str(mask_path), binary)
            if first_mask is None:
                first_mask = str(mask_path)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv2.contourArea) if contours else None
            polygon = []
            if contour is not None:
                epsilon = max(1.0, 0.002 * cv2.arcLength(contour, True))
                simplified = cv2.approxPolyDP(contour, epsilon, True)
                polygon = [[round(float(p[0][0]) / width, 6), round(float(p[0][1]) / height, 6)] for p in simplified]
            annotations.append({
                "class_name": obj.get("label") or "object",
                "confidence": float(obj.get("confidence") or 0),
                "box_2d": [y0, x0, y1, x1],
                "polygon": polygon,
                "mask_path": str(mask_path),
                "source": "gemini",
            })
        return annotations, first_mask

    async def _load_config(self, include_key: bool) -> dict[str, Any]:
        if self._session_factory is None:
            return {}
        async with self._session_factory() as db:
            result = await db.execute(select(AppSettings).where(AppSettings.key == "gemini"))
            row = result.scalar_one_or_none()
            config = dict(row.value or {}) if row else {}
        config["configured"] = bool(config.get("api_key_encrypted"))
        if include_key and config.get("api_key_encrypted"):
            config["api_key"] = decrypt_url(config["api_key_encrypted"])
        config.pop("api_key_encrypted", None)
        return config

    async def _mark_error(self, candidate_id: int, message: str):
        async with self._session_factory() as db:
            row = await db.get(GeminiReviewCandidate, candidate_id)
            if row:
                row.status = "error"
                row.error_message = message
                await db.commit()

    async def _commit(self, candidate: GeminiReviewCandidate):
        async with self._session_factory() as db:
            await db.merge(candidate)
            await db.commit()


gemini_training_service = GeminiTrainingService()
