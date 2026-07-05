import asyncio
import os
from pathlib import Path
from typing import Awaitable, Callable, Dict, Optional
import logging

import cv2

from app.services.video_capture_service import VideoCaptureService
from app.services.inference_service import InferencePipeline
from app.services.websocket_service import websocket_manager
from app.config import settings

logger = logging.getLogger(__name__)


class CameraManager:
    """
    Manages lifecycle of multiple camera pipelines.
    Each camera gets its own VideoCaptureService + InferencePipeline.
    """

    def __init__(self):
        self._captures: Dict[int, VideoCaptureService] = {}
        self._pipelines: Dict[int, InferencePipeline] = {}
        self._tasks: Dict[int, asyncio.Task] = {}
        self._redis = None
        self._lock = asyncio.Lock()
        self._source_finished_callback: Optional[Callable[[int, str], Awaitable[None]]] = None
        self._frame_analysis_callback: Optional[Callable[..., Awaitable[None]]] = None
        self._preview_encode_counts: Dict[int, int] = {}

    def set_redis(self, redis_client):
        self._redis = redis_client

    def set_source_finished_callback(self, callback: Callable[[int, str], Awaitable[None]]):
        self._source_finished_callback = callback

    def set_frame_analysis_callback(self, callback: Callable[..., Awaitable[None]]):
        self._frame_analysis_callback = callback

    async def start_camera(
        self,
        camera_id: int,
        rtsp_url: str,
        ai_mode: str = "balanced",
        max_fps: int = 25,
        runtime_settings: Optional[dict] = None,
        source_type: str = "rtsp",
        snapshot_interval_seconds: float = 1.0,
    ) -> bool:
        async with self._lock:
            if camera_id in self._tasks and not self._tasks[camera_id].done():
                logger.warning(f"Camera {camera_id} already running")
                return False

            # A completed file source leaves its final frame available for preview.
            # Starting it again replaces that finished runtime and begins at frame zero.
            old_capture = self._captures.pop(camera_id, None)
            if old_capture is not None:
                old_capture.stop()
            self._pipelines.pop(camera_id, None)
            self._tasks.pop(camera_id, None)
            self._latest_frames.pop(camera_id, None)
            self._latest_jpegs.pop(camera_id, None)
            self._frame_versions.pop(camera_id, None)
            self._preview_encode_counts.pop(camera_id, None)

            try:
                capture = VideoCaptureService(
                    rtsp_url=rtsp_url,
                    camera_id=camera_id,
                    max_fps=max_fps,
                    source_type=source_type,
                    snapshot_interval_seconds=snapshot_interval_seconds,
                )
                pipeline = InferencePipeline(
                    camera_id=camera_id,
                    ai_mode=ai_mode,
                    redis_client=self._redis,
                    runtime_settings=runtime_settings,
                )

                self._captures[camera_id] = capture
                self._pipelines[camera_id] = pipeline

                # Start capture thread
                capture.start()

                # Start async processing task
                task = asyncio.create_task(
                    self._processing_loop(camera_id),
                    name=f"pipeline-cam-{camera_id}"
                )
                self._tasks[camera_id] = task

                logger.info(f"Camera {camera_id} started [{source_type}/{ai_mode}]")
                return True

            except Exception as e:
                logger.error(f"Failed to start camera {camera_id}: {e}")
                return False

    async def stop_camera(self, camera_id: int) -> bool:
        async with self._lock:
            capture = self._captures.get(camera_id)
            if capture is not None and capture.source_type == "file":
                self._persist_preview_frame(camera_id)
            if camera_id in self._tasks:
                self._tasks[camera_id].cancel()
                try:
                    await asyncio.wait_for(self._tasks[camera_id], timeout=3.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                del self._tasks[camera_id]

            if camera_id in self._captures:
                self._captures[camera_id].stop()
                del self._captures[camera_id]

            if camera_id in self._pipelines:
                await self._pipelines[camera_id].finalize()
                del self._pipelines[camera_id]
            self._latest_frames.pop(camera_id, None)
            self._latest_jpegs.pop(camera_id, None)
            self._frame_versions.pop(camera_id, None)
            self._preview_encode_counts.pop(camera_id, None)

            logger.info(f"Camera {camera_id} stopped")
            return True

    async def _processing_loop(self, camera_id: int):
        """Main async loop — reads frames and runs inference."""
        capture = self._captures[camera_id]
        pipeline = self._pipelines[camera_id]

        logger.info(f"[Loop cam={camera_id}] Processing loop started")

        while True:
            try:
                frame = capture.get_frame()

                if frame is None:
                    if capture.is_finished():
                        status = capture.completion_status()
                        if status == "completed":
                            self._persist_preview_frame(camera_id)
                        await pipeline.finalize()
                        await websocket_manager.broadcast(camera_id, {
                            "type": "source_completed" if status == "completed" else "source_error",
                            "camera_id": camera_id,
                            "status": status,
                            "error": capture.last_error,
                        })
                        if self._source_finished_callback:
                            await self._source_finished_callback(camera_id, status)
                        logger.info("[Loop cam=%s] Recorded source %s", camera_id, status)
                        break
                    await asyncio.sleep(0.02)
                    continue

                source_frame = frame
                analysis_frame = frame
                if pipeline.input_resolution and frame.shape[1::-1] != pipeline.input_resolution:
                    analysis_frame = await asyncio.to_thread(
                        cv2.resize,
                        frame,
                        pipeline.input_resolution,
                        interpolation=cv2.INTER_AREA,
                    )

                metadata = await pipeline.process_frame(
                    analysis_frame,
                    source_frame=source_frame,
                )

                if metadata:
                    if self._frame_analysis_callback:
                        await self._frame_analysis_callback(
                            camera_id,
                            source_frame,
                            metadata,
                            pipeline.runtime_settings,
                        )
                    # Annotate frame
                    annotated = await asyncio.to_thread(
                        pipeline.draw_overlay, analysis_frame, metadata
                    )

                    # Broadcast metadata via WebSocket
                    await websocket_manager.broadcast(camera_id, metadata)

                    # Store latest annotated frame for MJPEG
                    self._latest_frames[camera_id] = annotated
                    await asyncio.to_thread(self._cache_preview_jpeg, camera_id, annotated)

                await asyncio.sleep(0)  # yield to event loop

            except asyncio.CancelledError:
                logger.info(f"[Loop cam={camera_id}] Cancelled")
                break
            except Exception as e:
                logger.error(f"[Loop cam={camera_id}] Error: {e}")
                await asyncio.sleep(0.1)

    # Separate latest frame store for MJPEG
    _latest_frames: Dict[int, any] = {}
    _latest_jpegs: Dict[int, bytes] = {}
    _frame_versions: Dict[int, int] = {}

    def get_latest_frame(self, camera_id: int):
        return self._latest_frames.get(camera_id)

    def _cache_preview_jpeg(self, camera_id: int, frame) -> Optional[bytes]:
        """Encode once per inference frame, independently of viewer count."""
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            return None
        encoded = buffer.tobytes()
        self._latest_jpegs[camera_id] = encoded
        self._frame_versions[camera_id] = self._frame_versions.get(camera_id, 0) + 1
        self._preview_encode_counts[camera_id] = self._preview_encode_counts.get(camera_id, 0) + 1
        return encoded

    def get_preview_jpeg(self, camera_id: int) -> Optional[bytes]:
        encoded = self._latest_jpegs.get(camera_id)
        if encoded is not None:
            return encoded
        frame = self._latest_frames.get(camera_id)
        return self._cache_preview_jpeg(camera_id, frame) if frame is not None else None

    def _preview_path(self, camera_id: int) -> Path:
        return Path(settings.FRAMES_PATH) / f"source_{camera_id}_preview.jpg"

    def _persist_preview_frame(self, camera_id: int) -> bool:
        """Atomically persist the latest frame so completed videos survive restarts."""
        frame = self._latest_frames.get(camera_id)
        if frame is None:
            return False
        destination = self._preview_path(camera_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
        try:
            if not cv2.imwrite(str(temporary), frame):
                return False
            os.replace(temporary, destination)
            return True
        except OSError as exc:
            logger.warning("Could not persist preview for source %s: %s", camera_id, exc)
            temporary.unlink(missing_ok=True)
            return False

    def restore_file_preview(self, camera_id: int, source_path: str) -> bool:
        """Restore a completed file preview from disk or decode a source frame."""
        if self._latest_frames.get(camera_id) is not None:
            return True

        preview_path = self._preview_path(camera_id)
        frame = cv2.imread(str(preview_path)) if preview_path.is_file() else None
        if frame is None:
            capture = cv2.VideoCapture(source_path)
            try:
                if not capture.isOpened():
                    return False
                total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if total_frames > 1:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames // 2))
                ok, frame = capture.read()
                if not ok or frame is None:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = capture.read()
                if not ok or frame is None:
                    return False
            finally:
                capture.release()

        self._latest_frames[camera_id] = frame
        self._cache_preview_jpeg(camera_id, frame)
        self._persist_preview_frame(camera_id)
        return True

    def delete_persisted_preview(self, camera_id: int):
        self._latest_frames.pop(camera_id, None)
        self._latest_jpegs.pop(camera_id, None)
        self._frame_versions.pop(camera_id, None)
        self._preview_encode_counts.pop(camera_id, None)
        self._preview_path(camera_id).unlink(missing_ok=True)

    def get_capture(self, camera_id: int) -> Optional[VideoCaptureService]:
        return self._captures.get(camera_id)

    def get_pipeline(self, camera_id: int) -> Optional[InferencePipeline]:
        return self._pipelines.get(camera_id)

    def is_running(self, camera_id: int) -> bool:
        task = self._tasks.get(camera_id)
        return task is not None and not task.done()

    def get_all_stats(self) -> list:
        stats = []
        for cam_id in self._captures:
            cap_stats = self._captures[cam_id].get_stats()
            pip_stats = self._pipelines[cam_id].get_stats() if cam_id in self._pipelines else {}
            stats.append({
                **cap_stats,
                **pip_stats,
                "is_running": self.is_running(cam_id),
                "preview_frames_encoded": self._preview_encode_counts.get(cam_id, 0),
            })
        return stats

    async def mjpeg_generator(self, camera_id: int):
        """Fan out already-encoded frames; viewers never trigger JPEG encoding."""
        last_version = -1
        while camera_id in self._captures or camera_id in self._latest_frames:
            version = self._frame_versions.get(camera_id, 0)
            if version == last_version:
                await asyncio.sleep(0.04)
                continue
            encoded = self.get_preview_jpeg(camera_id)
            if encoded is None:
                await asyncio.sleep(0.04)
                continue
            last_version = version
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded + b"\r\n"

    async def restart_camera(
        self, camera_id: int, rtsp_url: str, ai_mode: str, max_fps: int,
        runtime_settings: Optional[dict] = None,
        source_type: str = "rtsp",
        snapshot_interval_seconds: float = 1.0,
    ):
        await self.stop_camera(camera_id)
        await asyncio.sleep(1)
        return await self.start_camera(
            camera_id=camera_id,
            rtsp_url=rtsp_url,
            ai_mode=ai_mode,
            max_fps=max_fps,
            runtime_settings=runtime_settings,
            source_type=source_type,
            snapshot_interval_seconds=snapshot_interval_seconds,
        )


camera_manager = CameraManager()
