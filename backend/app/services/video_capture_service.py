import logging
import math
import os
import queue
import re
import threading
import time
from typing import Optional

# Some camera master playlists use valid but uncommon segment extensions
# (for example .ec3). OpenCV's bundled FFmpeg rejects them unless explicitly
# allowed. Deployments can override this with their own capture options.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "allowed_extensions;ALL")

import cv2
import httpx
import numpy as np

logger = logging.getLogger(__name__)


class VideoCaptureService:
    """Capture live sources, poll snapshots, or play a managed video file."""

    STREAM_TYPES = {"rtsp", "hls", "mjpeg"}
    SUPPORTED_TYPES = STREAM_TYPES | {"jpeg", "file"}
    MAX_SNAPSHOT_BYTES = 20 * 1024 * 1024

    def __init__(
        self,
        rtsp_url: str,
        camera_id: int,
        max_fps: int = 25,
        buffer_size: int = 2,
        reconnect_delay: float = 3.0,
        source_type: str = "rtsp",
        snapshot_interval_seconds: float = 1.0,
    ):
        source_type = source_type.lower()
        if source_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"Unsupported camera source type: {source_type}")

        self.rtsp_url = rtsp_url
        self.source_type = source_type
        self.snapshot_interval_seconds = max(snapshot_interval_seconds, 0.25)
        self.camera_id = camera_id
        self.max_fps = max_fps
        self.buffer_size = buffer_size
        self.reconnect_delay = reconnect_delay

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._connected = False
        self._finished = False
        self._failed = False
        self._last_frame_time = 0.0
        self._frame_interval = 1.0 / max_fps
        self._current_frame = 0
        self._total_frames = 0
        self._source_fps = 0.0
        self._started_at = 0.0

        self.frames_read = 0
        self.frames_dropped = 0
        self.reconnect_count = 0
        self.last_error: Optional[str] = None

    def start(self):
        self._stop_event.clear()
        self._finished = False
        self._failed = False
        self._current_frame = 0
        self._started_at = time.monotonic()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"capture-cam-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "[Capture cam=%s] Started [%s]: %s",
            self.camera_id,
            self.source_type,
            self._mask_url(self.rtsp_url),
        )

    def stop(self):
        self._stop_event.set()
        if self._cap:
            self._cap.release()
        if self._thread:
            self._thread.join(timeout=5.0)
        self._connected = False
        logger.info("[Capture cam=%s] Stopped", self.camera_id)

    def get_frame(self) -> Optional[np.ndarray]:
        if self.source_type == "file":
            try:
                return self._frame_queue.get_nowait()
            except queue.Empty:
                return None
        frame = None
        try:
            while True:
                frame = self._frame_queue.get_nowait()
        except queue.Empty:
            return frame

    def is_connected(self) -> bool:
        return self._connected

    def is_finished(self) -> bool:
        """Return True only after a recorded file reaches EOF or fails to open."""
        return self._finished

    def completion_status(self) -> str:
        return "error" if self._failed else "completed"

    def _capture_loop(self):
        if self.source_type == "file":
            self._file_loop()
            return
        if self.source_type == "jpeg":
            self._snapshot_loop()
            return

        while not self._stop_event.is_set():
            if not self._connect():
                self._stop_event.wait(self.reconnect_delay)
                continue

            self._connected = True
            self.last_error = None
            logger.info("[Capture cam=%s] Connected", self.camera_id)

            while not self._stop_event.is_set():
                if not self._cap or not self._cap.isOpened():
                    break
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    self.last_error = "Frame decode failed"
                    logger.warning("[Capture cam=%s] Frame read failed", self.camera_id)
                    break

                now = time.monotonic()
                if now - self._last_frame_time < self._frame_interval:
                    continue
                self._last_frame_time = now
                self._queue_frame(frame)

            self._connected = False
            if self._stop_event.is_set():
                break
            self.reconnect_count += 1
            logger.warning(
                "[Capture cam=%s] Disconnected, reconnecting in %.1fs",
                self.camera_id,
                self.reconnect_delay,
            )
            if self._cap:
                self._cap.release()
                self._cap = None
            self._stop_event.wait(self.reconnect_delay)

    def _file_loop(self):
        """Decode a finite recording with backpressure and no inference frame loss."""
        try:
            cap = self._open_capture(self.rtsp_url, timeout=15.0)
            if not cap.isOpened():
                cap.release()
                raise ValueError("FFmpeg could not open the recorded video")

            self._cap = cap
            source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if not math.isfinite(source_fps) or source_fps <= 0:
                source_fps = float(self.max_fps)
            self._source_fps = source_fps
            self._total_frames = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
            emit_step = source_fps / min(source_fps, float(self.max_fps))
            next_emit_frame = 0.0
            frame_index = 0
            self._connected = True
            self.last_error = None
            logger.info(
                "[Capture cam=%s] Recorded video opened: %.2f FPS, %s frames",
                self.camera_id,
                source_fps,
                self._total_frames or "unknown",
            )

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    self._finished = True
                    break

                self._current_frame = frame_index + 1
                frame_index += 1

                # Sample deterministically, then block until inference consumes
                # the frame. Uploaded evidence must never behave like a lossy
                # live stream simply because CPU inference is slower than FPS.
                source_index = frame_index - 1
                if source_index + 1e-9 < next_emit_frame:
                    continue
                next_emit_frame += emit_step
                self._queue_frame(frame)

        except Exception as exc:
            self.last_error = str(exc)
            self._failed = True
            self._finished = True
            logger.error("[Capture cam=%s] Recorded video failed: %s", self.camera_id, exc)
        finally:
            self._connected = False
            if self._cap:
                self._cap.release()
                self._cap = None

    def _snapshot_loop(self):
        timeout = max(5.0, min(self.snapshot_interval_seconds * 2, 15.0))
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"Cache-Control": "no-cache", "User-Agent": "VTP/2.0"},
        ) as client:
            while not self._stop_event.is_set():
                started = time.monotonic()
                try:
                    response = client.get(self.rtsp_url)
                    response.raise_for_status()
                    frame = self._decode_snapshot(response.content)
                    self._connected = True
                    self.last_error = None
                    self._queue_frame(frame)
                except Exception as exc:
                    self._connected = False
                    self.reconnect_count += 1
                    self.last_error = str(exc)
                    logger.warning("[Capture cam=%s] Snapshot fetch failed: %s", self.camera_id, exc)

                elapsed = time.monotonic() - started
                self._stop_event.wait(max(self.snapshot_interval_seconds - elapsed, 0.0))

    def _queue_frame(self, frame: np.ndarray):
        if self.source_type == "file":
            while not self._stop_event.is_set():
                try:
                    self._frame_queue.put(frame, timeout=0.1)
                    self.frames_read += 1
                    return
                except queue.Full:
                    continue
            return

        self.frames_read += 1
        if self._frame_queue.full():
            try:
                self._frame_queue.get_nowait()
                self.frames_dropped += 1
            except queue.Empty:
                pass
        self._frame_queue.put(frame)

    def _connect(self) -> bool:
        try:
            cap = self._open_capture(self.rtsp_url, timeout=15.0)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                self.last_error = "Failed to open stream"
                logger.error(
                    "[Capture cam=%s] Cannot open: %s",
                    self.camera_id,
                    self._mask_url(self.rtsp_url),
                )
                cap.release()
                return False
            self._cap = cap
            return True
        except Exception as exc:
            self.last_error = str(exc)
            logger.error("[Capture cam=%s] Connect error: %s", self.camera_id, exc)
            return False

    def get_stats(self) -> dict:
        progress = None
        if self.source_type == "file" and self._total_frames > 0:
            progress = min(100.0, self._current_frame * 100.0 / self._total_frames)
            if self._finished and not self._failed:
                progress = 100.0
        elapsed = max(0.0, time.monotonic() - self._started_at) if self._started_at else 0.0
        return {
            "camera_id": self.camera_id,
            "source_type": self.source_type,
            "connected": self._connected,
            "frames_read": self.frames_read,
            "capture_fps": round(self.frames_read / elapsed, 2) if elapsed else 0.0,
            "frames_dropped": self.frames_dropped,
            "reconnect_count": self.reconnect_count,
            "last_error": self.last_error,
            "finished": self._finished,
            "completion_status": self.completion_status() if self._finished else None,
            "current_frame": self._current_frame if self.source_type == "file" else None,
            "total_frames": self._total_frames if self.source_type == "file" else None,
            "source_fps": round(self._source_fps, 2) if self._source_fps > 0 else None,
            "progress_percent": round(progress, 1) if progress is not None else None,
        }

    @staticmethod
    def _mask_url(url: str) -> str:
        return re.sub(r"([a-z][a-z0-9+.-]*://)[^/@:]+:[^/@]+@", r"\1***:***@", url, flags=re.I)

    @staticmethod
    def _open_capture(url: str, timeout: float) -> cv2.VideoCapture:
        params = []
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            params.extend([cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(timeout * 1000)])
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            params.extend([cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(timeout * 1000)])
        if params:
            return cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)
        return cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    @classmethod
    def _decode_snapshot(cls, content: bytes) -> np.ndarray:
        if len(content) > cls.MAX_SNAPSHOT_BYTES:
            raise ValueError("Snapshot exceeds 20 MB")
        frame = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Response is not a valid JPEG image")
        return frame

    @staticmethod
    def _capture_metadata(cap: cv2.VideoCapture) -> dict:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        fourcc_value = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        codec = "".join(chr((fourcc_value >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ")
        try:
            backend = cap.getBackendName()
        except Exception:
            backend = "FFMPEG"
        return {
            "decoder": "opencv-ffmpeg",
            "backend": backend,
            "codec": codec or None,
            "fps": round(fps, 2) if math.isfinite(fps) and fps > 0 else None,
        }

    @classmethod
    def test_connection(cls, url: str, source_type: str = "rtsp", timeout: float = 8.0) -> dict:
        source_type = source_type.lower()
        if source_type not in cls.SUPPORTED_TYPES:
            return {
                "success": False,
                "source_type": source_type,
                "error_category": "unsupported_source_type",
                "error": f"Unsupported camera source type: {source_type}",
            }
        if source_type == "jpeg":
            return cls._test_snapshot(url, timeout)

        cap = None
        started = time.monotonic()
        try:
            cap = cls._open_capture(url, timeout)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                return {
                    "success": False,
                    "source_type": source_type,
                    "error_category": "open_failed",
                    "error": "FFmpeg could not open the stream",
                }

            while time.monotonic() - started < timeout:
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    return {
                        "success": True,
                        "source_type": source_type,
                        "width": width,
                        "height": height,
                        "latency_ms": int((time.monotonic() - started) * 1000),
                        **cls._capture_metadata(cap),
                        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if source_type == "file" else None,
                        "duration_seconds": cls._capture_duration(cap) if source_type == "file" else None,
                    }
                time.sleep(0.05)
            return {
                "success": False,
                "source_type": source_type,
                "error_category": "decode_timeout",
                "error": f"No decodable frame received within {timeout:g} seconds",
            }
        except Exception as exc:
            return {
                "success": False,
                "source_type": source_type,
                "error_category": "connection_error",
                "error": str(exc),
            }
        finally:
            if cap is not None:
                cap.release()

    @classmethod
    def _test_snapshot(cls, url: str, timeout: float) -> dict:
        started = time.monotonic()
        try:
            response = httpx.get(
                url,
                timeout=timeout,
                follow_redirects=True,
                headers={"Cache-Control": "no-cache", "User-Agent": "VTP/2.0"},
            )
            response.raise_for_status()
            frame = cls._decode_snapshot(response.content)
            height, width = frame.shape[:2]
            return {
                "success": True,
                "source_type": "jpeg",
                "width": width,
                "height": height,
                "latency_ms": int((time.monotonic() - started) * 1000),
                "decoder": "opencv-imdecode",
                "backend": "httpx",
                "codec": "JPEG",
                "content_type": response.headers.get("content-type"),
            }
        except httpx.HTTPStatusError as exc:
            return {
                "success": False,
                "source_type": "jpeg",
                "error_category": "http_status",
                "error": f"HTTP {exc.response.status_code}",
            }
        except (httpx.HTTPError, ValueError) as exc:
            category = "invalid_image" if isinstance(exc, ValueError) else "http_error"
            return {
                "success": False,
                "source_type": "jpeg",
                "error_category": category,
                "error": str(exc),
            }

    @staticmethod
    def _capture_duration(cap: cv2.VideoCapture) -> Optional[float]:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        if not math.isfinite(fps) or not math.isfinite(frames) or fps <= 0 or frames <= 0:
            return None
        return round(frames / fps, 3)
