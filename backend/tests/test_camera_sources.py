import unittest
import tempfile
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock, patch

import cv2
import numpy as np
from pydantic import ValidationError

from app.schemas.schemas import CameraCreate
from app.services.video_capture_service import VideoCaptureService


class _MJPEGHandler(BaseHTTPRequestHandler):
    frame = cv2.imencode(".jpg", np.full((48, 80, 3), 127, dtype=np.uint8))[1].tobytes()

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            for _ in range(30):
                self.wfile.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(self.frame)).encode()
                    + b"\r\n\r\n"
                    + self.frame
                    + b"\r\n"
                )
                self.wfile.flush()
                time.sleep(0.02)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, _format, *_args):
        pass


class CameraSourceSchemaTests(unittest.TestCase):
    def test_supported_source_types_accept_matching_urls(self):
        cases = [
            ("rtsp", "rtsp://camera.local/stream"),
            ("rtsp", "rtsps://camera.local/stream"),
            ("hls", "https://example.com/live.m3u8"),
            ("mjpeg", "http://example.com/video.mjpg"),
            ("jpeg", "https://example.com/latest.jpg"),
        ]
        for source_type, url in cases:
            with self.subTest(source_type=source_type, url=url):
                camera = CameraCreate(name="Test", source_type=source_type, rtsp_url=url)
                self.assertEqual(camera.source_type.value, source_type)

    def test_source_type_rejects_incompatible_scheme(self):
        with self.assertRaises(ValidationError):
            CameraCreate(name="Test", source_type="hls", rtsp_url="rtsp://camera.local/stream")

    def test_source_url_requires_host(self):
        with self.assertRaises(ValidationError):
            CameraCreate(name="Test", source_type="jpeg", rtsp_url="https:///latest.jpg")

    def test_recorded_video_cannot_be_created_with_a_local_path_in_json(self):
        with self.assertRaises(ValidationError):
            CameraCreate(name="Recording", source_type="file", rtsp_url="C:/videos/test.mp4")


class CameraSourceConnectionTests(unittest.TestCase):
    @staticmethod
    def _make_recording(path: Path, frame_count: int = 8, fps: float = 20.0):
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (80, 48),
        )
        if not writer.isOpened():
            raise unittest.SkipTest("OpenCV MJPG writer is unavailable")
        for index in range(frame_count):
            writer.write(np.full((48, 80, 3), index * 20, dtype=np.uint8))
        writer.release()

    def test_recorded_video_probe_returns_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "recording.avi"
            self._make_recording(path)
            result = VideoCaptureService.test_connection(str(path), source_type="file")

        self.assertTrue(result["success"], result)
        self.assertEqual(result["width"], 80)
        self.assertEqual(result["height"], 48)
        self.assertEqual(result["total_frames"], 8)
        self.assertGreater(result["duration_seconds"], 0)

    def test_recorded_video_reaches_eof_without_reconnect(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "recording.avi"
            self._make_recording(path, frame_count=6, fps=20.0)
            capture = VideoCaptureService(
                str(path),
                camera_id=99,
                source_type="file",
                max_fps=20,
            )
            capture.start()
            deadline = time.monotonic() + 3
            collected = 0
            while not capture.is_finished() and time.monotonic() < deadline:
                if capture.get_frame() is not None:
                    collected += 1
                time.sleep(0.02)
            while capture.get_frame() is not None:
                collected += 1
            stats = capture.get_stats()
            capture.stop()

        self.assertTrue(stats["finished"], stats)
        self.assertEqual(stats["completion_status"], "completed")
        self.assertEqual(stats["reconnect_count"], 0)
        self.assertEqual(stats["progress_percent"], 100.0)
        self.assertEqual(stats["frames_dropped"], 0)
        self.assertEqual(collected, 6)

    def test_mjpeg_stream_decodes_frame(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _MJPEGHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = VideoCaptureService.test_connection(
                f"http://127.0.0.1:{server.server_port}/video.mjpg",
                source_type="mjpeg",
                timeout=5,
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertTrue(result["success"], result)
        self.assertEqual(result["width"], 80)
        self.assertEqual(result["height"], 48)

    @patch("app.services.video_capture_service.httpx.get")
    def test_jpeg_snapshot_connection_decodes_image(self, get_mock):
        image = np.zeros((36, 64, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        response = Mock(content=encoded.tobytes(), headers={"content-type": "image/jpeg"})
        response.raise_for_status.return_value = None
        get_mock.return_value = response

        result = VideoCaptureService.test_connection(
            "https://example.com/latest.jpg",
            source_type="jpeg",
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["width"], 64)
        self.assertEqual(result["height"], 36)
        self.assertEqual(result["content_type"], "image/jpeg")

    @patch("app.services.video_capture_service.httpx.get")
    def test_jpeg_rejects_non_image_response(self, get_mock):
        response = Mock(content=b"not an image", headers={"content-type": "text/html"})
        response.raise_for_status.return_value = None
        get_mock.return_value = response

        result = VideoCaptureService.test_connection(
            "https://example.com/latest.jpg",
            source_type="jpeg",
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["error_category"], "invalid_image")

    def test_unknown_source_type_returns_diagnostic_error(self):
        result = VideoCaptureService.test_connection("https://example.com", source_type="dash")
        self.assertFalse(result["success"])
        self.assertEqual(result["error_category"], "unsupported_source_type")


if __name__ == "__main__":
    unittest.main()
