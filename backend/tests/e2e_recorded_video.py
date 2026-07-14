"""Docker-backed smoke test for the recorded-video API and finite-source lifecycle."""

import os
import tempfile
import time
from pathlib import Path

import cv2
import httpx
import numpy as np


BASE_URL = os.getenv("BEVP_E2E_BASE_URL", "http://127.0.0.1:8000/api/v1")
EMAIL = os.getenv("BEVP_E2E_EMAIL", "admin@bevp.local")
PASSWORD = os.getenv("BEVP_E2E_PASSWORD", "admin123")


def make_video(path: Path, seconds: int = 2, fps: int = 10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (320, 180))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the smoke-test video")
    rng = np.random.default_rng(42)
    for index in range(seconds * fps):
        frame = rng.integers(0, 80, size=(180, 320, 3), dtype=np.uint8)
        x = 20 + index * 8
        cv2.rectangle(frame, (x, 70), (min(x + 90, 319), 130), (220, 220, 220), -1)
        cv2.putText(frame, "AA1234BB", (x + 5, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        writer.write(frame)
    writer.release()


def wait_for_status(client: httpx.Client, video_id: int, expected: str, timeout: float = 90):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        videos = client.get("/videos").raise_for_status().json()
        last = next(video for video in videos if video["id"] == video_id)
        if last["status"] == expected:
            return last
        time.sleep(0.5)
    raise AssertionError(f"Timed out waiting for {expected}; last state: {last}")


def main():
    video_id = None
    with tempfile.TemporaryDirectory() as temp_dir, httpx.Client(base_url=BASE_URL, timeout=180) as client:
        path = Path(temp_dir) / "recorded-video-smoke.avi"
        make_video(path)

        if os.getenv("BEVP_E2E_LOCAL_AUTH") == "1":
            from app.utils.auth import create_access_token
            token = create_access_token({"sub": "1", "role": "admin", "email": EMAIL})
        else:
            login = client.post("/auth/login", json={"email": EMAIL, "password": PASSWORD})
            login.raise_for_status()
            token = login.json()["access_token"]
        client.headers["Authorization"] = f"Bearer {token}"

        try:
            with path.open("rb") as source:
                upload = client.post(
                    "/videos/upload",
                    data={"name": "Recorded video E2E", "ai_mode": "speed", "max_fps": "10"},
                    files={"file": (path.name, source, "video/x-msvideo")},
                )
            upload.raise_for_status()
            created = upload.json()
            video_id = created["id"]
            assert created["source_type"] == "file"
            assert created["source_file_name"] == path.name
            assert created["source_duration_seconds"] > 0

            probe = client.post(f"/videos/{video_id}/test").raise_for_status().json()
            assert probe["success"] is True
            assert probe["width"] == 320 and probe["height"] == 180

            first_start = client.post(f"/videos/{video_id}/start").raise_for_status().json()
            assert first_start["success"] is True
            completed = wait_for_status(client, video_id, "completed")
            assert completed["is_active"] is False
            assert completed["progress_percent"] == 100.0

            ticket = client.post(f"/stream/{video_id}/ticket").raise_for_status().json()["ticket"]
            with client.stream("GET", f"/stream/{video_id}", params={"ticket": ticket}, timeout=10) as stream:
                stream.raise_for_status()
                first_chunk = next(stream.iter_bytes())
                assert b"Content-Type: image/jpeg" in first_chunk

            second_start = client.post(f"/videos/{video_id}/start").raise_for_status().json()
            assert second_start["success"] is True
            wait_for_status(client, video_id, "completed")

            print("recorded-video-e2e: upload -> probe -> run -> EOF -> preview -> rerun: OK")
        finally:
            if video_id is not None:
                response = client.delete(f"/videos/{video_id}")
                response.raise_for_status()


if __name__ == "__main__":
    main()
