"""Create a small, ready-to-train vehicle detector dataset via Training API.

Run this inside the inference backend container, where saved vehicle crops are
available under /app/storage/crops. The script intentionally creates a smoke
dataset, not a production-quality training corpus.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from pathlib import Path

import httpx
from jose import jwt
from PIL import Image
from ultralytics import YOLO


API_URL = os.getenv("TRAINING_API_URL", "http://training-api:8001/api/v1/training")
SECRET_KEY = os.getenv(
    "SECRET_KEY", "change-me-super-secret-key-at-least-32-characters"
)
DATASET_NAME = "Test 1 - Vehicle Detector Smoke"
SOURCE_DIR = Path(os.getenv("CROPS_PATH", "/app/storage/crops"))
IMAGE_COUNT = 24
VEHICLE_MODEL_PATH = Path(
    os.getenv("VEHICLE_MODEL_PATH", "/app/models/vehicle_detector/yolo11n.pt")
)
COCO_VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck


def vehicle_identity(path: Path) -> str:
    """Group sequential crops from the same tracked vehicle."""
    stem_parts = path.stem.split("_")
    if stem_parts[-1].isdigit():
        stem_parts = stem_parts[:-1]
    return "_".join(stem_parts)


def select_vehicle_crops() -> list[Path]:
    candidates: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for path in SOURCE_DIR.glob("vehicle_*"):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        try:
            with Image.open(path) as image:
                width, height = image.size
        except OSError:
            continue
        if width < 96 or height < 64:
            continue
        candidates[vehicle_identity(path)].append((width * height, path))

    # One high-resolution crop per track/session prevents near-duplicate frames
    # from leaking across train and validation splits.
    selected = [
        max(track_crops, key=lambda item: item[0])[1]
        for _, track_crops in sorted(candidates.items())
    ]
    if len(selected) < IMAGE_COUNT:
        raise RuntimeError(
            f"Need {IMAGE_COUNT} distinct vehicle tracks, found {len(selected)}"
        )
    return selected[:IMAGE_COUNT]


def auth_headers() -> dict[str, str]:
    token = jwt.encode(
        {"sub": "1", "role": "admin", "email": "training-smoke@local"},
        SECRET_KEY,
        algorithm="HS256",
    )
    return {"Authorization": f"Bearer {token}"}


def checked(response: httpx.Response) -> httpx.Response:
    response.raise_for_status()
    return response


def predict_annotations(paths: list[Path]) -> tuple[dict[str, list[dict]], int]:
    if not VEHICLE_MODEL_PATH.exists():
        raise RuntimeError(f"Vehicle detector not found: {VEHICLE_MODEL_PATH}")

    model = YOLO(str(VEHICLE_MODEL_PATH))
    results = model.predict(
        source=[str(path) for path in paths],
        conf=0.25,
        device="cpu",
        verbose=False,
    )
    annotations: dict[str, list[dict]] = {}
    fallback_count = 0
    for path, result in zip(paths, results):
        boxes = []
        for box in result.boxes:
            if int(box.cls.item()) not in COCO_VEHICLE_CLASSES:
                continue
            x_center, y_center, width, height = box.xywhn[0].tolist()
            boxes.append(
                {
                    "annotation_type": "bbox",
                    "class_name": "vehicle",
                    "class_id": 0,
                    "x_center": x_center,
                    "y_center": y_center,
                    "bbox_width": width,
                    "bbox_height": height,
                    "confidence": float(box.conf.item()),
                    "is_auto": True,
                }
            )
        if not boxes:
            # These are detector-generated vehicle crops. Keep a conservative
            # full-crop fallback so the smoke dataset has no false negatives.
            boxes = [
                {
                    "annotation_type": "bbox",
                    "class_name": "vehicle",
                    "class_id": 0,
                    "x_center": 0.5,
                    "y_center": 0.5,
                    "bbox_width": 1.0,
                    "bbox_height": 1.0,
                    "confidence": None,
                    "is_auto": True,
                }
            ]
            fallback_count += 1
        annotations[path.name] = boxes
    return annotations, fallback_count


def replace_annotations(
    client: httpx.Client,
    images: list[dict],
    predicted: dict[str, list[dict]],
) -> int:
    annotation_count = 0
    for image in images:
        checked(client.delete(f"/annotations/image/{image['id']}/all"))
        boxes = predicted[image["filename"]]
        checked(
            client.post(
                f"/annotations/image/{image['id']}/bulk",
                json=boxes,
            )
        )
        annotation_count += len(boxes)
    return annotation_count


def main() -> None:
    selected = select_vehicle_crops()
    predicted, fallback_count = predict_annotations(selected)
    headers = auth_headers()

    with httpx.Client(base_url=API_URL, headers=headers, timeout=120) as client:
        existing = checked(client.get("/datasets")).json()
        for dataset in existing:
            if dataset["name"] == DATASET_NAME:
                dataset_id = dataset["id"]
                images = checked(
                    client.get(
                        f"/datasets/{dataset_id}/images", params={"limit": 500}
                    )
                ).json()
                annotation_count = replace_annotations(client, images, predicted)
                stats = checked(client.get(f"/datasets/{dataset_id}/stats")).json()
                split = dict(Counter(image["split"] for image in images))
                export = checked(
                    client.post(f"/datasets/{dataset_id}/export/yolo")
                ).json()
                print(f"Dataset already exists and is ready: id={dataset_id}")
                print(
                    f"Stats={stats}, annotations={annotation_count}, "
                    f"fallbacks={fallback_count}, split={split}, export={export['path']}"
                )
                return

        created = checked(
            client.post(
                "/datasets",
                json={
                    "name": DATASET_NAME,
                    "description": (
                        "Smoke dataset from saved vehicle crops. Intended only to "
                        "verify the end-to-end YOLO training pipeline."
                    ),
                    "model_type": "vehicle_detector",
                    "annotation_type": "bbox",
                    "classes": ["vehicle"],
                    "tags": ["smoke-test", "existing-crops", "single-class"],
                },
            )
        ).json()
        dataset_id = created["id"]

        try:
            files = [
                ("files", (path.name, path.read_bytes(), "image/jpeg"))
                for path in selected
            ]
            upload = checked(
                client.post(f"/datasets/{dataset_id}/images", files=files)
            ).json()
            if upload["uploaded"] != IMAGE_COUNT:
                raise RuntimeError(
                    f"Expected {IMAGE_COUNT} uploads, got {upload['uploaded']}"
                )

            images = checked(
                client.get(f"/datasets/{dataset_id}/images", params={"limit": 500})
            ).json()
            annotation_count = replace_annotations(client, images, predicted)

            split = checked(
                client.post(
                    f"/datasets/{dataset_id}/split",
                    json={
                        "train_ratio": 0.7,
                        "val_ratio": 0.2,
                        "test_ratio": 0.1,
                        "seed": 42,
                    },
                )
            ).json()["counts"]
            stats = checked(client.get(f"/datasets/{dataset_id}/stats")).json()
            export = checked(
                client.post(f"/datasets/{dataset_id}/export/yolo")
            ).json()
        except Exception:
            client.delete(f"/datasets/{dataset_id}")
            raise

    print(f"Created dataset id={dataset_id}: {DATASET_NAME}")
    print(
        f"Images={IMAGE_COUNT}, annotations={annotation_count}, "
        f"fallbacks={fallback_count}, split={split}"
    )
    print(f"Stats={stats}")
    print(f"YOLO export={export['path']}")
    print("Ready for a vehicle_detector job. Training was not started.")


if __name__ == "__main__":
    main()
