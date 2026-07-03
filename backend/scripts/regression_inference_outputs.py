"""Emit deterministic component outputs for CPU/GPU regression comparison."""

import json
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO

from app.models.model_registry import (
    PLATE_DETECTOR_OPENIMAGEMODELS,
    PLATE_DETECTOR_YOLO11N,
    PLATE_DETECTOR_YOLO8N,
)
from app.services.ocr_service import OCRService
from app.services.plate_detection_service import PlateDetectionService


def rounded_box(values):
    return [round(float(value), 3) for value in values]


video_path = Path("/app/storage/videos/474df4516aca40cb9021538603e66e79.mp4")
capture = cv2.VideoCapture(str(video_path))
frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count // 2))
ok, frame = capture.read()
capture.release()
if not ok:
    raise RuntimeError(f"Could not decode {video_path}")
frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)

vehicle_crop = cv2.imread(
    "/app/storage/crops/vehicle_19_749bcf120f274edf8427f659eb157c89_6.jpg"
)
plate_crop = cv2.imread(
    "/app/storage/crops/plate_19_749bcf120f274edf8427f659eb157c89_6.jpg"
)

output = {"vehicle": {}, "plate": {}, "ocr": {}}
for name, path in [
    ("yolo11n", "/app/models/vehicle_detector/yolo11n.pt"),
    ("yolo11s", "/app/models/vehicle_detector/yolo11s.pt"),
    ("yolo26n", "/app/models/vehicle_detector/yolo26n.pt"),
]:
    result = YOLO(path).predict(frame, conf=0.45, verbose=False)[0]
    output["vehicle"][name] = [
        {
            "class_id": int(box.cls[0]),
            "confidence": round(float(box.conf[0]), 5),
            "bbox": rounded_box(box.xyxy[0].tolist()),
        }
        for box in result.boxes
    ]

for config in [
    PLATE_DETECTOR_YOLO8N,
    PLATE_DETECTOR_OPENIMAGEMODELS,
    PLATE_DETECTOR_YOLO11N,
]:
    service = PlateDetectionService(config, confidence_threshold=0.1)
    output["plate"][config.name] = [
        {
            "confidence": round(detection.confidence, 5),
            "bbox": detection.bbox,
        }
        for detection in service.detect(vehicle_crop)
    ]

for engine in ["easyocr", "fastalpr", "paddleocr"]:
    result = OCRService(
        engine=engine,
        regex_profile="AUTO",
        confidence_threshold=0.1,
    ).run_ocr(plate_crop)
    output["ocr"][engine] = (
        {
            "text": result.text,
            "raw_text": result.raw_text,
            "regex_valid": result.regex_valid,
            "confidence": round(result.confidence, 5),
        }
        if result else None
    )

serialized = json.dumps(output, sort_keys=True, ensure_ascii=True)
if destination := os.environ.get("REGRESSION_OUTPUT_PATH"):
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    Path(destination).write_text(serialized + "\n", encoding="utf-8")
print(serialized)
