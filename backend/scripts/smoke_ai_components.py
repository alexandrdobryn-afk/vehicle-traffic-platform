"""Download model caches when needed and run one real inference per component."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np


def timed(name, callback):
    started = time.perf_counter()
    result = callback()
    print(json.dumps({"component": name, "ok": True, "seconds": round(time.perf_counter() - started, 2), "result": result}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("component", choices=["yolo26n", "yolo26s", "tracktrack", "fastalpr", "paddleocr", "rfdetr_nano", "rfdetr_medium", "openimagemodels_plate", "yolo11n_plate"])
    args = parser.parse_args()
    frame = np.zeros((384, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "AA1234BX", (180, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    if args.component.startswith("yolo26") or args.component == "yolo11n_plate":
        from ultralytics import YOLO
        path = (
            "/app/models/plate_detector/yolo11n_plate.pt"
            if args.component == "yolo11n_plate"
            else f"/app/models/vehicle_detector/{args.component}.pt"
        )
        timed(args.component, lambda: {"detections": len(YOLO(path).predict(frame, verbose=False)[0].boxes), "path": path})
    elif args.component == "openimagemodels_plate":
        from app.models.model_registry import PLATE_DETECTOR_OPENIMAGEMODELS
        from app.services.plate_detection_service import PlateDetectionService
        service = PlateDetectionService(PLATE_DETECTOR_OPENIMAGEMODELS, confidence_threshold=0.1)
        timed(args.component, lambda: {"detections": len(service.detect(frame)), "path": PLATE_DETECTOR_OPENIMAGEMODELS.path})
    elif args.component == "tracktrack":
        from app.services.tracking_service import TrackingService
        from app.services.vehicle_detection_service import Detection
        service = TrackingService("tracktrack")
        detections = [Detection([100, 100, 300, 300], 0.9, "car", 2)]
        timed(args.component, lambda: {"tracks": len(service.update(detections, frame)), "resolved": service.resolved_mode})
    elif args.component in {"fastalpr", "paddleocr"}:
        from app.services.ocr_service import OCRService
        service = OCRService(engine=args.component, regex_profile="AUTO", confidence_threshold=0.1)
        crop = frame[160:250, 150:490]
        timed(args.component, lambda: {"engine": service.resolved_engine, "prediction": getattr(service.run_ocr(crop), "text", None)})
    else:
        os.chdir("/app/models/vehicle_detector")
        from rfdetr import RFDETRMedium, RFDETRNano
        cls = RFDETRNano if args.component.endswith("nano") else RFDETRMedium
        model = cls()
        timed(args.component, lambda: {"detections": len(model.predict(frame, threshold=0.9))})


if __name__ == "__main__":
    main()
