"""Run one small inference smoke test per active drone-vision component."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np


def timed(name, callback):
    started = time.perf_counter()
    result = callback()
    print(json.dumps({
        "component": name,
        "ok": True,
        "seconds": round(time.perf_counter() - started, 2),
        "result": result,
    }))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("component", choices=[
        "yolo11n_object",
        "yolo11s_object",
        "rfdetr_nano",
        "rfdetr_medium",
        "tracker",
    ])
    args = parser.parse_args()

    frame = np.zeros((512, 768, 3), dtype=np.uint8)
    cv2.rectangle(frame, (120, 180), (220, 260), (255, 255, 255), -1)
    cv2.rectangle(frame, (420, 120), (500, 190), (180, 180, 180), -1)

    if args.component.startswith("yolo"):
        from ultralytics import YOLO
        model_id = args.component.replace("_object", "")
        path = f"/app/models/object_detector/{model_id}.pt"
        timed(args.component, lambda: {
            "detections": len(YOLO(path).predict(frame, verbose=False)[0].boxes),
            "path": path,
        })
    elif args.component == "tracker":
        from app.services.object_detection_service import Detection
        from app.services.tracking_service import TrackingService
        service = TrackingService("bytetrack")
        detections = [Detection([120, 180, 220, 260], 0.9, "object", 0)]
        timed(args.component, lambda: {
            "tracks": len(service.update(detections, frame)),
            "resolved": service.resolved_mode,
        })
    else:
        from rfdetr import RFDETRMedium, RFDETRNano
        cls = RFDETRNano if args.component.endswith("nano") else RFDETRMedium
        model = cls()
        timed(args.component, lambda: {
            "detections": len(model.predict(frame, threshold=0.9)),
        })


if __name__ == "__main__":
    main()
