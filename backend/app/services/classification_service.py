"""Generic project-trained ONNX image classifier."""

from typing import Dict, List
import logging
import json
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ClassificationService:
    def __init__(self, model_config, execution_device: str = "cpu", threshold: float = 0.35):
        self.model_config = model_config
        self.threshold = float(threshold)
        self.execution_device = execution_device
        self.session = None
        self.yolo_model = None
        self.labels = list(model_config.classes or [])
        self.model_path = getattr(model_config, "artifact_path", model_config.path)
        labels_path = Path(self.model_path).with_name("labels.json") if self.model_path else None
        if not self.labels and labels_path and labels_path.is_file():
            try:
                labels = json.loads(labels_path.read_text(encoding="utf-8"))
                self.labels = list(labels.get("classes", labels) if isinstance(labels, dict) else labels)
            except (OSError, ValueError, TypeError) as exc:
                logger.warning("Classifier labels load failed: %s", exc)
        if not model_config.available:
            return
        if self.model_path and self.model_path.lower().endswith(".pt"):
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(self.model_path)
                names = getattr(self.yolo_model, "names", None) or getattr(self.yolo_model.model, "names", None)
                if isinstance(names, dict):
                    self.labels = [str(names[index]) for index in sorted(names)]
                elif isinstance(names, list):
                    self.labels = [str(name) for name in names]
            except Exception as exc:
                logger.warning("Ultralytics classifier load failed: %s", exc)
            return
        try:
            import onnxruntime as ort
            providers = ["CPUExecutionProvider"]
            if execution_device.startswith("cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")
            self.session = ort.InferenceSession(self.model_path, providers=providers)
        except Exception as exc:
            logger.warning("Generic classifier load failed: %s", exc)

    @property
    def available(self) -> bool:
        return (self.session is not None or self.yolo_model is not None) and bool(self.labels)

    def classify(self, crop: np.ndarray, top_k: int = 3) -> Dict:
        if not self.available or crop is None or crop.size == 0:
            return {"label": "unknown", "confidence": 0.0, "top_k": [], "status": "unavailable"}
        if self.yolo_model is not None:
            result = self.yolo_model.predict(
                crop,
                imgsz=self.model_config.input_size[0] if self.model_config.input_size else 224,
                device=self.execution_device,
                verbose=False,
            )[0]
            probs = getattr(result, "probs", None)
            if probs is None:
                return {"label": "unknown", "confidence": 0.0, "top_k": [], "status": "unavailable"}
            indices = [int(index) for index in probs.top5[:max(1, int(top_k))]]
            confidences = [float(value) for value in probs.top5conf[:len(indices)]]
            names = result.names or {}
            ranked = [
                {
                    "label": str(names.get(index, self.labels[index] if index < len(self.labels) else index)),
                    "confidence": round(confidence, 4),
                }
                for index, confidence in zip(indices, confidences)
            ]
            best = ranked[0] if ranked else {"label": "unknown", "confidence": 0.0}
            accepted = best["confidence"] >= self.threshold
            return {"label": best["label"] if accepted else "unknown", "confidence": best["confidence"], "top_k": ranked, "status": "classified" if accepted else "uncertain"}
        width, height = self.model_config.input_size or (224, 224)
        image = cv2.cvtColor(cv2.resize(crop, (width, height)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = np.transpose(image, (2, 0, 1))[None]
        logits = np.asarray(self.session.run(None, {self.session.get_inputs()[0].name: tensor})[0]).reshape(-1)
        probabilities = np.exp(logits - logits.max())
        probabilities /= max(float(probabilities.sum()), 1e-9)
        indices = np.argsort(probabilities)[::-1][:max(1, int(top_k))]
        ranked: List[Dict] = [
            {"label": self.labels[int(index)] if int(index) < len(self.labels) else str(index), "confidence": round(float(probabilities[index]), 4)}
            for index in indices
        ]
        best = ranked[0]
        accepted = best["confidence"] >= self.threshold
        return {"label": best["label"] if accepted else "unknown", "confidence": best["confidence"], "top_k": ranked, "status": "classified" if accepted else "uncertain"}
