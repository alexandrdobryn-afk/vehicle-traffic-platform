from typing import Any, Dict, List, Optional
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BRAND_ALIASES = {
    "landrover": "Land Rover",
    "rangerover": "Range Rover",
    "mercedes": "Mercedes-Benz",
    "pegout": "Peugeot",
    "wolksvogen": "Volkswagen",
}


class BrandRecognitionService:
    """Detect a visible vehicle emblem and resolve a conservative track-level make."""

    def __init__(self, model_config=None, execution_device: Optional[str] = None):
        self.model = None
        self.model_config = model_config
        self.execution_device = execution_device or "cpu"
        if model_config and model_config.available:
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_config.path)
                logger.info("Brand detector loaded: %s", model_config.name)
            except Exception as exc:
                logger.warning("Brand detector load failed: %s", exc)

    @staticmethod
    def _display_name(raw: str) -> str:
        key = raw.strip().lower()
        return BRAND_ALIASES.get(key, key.replace("_", " ").title())

    def recognize_details(self, vehicle_crop: np.ndarray) -> Dict[str, Any]:
        if self.model is None or vehicle_crop is None or vehicle_crop.size == 0:
            return {"brand": "unknown", "confidence": 0.0, "candidates": [], "status": "unavailable"}
        try:
            # CCTV emblems occupy only a few source pixels. A 960px inference
            # canvas materially improves recall; sharpening/cropped variants
            # were rejected because they introduced confident false brands.
            result = self.model.predict(
                vehicle_crop,
                imgsz=960,
                conf=0.03,
                verbose=False,
                device=self.execution_device,
                half=False,
            )[0]
            scores: Dict[str, float] = {}
            if result.boxes is not None:
                for box in result.boxes:
                    index = int(box.cls[0].item())
                    brand = self._display_name(self.model.names[index])
                    scores[brand] = max(scores.get(brand, 0.0), float(box.conf[0].item()))
            candidates = [
                {"brand": brand, "confidence": round(confidence, 4)}
                for brand, confidence in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:3]
            ]
            if not candidates:
                return {"brand": "unknown", "confidence": 0.0, "candidates": [], "status": "not_found"}
            top = candidates[0]
            runner_up = candidates[1]["confidence"] if len(candidates) > 1 else 0.0
            margin = float(top["confidence"] - runner_up)
            accepted = float(top["confidence"]) >= 0.45 and margin >= 0.08
            return {
                "brand": top["brand"] if accepted else "unknown",
                "candidate_brand": top["brand"],
                "confidence": float(top["confidence"]),
                "margin": round(margin, 4),
                "candidates": candidates,
                "status": "recognized" if accepted else "ambiguous",
            }
        except Exception as exc:
            logger.debug("Brand recognition failed: %s", exc)
            return {"brand": "unknown", "confidence": 0.0, "candidates": [], "status": "error"}

    @staticmethod
    def update_voting(history: List[Dict[str, Any]], observation: Dict[str, Any], max_history: int = 10):
        history.append(observation)
        return history[-max_history:]

    @staticmethod
    def resolve_brand(history: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not history:
            return {"brand": "unknown", "confidence": 0.0, "support_count": 0, "sample_count": 0, "distribution": {}}
        scores: Dict[str, float] = {}
        support: Dict[str, int] = {}
        for observation in history:
            seen = set()
            for candidate in observation.get("candidates", []):
                brand = str(candidate.get("brand", "unknown"))
                confidence = float(candidate.get("confidence", 0.0))
                if brand == "unknown" or confidence <= 0:
                    continue
                scores[brand] = scores.get(brand, 0.0) + confidence
                if brand not in seen:
                    support[brand] = support.get(brand, 0) + 1
                    seen.add(brand)
        if not scores:
            return {"brand": "unknown", "confidence": 0.0, "support_count": 0, "sample_count": len(history), "distribution": {}}
        total = max(sum(scores.values()), 1e-6)
        winner = max(scores, key=scores.get)
        share = scores[winner] / total
        winner_support = support.get(winner, 0)
        average = scores[winner] / max(winner_support, 1)
        enough_evidence = (
            (winner_support >= 2 and share >= 0.56 and average >= 0.40)
            or (winner_support == 1 and share >= 0.68 and average >= 0.50)
        )
        confidence = average * (0.55 + share * 0.45)
        return {
            "brand": winner if enough_evidence else "unknown",
            "candidate_brand": winner,
            "confidence": min(float(confidence), 0.99),
            "support_count": winner_support,
            "sample_count": len(history),
            "distribution": {
                brand: round(score / total * 100.0, 1)
                for brand, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
            },
            "provisional": not enough_evidence or winner_support < 3,
        }
