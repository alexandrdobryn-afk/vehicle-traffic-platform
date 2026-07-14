"""Lightweight object embedding/ReID service.

The current implementation is an HSV histogram embedding. It is intentionally
small and dependency-light, but the API is shaped so a neural ReID model can
replace `_signature` later without changing the pipeline.
"""

from __future__ import annotations

from typing import Dict, Optional
import hashlib

import cv2
import numpy as np


class ReIdentificationService:
    def __init__(self, similarity_threshold: float = 0.72, model_name: str = "hsv_histogram_v1"):
        self.similarity_threshold = float(similarity_threshold)
        self.model_name = model_name
        self._gallery: Dict[int, np.ndarray] = {}

    @property
    def available(self) -> bool:
        return True

    def update(self, track_id: int, crop: np.ndarray) -> Dict:
        embedding = self.embed(crop)
        vector = embedding.pop("vector", None)
        if vector is None:
            return {"status": "empty", "similarity": 0.0, "nearest_track_id": None}
        signature = np.asarray(vector, dtype=np.float32)

        nearest_track_id: Optional[int] = None
        nearest_similarity = 0.0
        for candidate_id, candidate in self._gallery.items():
            if candidate_id == track_id:
                continue
            similarity = float(np.dot(signature, candidate))
            if similarity > nearest_similarity:
                nearest_track_id = candidate_id
                nearest_similarity = similarity

        previous = self._gallery.get(track_id)
        self._gallery[track_id] = (
            signature if previous is None else self._normalize(previous * 0.72 + signature * 0.28)
        )
        signature_hash = self._hash(self._gallery[track_id])
        status = "candidate_match" if nearest_similarity >= self.similarity_threshold else "indexed"
        return {
            "status": status,
            "similarity": round(nearest_similarity, 4),
            "nearest_track_id": nearest_track_id,
            "embedding_hash": signature_hash,
            "embedding_model": self.model_name,
            "embedding_dim": int(self._gallery[track_id].shape[0]),
        }

    def forget(self, track_id: int):
        self._gallery.pop(track_id, None)

    def embed(self, crop: np.ndarray) -> Dict:
        signature = self._signature(crop)
        if signature is None:
            return {
                "status": "empty",
                "vector": None,
                "embedding_hash": None,
                "embedding_model": self.model_name,
                "embedding_dim": 0,
            }
        return {
            "status": "embedded",
            "vector": signature.round(6).tolist(),
            "embedding_hash": self._hash(signature),
            "embedding_model": self.model_name,
            "embedding_dim": int(signature.shape[0]),
        }

    @staticmethod
    def _signature(crop: np.ndarray) -> Optional[np.ndarray]:
        if crop is None or crop.size == 0:
            return None
        resized = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 8, 4], [0, 180, 0, 256, 0, 256])
        return ReIdentificationService._normalize(hist.reshape(-1).astype(np.float32))

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-9:
            return vector
        return vector / norm

    @staticmethod
    def _hash(vector: np.ndarray) -> str:
        return hashlib.sha1(vector.astype(np.float32).tobytes()).hexdigest()[:12]
