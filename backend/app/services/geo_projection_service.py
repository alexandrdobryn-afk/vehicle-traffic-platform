"""Image-space geo projection contract for aerial object outputs."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


class GeoProjectionService:
    def __init__(self, telemetry: Optional[dict] = None, coordinate_output: bool = False):
        self.telemetry = telemetry or {}
        self.coordinate_output = bool(coordinate_output)

    @property
    def available(self) -> bool:
        return True

    def project(
        self,
        bbox: List[int],
        analysis_shape: Tuple[int, ...],
        source_shape: Tuple[int, ...],
    ) -> Dict:
        analysis_h, analysis_w = analysis_shape[:2]
        source_h, source_w = source_shape[:2]
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        normalized = {
            "x": round(center_x / max(analysis_w, 1), 6),
            "y": round(center_y / max(analysis_h, 1), 6),
        }
        source_bbox = [
            round(x1 * source_w / max(analysis_w, 1), 2),
            round(y1 * source_h / max(analysis_h, 1), 2),
            round(x2 * source_w / max(analysis_w, 1), 2),
            round(y2 * source_h / max(analysis_h, 1), 2),
        ]
        payload = {
            "status": "image_space",
            "normalized_center": normalized,
            "source_bbox": source_bbox,
            "telemetry": {
                key: self.telemetry.get(key)
                for key in ("latitude", "longitude", "altitude_m", "heading_deg", "camera_pitch_deg")
                if self.telemetry.get(key) is not None
            },
        }
        if self.coordinate_output and payload["telemetry"].get("latitude") is not None:
            payload["status"] = "telemetry_attached"
        return payload
