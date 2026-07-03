import cv2
import numpy as np
from typing import Any, List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

COLOR_CLASSES = [
    "black", "white", "gray", "silver",
    "red", "blue", "green", "yellow",
    "orange", "brown", "beige", "unknown"
]

CLASS_HTML_HEX = {
    "black": "#1A1A1A", "white": "#F5F5F5", "gray": "#808080",
    "silver": "#C0C0C0", "red": "#DC2626", "blue": "#2563EB",
    "green": "#16A34A", "yellow": "#CA8A04", "orange": "#EA580C",
    "brown": "#92400E", "beige": "#D4B896", "unknown": "#6B7280",
}

# HSV ranges for fallback KMeans color mapping
COLOR_HSV_RANGES = {
    "red":    [(0, 100, 100), (10, 255, 255), (160, 100, 100), (180, 255, 255)],
    "orange": [(11, 100, 100), (25, 255, 255)],
    "yellow": [(26, 100, 100), (35, 255, 255)],
    "green":  [(36, 50, 50), (85, 255, 255)],
    "blue":   [(86, 50, 50), (130, 255, 255)],
    "white":  [(0, 0, 200), (180, 30, 255)],
    "black":  [(0, 0, 0), (180, 255, 50)],
    "gray":   [(0, 0, 51), (180, 50, 199)],
    "silver": [(0, 0, 160), (180, 40, 220)],
    "brown":  [(10, 50, 50), (20, 200, 150)],
    "beige":  [(20, 20, 150), (35, 60, 220)],
}

# Canonical paint hue anchors (OpenCV hue: 0..179). Classification is based
# on the accumulated pixel mass nearest to these anchors, not on one KMeans
# cluster. That matters when reflections split one red body into several
# magenta/red clusters while a single blue reflection becomes the largest
# individual cluster.
COLOR_HUE_PALETTE = {
    "red": 0.0,
    "orange": 18.0,
    "yellow": 30.0,
    "green": 60.0,
    "blue": 110.0,
}


class ColorRecognitionService:
    """
    Vehicle color recognition.
    Primary: ONNX classifier (MobileNetV3)
    Fallback: HSV/KMeans analysis
    Multi-frame voting for stability.
    """

    def __init__(self, model_config=None, execution_device: Optional[str] = None):
        self.model_session = None
        self.model_config = model_config
        self.execution_device = execution_device or "cpu"
        self.scene_hue = None
        self._load_model()

    def update_scene_context(self, frame: np.ndarray):
        """Estimate the dominant illumination hue for fallback de-biasing."""
        if self.model_session is not None or frame is None or frame.size == 0:
            return
        sample = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        mask = (hsv[:, 1] >= 35) & (hsv[:, 2] >= 30) & (hsv[:, 2] <= 225)
        if np.count_nonzero(mask) < 100:
            self.scene_hue = None
            return
        histogram, _ = np.histogram(hsv[mask, 0], bins=180, range=(0, 180))
        self.scene_hue = float(np.argmax(histogram))

    def _load_model(self):
        if self.model_config and self.model_config.available:
            try:
                import onnxruntime as ort
                providers = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if self.execution_device.startswith("cuda")
                    and "CUDAExecutionProvider" in ort.get_available_providers()
                    else ["CPUExecutionProvider"]
                )
                self.model_session = ort.InferenceSession(
                    self.model_config.path,
                    providers=providers
                )
                logger.info(f"Color classifier loaded: {self.model_config.name}")
            except Exception as e:
                logger.warning(f"Color classifier load failed: {e}, using HSV fallback")
        else:
            logger.info("Color classifier: using HSV/KMeans fallback")

    def recognize(self, vehicle_crop: np.ndarray) -> Tuple[str, float]:
        """Recognize vehicle color from crop."""
        if vehicle_crop is None or vehicle_crop.size == 0:
            return "unknown", 0.0

        if self.model_session is not None:
            body_crop = self._extract_body_region(vehicle_crop)
            if body_crop is None or body_crop.size == 0:
                body_crop = vehicle_crop
            normalized = self._normalize_illumination(body_crop)
            return self._classify_onnx(normalized)
        # The fallback intentionally works on raw luminance. CLAHE makes black
        # paint look gray and amplifies blue sky/red lamp reflections.
        return self._classify_multizone(vehicle_crop)

    def recognize_details(self, vehicle_crop: np.ndarray) -> Dict[str, Any]:
        """Return a frame observation including measured HTML color and quality."""
        color, confidence = self.recognize(vehicle_crop)
        if vehicle_crop is None or vehicle_crop.size == 0:
            return {
                "color": "unknown", "confidence": 0.0,
                "hex_code": CLASS_HTML_HEX["unknown"], "quality_score": 0.0,
            }
        regions = self._extract_paint_regions(vehicle_crop)
        body_crop = self._combine_regions(regions)
        if body_crop is None or body_crop.size == 0:
            body_crop = self._extract_body_region(vehicle_crop)
        if body_crop is None or body_crop.size == 0:
            body_crop = vehicle_crop
        hex_code, quality_score = self._measure_html_color(body_crop, color)
        return {
            "color": color,
            "confidence": float(confidence),
            "hex_code": hex_code,
            "quality_score": quality_score,
        }

    def _measure_html_color(self, body_crop: np.ndarray, color: str) -> Tuple[str, float]:
        """Measure representative paint in LAB and return #RRGGBB plus frame quality."""
        try:
            hsv = cv2.cvtColor(body_crop, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            usable = v >= 25
            if color == "red":
                paint = usable & ((h <= 12) | (h >= 145)) & (s >= 45)
            elif color == "orange":
                paint = usable & (h >= 10) & (h <= 28) & (s >= 45) & (v >= 120)
            elif color == "yellow":
                paint = usable & (h >= 24) & (h <= 38) & (s >= 40)
            elif color == "green":
                paint = usable & (h >= 36) & (h <= 85) & (s >= 35)
            elif color == "blue":
                paint = usable & (h >= 86) & (h <= 145) & (s >= 35)
            elif color == "brown":
                paint = usable & (h >= 5) & (h <= 28) & (s >= 35) & (v < 150)
            elif color == "beige":
                paint = usable & (h >= 15) & (h <= 40) & (s >= 15) & (s <= 110) & (v >= 120)
            elif color == "black":
                paint = usable & (v < 80)
            elif color == "white":
                paint = usable & (s < 55) & (v >= 175)
            elif color == "silver":
                paint = usable & (s < 60) & (v >= 135)
            else:
                paint = usable & (s < 70) & (v >= 60) & (v < 180)

            if int(np.count_nonzero(paint)) < 30:
                paint = usable
            if int(np.count_nonzero(paint)) < 30:
                return CLASS_HTML_HEX.get(color, CLASS_HTML_HEX["unknown"]), 0.0

            lab = cv2.cvtColor(body_crop, cv2.COLOR_BGR2LAB)
            representative_lab = np.median(lab[paint], axis=0).astype(np.uint8).reshape(1, 1, 3)
            representative_bgr = cv2.cvtColor(representative_lab, cv2.COLOR_LAB2BGR)[0, 0]
            blue, green, red = (int(channel) for channel in representative_bgr)
            hex_code = f"#{red:02X}{green:02X}{blue:02X}"

            gray = cv2.cvtColor(body_crop, cv2.COLOR_BGR2GRAY)
            size_score = min(1.0, np.sqrt(body_crop.shape[0] * body_crop.shape[1]) / 220.0)
            sharpness_score = min(1.0, float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 300.0)
            median_value = float(np.median(v[usable])) if np.any(usable) else 0.0
            exposure_score = max(0.0, 1.0 - abs(median_value - 135.0) / 135.0)
            paint_ratio = float(np.count_nonzero(paint)) / max(1, int(np.count_nonzero(usable)))
            coverage_score = min(1.0, paint_ratio / 0.45)
            quality_score = (
                size_score * 0.25 + sharpness_score * 0.20
                + exposure_score * 0.25 + coverage_score * 0.30
            )
            return hex_code, round(float(np.clip(quality_score, 0.0, 1.0)), 3)
        except Exception as exc:
            logger.debug(f"HTML color measurement error: {exc}")
            return CLASS_HTML_HEX.get(color, CLASS_HTML_HEX["unknown"]), 0.0

    def _extract_body_region(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract vehicle body area, excluding:
        - windows (top portion)
        - wheels (bottom portion)
        - headlights (edges)
        """
        h, w = crop.shape[:2]
        # The upper middle of frontal/rear vehicles is usually glass. Prefer
        # the lower body band and trim edges where road/headlights dominate.
        y1 = int(h * 0.38)
        y2 = int(h * 0.80)
        x1 = int(w * 0.14)
        x2 = int(w * 0.86)

        body = crop[y1:y2, x1:x2]
        return body if body.size > 0 else None

    def _extract_paint_regions(self, crop: np.ndarray) -> List[np.ndarray]:
        """Return interior body zones while avoiding glass, lamps and bbox background."""
        h, w = crop.shape[:2]
        # Zones cover roof/hood/trunk, both painted shoulders and the central
        # bumper. They deliberately avoid the outer corners where headlights,
        # tail lights, wheels and background dominate.
        boxes = [
            (0.28, 0.14, 0.72, 0.30),
            (0.24, 0.57, 0.76, 0.73),
            (0.30, 0.72, 0.70, 0.87),
            (0.10, 0.32, 0.27, 0.68),
            (0.73, 0.32, 0.90, 0.68),
        ]
        regions: List[np.ndarray] = []
        for x1f, y1f, x2f, y2f in boxes:
            x1, x2 = int(w * x1f), int(w * x2f)
            y1, y2 = int(h * y1f), int(h * y2f)
            region = crop[y1:y2, x1:x2]
            if region.size and region.shape[0] >= 8 and region.shape[1] >= 8:
                regions.append(region)
        return regions

    @staticmethod
    def _combine_regions(regions: List[np.ndarray]) -> Optional[np.ndarray]:
        if not regions:
            return None
        normalized = [cv2.resize(region, (96, 64), interpolation=cv2.INTER_AREA) for region in regions]
        return np.hstack(normalized)

    def _classify_multizone(self, crop: np.ndarray) -> Tuple[str, float]:
        """Vote across independent paint zones so glass or one lamp cannot win."""
        regions = self._extract_paint_regions(crop)
        if not regions:
            body = self._extract_body_region(crop)
            return self._classify_hsv(body if body is not None else crop)

        combined = self._combine_regions(regions)
        if combined is not None:
            hsv = cv2.cvtColor(combined, cv2.COLOR_BGR2HSV).reshape(-1, 3)
            usable = hsv[hsv[:, 2] >= 20]
            if len(usable) >= 100:
                median_v = float(np.median(usable[:, 2]))
                dark_v = float(np.percentile(usable[:, 2], 30))
                strong_paint = (usable[:, 1] >= 70) & (usable[:, 2] >= 55)
                strong_ratio = float(np.mean(strong_paint))
                # Black paint commonly mirrors a blue sky across every glossy
                # panel, but those reflected pixels remain weakly saturated.
                # True dark-blue/red paint has a substantial strongly
                # saturated mass and is therefore not caught by this guard.
                if median_v < 135 and dark_v < 85 and strong_ratio < 0.10:
                    confidence = 0.48 + min(0.16, (135 - median_v) / 300)
                    return "black", min(confidence, 0.68)

        observations: List[Tuple[str, float]] = []
        for region in regions:
            color, confidence = self._classify_hsv(region)
            if color != "unknown" and confidence > 0:
                observations.append((color, confidence))
        if not observations:
            return "unknown", 0.0

        scores: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for color, confidence in observations:
            scores[color] = scores.get(color, 0.0) + confidence
            counts[color] = counts.get(color, 0) + 1

        chromatic_names = {"red", "blue", "green", "yellow", "orange", "brown", "beige"}
        best_color = max(scores, key=scores.get)
        # Reflected sky and a pair of tail lamps are usually confined to one
        # zone. A chromatic label must agree across at least two body zones.
        if best_color in chromatic_names and counts.get(best_color, 0) < 2:
            neutral_scores = {
                color: score for color, score in scores.items()
                if color not in chromatic_names and color != "unknown"
            }
            if neutral_scores:
                best_color = max(neutral_scores, key=neutral_scores.get)
            else:
                return "unknown", min(0.34, scores[best_color])

        # White and silver often split across highlighted and shaded panels.
        # Treat them as one light-paint family, then use raw zone luminance.
        light_score = scores.get("white", 0.0) + scores.get("silver", 0.0)
        if light_score > scores.get(best_color, 0.0):
            neutral_values = []
            for region in regions:
                hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).reshape(-1, 3)
                neutral_values.extend(hsv[hsv[:, 1] < 50, 2].tolist())
            median_light = float(np.median(neutral_values)) if neutral_values else 0.0
            best_color = "white" if median_light >= 178 else "silver"
            scores[best_color] = light_score

        total = max(sum(scores.values()), 1e-6)
        share = scores.get(best_color, 0.0) / total
        support = counts.get(best_color, 0)
        if best_color in {"white", "silver"}:
            support = counts.get("white", 0) + counts.get("silver", 0)
        confidence = 0.32 + min(0.24, share * 0.24) + min(0.14, support * 0.035)
        if share < 0.34 and support < 2:
            return "unknown", min(confidence, 0.45)
        return best_color, min(confidence, 0.72)

    def _normalize_illumination(self, img: np.ndarray) -> np.ndarray:
        """Apply illumination normalization using LAB color space."""
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_normalized = clahe.apply(l_channel)
            normalized_lab = cv2.merge([l_normalized, a, b])
            return cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)
        except Exception:
            return img

    def _classify_onnx(self, img: np.ndarray) -> Tuple[str, float]:
        """Run ONNX color classifier."""
        try:
            input_size = (224, 224)
            resized = cv2.resize(img, input_size)
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            inp = resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            inp = (inp - mean) / std
            inp = np.transpose(inp, (2, 0, 1))
            inp = np.expand_dims(inp, 0).astype(np.float32)

            input_name = self.model_session.get_inputs()[0].name
            outputs = self.model_session.run(None, {input_name: inp})
            probs = self._softmax(outputs[0][0])

            class_idx = int(np.argmax(probs))
            confidence = float(probs[class_idx])

            if class_idx < len(COLOR_CLASSES):
                return COLOR_CLASSES[class_idx], confidence
            return "unknown", confidence
        except Exception as e:
            logger.debug(f"ONNX color classification error: {e}")
            return self._classify_hsv(img)

    def _classify_hsv(self, img: np.ndarray) -> Tuple[str, float]:
        """
        HSV + KMeans dominant color detection.
        Fallback when no trained model is available.
        """
        try:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            pixels = hsv.reshape(-1, 3).astype(np.float32)
            saturation = pixels[:, 1]
            value = pixels[:, 2]
            # Saturated paint may legitimately reach 255 after CLAHE. Keep it;
            # highlights are filtered by saturation and palette agreement.
            usable = value >= 25
            chromatic = usable & (saturation >= 45)
            neutral = usable & (saturation < 45)

            usable_count = int(np.count_nonzero(usable))
            if usable_count < 50:
                return "unknown", 0.0

            chromatic_ratio = float(np.count_nonzero(chromatic)) / usable_count
            if chromatic_ratio >= 0.30:
                chromatic_pixels = pixels[chromatic]
            else:
                neutral_pixels = pixels[neutral]
                if len(neutral_pixels) < 50:
                    return "unknown", 0.0
                median_v = float(np.median(neutral_pixels[:, 2]))
                dark_quantile_v = float(np.percentile(neutral_pixels[:, 2], 30))
                # Glossy black paint contains bright sky/tree reflections, so
                # its median can look gray or even silver after CLAHE. A large
                # genuinely dark body mass is a more stable signal than the
                # median, while the median guard prevents a few shadows on a
                # bright silver vehicle from turning it black.
                if median_v < 65 or (dark_quantile_v < 95 and median_v < 150):
                    darkness = max(65 - median_v, 95 - dark_quantile_v)
                    return "black", min(0.58, 0.36 + darkness / 180)
                if median_v > 190:
                    return "white", min(0.58, 0.30 + (median_v - 190) / 150)
                return ("gray" if median_v < 145 else "silver"), 0.38

            hues = chromatic_pixels[:, 0]
            weights = 0.55 + chromatic_pixels[:, 1] / 510.0
            palette_names = list(COLOR_HUE_PALETTE)
            anchors = np.array([COLOR_HUE_PALETTE[name] for name in palette_names])
            distances = np.abs(hues[:, None] - anchors[None, :])
            distances = np.minimum(distances, 180.0 - distances)
            assignments = np.argmin(distances, axis=1)

            # Suppress the global illumination hue rather than relabelling a
            # whole vehicle because sky/camera tint appears on every surface.
            if self.scene_hue is not None:
                scene_distance = np.abs(hues - self.scene_hue)
                scene_distance = np.minimum(scene_distance, 180.0 - scene_distance)
                weights = np.where(scene_distance <= 10.0, weights * 0.30, weights)

            scores = np.array([
                float(weights[assignments == index].sum())
                for index in range(len(palette_names))
            ])
            winner = int(np.argmax(scores))
            color = palette_names[winner]
            share = float(scores[winner] / max(scores.sum(), 1e-6))

            if color == "blue" and self.scene_hue is not None:
                winner_hues = hues[assignments == winner]
                if len(winner_hues):
                    scene_distance = np.abs(winner_hues - self.scene_hue)
                    scene_distance = np.minimum(scene_distance, 180.0 - scene_distance)
                    if float(np.mean(scene_distance <= 12.0)) >= 0.65:
                        brightness = float(np.median(value[usable]))
                        if brightness < 65:
                            return "black", 0.40
                        if brightness > 195:
                            return "white", 0.42
                        if brightness > 150:
                            return "silver", 0.39
                        return "gray", 0.38

            # Dark orange pixels are perceived as brown vehicle paint.
            winner_pixels = chromatic_pixels[assignments == winner]
            if color == "orange" and len(winner_pixels) and float(np.median(winner_pixels[:, 2])) < 135:
                color = "brown"

            confidence = 0.34 + share * 0.36 + min(chromatic_ratio, 0.7) * 0.16
            if share < 0.34:
                return "unknown", min(confidence, 0.49)
            return color, min(confidence, 0.72)
        except Exception as e:
            logger.debug(f"HSV color classification error: {e}")
            return "unknown", 0.0

    def _hsv_to_color_name(self, hsv_pixel: np.ndarray) -> str:
        h, s, v = float(hsv_pixel[0]), float(hsv_pixel[1]), float(hsv_pixel[2])

        if v < 50:
            return "black"
        if s < 30 and v > 200:
            return "white"
        if s < 40:
            return "gray" if v < 160 else "silver"

        if h <= 10 or h >= 160:
            return "red"
        elif h <= 25:
            return "orange"
        elif h <= 35:
            return "yellow"
        elif h <= 85:
            return "green"
        elif h <= 130:
            return "blue"
        elif h <= 145:
            return "blue"
        else:
            return "brown"

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()

    # ─── Multi-frame Voting ───────────────────────────────────────

    def update_color_voting(
        self,
        history: List[Any],
        new_color: str,
        new_confidence: float,
        hex_code: Optional[str] = None,
        quality_score: float = 0.5,
        frame_index: Optional[int] = None,
        max_history: int = 15,
    ) -> List[Any]:
        """Add a measured observation; keep a bounded temporal window."""
        history.append({
            "color": new_color,
            "confidence": round(float(new_confidence), 4),
            "hex_code": hex_code or CLASS_HTML_HEX.get(new_color, CLASS_HTML_HEX["unknown"]),
            "quality_score": round(float(np.clip(quality_score, 0.0, 1.0)), 4),
            "frame_index": frame_index,
        })
        if len(history) > max_history:
            history = history[-max_history:]
        return history

    @staticmethod
    def _normalize_observation(item: Any) -> Dict[str, Any]:
        if isinstance(item, dict):
            color = str(item.get("color", "unknown"))
            return {
                "color": color,
                "confidence": float(item.get("confidence", 0.0)),
                "hex_code": str(item.get("hex_code") or CLASS_HTML_HEX.get(color, CLASS_HTML_HEX["unknown"])),
                "quality_score": float(item.get("quality_score", 0.5)),
                "frame_index": item.get("frame_index"),
            }
        color, confidence = item[:2]
        return {
            "color": str(color), "confidence": float(confidence),
            "hex_code": CLASS_HTML_HEX.get(str(color), CLASS_HTML_HEX["unknown"]),
            "quality_score": 0.5, "frame_index": None,
        }

    @staticmethod
    def _aggregate_html_hex(observations: List[Dict[str, Any]], weights: List[float], fallback: str) -> str:
        valid = []
        valid_weights = []
        for observation, weight in zip(observations, weights):
            code = observation.get("hex_code", "")
            if not isinstance(code, str) or len(code) != 7 or not code.startswith("#"):
                continue
            try:
                rgb = np.array([int(code[index:index + 2], 16) for index in (1, 3, 5)], dtype=np.uint8)
            except ValueError:
                continue
            bgr = rgb[::-1].reshape(1, 1, 3)
            valid.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32))
            valid_weights.append(weight)
        if not valid:
            return fallback
        mean_lab = np.average(np.stack(valid), axis=0, weights=np.array(valid_weights)).astype(np.uint8)
        blue, green, red = (int(value) for value in cv2.cvtColor(mean_lab.reshape(1, 1, 3), cv2.COLOR_LAB2BGR)[0, 0])
        return f"#{red:02X}{green:02X}{blue:02X}"

    def resolve_color_details(self, history: List[Any], sample_limit: int = 10) -> Dict[str, Any]:
        """Resolve the best-quality multi-frame sample into color, confidence and HTML code."""
        if not history:
            return {
                "color": "unknown", "confidence": 0.0,
                "hex_code": CLASS_HTML_HEX["unknown"], "sample_count": 0,
                "support_count": 0, "distribution": {}, "provisional": True,
            }

        normalized = [self._normalize_observation(item) for item in history]
        ranked = sorted(
            normalized,
            key=lambda item: item["confidence"] * (0.35 + item["quality_score"] * 0.65),
            reverse=True,
        )[:sample_limit]
        color_scores: Dict[str, float] = {}
        color_counts: Dict[str, int] = {}
        observation_weights = []
        for observation in ranked:
            weight = observation["confidence"] * (0.35 + observation["quality_score"] * 0.65)
            observation_weights.append(weight)
            color = observation["color"]
            color_scores[color] = color_scores.get(color, 0.0) + weight
            color_counts[color] = color_counts.get(color, 0) + 1

        total_score = max(sum(color_scores.values()), 1e-6)
        winning_color = max(color_scores, key=color_scores.get)
        winning_share = color_scores[winning_color] / total_score
        winners = [item for item in ranked if item["color"] == winning_color]
        winner_weights = [
            item["confidence"] * (0.35 + item["quality_score"] * 0.65)
            for item in winners
        ]
        average_confidence = float(np.average(
            [item["confidence"] for item in winners], weights=winner_weights
        )) if winners else 0.0
        confidence = average_confidence * (0.55 + winning_share * 0.45)
        final_color = winning_color
        if winning_color == "unknown" or confidence < 0.35:
            final_color = "unknown"

        distribution = {
            color: round(score / total_score * 100.0, 1)
            for color, score in sorted(color_scores.items(), key=lambda pair: pair[1], reverse=True)
        }
        fallback_hex = CLASS_HTML_HEX.get(winning_color, CLASS_HTML_HEX["unknown"])
        html_hex = self._aggregate_html_hex(winners, winner_weights, fallback_hex)
        support_count = color_counts.get(winning_color, 0)
        sample_count = len(ranked)
        return {
            "color": final_color,
            "candidate_color": winning_color,
            "confidence": min(float(confidence), 0.99),
            "hex_code": html_hex,
            "sample_count": sample_count,
            "support_count": support_count,
            "distribution": distribution,
            "provisional": sample_count < 5 or winning_share < 0.55 or confidence < 0.55,
        }

    def resolve_color(
        self,
        history: List[Any]
    ) -> Tuple[str, float]:
        """Backward-compatible color/confidence tuple."""
        result = self.resolve_color_details(history)
        return result["color"], result["confidence"]
