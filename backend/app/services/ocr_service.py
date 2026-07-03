import re
import cv2
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from collections import Counter
import logging

from app.services.plate_profiles import normalize_plate_for_profile, validate_plate

logger = logging.getLogger(__name__)


# ─── Regional Plate Regex Profiles ───────────────────────────────
@dataclass
class OCRResult:
    text: str
    confidence: float
    raw_text: str
    regex_valid: bool
    regex_score: float


@dataclass
class OCRCandidate:
    text: str
    raw_text: str
    confidence: float
    regex_valid: bool
    regex_score: float
    image_quality_score: float
    count: int = 1
    character_votes: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class VotingResult:
    final_plate: Optional[str]
    confidence: float
    status: str   # searching, candidate, verified, low_confidence, invalid
    candidates: List[OCRCandidate] = field(default_factory=list)


class OCRService:
    """
    Multi-engine OCR with fallback chain:
    PaddleOCR → EasyOCR → Tesseract
    + Regex validation + Temporal voting
    """

    def __init__(
        self,
        engine: str = "paddleocr",
        model_config=None,
        regex_profile: str = "UA",
        confidence_threshold: float = 0.60,
        voting_window: int = 10,
        execution_device: Optional[str] = None,
    ):
        self.engine = engine
        self.model_config = model_config
        self.regex_profile = regex_profile
        self.confidence_threshold = confidence_threshold
        self.voting_window = voting_window
        self.execution_device = execution_device

        self._paddle = None
        self._easyocr = None
        self._fast_plate_ocr = None
        self._lprnet_session = None
        self._tesseract_available = False
        self.resolved_engine = engine

        self._init_engine()

    def _init_engine(self):
        if self.engine == "paddleocr":
            self._init_paddle()
        elif self.engine == "lprnet":
            self._init_lprnet()
        elif self.engine == "easyocr":
            self._init_easyocr()
        elif self.engine == "fastalpr":
            self._init_fast_plate_ocr()
        else:
            self._init_paddle()

    def _init_paddle(self):
        try:
            from paddleocr import PaddleOCR
            try:
                # PaddleOCR 3.x API.
                self._paddle = PaddleOCR(
                    lang="en",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=True,
                    device="gpu" if self._has_gpu() else "cpu",
                )
            except (TypeError, ValueError):
                # PaddleOCR 2.x compatibility.
                self._paddle = PaddleOCR(
                    use_angle_cls=True,
                    lang="en",
                    use_gpu=self._has_gpu(),
                    show_log=False,
                )
            self.resolved_engine = "paddleocr"
            logger.info("PaddleOCR initialized")
        except Exception as e:
            logger.error("PaddleOCR init failed: %s", e)
            raise RuntimeError(f"PaddleOCR is unavailable: {e}") from e

    def _init_fast_plate_ocr(self):
        """Load FastALPR's ONNX OCR component for already-cropped plates."""
        try:
            from fast_plate_ocr import LicensePlateRecognizer
            self._fast_plate_ocr = LicensePlateRecognizer("cct-xs-v2-global-model")
            self.resolved_engine = "fastalpr"
            logger.info("FastPlateOCR initialized (FastALPR OCR backend)")
        except Exception as e:
            logger.error("FastPlateOCR init failed: %s", e)
            raise RuntimeError(f"FastALPR OCR is unavailable: {e}") from e

    def _init_easyocr(self):
        try:
            import easyocr
            self._easyocr = easyocr.Reader(
                ["en"],
                gpu=self._has_gpu(),
                verbose=False
            )
            self.resolved_engine = "easyocr"
            logger.info("EasyOCR initialized")
        except Exception as e:
            logger.warning(f"EasyOCR init failed: {e}, falling back to Tesseract")
            self._init_tesseract()

    def _init_lprnet(self):
        """LPRNet ONNX lightweight inference."""
        try:
            import onnxruntime as ort
            if self.model_config and self.model_config.available:
                self._lprnet_session = ort.InferenceSession(
                    self.model_config.path,
                    providers=(
                        ["CUDAExecutionProvider", "CPUExecutionProvider"]
                        if self._has_gpu() and "CUDAExecutionProvider" in ort.get_available_providers()
                        else ["CPUExecutionProvider"]
                    ),
                )
                logger.info("LPRNet ONNX initialized")
                self.resolved_engine = "lprnet"
            else:
                raise FileNotFoundError("LPRNet ONNX weights are not installed")
        except Exception as e:
            logger.error("LPRNet init failed: %s", e)
            raise RuntimeError(f"LPRNet is unavailable: {e}") from e

    def _init_tesseract(self):
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self._tesseract_available = True
            logger.info("Tesseract initialized as fallback")
        except Exception:
            logger.warning("Tesseract not available")

    def _has_gpu(self) -> bool:
        if self.execution_device is not None and not self.execution_device.startswith("cuda"):
            return False
        if self.engine == "paddleocr":
            try:
                import paddle
                return bool(
                    paddle.is_compiled_with_cuda()
                    and paddle.device.cuda.device_count() > 0
                )
            except Exception:
                return False
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    # ─── OCR ───────────────────────────────────────────────────────

    def run_ocr(self, plate_crop: np.ndarray, quality_score: float = 1.0) -> Optional[OCRResult]:
        """Run OCR over several plate-specific image variants and keep the best."""
        if plate_crop is None or plate_crop.size == 0:
            return None

        candidates = []
        for variant in self._prepare_variants(plate_crop):
            raw, conf = self._run_engine(variant)
            if raw is None:
                continue
            cleaned = self._clean_text(raw)
            if not cleaned:
                continue
            regex_valid, regex_score = self._validate_regex(cleaned)
            candidates.append((regex_valid, regex_score, conf, cleaned, raw))
            if regex_valid and conf >= self.confidence_threshold * 0.8:
                break

        if not candidates:
            return None

        # Prefer a valid regional format, then OCR confidence. This prevents a
        # confident piece of surrounding text from beating a plausible plate.
        regex_valid, regex_score, conf, cleaned, raw = max(
            candidates,
            key=lambda item: (item[0], item[1], item[2]),
        )
        if conf < self.confidence_threshold * 0.5:
            return None

        return OCRResult(
            text=cleaned,
            confidence=conf,
            raw_text=raw,
            regex_valid=regex_valid,
            regex_score=regex_score,
        )

    def _prepare_variants(self, crop: np.ndarray) -> List[np.ndarray]:
        """Create raw, contrast-enhanced and thresholded OCR inputs."""
        if len(crop.shape) == 2:
            bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        else:
            bgr = crop

        original = bgr
        h, w = bgr.shape[:2]
        if w < 240:
            scale = 240 / max(w, 1)
            bgr = cv2.resize(
                bgr,
                (240, max(32, int(h * scale))),
                interpolation=cv2.INTER_CUBIC,
            )

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        denoised = cv2.bilateralFilter(clahe, 7, 35, 35)
        thresholded = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            7,
        )
        variants = [
            bgr,
            cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR),
        ]
        # Very short but readable plates (for example 79x18 px) lose character
        # strokes when enlarged only to 240 px. A larger cubic/Lanczos pass
        # plus unsharp masking recovered text that the normal variants missed.
        original_h, original_w = original.shape[:2]
        if original_w < 180 and original_h < 45 and original_w / max(original_h, 1) >= 2.0:
            target_w = 420
            target_h = max(64, round(original_h * target_w / max(original_w, 1)))
            large_cubic = cv2.resize(original, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            blurred = cv2.GaussianBlur(large_cubic, (0, 0), 2.0)
            sharpened = cv2.addWeighted(large_cubic, 1.8, blurred, -0.8, 0)
            large_lanczos = cv2.resize(original, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            variants = [sharpened, large_lanczos] + variants
        return variants

    def _run_engine(self, crop: np.ndarray) -> Tuple[Optional[str], float]:
        """Run primary engine with fallback chain."""
        if self._lprnet_session is not None:
            return self._run_lprnet(crop)

        if self._fast_plate_ocr is not None:
            return self._run_fast_plate_ocr(crop)

        if self._paddle is not None:
            return self._run_paddle(crop)

        # OCR failure on one crop is different from engine initialization
        # failure. Lazily initialize the next engine so the documented fallback
        # chain also works when the primary returns no text.
        if self._easyocr is not None:
            return self._run_easyocr(crop)

        if not self._tesseract_available:
            self._init_tesseract()
        if self._tesseract_available:
            return self._run_tesseract(crop)

        return None, 0.0

    def _run_fast_plate_ocr(self, crop: np.ndarray) -> Tuple[Optional[str], float]:
        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if crop.ndim == 3 else crop
            prediction = self._fast_plate_ocr.run_one(rgb, return_confidence=True)
            text = getattr(prediction, "plate", "")
            probabilities = getattr(prediction, "char_probs", None) or []
            confidence = float(np.mean(probabilities)) if len(probabilities) else 0.0
            return text, confidence
        except Exception as e:
            logger.debug("FastPlateOCR error: %s", e)
            return None, 0.0

    def _run_paddle(self, crop: np.ndarray) -> Tuple[Optional[str], float]:
        try:
            try:
                result = self._paddle.predict(crop)
            except (AttributeError, TypeError):
                result = self._paddle.ocr(crop, cls=True)
            # PaddleOCR 3.x result objects expose a JSON-like payload.
            if result and hasattr(result[0], "json"):
                payload = result[0].json
                payload = payload() if callable(payload) else payload
                data = payload.get("res", payload) if isinstance(payload, dict) else {}
                texts = data.get("rec_texts", [])
                scores = data.get("rec_scores", [])
                if texts:
                    return "".join(texts).replace(" ", ""), float(np.mean(scores or [0.0]))
            if not result or not result[0]:
                return None, 0.0

            texts = []
            for line in result[0]:
                text, conf = line[1]
                texts.append((text, conf))

            if not texts:
                return None, 0.0

            # Combine all text boxes
            full_text = "".join(t[0] for t in texts).replace(" ", "")
            avg_conf = sum(t[1] for t in texts) / len(texts)
            return full_text, float(avg_conf)
        except Exception as e:
            logger.debug(f"PaddleOCR error: {e}")
            return None, 0.0

    def _run_easyocr(self, crop: np.ndarray) -> Tuple[Optional[str], float]:
        try:
            results = self._easyocr.readtext(
                crop,
                detail=1,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                paragraph=False,
                decoder="greedy",
                width_ths=0.8,
                mag_ratio=1.0,
            )
            if not results:
                return None, 0.0
            text = "".join(r[1] for r in results).replace(" ", "")
            conf = sum(r[2] for r in results) / len(results)
            return text, float(conf)
        except Exception as e:
            logger.debug(f"EasyOCR error: {e}")
            return None, 0.0

    def _run_tesseract(self, crop: np.ndarray) -> Tuple[Optional[str], float]:
        try:
            import pytesseract
            config = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            text = pytesseract.image_to_string(crop, config=config).strip()
            conf_data = pytesseract.image_to_data(crop, output_type=pytesseract.Output.DICT, config=config)
            confs = [c for c in conf_data["conf"] if c != -1]
            avg_conf = sum(confs) / len(confs) / 100 if confs else 0.3
            return text, avg_conf
        except Exception as e:
            logger.debug(f"Tesseract error: {e}")
            return None, 0.0

    def _run_lprnet(self, crop: np.ndarray) -> Tuple[Optional[str], float]:
        try:
            # LPRNet specific preprocessing
            chars_table = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            input_img = cv2.resize(crop, (128, 32))
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            input_img = input_img.astype(np.float32) / 255.0
            input_img = (input_img - 0.5) / 0.5
            input_img = np.transpose(input_img, (2, 0, 1))
            input_img = np.expand_dims(input_img, 0)

            outputs = self._lprnet_session.run(None, {"input": input_img})
            output = outputs[0]
            if output.ndim != 3:
                raise ValueError(f"Unexpected LPRNet output shape: {output.shape}")
            if output.shape[1] == 1:  # T, B, C
                logits = output[:, 0, :]
            elif output.shape[0] == 1:  # B, T, C
                logits = output[0]
            else:
                raise ValueError(f"Unsupported LPRNet output shape: {output.shape}")

            logits = logits - np.max(logits, axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= np.sum(probs, axis=1, keepdims=True)

            # CTC decode
            pred = np.argmax(probs, axis=1)
            blank_idx = len(chars_table)
            chars = []
            selected_confidences = []
            prev = -1
            for timestep, p in enumerate(pred):
                if p != prev and p != blank_idx:
                    chars.append(chars_table[p])
                    selected_confidences.append(float(probs[timestep, p]))
                prev = p

            text = "".join(chars)
            conf = float(np.mean(selected_confidences)) if selected_confidences else 0.0
            return text, conf
        except Exception as e:
            logger.debug(f"LPRNet error: {e}")
            return None, 0.0

    # ─── Text cleanup ──────────────────────────────────────────────

    def _clean_text(self, text: str) -> str:
        """Normalize OCR output."""
        return normalize_plate_for_profile(text, self.regex_profile)

    # ─── Regex Validation ─────────────────────────────────────────

    def _validate_regex(self, text: str) -> Tuple[bool, float]:
        """Check text against regional plate patterns."""
        return validate_plate(text, self.regex_profile)

    # ─── Temporal Voting ──────────────────────────────────────────

    def update_voting(
        self,
        candidates: List[OCRCandidate],
        new_result: OCRResult,
        quality_score: float,
    ) -> List[OCRCandidate]:
        """Add new OCR result to voting pool."""
        found = False
        for c in candidates:
            if self._same_plate_candidate(c.text, new_result.text):
                c.count += 1
                if len(c.text) == len(new_result.text):
                    if not c.character_votes:
                        self._add_character_votes(
                            c.character_votes,
                            c.text,
                            c.confidence * max(0.25, c.image_quality_score),
                        )
                    self._add_character_votes(
                        c.character_votes,
                        new_result.text,
                        new_result.confidence * max(0.25, quality_score),
                    )
                    consensus = "".join(
                        max(c.character_votes[str(index)].items(), key=lambda item: item[1])[0]
                        for index in range(len(c.text))
                    )
                    consensus_valid, consensus_score = self._validate_regex(consensus)
                    c.text = consensus
                    c.regex_valid = c.regex_valid or consensus_valid
                    c.regex_score = max(c.regex_score, consensus_score)
                if (new_result.regex_score, new_result.confidence) > (c.regex_score, c.confidence):
                    c.raw_text = new_result.raw_text
                c.confidence = max(c.confidence, new_result.confidence)
                c.image_quality_score = max(c.image_quality_score, quality_score)
                c.regex_valid = c.regex_valid or new_result.regex_valid
                c.regex_score = max(c.regex_score, new_result.regex_score)
                found = True
                break

        if not found:
            candidates.append(OCRCandidate(
                text=new_result.text,
                raw_text=new_result.raw_text,
                confidence=new_result.confidence,
                regex_valid=new_result.regex_valid,
                regex_score=new_result.regex_score,
                image_quality_score=quality_score,
                count=1,
                character_votes={},
            ))

        # Keep only top candidates (by composite score)
        candidates.sort(key=lambda c: self._composite_score(c), reverse=True)
        return candidates[:self.voting_window]

    @staticmethod
    def _add_character_votes(votes: Dict[str, Dict[str, float]], text: str, weight: float):
        for index, character in enumerate(text):
            position = votes.setdefault(str(index), {})
            position[character] = position.get(character, 0.0) + float(weight)

    def resolve_plate(
        self,
        candidates: List[OCRCandidate],
        total_frames: int = 0,
    ) -> VotingResult:
        """Determine final plate from voting candidates."""
        if not candidates:
            return VotingResult(
                final_plate=None,
                confidence=0.0,
                status="searching",
                candidates=[]
            )

        best = candidates[0]
        composite = self._composite_score(best)

        # Determine status thresholds
        if not best.regex_valid:
            status = "invalid"
        elif composite >= 0.68 and best.count >= 3:
            status = "verified"
        elif composite >= 0.48 and best.count >= 2:
            status = "candidate"
        elif composite >= 0.35:
            status = "low_confidence"
        else:
            status = "searching"

        return VotingResult(
            final_plate=best.text if status in ("verified", "candidate", "low_confidence") else None,
            confidence=composite,
            status=status,
            candidates=candidates,
        )

    def _composite_score(self, c: OCRCandidate) -> float:
        """Weighted composite score for plate candidate."""
        freq_score = min(c.count / 5.0, 1.0)
        return (
            c.confidence * 0.40
            + freq_score * 0.25
            + c.regex_score * 0.20
            + c.image_quality_score * 0.15
        )

    @staticmethod
    def _same_plate_candidate(left: str, right: str) -> bool:
        """Cluster one-character OCR jitter instead of requiring exact strings."""
        if left == right:
            return True
        if not (5 <= len(left) <= 10 and 5 <= len(right) <= 10):
            return False
        return OCRService._edit_distance(left, right) <= (2 if abs(len(left) - len(right)) <= 1 else 1)

    @staticmethod
    def _edit_distance(left: str, right: str) -> int:
        previous = list(range(len(right) + 1))
        for i, left_char in enumerate(left, 1):
            current = [i]
            for j, right_char in enumerate(right, 1):
                current.append(min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_char != right_char),
                ))
            previous = current
        return previous[-1]
