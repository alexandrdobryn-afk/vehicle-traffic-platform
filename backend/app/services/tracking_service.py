import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import logging
import time

logger = logging.getLogger(__name__)


class TrackerDetections:
    """Minimal Ultralytics Results-compatible detection container."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls

    @property
    def xywh(self) -> np.ndarray:
        boxes = self.xyxy.copy()
        boxes[:, 2:4] -= boxes[:, 0:2]
        boxes[:, 0:2] += boxes[:, 2:4] / 2
        return boxes

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, index):
        return TrackerDetections(self.xyxy[index], self.conf[index], self.cls[index])


@dataclass
class TrackedObject:
    track_id: int
    bbox: List[int]       # [x1, y1, x2, y2]
    confidence: float
    class_name: str
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    predicted: bool = False
    motion_vector: List[float] = field(default_factory=list)
    appearance_signature: List[float] = field(default_factory=list)


class TrackingService:
    """
    Multi-tracker service.
    ByteTrack, BoT-SORT, OC-SORT, DeepSORT and StrongSORT-compatible adapters.
    Falls back gracefully when libraries not available.
    """

    def __init__(
        self,
        tracker_mode: str = "bytetrack",
        ai_mode: str = "balanced",
        identity_stitching: bool = True,
    ):
        self.tracker_mode = tracker_mode
        self.resolved_mode = tracker_mode
        self.ai_mode = ai_mode
        self.identity_stitching = identity_stitching
        self._tracker = None
        self._simple_tracker = None
        self._identity_map: Dict[Tuple[int, str], int] = {}
        self._identity_state: Dict[int, dict] = {}
        self._next_identity_id = 1
        self._identity_frame = 0
        self._init_tracker()

    def _apply_identity_layer(
        self,
        tracks: List[TrackedObject],
        frame: Optional[np.ndarray] = None,
    ) -> List[TrackedObject]:
        if not self.identity_stitching:
            return tracks
        return self._stabilize_track_ids(tracks, frame)

    @staticmethod
    def _class_group(class_name: str) -> str:
        """Keep raw tracker IDs from merging visibly different object classes."""
        normalized = (class_name or "unknown").strip().lower() or "unknown"
        vehicle_like = {
            "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        }
        return "vehicle_like" if normalized in vehicle_like else normalized

    def _stabilize_track_ids(
        self,
        tracks: List[TrackedObject],
        frame: Optional[np.ndarray] = None,
    ) -> List[TrackedObject]:
        self._identity_frame += 1
        assigned = set()
        for track in tracks:
            raw_id = track.track_id
            group = self._class_group(track.class_name)
            signature = self._appearance_signature(frame, track.bbox)
            track.appearance_signature = signature
            key = (raw_id, group)
            if key not in self._identity_map:
                # ByteTrack can issue a fresh raw ID after short occlusions.
                # Stitch only plausible preceding tracks with the same coarse
                # object group. This is intentionally motion/appearance based:
                # on 4K video processed at low FPS, a nearby object can grow so
                # much between frames that plain IoU becomes close to zero.
                candidates = []
                for stable_id, state in self._identity_state.items():
                    gap = self._identity_frame - state["frame"]
                    if stable_id in assigned or state["group"] != group or not (1 <= gap <= 30):
                        continue
                    score = self._identity_candidate_score(track, state, gap, signature)
                    if score > 0:
                        candidates.append((score, stable_id))
                if candidates:
                    self._identity_map[key] = max(candidates)[1]
                else:
                    self._identity_map[key] = self._next_identity_id
                    self._next_identity_id += 1
            stable_id = self._identity_map[key]
            track.track_id = stable_id
            assigned.add(stable_id)
            previous = self._identity_state.get(stable_id)
            velocity = (
                self._center_delta(previous["bbox"], track.bbox)
                if previous is not None else [0.0, 0.0]
            )
            self._identity_state[stable_id] = {
                "bbox": list(track.bbox),
                "group": group,
                "frame": self._identity_frame,
                "velocity": velocity,
                "signature": signature or (previous or {}).get("signature", []),
            }
        self._identity_state = {
            stable_id: state for stable_id, state in self._identity_state.items()
            if self._identity_frame - state["frame"] <= 90
        }
        return self._dedupe_stable_tracks(tracks)

    def _identity_candidate_score(
        self,
        track: TrackedObject,
        state: dict,
        gap: int,
        signature: List[float],
    ) -> float:
        overlap = self._bbox_iou(track.bbox, state["bbox"])
        contained = self._contained_overlap(track.bbox, state["bbox"])
        center_score = self._center_continuity_score(track.bbox, state["bbox"])
        motion_score = self._motion_continuity_score(track.bbox, state, gap)
        appearance = self._appearance_similarity(signature, state.get("signature", []))

        if overlap >= 0.04 or contained >= 0.55:
            return max(overlap, contained) + 0.15
        if center_score >= 0.25 and (appearance >= 0.68 or gap <= 2):
            return center_score * 0.65 + appearance * 0.35
        if motion_score >= 0.32 and appearance >= 0.70:
            return motion_score * 0.70 + appearance * 0.30
        if state.get("group") == "vehicle_like" and gap <= 4 and motion_score >= 0.52:
            return motion_score * 0.85
        return 0.0

    @classmethod
    def _dedupe_stable_tracks(cls, tracks: List[TrackedObject]) -> List[TrackedObject]:
        """ByteTrack/BoT-SORT can briefly return two raw tracks mapped to one stable ID."""
        by_id: Dict[int, TrackedObject] = {}
        for track in tracks:
            current = by_id.get(track.track_id)
            if current is None or cls._track_rank(track) > cls._track_rank(current):
                by_id[track.track_id] = track
        return list(by_id.values())

    @staticmethod
    def _track_rank(track: TrackedObject) -> tuple:
        x1, y1, x2, y2 = track.bbox
        area = max(0, x2 - x1) * max(0, y2 - y1)
        return (float(track.confidence), area)

    @staticmethod
    def _bbox_iou(left: List[int], right: List[int]) -> float:
        x1, y1 = max(left[0], right[0]), max(left[1], right[1])
        x2, y2 = min(left[2], right[2]), min(left[3], right[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
        right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
        union = left_area + right_area - intersection
        return intersection / union if union > 0 else 0.0

    @classmethod
    def _contained_overlap(cls, left: List[int], right: List[int]) -> float:
        x1, y1 = max(left[0], right[0]), max(left[1], right[1])
        x2, y2 = min(left[2], right[2]), min(left[3], right[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
        right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
        return intersection / max(1.0, min(left_area, right_area))

    @staticmethod
    def _center_continuity_score(left: List[int], right: List[int]) -> float:
        left_w, left_h = max(1, left[2] - left[0]), max(1, left[3] - left[1])
        right_w, right_h = max(1, right[2] - right[0]), max(1, right[3] - right[1])
        left_center = ((left[0] + left[2]) / 2, (left[1] + left[3]) / 2)
        right_center = ((right[0] + right[2]) / 2, (right[1] + right[3]) / 2)
        distance = float(np.hypot(left_center[0] - right_center[0], left_center[1] - right_center[1]))
        size_gate = max(120.0, 1.15 * max(left_w, left_h, right_w, right_h))
        if distance > size_gate:
            return 0.0
        area_ratio = (left_w * left_h) / max(1, right_w * right_h)
        if area_ratio < 0.05 or area_ratio > 20.0:
            return 0.0
        return 1.0 - (distance / size_gate)

    @staticmethod
    def _center_delta(previous: List[int], current: List[int]) -> List[float]:
        previous_center = ((previous[0] + previous[2]) / 2, (previous[1] + previous[3]) / 2)
        current_center = ((current[0] + current[2]) / 2, (current[1] + current[3]) / 2)
        return [
            float(current_center[0] - previous_center[0]),
            float(current_center[1] - previous_center[1]),
        ]

    @staticmethod
    def _center(box: List[int]) -> Tuple[float, float]:
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    def _motion_continuity_score(self, bbox: List[int], state: dict, gap: int) -> float:
        prev_box = state["bbox"]
        prev_center = self._center(prev_box)
        velocity = state.get("velocity") or [0.0, 0.0]
        predicted_center = (
            prev_center[0] + float(velocity[0]) * max(1, gap),
            prev_center[1] + float(velocity[1]) * max(1, gap),
        )
        current_center = self._center(bbox)
        distance = float(np.hypot(
            current_center[0] - predicted_center[0],
            current_center[1] - predicted_center[1],
        ))
        left_w, left_h = max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])
        right_w, right_h = max(1, prev_box[2] - prev_box[0]), max(1, prev_box[3] - prev_box[1])
        gate = max(220.0, 1.55 * max(left_w, left_h, right_w, right_h) + 120.0 * max(1, gap))
        if distance > gate:
            return 0.0
        area_ratio = (left_w * left_h) / max(1, right_w * right_h)
        if area_ratio < 0.04 or area_ratio > 24.0:
            return 0.0
        return 1.0 - (distance / gate)

    @staticmethod
    def _appearance_signature(frame: Optional[np.ndarray], bbox: List[int]) -> List[float]:
        if frame is None or frame.size == 0:
            return []
        frame_h, frame_w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, min(frame_w, x1)), max(0, min(frame_h, y1))
        x2, y2 = max(0, min(frame_w, x2)), max(0, min(frame_h, y2))
        if x2 <= x1 or y2 <= y1:
            return []
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return []
        histograms = [
            np.histogram(crop[:, :, channel], bins=12, range=(0, 256))[0].astype(np.float32)
            for channel in range(min(3, crop.shape[2]))
        ]
        signature = np.concatenate(histograms)
        norm = float(np.linalg.norm(signature))
        if norm <= 0:
            return []
        return (signature / norm).round(5).tolist()

    @staticmethod
    def _appearance_similarity(left: List[float], right: List[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        left_vector = np.asarray(left, dtype=np.float32)
        right_vector = np.asarray(right, dtype=np.float32)
        denom = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
        return float(np.dot(left_vector, right_vector) / denom) if denom > 0 else 0.0

    def _init_tracker(self):
        if self.tracker_mode == "bytetrack":
            self._init_bytetrack()
        elif self.tracker_mode == "ocsort":
            self._init_ocsort()
        elif self.tracker_mode == "deepsort":
            self._init_deepsort()
        elif self.tracker_mode == "botsort":
            self._init_botsort("botsort")
        elif self.tracker_mode == "strongsort":
            logger.warning("StrongSORT ReID runtime is not installed; falling back to BoT-SORT")
            self._init_botsort("botsort")
        else:
            self._init_bytetrack()

    def _init_bytetrack(self):
        try:
            from ultralytics.trackers import BYTETracker
            from types import SimpleNamespace
            args = SimpleNamespace(
                track_high_thresh=0.35,
                track_low_thresh=0.05,
                new_track_thresh=0.45,
                track_buffer=90,
                match_thresh=0.55,
                fuse_score=True,
            )
            try:
                self._tracker = BYTETracker(args)
            except TypeError:
                self._tracker = BYTETracker(args, frame_rate=25)
            logger.info("ByteTrack initialized")
            self.resolved_mode = "bytetrack"
        except Exception as e:
            logger.warning(f"ByteTrack init failed: {e}, using SimpleTracker")
            self._init_simple()

    def _init_ocsort(self):
        try:
            from ocsort import OCSort
            self._tracker = OCSort(
                det_thresh=0.45,
                iou_threshold=0.3,
                asso_func="iou",
                delta_t=3,
                inertia=0.2,
            )
            logger.info("OC-SORT initialized")
            self.resolved_mode = "ocsort"
        except Exception as e:
            logger.warning(f"OC-SORT init failed: {e}, falling back to ByteTrack")
            self._init_bytetrack()

    def _init_deepsort(self):
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self._tracker = DeepSort(
                max_age=30,
                n_init=3,
                max_iou_distance=0.7,
                embedder="mobilenet",
                half=True,
                bgr=True,
                embedder_gpu=True,
            )
            logger.info("DeepSORT initialized")
            self.resolved_mode = "deepsort"
        except Exception as e:
            logger.warning(f"DeepSORT init failed: {e}, falling back to ByteTrack")
            self._init_bytetrack()

    def _init_botsort(self, resolved_mode: str = "botsort"):
        try:
            from ultralytics.trackers import BOTSORT
            from types import SimpleNamespace
            args = SimpleNamespace(
                track_high_thresh=0.35,
                track_low_thresh=0.05,
                new_track_thresh=0.45,
                track_buffer=90,
                match_thresh=0.55,
                proximity_thresh=0.5,
                appearance_thresh=0.25,
                with_reid=False,  # ReID disabled unless model provided
                gmc_method="sparseOptFlow",
                model="auto",
                fuse_score=True,
            )
            try:
                self._tracker = BOTSORT(args)
            except TypeError:
                self._tracker = BOTSORT(args, frame_rate=25)
            logger.info("%s initialized (without external ReID)", resolved_mode)
            self.resolved_mode = resolved_mode
        except Exception as e:
            logger.warning(f"{resolved_mode} init failed: {e}, falling back to ByteTrack")
            self._init_bytetrack()

    def _init_simple(self):
        """Pure IoU-based simple tracker as last resort fallback."""
        self._simple_tracker = SimpleIoUTracker()
        self.resolved_mode = "simple_iou"
        logger.info("SimpleIoUTracker initialized as fallback")

    def update(
        self,
        detections: List,  # List[Detection] from ObjectDetectionService
        frame: np.ndarray,
    ) -> List[TrackedObject]:
        """
        Update tracker with new detections.
        Returns list of tracked objects with stable IDs.
        """
        if self._simple_tracker:
            return self._apply_identity_layer(self._simple_tracker.update(detections), frame)

        try:
            if self.resolved_mode == "deepsort":
                return self._apply_identity_layer(self._update_deepsort_tracker(detections, frame), frame)
            if self.resolved_mode == "ocsort":
                return self._apply_identity_layer(self._update_ocsort_tracker(detections, frame), frame)
            return self._apply_identity_layer(self._update_ultralytics_tracker(detections, frame), frame)
        except Exception as e:
            logger.error(f"Tracker update error: {e}")
            if self._simple_tracker is None:
                self._init_simple()
            return self._apply_identity_layer(self._simple_tracker.update(detections), frame)

    def _update_ultralytics_tracker(
        self,
        detections: List,
        frame: np.ndarray,
    ) -> List[TrackedObject]:
        """Update Ultralytics-based tracker (ByteTrack/BoT-SORT/StrongSORT-compatible)."""
        xyxy = np.asarray([d.bbox for d in detections], dtype=np.float32).reshape(-1, 4)
        conf = np.asarray([d.confidence for d in detections], dtype=np.float32)
        classes = np.asarray([d.class_id for d in detections], dtype=np.float32)
        tracks = self._tracker.update(TrackerDetections(xyxy, conf, classes), frame)

        result = []
        for track in tracks:
            if isinstance(track, np.ndarray) or isinstance(track, (list, tuple)):
                # Current Ultralytics format:
                # x1, y1, x2, y2, track_id, score, class_id, detection_index
                if len(track) < 7:
                    continue
                x1, y1, x2, y2 = map(int, track[:4])
                track_id = int(track[4])
                conf = float(track[5])
                cls_id = int(track[6])
                det_index = int(track[7]) if len(track) > 7 else -1
                cls_name = (
                    detections[det_index].class_name
                    if 0 <= det_index < len(detections)
                    else next((d.class_name for d in detections if d.class_id == cls_id), "unknown")
                )
                result.append(TrackedObject(
                    track_id=track_id,
                    bbox=[x1, y1, x2, y2],
                    confidence=conf,
                    class_name=cls_name,
                ))
                continue
            if hasattr(track, "tlbr"):
                x1, y1, x2, y2 = map(int, track.tlbr)
            elif hasattr(track, "xyxy"):
                x1, y1, x2, y2 = map(int, track.xyxy)
            else:
                continue

            track_id = int(track.track_id)
            conf = float(getattr(track, "score", 0.5))

            # Map class id back to name
            cls_id = int(getattr(track, "cls", 2))
            cls_name = next(
                (d.class_name for d in detections if d.class_id == cls_id),
                "unknown",
            )

            result.append(TrackedObject(
                track_id=track_id,
                bbox=[x1, y1, x2, y2],
                confidence=conf,
                class_name=cls_name,
            ))

        return result

    def _update_ocsort_tracker(
        self,
        detections: List,
        frame: np.ndarray,
    ) -> List[TrackedObject]:
        """Update OC-SORT, whose public package uses raw xyxy+score arrays."""
        det_array = np.asarray(
            [[*det.bbox, det.confidence] for det in detections],
            dtype=np.float32,
        ).reshape(-1, 5)
        frame_shape = frame.shape[:2] if frame is not None and frame.size else (0, 0)
        try:
            tracks = self._tracker.update(det_array, frame_shape, frame_shape)
        except TypeError:
            tracks = self._tracker.update(det_array)

        result = []
        for track in tracks:
            if not isinstance(track, (np.ndarray, list, tuple)) or len(track) < 5:
                continue
            x1, y1, x2, y2 = map(int, track[:4])
            track_id = int(track[4])
            cls_name = "unknown"
            cls_id = -1
            confidence = 1.0
            best_iou = 0.0
            for det in detections:
                iou = self._bbox_iou([x1, y1, x2, y2], det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    cls_name = det.class_name
                    cls_id = det.class_id
                    confidence = det.confidence
            result.append(TrackedObject(
                track_id=track_id,
                bbox=[x1, y1, x2, y2],
                confidence=float(confidence),
                class_name=cls_name,
            ))
        return result

    def _update_deepsort_tracker(
        self,
        detections: List,
        frame: np.ndarray,
    ) -> List[TrackedObject]:
        raw_detections = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            raw_detections.append(([x1, y1, max(0, x2 - x1), max(0, y2 - y1)], det.confidence, det.class_name))
        tracks = self._tracker.update_tracks(raw_detections, frame=frame)
        result = []
        for track in tracks:
            if hasattr(track, "is_confirmed") and not track.is_confirmed():
                continue
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            result.append(TrackedObject(
                track_id=int(track.track_id),
                bbox=[x1, y1, x2, y2],
                confidence=1.0,
                class_name=str(getattr(track, "det_class", None) or "unknown"),
            ))
        return result

    def reset(self):
        """Reset tracker state (e.g., on camera reconnect)."""
        self._identity_map.clear()
        self._identity_state.clear()
        self._next_identity_id = 1
        self._identity_frame = 0
        self._init_tracker()


class SimpleIoUTracker:
    """
    Simple IoU-based tracker.
    Assigns IDs based on overlap with previous detections.
    No external dependencies.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: Dict[int, dict] = {}
        self._next_id = 1

    def update(self, detections: List) -> List[TrackedObject]:
        if not detections:
            # Age all tracks
            for tid in list(self.tracks.keys()):
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]
            return []

        det_boxes = [d.bbox for d in detections]
        matched = {}

        # Match new detections to existing tracks
        for det_idx, det_box in enumerate(det_boxes):
            best_iou = self.iou_threshold
            best_tid = None

            for tid, track in self.tracks.items():
                if tid in matched.values():
                    continue
                iou = self._iou(det_box, track["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None:
                matched[det_idx] = best_tid
            else:
                # New track
                matched[det_idx] = self._next_id
                self.tracks[self._next_id] = {"bbox": det_box, "age": 0, "hits": 0}
                self._next_id += 1

        # Update tracks
        new_tracks = {}
        result = []
        for det_idx, tid in matched.items():
            det = detections[det_idx]
            new_tracks[tid] = {
                "bbox": det.bbox,
                "age": 0,
                "hits": self.tracks.get(tid, {}).get("hits", 0) + 1,
            }
            result.append(TrackedObject(
                track_id=tid,
                bbox=det.bbox,
                confidence=det.confidence,
                class_name=det.class_name,
                hits=new_tracks[tid]["hits"],
            ))

        # Keep unmatched old tracks (age them)
        for tid, track in self.tracks.items():
            if tid not in new_tracks:
                track["age"] += 1
                if track["age"] <= self.max_age:
                    new_tracks[tid] = track

        self.tracks = new_tracks
        return result

    def _iou(self, box1: List[int], box2: List[int]) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0
