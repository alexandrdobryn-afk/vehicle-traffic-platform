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


class TrackingService:
    """
    Multi-tracker service.
    ByteTrack (speed), OC-SORT (balanced), BoT-SORT (quality).
    Falls back gracefully when libraries not available.
    """

    def __init__(self, tracker_mode: str = "bytetrack", ai_mode: str = "balanced"):
        self.tracker_mode = tracker_mode
        self.resolved_mode = tracker_mode
        self.ai_mode = ai_mode
        self._tracker = None
        self._simple_tracker = None
        self._identity_map: Dict[Tuple[int, str], int] = {}
        self._identity_state: Dict[int, dict] = {}
        self._next_identity_id = 1
        self._identity_frame = 0
        self._init_tracker()

    @staticmethod
    def _class_group(class_name: str) -> str:
        """Prevent a two-wheeler track from absorbing a passing car and vice versa."""
        return "two_wheel" if class_name == "motorcycle" else "four_wheel"

    def _stabilize_track_ids(self, tracks: List[TrackedObject]) -> List[TrackedObject]:
        self._identity_frame += 1
        assigned = set()
        for track in tracks:
            raw_id = track.track_id
            group = self._class_group(track.class_name)
            key = (raw_id, group)
            if key not in self._identity_map:
                # ByteTrack can issue a fresh raw ID when a close vehicle is
                # briefly occluded or changes detector class (car -> truck).
                # Stitch only an immediately preceding, overlapping track.
                candidates = []
                for stable_id, state in self._identity_state.items():
                    gap = self._identity_frame - state["frame"]
                    if stable_id in assigned or state["group"] != group or not (1 <= gap <= 5):
                        continue
                    overlap = self._bbox_iou(track.bbox, state["bbox"])
                    if overlap >= 0.20:
                        candidates.append((overlap, stable_id))
                if candidates:
                    self._identity_map[key] = max(candidates)[1]
                else:
                    self._identity_map[key] = self._next_identity_id
                    self._next_identity_id += 1
            stable_id = self._identity_map[key]
            track.track_id = stable_id
            assigned.add(stable_id)
            self._identity_state[stable_id] = {
                "bbox": list(track.bbox), "group": group, "frame": self._identity_frame,
            }
        self._identity_state = {
            stable_id: state for stable_id, state in self._identity_state.items()
            if self._identity_frame - state["frame"] <= 60
        }
        return tracks

    @staticmethod
    def _bbox_iou(left: List[int], right: List[int]) -> float:
        x1, y1 = max(left[0], right[0]), max(left[1], right[1])
        x2, y2 = min(left[2], right[2]), min(left[3], right[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
        right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
        union = left_area + right_area - intersection
        return intersection / union if union > 0 else 0.0

    def _init_tracker(self):
        if self.tracker_mode == "bytetrack":
            self._init_bytetrack()
        elif self.tracker_mode == "tracktrack":
            self._init_tracktrack()
        elif self.tracker_mode == "ocsort":
            self._init_ocsort()
        elif self.tracker_mode in ("botsort", "strongsort"):
            self._init_botsort()
        else:
            self._init_bytetrack()

    def _init_bytetrack(self):
        try:
            from ultralytics.trackers import BYTETracker
            from types import SimpleNamespace
            args = SimpleNamespace(
                track_high_thresh=0.5,
                track_low_thresh=0.1,
                new_track_thresh=0.6,
                track_buffer=30,
                match_thresh=0.8,
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

    def _init_tracktrack(self):
        """Initialize the TrackTrack implementation bundled with Ultralytics."""
        try:
            from pathlib import Path
            from types import SimpleNamespace
            import yaml
            import ultralytics
            from ultralytics.trackers.track_tracker import TRACKTRACK

            config_path = Path(ultralytics.__file__).parent / "cfg" / "trackers" / "tracktrack.yaml"
            with config_path.open("r", encoding="utf-8") as config_file:
                args = SimpleNamespace(**yaml.safe_load(config_file))
            self._tracker = TRACKTRACK(args)
            self.resolved_mode = "tracktrack"
            logger.info("TrackTrack initialized from %s", config_path)
        except Exception as e:
            logger.error("TrackTrack init failed: %s", e)
            raise RuntimeError(f"TrackTrack is unavailable: {e}") from e

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

    def _init_botsort(self):
        try:
            from ultralytics.trackers import BOTSORT
            from types import SimpleNamespace
            args = SimpleNamespace(
                track_high_thresh=0.5,
                track_low_thresh=0.1,
                new_track_thresh=0.6,
                track_buffer=30,
                match_thresh=0.8,
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
            logger.info("BoT-SORT initialized (without ReID)")
            self.resolved_mode = "botsort"
        except Exception as e:
            logger.warning(f"BoT-SORT init failed: {e}, falling back to ByteTrack")
            self._init_bytetrack()

    def _init_simple(self):
        """Pure IoU-based simple tracker as last resort fallback."""
        self._simple_tracker = SimpleIoUTracker()
        self.resolved_mode = "simple_iou"
        logger.info("SimpleIoUTracker initialized as fallback")

    def update(
        self,
        detections: List,  # List[Detection] from VehicleDetectionService
        frame: np.ndarray,
    ) -> List[TrackedObject]:
        """
        Update tracker with new detections.
        Returns list of tracked objects with stable IDs.
        """
        if self._simple_tracker:
            return self._stabilize_track_ids(self._simple_tracker.update(detections))

        try:
            return self._stabilize_track_ids(self._update_ultralytics_tracker(detections, frame))
        except Exception as e:
            logger.error(f"Tracker update error: {e}")
            if self._simple_tracker is None:
                self._init_simple()
            return self._stabilize_track_ids(self._simple_tracker.update(detections))

    def _update_ultralytics_tracker(
        self,
        detections: List,
        frame: np.ndarray,
    ) -> List[TrackedObject]:
        """Update Ultralytics-based tracker (ByteTrack/BoT-SORT)."""
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
