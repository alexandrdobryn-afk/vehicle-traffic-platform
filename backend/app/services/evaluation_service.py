"""Deterministic, dependency-light evaluation used by the API and CI.

This is the platform baseline evaluator.  It intentionally operates on a
frozen list of labeled frames; external COCO/TrackEval adapters can feed the
same persisted EvaluationRun records without changing the public API.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


def bbox_iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


class EvaluationService:
    def evaluate_detection_suite(self, frames: list[dict[str, Any]]) -> dict[str, Any]:
        thresholds = [round(0.5 + index * 0.05, 2) for index in range(10)]
        ap_by_threshold = {str(threshold): self._mean_average_precision(frames, threshold) for threshold in thresholds}
        result = self.evaluate_detection(frames, 0.5)
        result.update({
            "ap50": ap_by_threshold["0.5"],
            "map50_95": round(sum(ap_by_threshold.values()) / len(ap_by_threshold), 6),
            "ap_by_iou": ap_by_threshold,
        })
        return result

    def evaluate_detection(self, frames: list[dict[str, Any]], iou_threshold: float = 0.5) -> dict[str, Any]:
        totals = defaultdict(int)
        per_class = defaultdict(lambda: defaultdict(int))
        confidences: list[float] = []

        for frame in frames:
            ground_truth = list(frame.get("ground_truth") or [])
            predictions = sorted(
                frame.get("predictions") or [],
                key=lambda item: float(item.get("confidence", 0.0)),
                reverse=True,
            )
            matched: set[int] = set()
            for prediction in predictions:
                label = str(prediction.get("class_name", "unknown"))
                best_index, best_iou = None, 0.0
                for index, truth in enumerate(ground_truth):
                    if index in matched or str(truth.get("class_name", "unknown")) != label:
                        continue
                    overlap = bbox_iou(prediction["bbox"], truth["bbox"])
                    if overlap > best_iou:
                        best_index, best_iou = index, overlap
                confidences.append(float(prediction.get("confidence", 0.0)))
                if best_index is not None and best_iou >= iou_threshold:
                    matched.add(best_index)
                    totals["tp"] += 1
                    per_class[label]["tp"] += 1
                else:
                    totals["fp"] += 1
                    per_class[label]["fp"] += 1
            for index, truth in enumerate(ground_truth):
                if index not in matched:
                    label = str(truth.get("class_name", "unknown"))
                    totals["fn"] += 1
                    per_class[label]["fn"] += 1

        summary = self._scores(totals)
        summary.update({
            "iou_threshold": round(iou_threshold, 3),
            "frames": len(frames),
            "ground_truth_count": totals["tp"] + totals["fn"],
            "prediction_count": totals["tp"] + totals["fp"],
            "mean_prediction_confidence": round(sum(confidences) / len(confidences), 6) if confidences else 0.0,
            "per_class": {label: self._scores(values) for label, values in sorted(per_class.items())},
        })
        return summary

    @staticmethod
    def evaluate_tracking(summary: dict[str, int]) -> dict[str, float | int]:
        ground_truth = max(0, int(summary.get("ground_truth_detections", 0)))
        false_positives = max(0, int(summary.get("false_positives", 0)))
        false_negatives = max(0, int(summary.get("false_negatives", 0)))
        id_switches = max(0, int(summary.get("id_switches", 0)))
        idtp = max(0, int(summary.get("id_true_positives", 0)))
        idfp = max(0, int(summary.get("id_false_positives", false_positives)))
        idfn = max(0, int(summary.get("id_false_negatives", false_negatives)))
        mota = 1.0 - ((false_positives + false_negatives + id_switches) / ground_truth) if ground_truth else 0.0
        idf1_denominator = 2 * idtp + idfp + idfn
        return {
            "mota": round(mota, 6),
            "idf1": round((2 * idtp) / idf1_denominator, 6) if idf1_denominator else 0.0,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "id_switches": id_switches,
            "ground_truth_detections": ground_truth,
        }

    @staticmethod
    def evaluate_classification(items: list[dict[str, Any]]) -> dict[str, Any]:
        labels = sorted({str(item.get("ground_truth", "unknown")) for item in items})
        correct = sum(str(item.get("ground_truth")) == str(item.get("prediction")) for item in items)
        per_class = {}
        for label in labels:
            tp = sum(item.get("ground_truth") == label and item.get("prediction") == label for item in items)
            fp = sum(item.get("ground_truth") != label and item.get("prediction") == label for item in items)
            fn = sum(item.get("ground_truth") == label and item.get("prediction") != label for item in items)
            per_class[label] = EvaluationService._scores({"tp": tp, "fp": fp, "fn": fn})
        return {
            "samples": len(items),
            "accuracy": round(correct / len(items), 6) if items else 0.0,
            "macro_f1": round(sum(value["f1"] for value in per_class.values()) / len(per_class), 6) if per_class else 0.0,
            "per_class": per_class,
        }

    @staticmethod
    def evaluate_segmentation(items: list[dict[str, Any]]) -> dict[str, Any]:
        values = [max(0.0, min(1.0, float(item.get("iou", 0.0)))) for item in items]
        return {
            "samples": len(values),
            "mean_iou": round(sum(values) / len(values), 6) if values else 0.0,
            "iou50_rate": round(sum(value >= 0.5 for value in values) / len(values), 6) if values else 0.0,
            "iou75_rate": round(sum(value >= 0.75 for value in values) / len(values), 6) if values else 0.0,
        }

    def _mean_average_precision(self, frames: list[dict[str, Any]], iou_threshold: float) -> float:
        classes = sorted({
            str(item.get("class_name", "unknown"))
            for frame in frames for item in (frame.get("ground_truth") or [])
        })
        class_aps = []
        for label in classes:
            ground_truth_count = sum(
                str(item.get("class_name", "unknown")) == label
                for frame in frames for item in (frame.get("ground_truth") or [])
            )
            predictions = []
            for frame_index, frame in enumerate(frames):
                for prediction in frame.get("predictions") or []:
                    if str(prediction.get("class_name", "unknown")) == label:
                        predictions.append((float(prediction.get("confidence", 0.0)), frame_index, prediction))
            predictions.sort(reverse=True, key=lambda item: item[0])
            matched: dict[int, set[int]] = defaultdict(set)
            true_positive, false_positive = [], []
            for _, frame_index, prediction in predictions:
                truths = frames[frame_index].get("ground_truth") or []
                best_index, best_iou = None, 0.0
                for index, truth in enumerate(truths):
                    if index in matched[frame_index] or str(truth.get("class_name", "unknown")) != label:
                        continue
                    overlap = bbox_iou(prediction["bbox"], truth["bbox"])
                    if overlap > best_iou:
                        best_index, best_iou = index, overlap
                hit = best_index is not None and best_iou >= iou_threshold
                if hit:
                    matched[frame_index].add(best_index)
                true_positive.append(1 if hit else 0)
                false_positive.append(0 if hit else 1)
            if not ground_truth_count:
                continue
            cumulative_tp = cumulative_fp = 0
            precision, recall = [], []
            for tp, fp in zip(true_positive, false_positive):
                cumulative_tp += tp
                cumulative_fp += fp
                precision.append(cumulative_tp / max(1, cumulative_tp + cumulative_fp))
                recall.append(cumulative_tp / ground_truth_count)
            interpolated = []
            for point in range(101):
                target = point / 100
                interpolated.append(max((p for p, r in zip(precision, recall) if r >= target), default=0.0))
            class_aps.append(sum(interpolated) / 101)
        return round(sum(class_aps) / len(class_aps), 6) if class_aps else 0.0

    @staticmethod
    def _scores(counts: dict[str, int]) -> dict[str, float | int]:
        tp, fp, fn = int(counts["tp"]), int(counts["fp"]), int(counts["fn"])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6),
            "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        }


evaluation_service = EvaluationService()
