"""
Validation Worker — runs after training to compute final metrics
and benchmark against the current production model.
"""
import os
import logging
from pathlib import Path
from app.celery_app import celery_app
from app.config import settings
from app.services.lifecycle_service import lifecycle_service
from app.utils.geometry import abs_box_to_norm, bbox_iou, norm_box_to_abs

logger = logging.getLogger(__name__)


@celery_app.task(name="app.workers.validation_worker.run_validation", bind=True)
def run_validation(self, model_version_id: int):
    """Full validation + benchmark pipeline."""
    db = _get_db()
    try:
        from app.models.database import ModelVersion, DeployStatus
        mv = db.query(ModelVersion).filter(ModelVersion.id == model_version_id).first()
        if not mv:
            return {"success": False}

        model_type = mv.model_type
        weights = mv.weights_path

        if not weights or not os.path.exists(weights):
            mv.validation_passed = False
            mv.auto_test_results = {"error": "Weights not found"}
            db.commit()
            return {"success": False, "error": "Weights not found"}

        # Run validation
        if model_type in ("object_detector", "object_segmenter"):
            val_results = _validate_yolo(mv, db)
        elif model_type == "object_classifier":
            val_results = _validate_classifier(mv, db)
        else:
            val_results = {"error": f"Unsupported model type: {model_type}"}

        # Auto tests
        auto_tests = _run_auto_tests(mv)
        smoke_ms = (auto_tests.get("tests") or {}).get("synthetic_smoke_inference_ms")
        if smoke_ms is not None and "error" not in val_results:
            val_results["latency_p50_ms"] = smoke_ms
            val_results["latency_p95_ms"] = smoke_ms

        # Benchmark vs production
        benchmark = _benchmark_vs_production(mv, db, val_results)

        mv.auto_test_results = auto_tests
        mv.benchmark_vs_prev = benchmark
        validation_ok = "error" not in val_results and val_results.get("samples", 1) > 0
        if validation_ok:
            mv.metrics = val_results
            lifecycle_service.build_report_for_model(db, mv, val_results, benchmark)
        else:
            mv.validation_passed = False
            mv.gate_result = "rejected"
            mv.gate_reasons = [val_results.get("error", "validation produced no usable samples")]
            mv.deploy_status = "rejected"
        if not auto_tests.get("passed", False):
            mv.validation_passed = False
            mv.gate_result = "rejected"
            mv.gate_reasons = list(mv.gate_reasons or []) + ["required auto tests failed"]
            mv.deploy_status = "rejected"
        db.commit()

        logger.info(f"Validation complete for model {mv.id}: passed={mv.validation_passed}")
        return {
            "success": True,
            "validation_passed": mv.validation_passed,
            "auto_tests": auto_tests,
            "benchmark": benchmark,
        }
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}
    finally:
        db.close()


def _validate_yolo(mv, db) -> dict:
    try:
        from ultralytics import YOLO
        from app.models.database import Dataset, TrainingJob

        job = db.query(TrainingJob).filter(TrainingJob.id == mv.job_id).first()
        if not job:
            return {"error": "Training job not found"}

        dataset = db.query(Dataset).filter(Dataset.id == job.dataset_id).first()
        if not dataset:
            return {"error": "Dataset not found"}

        export_dir = Path(settings.EXPORTS_PATH) / f"dataset_{dataset.id}_yolo"
        data_yaml = str(export_dir / "data.yaml")
        if not os.path.exists(data_yaml):
            from app.workers.training_worker import _export_dataset_sync
            _export_dataset_sync(db, dataset.id, export_dir, dataset, training_mode=job.training_mode)

        model = YOLO(mv.weights_path)
        results = model.val(data=data_yaml, verbose=False)
        metrics = results.results_dict
        error_metrics = _collect_yolo_error_metrics(model, job, db)

        metric_suffix = "M" if model_type.endswith("_segmenter") else "B"
        return {
            "map50": float(metrics.get(f"metrics/mAP50({metric_suffix})", 0)),
            "map50_95": float(metrics.get(f"metrics/mAP50-95({metric_suffix})", 0)),
            "precision": float(metrics.get(f"metrics/precision({metric_suffix})", 0)),
            "recall": float(metrics.get(f"metrics/recall({metric_suffix})", 0)),
            "metric_type": "mask" if metric_suffix == "M" else "box",
            "fitness": float(metrics.get("fitness", 0)),
            **error_metrics,
        }
    except Exception as e:
        logger.warning(f"YOLO validation error: {e}")
        return {"error": str(e)}


def _collect_yolo_error_metrics(model, job, db) -> dict:
    from collections import defaultdict
    from app.models.database import Annotation, Dataset, DatasetImage

    policy = job.evaluation_policy or {}
    iou_threshold = float(policy.get("error_iou_threshold", 0.5))
    conf_threshold = float(policy.get("error_confidence_threshold", 0.25))
    small_area_threshold = float(policy.get("small_object_area_threshold", 0.01))
    max_errors = int(policy.get("max_error_items", 300))

    dataset = db.query(Dataset).filter(Dataset.id == job.dataset_id).first()
    class_names = list(dataset.classes or []) if dataset else []
    val_imgs = db.query(DatasetImage).filter_by(dataset_id=job.dataset_id, split="val").all()
    errors = []
    false_positives = 0
    false_negatives = 0
    wrong_class = 0
    matched_gt = 0
    gt_total = 0
    gt_small = 0
    matched_small = 0
    pred_total = 0
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "false_positives": 0, "false_negatives": 0})
    confusion = defaultdict(lambda: defaultdict(int))
    slice_totals = defaultdict(lambda: {"samples": 0, "tp": 0, "fp": 0, "fn": 0})

    def class_label(class_id, fallback="unknown"):
        try:
            idx = int(class_id)
        except (TypeError, ValueError):
            return str(fallback or "unknown")
        if 0 <= idx < len(class_names):
            return str(class_names[idx])
        return str(fallback or f"class_{idx}")

    def image_slices(image):
        values = []
        for tag in (image.scene_tags or []):
            values.append(f"scene:{tag}")
        for tag in (image.quality_tags or []):
            values.append(f"quality:{tag}")
        metadata = image.frame_metadata or {}
        for key in ("daypart", "weather", "area_type", "altitude_band", "density"):
            if metadata.get(key):
                values.append(f"{key}:{metadata[key]}")
        return values or ["slice:unspecified"]

    for img in val_imgs:
        if not img.file_path or not os.path.exists(img.file_path):
            continue
        width, height = img.width or 0, img.height or 0
        if width <= 0 or height <= 0:
            try:
                from PIL import Image
                with Image.open(img.file_path) as im:
                    width, height = im.size
            except Exception:
                continue

        gt = []
        image_tp = image_fp = image_fn = 0
        annotation_type = "segmentation" if job.model_type.endswith("_segmenter") else "bbox"
        annotations = db.query(Annotation).filter(
            Annotation.image_id == img.id,
            Annotation.annotation_type == annotation_type,
            Annotation.is_verified == True,
        ).all()
        for ann in annotations:
            box = norm_box_to_abs(
                ann.x_center, ann.y_center, ann.bbox_width, ann.bbox_height, width, height
            )
            if box is None:
                continue
            area_ratio = ((box[2] - box[0]) * (box[3] - box[1])) / max(width * height, 1)
            gt.append({
                "ann": ann,
                "box": box,
                "matched": False,
                "small": area_ratio <= small_area_threshold,
                "class_name": class_label(ann.class_id, ann.class_name),
            })
            gt_total += 1
            if area_ratio <= small_area_threshold:
                gt_small += 1

        predictions = []
        result_list = model(img.file_path, conf=conf_threshold, verbose=False)
        for result in result_list:
            for box in result.boxes:
                xyxy = [float(v) for v in box.xyxy[0].tolist()]
                cls_id = int(box.cls[0])
                predictions.append({
                    "box": xyxy,
                    "class_id": cls_id,
                    "class_name": str(result.names.get(cls_id, "unknown")),
                    "confidence": float(box.conf[0]),
                })
        predictions.sort(key=lambda item: item["confidence"], reverse=True)
        pred_total += len(predictions)

        for pred in predictions:
            best = None
            best_iou = 0.0
            for item in gt:
                if item["matched"]:
                    continue
                iou = bbox_iou(pred["box"], item["box"])
                if iou > best_iou:
                    best_iou = iou
                    best = item

            if best and best_iou >= iou_threshold:
                best["matched"] = True
                expected_name = best["class_name"]
                if int(best["ann"].class_id or 0) == pred["class_id"]:
                    matched_gt += 1
                    image_tp += 1
                    per_class[expected_name]["tp"] += 1
                    confusion[expected_name][expected_name] += 1
                    if best["small"]:
                        matched_small += 1
                else:
                    wrong_class += 1
                    image_fp += 1
                    image_fn += 1
                    per_class[pred["class_name"]]["fp"] += 1
                    per_class[pred["class_name"]]["false_positives"] += 1
                    per_class[expected_name]["fn"] += 1
                    per_class[expected_name]["false_negatives"] += 1
                    confusion[expected_name][pred["class_name"]] += 1
                    if len(errors) < max_errors:
                        errors.append({
                            "dataset_image_id": img.id,
                            "error_type": "wrong_class",
                            "class_name": pred["class_name"],
                            "confidence": pred["confidence"],
                            "priority_score": 80.0,
                            "bbox": abs_box_to_norm(pred["box"], width, height),
                            "details": {
                                "expected_class_id": best["ann"].class_id,
                                "predicted_class_id": pred["class_id"],
                                "iou": round(best_iou, 4),
                            },
                        })
            else:
                false_positives += 1
                image_fp += 1
                per_class[pred["class_name"]]["fp"] += 1
                per_class[pred["class_name"]]["false_positives"] += 1
                confusion["background"][pred["class_name"]] += 1
                if len(errors) < max_errors:
                    errors.append({
                        "dataset_image_id": img.id,
                        "error_type": "false_positive",
                        "class_name": pred["class_name"],
                        "confidence": pred["confidence"],
                        "priority_score": 60.0 + min(pred["confidence"] * 20, 20),
                        "bbox": abs_box_to_norm(pred["box"], width, height),
                        "details": {"best_iou": round(best_iou, 4)},
                    })

        for item in gt:
            if item["matched"]:
                continue
            false_negatives += 1
            image_fn += 1
            per_class[item["class_name"]]["fn"] += 1
            per_class[item["class_name"]]["false_negatives"] += 1
            confusion[item["class_name"]]["missed"] += 1
            priority = 90.0 if item["small"] else 75.0
            if len(errors) < max_errors:
                errors.append({
                    "dataset_image_id": img.id,
                    "error_type": "small_object_miss" if item["small"] else "false_negative",
                    "class_name": item["ann"].class_name,
                    "confidence": None,
                    "priority_score": priority,
                    "bbox": abs_box_to_norm(item["box"], width, height),
                    "details": {
                        "class_id": item["ann"].class_id,
                        "small_object": item["small"],
                    },
                })
        for slice_name in image_slices(img):
            slice_totals[slice_name]["samples"] += 1
            slice_totals[slice_name]["tp"] += image_tp
            slice_totals[slice_name]["fp"] += image_fp
            slice_totals[slice_name]["fn"] += image_fn

    samples = len([img for img in val_imgs if img.file_path and os.path.exists(img.file_path)])
    recall_small = matched_small / gt_small if gt_small else None

    def scores(values):
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            **values,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }

    per_class_metrics = {name: scores(values) for name, values in sorted(per_class.items())}
    slice_metrics = {name: scores(values) for name, values in sorted(slice_totals.items())}
    return {
        "samples": samples,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "wrong_class": wrong_class,
        "predictions": pred_total,
        "ground_truth": gt_total,
        "gt_small": gt_small,
        "recall_small": recall_small,
        "ap_small": recall_small,
        "fp_per_frame": false_positives / samples if samples else None,
        "fn_per_frame": false_negatives / samples if samples else None,
        "per_class": per_class_metrics,
        "slices": slice_metrics,
        "confusion_matrix": {row: dict(cols) for row, cols in sorted(confusion.items())},
        "errors": errors,
    }

def _validate_classifier(mv, db) -> dict:
    try:
        import torch
        import torch.nn as nn
        from torchvision import transforms, models
        from torch.utils.data import DataLoader, Dataset as TDS
        from PIL import Image
        from app.models.database import Dataset, DatasetImage, Annotation, TrainingJob

        job = db.query(TrainingJob).filter(TrainingJob.id == mv.job_id).first()
        if not job:
            return {}

        dataset = db.query(Dataset).filter(Dataset.id == job.dataset_id).first()
        classes = dataset.classes or []
        n_classes = len(classes)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if mv.architecture == "resnet18":
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, n_classes)
        elif mv.architecture == "efficientnet":
            model = models.efficientnet_b0(pretrained=False)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
        else:
            model = models.mobilenet_v3_small(pretrained=False)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, n_classes)

        model.load_state_dict(torch.load(mv.weights_path, map_location=device))
        model.eval().to(device)

        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        val_imgs = db.query(DatasetImage).filter_by(
            dataset_id=job.dataset_id, split="val"
        ).all()

        correct = total = 0
        class_to_idx = {c: i for i, c in enumerate(classes)}
        with torch.no_grad():
            for img_rec in val_imgs:
                if not os.path.exists(img_rec.file_path):
                    continue
                ann = db.query(Annotation).filter_by(image_id=img_rec.id).first()
                if not ann or not ann.label:
                    continue
                img = tf(Image.open(img_rec.file_path).convert("RGB")).unsqueeze(0).to(device)
                out = model(img)
                pred = out.argmax(1).item()
                label = class_to_idx.get(ann.label, 0)
                if pred == label:
                    correct += 1
                total += 1

        acc = correct / total if total else 0
        return {"accuracy": round(acc, 4), "samples": total}
    except Exception as e:
        logger.warning(f"Classifier validation error: {e}")
        return {"error": str(e)}


def _run_auto_tests(mv) -> dict:
    """Size check, speed test, ONNX check."""
    results = {"passed": True, "tests": {}}

    # 1. Weights exist
    weights_ok = mv.weights_path and os.path.exists(mv.weights_path)
    results["tests"]["weights_exist"] = weights_ok
    if not weights_ok:
        results["passed"] = False

    # 2. File size sanity
    if weights_ok:
        size_mb = os.path.getsize(mv.weights_path) / 1e6
        results["tests"]["file_size_mb"] = round(size_mb, 1)
        results["tests"]["size_ok"] = size_mb > 0.1
        if not results["tests"]["size_ok"]:
            results["passed"] = False

    # 3. ONNX check
    if mv.onnx_path and os.path.exists(mv.onnx_path):
        try:
            import onnx
            model = onnx.load(mv.onnx_path)
            onnx.checker.check_model(model)
            results["tests"]["onnx_valid"] = True
        except Exception as e:
            results["tests"]["onnx_valid"] = False
            results["tests"]["onnx_error"] = str(e)
            results["passed"] = False
    else:
        results["tests"]["onnx_valid"] = None  # not exported yet

    # 4. Speed test (inference time)
    if mv.model_type in ("object_detector", "object_segmenter") and weights_ok:
        try:
            import time
            import numpy as np
            from ultralytics import YOLO
            model = YOLO(mv.weights_path)
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            t0 = time.time()
            for _ in range(5):
                model(dummy, verbose=False)
            avg_ms = (time.time() - t0) / 5 * 1000
            results["tests"]["synthetic_smoke_inference_ms"] = round(avg_ms, 1)
            results["tests"]["inference_smoke_ok"] = True
        except Exception as e:
            results["tests"]["speed_error"] = str(e)
            results["passed"] = False

    return results


def _benchmark_vs_production(mv, db, new_metrics: dict) -> dict:
    """Compare new model metrics vs current production model of same type."""
    from app.models.database import ModelVersion
    prod = db.query(ModelVersion).filter(
        ModelVersion.model_type == mv.model_type,
        ModelVersion.is_production == True,
    ).first()

    if not prod or not prod.metrics:
        return {"note": "No production model to compare against", "is_first": True}

    old = prod.metrics
    delta = {}
    for key in ["map50", "map50_95", "precision", "recall", "accuracy"]:
        if key in new_metrics and key in old:
            delta[f"{key}_delta"] = round(new_metrics[key] - old[key], 4)
            delta[f"{key}_new"] = round(new_metrics.get(key, 0), 4)
            delta[f"{key}_old"] = round(old.get(key, 0), 4)

    primary_metric = "map50" if "map50" in new_metrics else "accuracy"
    new_val = new_metrics.get(primary_metric, 0)
    old_val = old.get(primary_metric, 0)
    delta["is_improvement"] = new_val >= old_val
    delta["primary_metric"] = primary_metric
    delta["production_model_id"] = prod.id
    delta["production_model_name"] = prod.name

    return delta


def _get_db():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    engine = create_engine(settings.DATABASE_SYNC_URL)
    Session = sessionmaker(bind=engine)
    return Session()
