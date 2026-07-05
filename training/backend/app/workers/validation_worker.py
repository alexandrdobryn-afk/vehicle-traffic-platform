"""
Validation Worker — runs after training to compute final metrics
and benchmark against the current production model.
"""
import os
import logging
from pathlib import Path
from app.celery_app import celery_app
from app.config import settings

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
        if model_type in ("vehicle_detector", "plate_detector", "vehicle_segmenter", "plate_segmenter"):
            val_results = _validate_yolo(mv, db)
        elif model_type == "color_classifier":
            val_results = _validate_classifier(mv, db)
        elif model_type == "ocr":
            val_results = _validate_ocr(mv, db)
        else:
            val_results = {"error": f"Unsupported model type: {model_type}"}

        # Auto tests
        auto_tests = _run_auto_tests(mv)

        # Benchmark vs production
        benchmark = _benchmark_vs_production(mv, db, val_results)

        mv.auto_test_results = auto_tests
        mv.benchmark_vs_prev = benchmark
        validation_ok = "error" not in val_results and val_results.get("samples", 1) > 0
        mv.validation_passed = bool(auto_tests.get("passed", False) and validation_ok)
        if validation_ok:
            mv.metrics = val_results
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
            return {"error": "Dataset export is missing"}

        model = YOLO(mv.weights_path)
        results = model.val(data=data_yaml, verbose=False)
        metrics = results.results_dict

        return {
            "map50": float(metrics.get("metrics/mAP50(B)", 0)),
            "map50_95": float(metrics.get("metrics/mAP50-95(B)", 0)),
            "precision": float(metrics.get("metrics/precision(B)", 0)),
            "recall": float(metrics.get("metrics/recall(B)", 0)),
            "fitness": float(metrics.get("fitness", 0)),
        }
    except Exception as e:
        logger.warning(f"YOLO validation error: {e}")
        return {"error": str(e)}


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


def _validate_ocr(mv, db) -> dict:
    """Evaluate exact-plate and normalized character accuracy on the val split."""
    try:
        import cv2
        import numpy as np
        import onnxruntime as ort
        from app.models.database import Annotation, DatasetImage, TrainingJob

        if not mv.onnx_path or not os.path.exists(mv.onnx_path):
            return {"error": "OCR ONNX artifact is missing"}
        job = db.query(TrainingJob).filter(TrainingJob.id == mv.job_id).first()
        if not job:
            return {"error": "Training job not found"}

        chars_table = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        blank_idx = len(chars_table)
        cyrillic_to_latin = str.maketrans({
            "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H",
            "І": "I", "К": "K", "М": "M", "О": "O", "Р": "P",
            "Т": "T", "Х": "X",
        })
        session = ort.InferenceSession(mv.onnx_path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name

        def normalize(text):
            normalized = (text or "").upper().translate(cyrillic_to_latin)
            return "".join(char for char in normalized if char in chars_table)

        def decode(output):
            logits = output[:, 0, :] if output.shape[1] == 1 else output[0]
            tokens = logits.argmax(axis=1)
            previous = -1
            result = []
            for token in tokens:
                if token != previous and token != blank_idx:
                    result.append(chars_table[int(token)])
                previous = token
            return "".join(result)

        def edit_distance(left, right):
            row = list(range(len(right) + 1))
            for i, left_char in enumerate(left, 1):
                next_row = [i]
                for j, right_char in enumerate(right, 1):
                    next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (left_char != right_char)))
                row = next_row
            return row[-1]

        images = db.query(DatasetImage).filter_by(dataset_id=job.dataset_id, split="val").all()
        exact = samples = char_errors = char_total = 0
        for image_record in images:
            annotation = (
                db.query(Annotation)
                .filter(
                    Annotation.image_id == image_record.id,
                    Annotation.annotation_type == "ocr",
                    Annotation.is_verified == True,
                )
                .first()
            )
            expected = normalize(annotation.ocr_text if annotation else "")
            image = cv2.imread(image_record.file_path)
            if image is None or not expected:
                continue
            image = cv2.resize(image, (128, 32))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            image = ((image - 0.5) / 0.5).transpose(2, 0, 1)[None, ...]
            predicted = decode(session.run(None, {input_name: image})[0])
            exact += int(predicted == expected)
            char_errors += edit_distance(predicted, expected)
            char_total += max(len(expected), 1)
            samples += 1

        if samples == 0:
            return {"error": "No verified OCR samples in validation split", "samples": 0}
        return {
            "plate_accuracy": round(exact / samples, 4),
            "char_accuracy": round(max(0.0, 1.0 - char_errors / char_total), 4),
            "samples": samples,
        }
    except Exception as e:
        logger.warning(f"OCR validation error: {e}")
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
    if mv.model_type in ("vehicle_detector", "plate_detector", "vehicle_segmenter", "plate_segmenter") and weights_ok:
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
