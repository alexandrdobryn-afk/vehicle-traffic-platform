from app.services.evaluation_service import EvaluationService


def test_detection_metrics_match_by_class_and_iou():
    metrics = EvaluationService().evaluate_detection([{
        "ground_truth": [
            {"class_name": "person", "bbox": [0, 0, 100, 100]},
            {"class_name": "target", "bbox": [200, 200, 300, 300]},
        ],
        "predictions": [
            {"class_name": "person", "bbox": [0, 0, 100, 100], "confidence": 0.9},
            {"class_name": "person", "bbox": [200, 200, 300, 300], "confidence": 0.8},
        ],
    }])
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5


def test_tracking_summary_is_deterministic():
    metrics = EvaluationService.evaluate_tracking({
        "ground_truth_detections": 100,
        "false_positives": 5,
        "false_negatives": 10,
        "id_switches": 2,
        "id_true_positives": 85,
        "id_false_positives": 5,
        "id_false_negatives": 10,
    })
    assert metrics["mota"] == 0.83
    assert metrics["idf1"] == 0.918919


def test_detection_suite_computes_average_precision_across_iou_thresholds():
    metrics = EvaluationService().evaluate_detection_suite([{
        "ground_truth": [{"class_name": "person", "bbox": [0, 0, 100, 100]}],
        "predictions": [{"class_name": "person", "bbox": [0, 0, 100, 100], "confidence": 0.95}],
    }])
    assert metrics["ap50"] == 1.0
    assert metrics["map50_95"] == 1.0
    assert len(metrics["ap_by_iou"]) == 10


def test_classification_and_segmentation_summaries():
    classification = EvaluationService.evaluate_classification([
        {"ground_truth": "target", "prediction": "target"},
        {"ground_truth": "object", "prediction": "target"},
    ])
    segmentation = EvaluationService.evaluate_segmentation([{"iou": 0.8}, {"iou": 0.4}])
    assert classification["accuracy"] == 0.5
    assert segmentation == {"samples": 2, "mean_iou": 0.6, "iou50_rate": 0.5, "iou75_rate": 0.5}
