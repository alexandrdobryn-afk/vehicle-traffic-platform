"""Capability map for the full aerial CV platform roadmap.

This module is intentionally explicit: the UI must show what is ready, what is
partial, and what still needs adapters, weights or external runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from importlib.util import find_spec
from typing import Literal

from app.models.model_registry import model_registry

CapabilityStatus = Literal["ready", "partial", "planned", "blocked"]


@dataclass(frozen=True)
class ModuleCapability:
    id: str
    order: int
    name: str
    group: str
    status: CapabilityStatus
    ui_ready: bool
    api_ready: bool
    runtime_ready: bool
    training_ready: bool
    implemented: list[str]
    missing: list[str]
    dependencies: list[str]
    next_steps: list[str]


def _package_available(name: str) -> bool:
    return find_spec(name) is not None


def _model_available(name: str) -> bool:
    return any(model.name == name and model.available for model in model_registry.all_models())


class ModuleCapabilityService:
    def list_modules(self) -> list[dict]:
        yolo_ready = _model_available("yolo11s_object") or _model_available("yolo11n_object")
        segmenter_ready = _model_available("object_segmenter_production")
        classifier_ready = _model_available("object_classifier_production")
        paddle_ready = _package_available("paddleocr")
        easyocr_ready = _package_available("easyocr")

        modules = [
            ModuleCapability(
                "video_image_sources", 1, "Video and image sources", "input",
                "ready", True, True, True, False,
                ["Recorded video upload", "RTSP/HLS/MJPEG/JPEG/USB/drone source types", "stream tickets", "frame extraction for training"],
                ["Dedicated still-image inference workspace", "bulk image source runner"],
                ["OpenCV", "FastAPI upload storage"],
                ["Add batch image analysis page", "Store source-level test reports"],
            ),
            ModuleCapability(
                "small_object", 2, "Small Object Detection", "vision",
                "partial", True, True, True, True,
                ["Tile inference", "tile NMS", "source-resolution preservation", "scale processing", "tiled training export"],
                ["SAHI adapter", "image pyramid policy", "dynamic crop scheduler", "multi-scale ensemble"],
                ["OpenCV"],
                ["Add SAHI-compatible adapter", "Add UI preset for tile size/overlap/empty tiles", "Benchmark tiled vs full-frame"],
            ),
            ModuleCapability(
                "object_detection", 3, "Object Detection", "vision",
                "partial" if yolo_ready else "blocked", True, True, yolo_ready, True,
                ["YOLO adapter", "RF-DETR adapter", "RT-DETR adapter", "manual model selection"],
                ["EfficientDet", "Faster R-CNN", "RetinaNet", "FCOS", "CenterNet", "DINO", "Grounding DINO"],
                ["ultralytics", "rfdetr optional package", "model weights"],
                ["Add detector plugin interface", "Add adapter tests per detector family", "Add text-prompt detector slot for Grounding DINO"],
            ),
            ModuleCapability(
                "segmentation", 4, "Segmentation", "vision",
                "partial", True, True, segmenter_ready, True,
                ["YOLO segmentation adapter", "polygon output", "segmentation annotation schema", "segmentation YOLO export"],
                ["SAM", "MobileSAM", "FastSAM", "SegFormer", "Mask R-CNN"],
                ["ultralytics", "segmentation weights"],
                ["Add SAM/FastSAM proposal adapters", "Add mask refinement workflow", "Add mask mAP reporting"],
            ),
            ModuleCapability(
                "classification", 5, "Classification", "vision",
                "partial", True, True, classifier_ready, True,
                ["ONNX classifier inference", "MobileNetV3/EfficientNet/ResNet18 training"],
                ["Domain taxonomies", "hierarchical labels", "classifier model pack"],
                ["onnxruntime", "torchvision"],
                ["Add taxonomy editor", "Add per-project class hierarchy", "Add classifier validation dashboard"],
            ),
            ModuleCapability(
                "ocr", 6, "OCR", "vision",
                "partial", True, True, paddle_ready or easyocr_ready, False,
                ["PaddleOCR/EasyOCR registry entries", "source-level OCR settings", "optional OCR runtime stage", "OCR attributes in object output"],
                ["OCR annotation UI", "sign/text dataset export", "OCR training path", "bundled OCR model pack"],
                ["paddleocr optional", "easyocr optional"],
                ["Install selected OCR engine", "Add text annotation type in UI", "Add OCR accuracy report"],
            ),
            ModuleCapability(
                "tracking", 7, "Object Tracking", "motion",
                "partial", True, True, True, False,
                ["ByteTrack", "BoT-SORT", "OC-SORT", "DeepSORT adapter", "simple IoU fallback", "stable ID stitching"],
                ["True StrongSORT ReID integration", "tracker benchmark runner"],
                ["ultralytics", "ocsort optional", "deep-sort-realtime optional"],
                ["Add tracker availability endpoint", "Add tracker benchmark on frozen sequences"],
            ),
            ModuleCapability(
                "kalman", 8, "Kalman filtering", "motion",
                "partial", True, True, True, False,
                ["Constant-velocity Kalman smoothing", "short dropout prediction", "motion vector output"],
                ["Extended Kalman Filter", "Unscented Kalman Filter", "camera-motion-aware prediction"],
                ["numpy"],
                ["Add EKF/UKF strategy interface", "Expose prediction parameters in UI", "Evaluate ID recovery impact"],
            ),
            ModuleCapability(
                "reid", 9, "Re-Identification", "motion",
                "partial", True, True, True, False,
                ["Short-gap IoU/class identity stitching", "source-level ReID settings", "lightweight HSV appearance signature", "candidate match output"],
                ["OSNet", "FastReID", "TorchReID", "persistent embedding store", "ReID metrics"],
                ["OpenCV", "torchreid or fastreid optional for production embeddings"],
                ["Add OSNet/FastReID adapter", "Persist appearance gallery", "Wire BoT-SORT/StrongSORT appearance matching"],
            ),
            ModuleCapability(
                "super_resolution", 10, "Super Resolution", "image_processing",
                "partial", True, True, True, False,
                ["source-level SR settings", "crop-level conditional OpenCV Lanczos enhancement", "GPU budget guard fields"],
                ["Real-ESRGAN", "SwinIR", "BasicSR", "model-quality SR adapter"],
                ["OpenCV", "realesrgan/basicsr optional"],
                ["Add Real-ESRGAN/SwinIR adapters", "Benchmark accuracy vs latency", "Expose before/after preview"],
            ),
            ModuleCapability(
                "image_enhancement", 11, "Image Enhancement", "image_processing",
                "partial", True, True, True, False,
                ["CLAHE contrast enhancement inside aerial preprocessing", "frame quality checks"],
                ["Deblur", "denoise", "HDR processing", "white balance", "video stabilization"],
                ["OpenCV"],
                ["Add enhancement chain config", "Add before/after preview", "Add per-source enhancement presets"],
            ),
            ModuleCapability(
                "video_processing", 12, "Video Processing", "input",
                "partial", True, True, True, False,
                ["Recorded video upload", "decode validation", "frame extraction", "frame sampling", "progress tracking"],
                ["Key frame detection", "motion compensation", "optical flow module"],
                ["OpenCV", "ffmpeg"],
                ["Add key-frame extractor", "Add optical-flow diagnostics", "Add motion-compensated sampling"],
            ),
            ModuleCapability(
                "geo", 13, "Geo module", "geo",
                "partial", True, True, True, False,
                ["Source location text field", "source-level geo settings", "image-space object coordinates", "telemetry passthrough contract"],
                ["GPS/EXIF extraction", "altitude", "heading", "camera angle calibration", "real-world object coordinates"],
                ["EXIF/metadata parsers", "optional GIS library"],
                ["Add telemetry upload/schema", "Add EXIF parser", "Map object tracks to coordinates when calibration exists"],
            ),
            ModuleCapability(
                "dataset", 14, "Dataset Module", "training",
                "ready", True, True, True, False,
                ["Image/video import", "annotation CRUD", "auto-annotation", "dataset versions", "freeze/hash", "train/val/test split", "YOLO export"],
                ["COCO import/export", "segmentation mask file export polishing", "dataset diff UI"],
                ["PostgreSQL", "OpenCV"],
                ["Add COCO export/import", "Add dataset comparison view", "Add data-quality gates"],
            ),
            ModuleCapability(
                "training", 15, "Training Module", "training",
                "partial", True, True, True, True,
                ["YOLO detector/segmenter training", "classifier training", "baseline inference", "tiled training", "augmentation config", "CUDA worker"],
                ["Distributed training", "hyperparameter search", "full auto augmentation", "RF-DETR fine-tuning"],
                ["PyTorch CUDA", "Ultralytics", "Celery"],
                ["Add HPO jobs", "Add distributed strategy field", "Add RF-DETR trainer adapter"],
            ),
            ModuleCapability(
                "evaluation", 16, "Evaluation Module", "quality",
                "partial", True, True, True, False,
                ["Precision", "recall", "F1", "AP50", "mAP50-95", "classification accuracy", "segmentation IoU", "MOTA", "IDF1"],
                ["HOTA", "full benchmark runner", "GPU/CPU/RAM timeline", "latency distribution"],
                ["PostgreSQL"],
                ["Add benchmark-run table", "Add HOTA/TrackEval adapter", "Add runtime metric sampler"],
            ),
            ModuleCapability(
                "production_export", 17, "Production export/runtime", "deployment",
                "partial", True, True, True, False,
                ["ONNX export", "TensorRT export", "CUDA runtime policy", "ONNX Runtime provider detection"],
                ["OpenVINO", "artifact compatibility matrix", "deployment smoke tests"],
                ["onnxruntime", "TensorRT on CUDA host"],
                ["Add OpenVINO export", "Add artifact validation jobs", "Block deploy if smoke test fails"],
            ),
            ModuleCapability(
                "cpp_runtime", 18, "C++ Runtime", "deployment",
                "partial", False, False, False, False,
                ["C++ ONNX detector skeleton"],
                ["TensorRT C++ runtime", "CUDA streams", "multi-threaded pipeline", "packaged binary"],
                ["OpenCV C++", "ONNX Runtime C++", "TensorRT"],
                ["Add C++ build/test workflow", "Add runtime config parity", "Add perf comparison against Python"],
            ),
            ModuleCapability(
                "edge", 19, "Edge Module", "deployment",
                "planned", False, False, False, False,
                ["Edge-oriented ONNX/TensorRT direction"],
                ["Jetson packaging", "industrial PC profile", "offline install", "edge health checks"],
                ["JetPack/TensorRT per target"],
                ["Add target profiles", "Add Jetson Dockerfile", "Add edge validation checklist"],
            ),
            ModuleCapability(
                "api", 20, "API Module", "integration",
                "partial", True, True, True, False,
                ["REST API", "WebSocket live stream/events"],
                ["gRPC service", "OpenAPI export polishing", "client SDK"],
                ["FastAPI"],
                ["Add gRPC proto", "Add API contract tests", "Generate client SDK"],
            ),
            ModuleCapability(
                "vlm", 21, "LLM/VLM Module", "assistant",
                "partial", True, True, True, False,
                ["Gemini settings", "Gemini candidate review", "human-approved mask import"],
                ["OpenAI vision adapter", "Qwen-VL", "InternVL", "report generation", "provider comparison"],
                ["google-genai optional"],
                ["Add provider abstraction", "Add report generator", "Add cost/latency guardrails"],
            ),
            ModuleCapability(
                "simulation", 22, "Simulation Module", "simulation",
                "planned", False, False, False, False,
                ["Unity frame schema", "simulation README"],
                ["Unity connector", "synthetic dataset generator", "weather/light/traffic/person scenarios"],
                ["Unity project/runtime"],
                ["Add ingestion endpoint for simulation frames", "Add synthetic dataset provenance", "Add scenario manifest"],
            ),
        ]
        return [asdict(module) for module in modules]

    def summary(self) -> dict:
        modules = self.list_modules()
        counts = {"ready": 0, "partial": 0, "planned": 0, "blocked": 0}
        for module in modules:
            counts[module["status"]] += 1
        return {"total": len(modules), "counts": counts, "modules": modules}


module_capability_service = ModuleCapabilityService()
