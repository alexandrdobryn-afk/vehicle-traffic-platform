import { AIMode } from '@/types'

export type AiModelRole = 'vehicle' | 'tracking' | 'plate' | 'ocr' | 'color' | 'brand'

export type RuntimeModel = {
  role: AiModelRole
  name: string
  format: string
  available: boolean
}

export type RuntimeMode = {
  id: AIMode
  frame_skip_default: number
  tracker_default: string
  models: RuntimeModel[]
}

export type AiModeProfile = {
  id: AIMode
  label: string
  eyebrow: string
  summary: string
  bestFor: string
  speed: number
  accuracy: number
  compute: number
  accent: string
  dot: string
  framePolicy: string
  modelPreferences: Record<AiModelRole, string>
  pipeline: { title: string; detail: string }[]
  warning?: string
}

export const AI_MODE_PROFILES: Record<AIMode, AiModeProfile> = {
  speed: {
    id: 'speed',
    label: 'Speed',
    eyebrow: 'Maximum throughput',
    summary: 'Processes more camera feeds with lightweight model preferences and aggressive frame sampling.',
    bestFor: 'Edge devices, many cameras, real-time alerts',
    speed: 5,
    accuracy: 3,
    compute: 2,
    accent: 'from-amber-500/20 via-orange-500/10 to-transparent',
    dot: 'bg-amber-400',
    framePolicy: 'Default: analyze every 3rd frame; low-quality frames may be skipped.',
    modelPreferences: {
      vehicle: 'Production TensorRT/PT → YOLO11n TensorRT/PT',
      tracking: 'ByteTrack',
      plate: 'OpenImageModels ONNX plate (384)',
      ocr: 'LPRNet → EasyOCR fallback',
      color: 'MobileNetV3 ONNX → HSV/KMeans fallback',
      brand: 'BrandEye logo detector → conservative multi-frame vote',
    },
    pipeline: [
      { title: 'Sample frames', detail: 'Reduce redundant inference work' },
      { title: 'Detect + track', detail: 'Light vehicle detector with ByteTrack' },
      { title: 'Read plate', detail: 'Plate crop, OCR and UA validation' },
      { title: 'Vote + emit', detail: 'Multi-frame result and events' },
    ],
  },
  balanced: {
    id: 'balanced',
    label: 'Balanced',
    eyebrow: 'Production default',
    summary: 'Balances stable recognition with practical throughput and is the safest starting point for most cameras.',
    bestFor: 'Daily production use, mixed day/night traffic',
    speed: 4,
    accuracy: 4,
    compute: 3,
    accent: 'from-cyan-500/20 via-blue-500/10 to-transparent',
    dot: 'bg-cyan-400',
    framePolicy: 'Default: analyze every 2nd frame; reject frames that are too poor for reliable recognition.',
    modelPreferences: {
      vehicle: 'Production model → YOLO11s → YOLO11n',
      tracking: 'ByteTrack',
      plate: 'Production plate model → YOLOv8n plate',
      ocr: 'Production OCR → EasyOCR → LPRNet',
      color: 'MobileNetV3 ONNX → HSV/KMeans fallback',
      brand: 'BrandEye logo detector → conservative multi-frame vote',
    },
    pipeline: [
      { title: 'Quality gate', detail: 'Check blur, light and usable detail' },
      { title: 'Detect + track', detail: 'Stable detector with ByteTrack' },
      { title: 'Plate + attributes', detail: 'OCR, color and vehicle brand voting' },
      { title: 'Validate + emit', detail: 'Regex, temporal voting and events' },
    ],
  },
  quality: {
    id: 'quality',
    label: 'Quality',
    eyebrow: 'Maximum detail',
    summary: 'Prioritizes difficult and distant vehicles, processes every frame by default and accepts higher latency.',
    bestFor: 'Forensics, difficult angles, low-confidence plates',
    speed: 2,
    accuracy: 5,
    compute: 5,
    accent: 'from-violet-500/20 via-fuchsia-500/10 to-transparent',
    dot: 'bg-violet-400',
    framePolicy: 'Default: analyze every frame and keep difficult frames for the heavier recognition path.',
    modelPreferences: {
      vehicle: 'Production model → RF-DETR → YOLO11s',
      tracking: 'BoT-SORT',
      plate: 'YOLO11n plate (experimental)',
      ocr: 'Production OCR → EasyOCR → LPRNet',
      color: 'MobileNetV3 ONNX → HSV/KMeans fallback',
      brand: 'BrandEye logo detector → conservative multi-frame vote',
    },
    pipeline: [
      { title: 'Keep all frames', detail: 'Avoid early rejection of difficult scenes' },
      { title: 'Detailed detection', detail: 'RF-DETR when deployed, otherwise resolved YOLO' },
      { title: 'Robust tracking', detail: 'BoT-SORT profile for harder motion' },
      { title: 'Validate + vote', detail: 'Full OCR and multi-frame consensus' },
    ],
    warning: 'RF-DETR is optional. The resolved model below is the model actually available now.',
  },
  hybrid: {
    id: 'hybrid',
    label: 'Hybrid',
    eyebrow: 'Adaptive escalation',
    summary: 'Starts with the balanced YOLO path and can escalate uncertain or crowded scenes to a quality re-check.',
    bestFor: 'Variable traffic where difficult scenes appear occasionally',
    speed: 3,
    accuracy: 5,
    compute: 4,
    accent: 'from-emerald-500/20 via-cyan-500/10 to-transparent',
    dot: 'bg-emerald-400',
    framePolicy: 'Default: balanced frame sampling with conditional quality re-checks.',
    modelPreferences: {
      vehicle: 'YOLO production path + optional RF-DETR re-check',
      tracking: 'ByteTrack',
      plate: 'Production plate model → YOLOv8n plate',
      ocr: 'Production OCR → EasyOCR → LPRNet',
      color: 'MobileNetV3 ONNX → HSV/KMeans fallback',
      brand: 'BrandEye logo detector → conservative multi-frame vote',
    },
    pipeline: [
      { title: 'Fast first pass', detail: 'YOLO detects normal traffic' },
      { title: 'Assess uncertainty', detail: 'Low confidence or crowded scene trigger' },
      { title: 'Quality re-check', detail: 'RF-DETR only when deployed and loaded' },
      { title: 'Merge + vote', detail: 'Tracking, OCR and final consensus' },
    ],
    warning: 'Hybrid uses YOLO11s normally and performs scheduled or uncertainty-triggered RF-DETR Medium re-checks.',
  },
  practical: {
    id: 'practical',
    label: 'Practical NextGen',
    eyebrow: 'YOLO26 + PaddleOCR',
    summary: 'Experimental practical preset for testing YOLO26-style detection with stronger OCR and temporal voting.',
    bestFor: 'A/B tests on mixed camera footage before production promotion',
    speed: 4,
    accuracy: 4,
    compute: 4,
    accent: 'from-sky-500/20 via-cyan-500/10 to-transparent',
    dot: 'bg-sky-400',
    framePolicy: 'Default: balanced sampling with PaddleOCR.',
    modelPreferences: {
      vehicle: 'YOLO26n PT (TensorRT only on a compatible NVIDIA host)',
      tracking: 'TrackTrack',
      plate: 'YOLOv8n plate',
      ocr: 'PaddleOCR',
      color: 'MobileNetV3 ONNX → HSV/KMeans fallback',
      brand: 'BrandEye logo detector → conservative multi-frame vote',
    },
    pipeline: [
      { title: 'Quality gate', detail: 'Keep usable frames and score plate crops' },
      { title: 'YOLO26 detection', detail: 'Use the installed YOLO26 detector' },
      { title: 'Track + plate', detail: 'Track vehicles, crop plates and read the best frames' },
      { title: 'Temporal vote', detail: 'Combine OCR candidates across frames' },
    ],
    warning: 'This mode starts only when its required components initialize successfully; it does not silently turn into Balanced.',
  },
  max_accuracy: {
    id: 'max_accuracy',
    label: 'Maximum Accuracy',
    eyebrow: 'RF-DETR + PaddleOCR',
    summary: 'Heavier preset for difficult scenes with RF-DETR Medium, TrackTrack, PaddleOCR and composite confidence scoring.',
    bestFor: 'Forensic review, low-confidence plates, model comparison runs',
    speed: 2,
    accuracy: 5,
    compute: 5,
    accent: 'from-fuchsia-500/20 via-violet-500/10 to-transparent',
    dot: 'bg-fuchsia-400',
    framePolicy: 'Default: analyze every frame and preserve difficult frames for heavier checks.',
    modelPreferences: {
      vehicle: 'RF-DETR Medium → YOLO26s → YOLO11s fallback',
      tracking: 'TrackTrack',
      plate: 'YOLO11n plate (experimental)',
      ocr: 'PaddleOCR',
      color: 'MobileNetV3 ONNX → HSV/KMeans fallback',
      brand: 'BrandEye logo detector → conservative multi-frame vote',
    },
    pipeline: [
      { title: 'Keep all frames', detail: 'Avoid early rejection of difficult scenes' },
      { title: 'Heavy detection', detail: 'RF-DETR Medium' },
      { title: 'OCR + confidence', detail: 'PaddleOCR, temporal voting and composite confidence' },
      { title: 'Confidence score', detail: 'Expose the effective pipeline for benchmark comparison' },
    ],
    warning: 'Embedding-based Vehicle ReID is not part of this build; short gaps are handled by tracker identity stitching.',
  },
  edge_onnx: {
    id: 'edge_onnx',
    label: 'ONNX Edge',
    eyebrow: 'Fast path / TensorRT',
    summary: 'Fast ONNX preset with YOLO26 detection, ByteTrack and FastPlateOCR.',
    bestFor: 'Jetson, NVIDIA GPU export tests, high camera counts',
    speed: 5,
    accuracy: 3,
    compute: 2,
    accent: 'from-lime-500/20 via-emerald-500/10 to-transparent',
    dot: 'bg-lime-400',
    framePolicy: 'Default: speed sampling with TensorRT/ONNX preference when artifacts exist.',
    modelPreferences: {
      vehicle: 'YOLO26n TensorRT/PT → YOLO11n fallback',
      tracking: 'ByteTrack',
      plate: 'OpenImageModels ONNX plate (384)',
      ocr: 'FastPlateOCR (FastALPR ONNX OCR)',
      color: 'MobileNetV3 ONNX → HSV/KMeans fallback',
      brand: 'BrandEye logo detector → conservative multi-frame vote',
    },
    pipeline: [
      { title: 'Sample frames', detail: 'Favor throughput over per-frame detail' },
      { title: 'TensorRT/ONNX target', detail: 'Use optimized artifacts only when built for this GPU' },
      { title: 'ByteTrack', detail: 'Low-overhead tracking' },
      { title: 'Fast OCR vote', detail: 'FastPlateOCR ONNX plus temporal voting' },
    ],
    warning: 'TensorRT engines are machine-specific. Build them on the target NVIDIA GPU before treating this as a real speed result.',
  },
}

export const AI_MODE_OPTIONS = Object.values(AI_MODE_PROFILES)
