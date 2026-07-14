import { AIMode } from '@/types'

export type AiModelRole = 'detection' | 'tracking' | 'prediction' | 'classification' | 'segmentation'

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

export type ArchitectureFamily = 'CNN / YOLO' | 'Transformer / DETR' | 'Hybrid CNN + Transformer' | 'Algorithm' | 'Optional head'

export type ModelChoiceMeta = {
  label: string
  family: ArchitectureFamily
  short: string
  detail: string
  availableByDefault?: boolean
}

export type ManualPipelineOption = {
  value: string
  label: string
  family: ArchitectureFamily
  detail: string
  available: boolean
}

export type AiModeProfile = {
  id: AIMode
  label: string
  eyebrow: string
  architecture: ArchitectureFamily
  detectorChoice: string
  trackerChoice: string
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

export const MODEL_CHOICE_META: Record<string, ModelChoiceMeta> = {
  object_detector_production: {
    label: 'Production object detector',
    family: 'CNN / YOLO',
    short: 'Approved project model',
    detail: 'The promoted detector produced by our training loop. When present, it should be the main production choice.',
  },
  yolo11n_object: {
    label: 'YOLO11n object detector',
    family: 'CNN / YOLO',
    short: 'Fast CNN detector',
    detail: 'Small YOLO model for high FPS. Good for quick review and weaker hardware, but can miss very small or difficult objects.',
  },
  yolo11s_object: {
    label: 'YOLO11s object detector',
    family: 'CNN / YOLO',
    short: 'Standard CNN detector',
    detail: 'Balanced YOLO model for general video and camera processing. This is the safest default before project fine-tuning.',
  },
  rfdetr_nano_object: {
    label: 'RF-DETR Nano',
    family: 'Transformer / DETR',
    short: 'Light DETR-family detector',
    detail: 'Transformer detector from the RF-DETR family. It can be more robust on complex scenes, but is heavier than YOLO.',
  },
  rfdetr_medium_object: {
    label: 'RF-DETR Medium',
    family: 'Transformer / DETR',
    short: 'Accurate DETR-family detector',
    detail: 'Heavier RF-DETR model for difficult objects and review runs. Use when accuracy matters more than speed.',
  },
  rtdetr_aerial: {
    label: 'RT-DETR Large',
    family: 'Transformer / DETR',
    short: 'RT-DETR detector',
    detail: 'Real-time DETR model through Ultralytics RTDETR. It is heavier than YOLO and useful as a transformer baseline.',
  },
  bytetrack: {
    label: 'ByteTrack',
    family: 'Algorithm',
    short: 'Fast tracker',
    detail: 'Good default tracker for stable boxes and real-time processing.',
    availableByDefault: true,
  },
  botsort: {
    label: 'BoT-SORT',
    family: 'Algorithm',
    short: 'Robust tracker',
    detail: 'Heavier tracker for difficult motion and occlusion.',
    availableByDefault: true,
  },
  ocsort: {
    label: 'OC-SORT',
    family: 'Algorithm',
    short: 'Motion-focused tracker',
    detail: 'Motion-first tracker for smooth object movement and camera motion. No appearance model, so it is lighter than DeepSORT.',
  },
  deepsort: {
    label: 'DeepSORT',
    family: 'Algorithm',
    short: 'Appearance-aware tracker',
    detail: 'Uses crop appearance features in addition to motion. Better around crossings and short occlusions, but heavier.',
  },
  strongsort: {
    label: 'StrongSORT',
    family: 'Algorithm',
    short: 'Appearance-aware tracker',
    detail: 'A stronger DeepSORT-style ReID tracker. Disabled until a compatible ReID runtime is added without breaking the CUDA stack.',
  },
}

export const EMPTY_MODEL_OPTION: ManualPipelineOption = {
  value: '',
  label: 'Recommended automatically',
  family: 'Algorithm',
  detail: 'Use the default component for the selected pipeline mode.',
  available: true,
}

export const MANUAL_PIPELINE_OPTIONS = {
  object_detector: [
    EMPTY_MODEL_OPTION,
    'object_detector_production',
    'yolo11n_object',
    'yolo11s_object',
    'rfdetr_nano_object',
    'rfdetr_medium_object',
    'rtdetr_aerial',
  ].map((value) => typeof value === 'string' ? toManualOption(value) : value),
  tracker: [
    EMPTY_MODEL_OPTION,
    'bytetrack',
    'botsort',
    'ocsort',
    'deepsort',
    'strongsort',
  ].map((value) => typeof value === 'string' ? toManualOption(value) : value),
  verifier_detector: [
    EMPTY_MODEL_OPTION,
    'rfdetr_nano_object',
    'rfdetr_medium_object',
  ].map((value) => typeof value === 'string' ? toManualOption(value) : value),
}

function toManualOption(value: string): ManualPipelineOption {
  const meta = MODEL_CHOICE_META[value]
  return {
    value,
    label: meta.label,
    family: meta.family,
    detail: meta.detail,
    available: Boolean(meta.availableByDefault),
  }
}

const common = {
  prediction: 'Kalman trajectory prediction',
  classification: 'Project object classifier when installed',
  segmentation: 'Project segmenter when installed',
}

export const AI_MODE_PROFILES: Record<AIMode, AiModeProfile> = {
  speed: {
    id: 'speed',
    label: 'CNN Fast',
    eyebrow: 'YOLO11n',
    architecture: 'CNN / YOLO',
    detectorChoice: 'yolo11n_object',
    trackerChoice: 'bytetrack',
    summary: 'Fast CNN detector for quick review, several sources and weaker hardware. It keeps tiled inference but samples fewer frames.',
    bestFor: 'Live preview, many sources, first pass over long videos',
    speed: 5,
    accuracy: 3,
    compute: 2,
    accent: 'from-amber-500/20 via-orange-500/10 to-transparent',
    dot: 'bg-amber-400',
    framePolicy: 'Uses the lightweight YOLO11n detector and speed-oriented frame sampling.',
    modelPreferences: { detection: 'YOLO11n object detector', tracking: 'ByteTrack', prediction: common.prediction, classification: common.classification, segmentation: common.segmentation },
    pipeline: [
      { title: 'Keep source resolution', detail: 'Read the original frame and split it into overlapping tiles.' },
      { title: 'CNN detection', detail: 'Run YOLO11n on tiles for fast object proposals.' },
      { title: 'Track objects', detail: 'Use ByteTrack and short Kalman prediction to keep IDs stable.' },
      { title: 'Store review data', detail: 'Save crops, confidence and trajectories for later training review.' },
    ],
  },
  balanced: {
    id: 'balanced',
    label: 'Standard',
    eyebrow: 'Single detector',
    architecture: 'CNN / YOLO',
    detectorChoice: 'yolo11s_object',
    trackerChoice: 'bytetrack',
    summary: 'Runs one selected detector, then sends its detections to the selected tracker and optional analysis modules.',
    bestFor: 'Normal video and camera processing with one trusted detector',
    speed: 4,
    accuracy: 4,
    compute: 3,
    accent: 'from-cyan-500/20 via-blue-500/10 to-transparent',
    dot: 'bg-cyan-400',
    framePolicy: 'Uses the selected detector on tiled frames. No second model re-check is run.',
    modelPreferences: { detection: 'YOLO11s object detector', tracking: 'ByteTrack', prediction: common.prediction, classification: common.classification, segmentation: common.segmentation },
    pipeline: [
      { title: 'Prepare tiles', detail: 'Preserve small details without shrinking the full aerial frame.' },
      { title: 'Selected detector', detail: 'Run exactly one detector selected on the left.' },
      { title: 'Selected tracker', detail: 'Track detections with the selected association algorithm.' },
      { title: 'Optional modules', detail: 'Run classifier, segmentation, OCR or geo only when enabled.' },
    ],
  },
  hybrid: {
    id: 'hybrid',
    label: 'Double verification',
    eyebrow: 'YOLO -> RF-DETR',
    architecture: 'Hybrid CNN + Transformer',
    detectorChoice: 'yolo11s_object',
    trackerChoice: 'bytetrack',
    summary: 'Runs YOLO first, then uses RF-DETR as a verifier for uncertain or crowded frames before tracking.',
    bestFor: 'Difficult scenes where YOLO needs a transformer second opinion',
    speed: 3,
    accuracy: 5,
    compute: 4,
    accent: 'from-emerald-500/20 via-cyan-500/10 to-transparent',
    dot: 'bg-emerald-400',
    framePolicy: 'First detector must be YOLO. The verifier must be RF-DETR Nano or RF-DETR Medium.',
    modelPreferences: { detection: 'YOLO11s + RF-DETR Medium re-check', tracking: 'ByteTrack', prediction: common.prediction, classification: common.classification, segmentation: common.segmentation },
    pipeline: [
      { title: 'YOLO first pass', detail: 'YOLO finds candidate objects quickly across tiles.' },
      { title: 'Uncertainty check', detail: 'Crowded or weak detections are selected for a heavier pass.' },
      { title: 'RF-DETR verifier', detail: 'The selected RF-DETR model validates difficult regions.' },
      { title: 'Merge + track', detail: 'Merged detections are tracked with short Kalman smoothing.' },
    ],
    warning: 'Double verification is restricted to YOLO first pass plus RF-DETR verifier. Other pairings are blocked to keep merge behavior predictable.',
  },
  quality: {
    id: 'quality',
    label: 'Transformer Accurate',
    eyebrow: 'RF-DETR',
    architecture: 'Transformer / DETR',
    detectorChoice: 'rfdetr_medium_object',
    trackerChoice: 'botsort',
    summary: 'Heavy transformer detector for offline review. Hidden from the default selector until explicitly needed.',
    bestFor: 'Offline accuracy checks and difficult footage',
    speed: 2,
    accuracy: 5,
    compute: 5,
    accent: 'from-violet-500/20 via-fuchsia-500/10 to-transparent',
    dot: 'bg-violet-400',
    framePolicy: 'Runs the heavier detector on dense tiled inference.',
    modelPreferences: { detection: 'RF-DETR Medium object detector', tracking: 'BoT-SORT', prediction: common.prediction, classification: common.classification, segmentation: common.segmentation },
    pipeline: [
      { title: 'Full detail input', detail: 'Keep source resolution and tile aggressively.' },
      { title: 'Transformer detection', detail: 'Run RF-DETR for more expensive object proposals.' },
      { title: 'Robust tracking', detail: 'Prefer BoT-SORT for harder motion.' },
      { title: 'Detailed output', detail: 'Store confidence, trajectories and enabled masks.' },
    ],
  },
  practical: {
    id: 'practical',
    label: 'Production Model',
    eyebrow: 'Approved artifact',
    architecture: 'CNN / YOLO',
    detectorChoice: 'object_detector_production',
    trackerChoice: 'ocsort',
    summary: 'Uses the promoted project model when one exists. Kept for compatibility with saved sources.',
    bestFor: 'Validated project-specific model runs',
    speed: 4,
    accuracy: 4,
    compute: 4,
    accent: 'from-sky-500/20 via-cyan-500/10 to-transparent',
    dot: 'bg-sky-400',
    framePolicy: 'Uses the promoted detector and source-local settings.',
    modelPreferences: { detection: 'Production object detector', tracking: 'OC-SORT', prediction: common.prediction, classification: common.classification, segmentation: common.segmentation },
    pipeline: [
      { title: 'Load approved model', detail: 'Resolve the promoted detector artifact.' },
      { title: 'Run tiled detection', detail: 'Use the project detector on aerial tiles.' },
      { title: 'Track + predict', detail: 'Associate objects and smooth short gaps.' },
      { title: 'Record provenance', detail: 'Keep model and source settings with the run.' },
    ],
  },
  max_accuracy: {
    id: 'max_accuracy',
    label: 'Maximum Accuracy',
    eyebrow: 'Legacy alias',
    architecture: 'Transformer / DETR',
    detectorChoice: 'rfdetr_medium_object',
    trackerChoice: 'botsort',
    summary: 'Compatibility alias for heavy RF-DETR review runs.',
    bestFor: 'Existing sources that already use this mode',
    speed: 1,
    accuracy: 5,
    compute: 5,
    accent: 'from-fuchsia-500/20 via-violet-500/10 to-transparent',
    dot: 'bg-fuchsia-400',
    framePolicy: 'Analyzes every frame with the heaviest available detector.',
    modelPreferences: { detection: 'RF-DETR Medium object detector', tracking: 'BoT-SORT', prediction: common.prediction, classification: common.classification, segmentation: common.segmentation },
    pipeline: [
      { title: 'Full detail input', detail: 'Keep all useful spatial detail.' },
      { title: 'Dense detection', detail: 'Use the heaviest installed detector.' },
      { title: 'Robust tracking', detail: 'Preserve identity through difficult motion.' },
      { title: 'Export results', detail: 'Store tracks, confidence and enabled masks.' },
    ],
  },
  edge_onnx: {
    id: 'edge_onnx',
    label: 'Edge Optimized',
    eyebrow: 'ONNX/TensorRT',
    architecture: 'CNN / YOLO',
    detectorChoice: 'object_detector_production',
    trackerChoice: 'bytetrack',
    summary: 'Compatibility mode for optimized artifacts. Hidden until ONNX/TensorRT models are promoted.',
    bestFor: 'Jetson and optimized NVIDIA edge deployments',
    speed: 5,
    accuracy: 3,
    compute: 2,
    accent: 'from-lime-500/20 via-emerald-500/10 to-transparent',
    dot: 'bg-lime-400',
    framePolicy: 'Uses speed-oriented sampling and optimized artifacts when installed.',
    modelPreferences: { detection: 'ONNX/TensorRT object detector', tracking: 'ByteTrack', prediction: common.prediction, classification: common.classification, segmentation: common.segmentation },
    pipeline: [
      { title: 'Prepare tiles', detail: 'Split the frame without global downscaling.' },
      { title: 'Optimized inference', detail: 'Use installed ONNX or TensorRT artifacts.' },
      { title: 'Fast tracking', detail: 'Maintain IDs with ByteTrack.' },
      { title: 'Emit results', detail: 'Stream detections and performance data.' },
    ],
  },
}

export const AI_MODE_OPTIONS = ['balanced', 'hybrid'].map((id) => AI_MODE_PROFILES[id as AIMode])
