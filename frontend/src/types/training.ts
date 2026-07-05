export type ModelType = 'vehicle_detector' | 'plate_detector' | 'vehicle_segmenter' | 'plate_segmenter' | 'ocr' | 'color_classifier'
export type AnnotationType = 'bbox' | 'segmentation' | 'classification' | 'ocr'
export type JobStatus = 'queued' | 'preparing' | 'training' | 'validation' | 'export' | 'completed' | 'failed' | 'cancelled'
export type DeployStatus = 'pending' | 'approved' | 'deployed' | 'rejected' | 'rolled_back'

export interface TDataset {
  id: number
  name: string
  description: string | null
  model_type: ModelType
  annotation_type: AnnotationType
  classes: string[]
  status: string
  version: string
  image_count: number
  video_count: number
  annotation_count: number
  author_email: string | null
  tags: string[]
  created_at: string
  updated_at: string
}

export interface TDatasetImage {
  id: number
  dataset_id: number
  filename: string
  file_path: string
  width: number | null
  height: number | null
  source: string
  split: string | null
  is_annotated: boolean
  created_at: string
}

export interface TAnnotation {
  id: number
  image_id: number
  annotation_type: AnnotationType
  class_name: string | null
  class_id: number | null
  x_center: number | null
  y_center: number | null
  bbox_width: number | null
  bbox_height: number | null
  polygon: number[][] | null
  mask_path: string | null
  provenance: Record<string, unknown> | null
  ocr_text: string | null
  label: string | null
  confidence: number | null
  is_auto: boolean
  is_verified: boolean
  created_at: string
}

export interface TTrainingJob {
  id: number
  name: string
  model_type: ModelType
  architecture: string
  dataset_id: number
  status: JobStatus
  celery_task_id: string | null
  hyperparams: Record<string, unknown> | null
  current_epoch: number
  total_epochs: number
  progress_pct: number
  eta_seconds: number | null
  best_metrics: Record<string, number> | null
  final_metrics: Record<string, number> | null
  gpu_device: string | null
  error_message: string | null
  author_email: string | null
  started_at: string | null
  finished_at: string | null
  created_at: string
}

export interface TMetrics {
  epoch: number
  train_loss: number | null
  val_loss: number | null
  precision: number | null
  recall: number | null
  map50: number | null
  map50_95: number | null
  accuracy: number | null
  gpu_memory_mb: number | null
  gpu_utilization: number | null
  lr: number | null
  timestamp: string
}

export interface TModelVersion {
  id: number
  name: string
  model_type: ModelType
  architecture: string
  version: string
  version_number: number
  description: string | null
  changelog: string | null
  dataset_name: string | null
  weights_path: string | null
  onnx_path: string | null
  trt_path: string | null
  metrics: Record<string, number> | null
  hyperparams: Record<string, unknown> | null
  deploy_status: DeployStatus
  is_production: boolean
  validation_passed: boolean | null
  auto_test_results: Record<string, unknown> | null
  benchmark_vs_prev: Record<string, unknown> | null
  author_email: string | null
  approved_by: string | null
  deployed_at: string | null
  created_at: string
}

export interface TGPUInfo {
  id: number
  name: string
  memory_total_mb: number
  memory_used_mb: number
  memory_free_mb: number
  utilization_pct: number
  temperature: number | null
  is_available: boolean
}

export interface TArchitecture {
  id: string
  name: string
  desc: string
  params: string
}

export interface TrainingProgress {
  job_id: number
  status: JobStatus
  current_epoch: number
  total_epochs: number
  progress_pct: number
  eta_seconds: number | null
  latest_metrics: Record<string, number | null> | null
  gpu_utilization: number | null
  gpu_memory_mb: number | null
  message: string | null
}

export const MODEL_TYPE_LABELS: Record<ModelType, string> = {
  vehicle_detector: 'Vehicle Detector',
  plate_detector: 'Plate Detector',
  vehicle_segmenter: 'Vehicle Segmenter',
  plate_segmenter: 'Plate Segmenter',
  ocr: 'OCR',
  color_classifier: 'Color Classifier',
}

export const JOB_STATUS_COLORS: Record<JobStatus, string> = {
  queued: 'text-yellow-400',
  preparing: 'text-blue-400',
  training: 'text-cyan-400',
  validation: 'text-violet-400',
  export: 'text-indigo-400',
  completed: 'text-emerald-400',
  failed: 'text-red-400',
  cancelled: 'text-gray-400',
}

export const DEPLOY_STATUS_COLORS: Record<DeployStatus, string> = {
  pending: 'text-yellow-400',
  approved: 'text-blue-400',
  deployed: 'text-emerald-400',
  rejected: 'text-red-400',
  rolled_back: 'text-gray-400',
}
