export type ModelType = 'object_detector' | 'object_segmenter' | 'object_classifier'
export type AnnotationType = 'bbox' | 'segmentation' | 'classification'
export type JobStatus = 'queued' | 'preparing' | 'training' | 'validation' | 'export' | 'completed' | 'failed' | 'cancelled'
export type DeployStatus = 'pending' | 'candidate' | 'approved' | 'deployed' | 'rejected' | 'rolled_back'
export type FrameStatus = 'unlabeled' | 'auto_labeled' | 'needs_review' | 'reviewed' | 'approved' | 'rejected' | 'hard_negative' | 'training_ready'
export type TrainingMode = 'baseline_inference' | 'head_finetune' | 'full_finetune' | 'tiled_training' | 'hard_negative_training' | 'semi_supervised' | 'continual_retraining'

export type MetricBag = {
  map50?: number
  map50_95?: number
  precision?: number
  recall?: number
  accuracy?: number
  train_loss?: number
  val_loss?: number
  epoch?: number
  validation_queued?: boolean
} & Record<string, unknown>

export interface TDataset {
  id: number
  name: string
  description: string | null
  model_type: ModelType
  annotation_type: AnnotationType
  classes: string[]
  status: string
  version: string
  parent_dataset_id: number | null
  content_hash: string | null
  is_frozen: boolean
  frozen_at: string | null
  lineage: Record<string, unknown>
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
  frame_status: FrameStatus
  review_priority: number
  review_reason: string | null
  scene_tags: string[]
  quality_tags: string[]
  frame_metadata: Record<string, unknown>
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
  training_mode: TrainingMode
  hyperparams: Record<string, unknown> | null
  base_model: TBaseModel | null
  tile_config: Record<string, unknown>
  evaluation_policy: Record<string, unknown>
  current_epoch: number
  total_epochs: number
  progress_pct: number
  eta_seconds: number | null
  best_metrics: MetricBag | null
  final_metrics: MetricBag | null
  gpu_device: string | null
  error_message: string | null
  author_email: string | null
  started_at: string | null
  finished_at: string | null
  created_at: string
}

export interface TBaseModel {
  id: string
  label: string
  source: 'registry' | 'installed' | 'ultralytics' | 'torchvision'
  architecture: string
  version: string
  model_version_id?: number
  path: string | null
  available: boolean
  is_production: boolean
  metrics: MetricBag
  training: Record<string, unknown>
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
  metrics: MetricBag | null
  hyperparams: Record<string, unknown> | null
  artifact_metadata: Record<string, any> | null
  deploy_status: DeployStatus
  is_production: boolean
  validation_passed: boolean | null
  auto_test_results: Record<string, unknown> | null
  benchmark_vs_prev: Record<string, unknown> | null
  gate_result: 'approved' | 'candidate' | 'rejected' | null
  gate_reasons: string[]
  evaluation_report_id: number | null
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

export interface TActiveLearningItem {
  id: number
  dataset_id: number
  image_id: number
  model_version_id: number | null
  evaluation_error_id: number | null
  reason: string
  priority_score: number
  status: 'open' | 'in_review' | 'annotated' | 'approved' | 'rejected' | 'skipped'
  suggested_class: string | null
  source: string
  details: Record<string, unknown>
  reviewer_email: string | null
  created_at: string
  updated_at: string
}

export interface TEvaluationReport {
  id: number
  model_version_id: number
  dataset_id: number | null
  job_id: number | null
  summary: Record<string, number | string | null>
  per_class_metrics: Record<string, unknown>
  slice_metrics: Record<string, unknown>
  speed_metrics: Record<string, number | string | null>
  confusion_matrix: unknown
  gate_result: 'approved' | 'candidate' | 'rejected'
  gate_reasons: string[]
  created_at: string
}

export interface TEvaluationError {
  id: number
  report_id: number
  dataset_image_id: number | null
  error_type: string
  class_name: string | null
  confidence: number | null
  priority_score: number
  bbox: unknown
  details: Record<string, unknown>
  created_at: string
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
  object_detector: 'Object Detector',
  object_segmenter: 'Object Segmenter',
  object_classifier: 'Object Classifier',
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
  candidate: 'text-amber-400',
  approved: 'text-blue-400',
  deployed: 'text-emerald-400',
  rejected: 'text-red-400',
  rolled_back: 'text-gray-400',
}
