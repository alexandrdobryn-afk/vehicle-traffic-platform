export type AIMode = 'speed' | 'balanced' | 'quality' | 'hybrid' | 'practical' | 'max_accuracy' | 'edge_onnx'
export type PipelineMode = 'automatic' | 'manual'
export type CameraStatus = 'online' | 'offline' | 'error' | 'completed'
export type CameraSourceType = 'rtsp' | 'hls' | 'mjpeg' | 'jpeg' | 'file' | 'usb' | 'drone'
export type TaskProfile = 'aerial_small_objects'
export type UserRole = 'admin' | 'operator' | 'viewer'

export interface Camera {
  id: number
  project_id: number | null
  pipeline_id: number | null
  name: string
  location: string | null
  status: CameraStatus
  source_type: CameraSourceType
  source_file_name: string | null
  source_duration_seconds: number | null
  source_fps: number | null
  progress_percent: number | null
  snapshot_interval_seconds: number
  ai_mode: AIMode
  task_profile: TaskProfile
  pipeline_mode: PipelineMode
  pipeline_config: Record<string, unknown> | null
  priority: number
  max_fps: number
  is_active: boolean
  save_crops: boolean
  anonymization: boolean
  gemini_enabled: boolean
  gemini_verify_predictions: boolean
  gemini_collect_training: boolean
  gemini_sample_interval_seconds: number
  gemini_max_candidates_per_run: number
  created_at: string
  updated_at: string
}

export interface ObjectTrack {
  id: number
  source_id: number
  processing_run_id: string
  track_id: number
  object_class: string
  confidence: number
  trajectory: number[][]
  speed_pixels_per_second: number
  direction_degrees: number | null
  state: string
  attributes: Record<string, unknown>
  best_crop_path: string | null
  last_bbox: number[]
  first_video_timestamp_seconds: number | null
  last_video_timestamp_seconds: number | null
  first_seen: string
  last_seen: string
  duration_seconds: number
}

export interface WSObject {
  track_id: number
  bbox: [number, number, number, number]
  object_class: string
  detection_confidence: number
  trajectory?: number[][]
  speed_pixels_per_second?: number
  direction_degrees?: number | null
  state?: string
  predicted?: boolean
  attributes?: Record<string, unknown>
}

export interface WSFrame {
  camera_id: number
  timestamp: string
  fps: number
  latency_ms: number
  frame_width: number
  frame_height: number
  source_frame_width?: number
  source_frame_height?: number
  processing_run_id?: string
  stage_counts?: Record<string, number>
  objects: WSObject[]
  type?: string
}

export interface Event {
  id: number
  camera_id: number
  event_type: string
  payload_json: Record<string, unknown> | null
  frame_path: string | null
  created_at: string
}

export interface AppSettings {
  execution_provider: 'auto' | 'cpu' | 'cuda'
  runtime_fallback: 'fail_closed' | 'allow_cpu'
  gpu_device_index: number
  inference_precision: 'fp32'
}

export interface GeminiSettings {
  configured: boolean
  enabled: boolean
  model: string
  request_timeout_seconds: number
  max_concurrent_requests: number
  api_key_masked: string | null
}

export type GeminiCandidateStatus = 'processing' | 'ready' | 'approved' | 'rejected' | 'imported' | 'error'

export interface GeminiAnnotation {
  class_name: string
  confidence: number
  box_2d: [number, number, number, number]
  polygon: number[][]
  mask_path?: string
  source: string
  human_edited?: boolean
}

export interface GeminiCandidate {
  id: number
  camera_id: number
  processing_run_id: string
  track_id: number | null
  status: GeminiCandidateStatus
  selection_reason: string
  frame_width: number
  frame_height: number
  target_model_type: string
  local_predictions: Array<Record<string, unknown>>
  gemini_verification: { local_predictions_correct: boolean; summary: string; confidence: number } | null
  proposed_annotations: GeminiAnnotation[]
  provider_model: string | null
  usage_metadata: Record<string, unknown> | null
  error_message: string | null
  reviewed_by: string | null
  reviewed_at: string | null
  created_at: string
  image_url: string
  mask_url: string | null
}

export interface RuntimeStatus {
  requested: 'auto' | 'cpu' | 'cuda'
  resolved: 'cpu' | 'cuda' | null
  device: string | null
  available: boolean
  fallback_policy: 'fail_closed' | 'allow_cpu'
  fallback_reason: string | null
  gpu_device_index: number
  precision: 'fp32'
  capabilities: {
    cpu: { available: boolean }
    cuda: { available: boolean; devices: Array<{ index: number; name: string; memory_mb: number; compute_capability: string }> }
    torch_version: string | null
    onnxruntime_version: string | null
    onnxruntime_providers: string[]
    container_runtime: string
  }
}

export type ExperimentTaskType = 'detection' | 'tracking' | 'segmentation' | 'classification' | 'pipeline' | 'replay' | 'profiling'
export type ExperimentStatus = 'pending' | 'running' | 'completed' | 'failed'

export interface EvaluationRun {
  id: number
  project_id: number
  name: string
  task_type: ExperimentTaskType
  model_name: string
  dataset_ref: string
  status: ExperimentStatus
  config: Record<string, unknown>
  metrics: Record<string, unknown> | null
  error_message: string | null
  started_at: string | null
  finished_at: string | null
  created_at: string
}

export type ModuleCapabilityStatus = 'ready' | 'partial' | 'planned' | 'blocked'

export interface ModuleCapability {
  id: string
  order: number
  name: string
  group: string
  status: ModuleCapabilityStatus
  ui_ready: boolean
  api_ready: boolean
  runtime_ready: boolean
  training_ready: boolean
  implemented: string[]
  missing: string[]
  dependencies: string[]
  next_steps: string[]
}

export interface ModuleCapabilitySummary {
  total: number
  counts: Record<ModuleCapabilityStatus, number>
  modules: ModuleCapability[]
}

export const EVENT_LABELS: Record<string, string> = {
  object_entered: 'Object entered',
  object_left: 'Object left',
  track_checkpoint: 'Track updated',
  camera_offline: 'Source offline',
  pipeline_error: 'Inference error',
  low_quality_frame: 'Low-quality frame',
}

export const EVENT_COLORS: Record<string, string> = {
  object_entered: 'text-emerald-400',
  object_left: 'text-slate-400',
  track_checkpoint: 'text-cyan-400',
  camera_offline: 'text-orange-400',
  pipeline_error: 'text-red-400',
  low_quality_frame: 'text-yellow-400',
}
