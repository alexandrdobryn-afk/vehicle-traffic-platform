export type AIMode = 'speed' | 'balanced' | 'quality' | 'hybrid' | 'practical' | 'max_accuracy' | 'edge_onnx'
export type PipelineMode = 'automatic' | 'manual'
export type PlateStatus = 'searching' | 'candidate' | 'verified' | 'low_confidence' | 'invalid'
export type CameraStatus = 'online' | 'offline' | 'error' | 'completed'
export type CameraSourceType = 'rtsp' | 'hls' | 'mjpeg' | 'jpeg' | 'file'
export type UserRole = 'admin' | 'operator' | 'viewer'

export interface Camera {
  id: number
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

export interface VehicleTrack {
  id: number
  camera_id: number
  track_id: number
  vehicle_class: string
  final_plate: string | null
  plate_status: PlateStatus
  final_plate_confidence: number
  color: string
  color_confidence: number
  vehicle_make: string
  make_confidence: number
  first_seen: string
  last_seen: string
  duration_seconds: number
  best_vehicle_crop_path: string | null
  best_plate_crop_path: string | null
  processing_run_id: string
  recognition_diagnostics: RecognitionDiagnostics | null
}

export interface RecognitionDiagnostics {
  plate?: {
    status?: string
    reason?: string
    detail?: string
    detector_confidence?: number
    ocr_confidence?: number
    quality_score?: number
    crop_size?: [number, number]
    best_crop_score?: number
    best_crop_quality_score?: number
    best_crop_size?: [number, number]
    best_crop_detector_confidence?: number
    profile?: string
  }
  color?: {
    status?: string
    value?: string
    confidence?: number
    engine?: string
    html_hex?: string
    candidate_value?: string
    sample_count?: number
    support_count?: number
    distribution?: Record<string, number>
    provisional?: boolean
  }
  brand?: {
    status?: string
    value?: string
    candidate_value?: string
    confidence?: number
    engine?: string
    sample_count?: number
    support_count?: number
    distribution?: Record<string, number>
    provisional?: boolean
  }
}

export interface ActiveTrack {
  track_id: number
  camera_id: number
  vehicle_class: string
  bbox: [number, number, number, number]
  color: string
  color_confidence: number
  vehicle_make: string
  make_confidence: number
  plate: string | null
  plate_status: PlateStatus
  plate_confidence: number
  first_seen: string
  last_seen: string
}

export interface WSObject {
  track_id: number
  bbox: [number, number, number, number]
  vehicle_class: string
  plate: string | null
  plate_status: PlateStatus
  plate_confidence: number
  color: string
  color_confidence: number
  vehicle_make: string
  make_confidence: number
  recognition_diagnostics?: RecognitionDiagnostics
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
  vehicle_track_id: number | null
  event_type: string
  payload_json: Record<string, unknown> | null
  frame_path: string | null
  created_at: string
}

export interface WatchlistEntry {
  id: number
  plate_number: string
  description: string | null
  alert_channels: string[]
  is_active: boolean
  created_at: string
}

export interface AnalyticsSummary {
  total_vehicles_today: number
  total_plates_recognized: number
  ocr_success_rate: number
  active_cameras: number
  avg_fps: number
  avg_latency_ms: number
  watchlist_matches_today: number
}

export interface TrafficPoint {
  timestamp: string
  count: number
}

export interface ColorDist {
  color: string
  count: number
  percentage: number
}

export interface AppSettings {
  execution_provider: 'auto' | 'cpu' | 'cuda'
  runtime_fallback: 'fail_closed' | 'allow_cpu'
  gpu_device_index: number
  inference_precision: 'fp32'
  tracker_mode: string
  ocr_engine: string
  plate_regex_profile: string
  vehicle_confidence_threshold: number
  plate_confidence_threshold: number
  ocr_threshold: number
  frame_skip: number
  input_resolution: string
  max_fps_per_camera: number
  recorded_analysis_fps: number
  ocr_voting_window: number
  track_missing_grace_frames: number
  minimum_plate_width: number
  minimum_plate_height: number
  save_crops: boolean
  anonymization_mode: boolean
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
    cuda: {
      available: boolean
      devices: Array<{
        index: number
        name: string
        memory_mb: number
        compute_capability: string
      }>
    }
    torch_version: string | null
    onnxruntime_version: string | null
    onnxruntime_providers: string[]
    container_runtime: string
  }
}

export const COLOR_HEX: Record<string, string> = {
  black: '#1a1a1a',
  white: '#f5f5f5',
  gray: '#808080',
  silver: '#c0c0c0',
  red: '#dc2626',
  blue: '#2563eb',
  green: '#16a34a',
  yellow: '#ca8a04',
  orange: '#ea580c',
  brown: '#92400e',
  beige: '#d4b896',
  unknown: '#6b7280',
}

export const EVENT_LABELS: Record<string, string> = {
  vehicle_entered: 'Vehicle Entered',
  plate_detected: 'Plate Detected',
  plate_verified: 'Plate Verified',
  vehicle_left: 'Vehicle Left',
  watchlist_match: 'Watchlist Match',
  camera_offline: 'Camera Offline',
  pipeline_error: 'Pipeline Error',
  low_quality_frame: 'Low Quality Frame',
}

export const EVENT_COLORS: Record<string, string> = {
  vehicle_entered: 'text-green-500',
  plate_detected: 'text-blue-500',
  plate_verified: 'text-emerald-500',
  vehicle_left: 'text-gray-400',
  watchlist_match: 'text-red-500',
  camera_offline: 'text-orange-500',
  pipeline_error: 'text-red-400',
  low_quality_frame: 'text-yellow-500',
}
