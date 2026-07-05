'use client'

import { useEffect, useRef, useState } from 'react'
import {
  AlertCircle, CheckCircle2, FileVideo, Play, Plus, RefreshCw,
  Square, Trash2, Upload, X,
} from 'lucide-react'
import toast from 'react-hot-toast'

import AppShell from '@/components/shared/AppShell'
import AiModeExplainer from '@/components/shared/AiModeExplainer'
import LocalRecognitionSettings, { LOCAL_RECOGNITION_DEFAULTS } from '@/components/shared/LocalRecognitionSettings'
import api from '@/lib/api'
import { AI_MODE_OPTIONS } from '@/lib/aiModeProfiles'
import { useTranslation } from '@/lib/i18n'
import { formatDuration } from '@/lib/utils'
import { useStreamUrl } from '@/hooks/useStreamUrl'
import { AIMode, Camera, PipelineMode } from '@/types'

const EMPTY_FORM = {
  name: '',
  location: '',
  ai_mode: 'balanced' as AIMode,
  pipeline_mode: 'automatic' as PipelineMode,
  pipeline_config: {
    vehicle_detector: '',
    tracker: '',
    plate_detector: '',
    ocr_engine: '',
    ...LOCAL_RECOGNITION_DEFAULTS,
  },
  max_fps: 25,
  priority: 1,
  save_crops: true,
  anonymization: false,
  gemini_enabled: false,
  gemini_verify_predictions: true,
  gemini_collect_training: true,
  gemini_sample_interval_seconds: 30,
  gemini_max_candidates_per_run: 25,
}

const MANUAL_OPTIONS = {
  vehicle_detector: [
    { value: '', label: 'Auto from selected mode', available: true },
    { value: 'yolo11n_vehicle', label: 'YOLO11n vehicle', available: true },
    { value: 'yolo11s_vehicle', label: 'YOLO11s vehicle', available: true },
    { value: 'yolo26n_vehicle', label: 'YOLO26n vehicle', available: true },
    { value: 'yolo26s_vehicle', label: 'YOLO26s vehicle', available: true },
    { value: 'yolo26n_vehicle_trt', label: 'YOLO26n TensorRT', available: false },
    { value: 'rfdetr_nano_vehicle', label: 'RF-DETR Nano', available: false },
    { value: 'rfdetr_medium_vehicle', label: 'RF-DETR Medium', available: false },
  ],
  tracker: [
    { value: '', label: 'Auto from selected mode', available: true },
    { value: 'bytetrack', label: 'ByteTrack', available: true },
    { value: 'botsort', label: 'BoT-SORT', available: true },
    { value: 'tracktrack', label: 'TrackTrack', available: true },
  ],
  plate_detector: [
    { value: '', label: 'Auto from selected mode', available: true },
    { value: 'yolov8n_plate', label: 'YOLOv8n plate', available: true },
    { value: 'openimagemodels_plate_384', label: 'OpenImageModels ONNX plate (384)', available: true },
    { value: 'yolo11n_plate', label: 'YOLO11n plate', available: true },
    { value: 'yolov8n_plate_trt', label: 'YOLOv8n plate TensorRT', available: false },
  ],
  ocr_engine: [
    { value: '', label: 'Auto from selected mode', available: true },
    { value: 'easyocr', label: 'EasyOCR', available: true },
    { value: 'paddleocr', label: 'PaddleOCR', available: false },
    { value: 'lprnet', label: 'LPRNet ONNX', available: false },
    { value: 'fastalpr', label: 'FastALPR target', available: false },
  ],
}

function statusClasses(status: Camera['status']) {
  if (status === 'online') return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
  if (status === 'completed') return 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20'
  if (status === 'error') return 'bg-red-500/10 text-red-400 border-red-500/20'
  return 'bg-muted text-muted-foreground border-border'
}

function VideoPreview({ video }: { video: Camera }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [visible, setVisible] = useState(false)
  useEffect(() => {
    const element = containerRef.current
    if (!element) return
    const observer = new IntersectionObserver(
      ([entry]) => setVisible(entry.isIntersecting),
      { rootMargin: '200px' },
    )
    observer.observe(element)
    return () => observer.disconnect()
  }, [])
  const resource = video.is_active ? 'stream' : 'snapshot'
  const streamUrl = useStreamUrl(
    video.id,
    visible && (video.is_active || video.status === 'completed'),
    resource,
  )
  return (
    <div ref={containerRef} className="aspect-video bg-black">
      {streamUrl && <img src={streamUrl} alt={video.name} className="w-full h-full object-contain" />}
    </div>
  )
}

export default function VideosPage() {
  const { t, locale } = useTranslation()
  const [videos, setVideos] = useState<Camera[]>([])
  const [showUpload, setShowUpload] = useState(false)
  const [form, setForm] = useState({ ...EMPTY_FORM })
  const [file, setFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [busy, setBusy] = useState<number | null>(null)
  const [manualOptions, setManualOptions] = useState(MANUAL_OPTIONS)
  const inputRef = useRef<HTMLInputElement>(null)

  const load = async () => {
    try {
      const response = await api.get('/videos')
      setVideos(response.data)
    } catch {}
  }

  useEffect(() => {
    load()
    api.get('/health/ai-modes').then(({ data }) => {
      const groups: Record<string, any[]> = {
        vehicle_detector: data.manual_options?.vehicle_detectors || [],
        tracker: data.manual_options?.trackers || [],
        plate_detector: data.manual_options?.plate_detectors || [],
        ocr_engine: data.manual_options?.ocr_engines || [],
      }
      setManualOptions(Object.fromEntries(Object.entries(MANUAL_OPTIONS).map(([key, options]) => [key, options.map((option) => {
        if (!option.value) return option
        const match = groups[key]?.find((entry) => entry.name === option.value || (option.value === 'lprnet' && String(entry.name).startsWith('lprnet')))
        return { ...option, available: Boolean(match?.available) }
      })])) as typeof MANUAL_OPTIONS)
    }).catch(() => {})
    const timer = setInterval(load, 2000)
    return () => clearInterval(timer)
  }, [])

  const openUpload = () => {
    setForm({ ...EMPTY_FORM })
    setFile(null)
    setUploadProgress(0)
    setShowUpload(true)
  }

  const chooseFile = (selected: File | null) => {
    setFile(selected)
    if (selected && !form.name) {
      setForm((current) => ({ ...current, name: selected.name.replace(/\.[^.]+$/, '') }))
    }
  }

  const uploadVideo = async () => {
    if (!file || !form.name.trim()) return
    const body = new FormData()
    body.append('file', file)
    Object.entries(form).forEach(([key, value]) => {
      if (key === 'pipeline_config') {
        body.append(key, JSON.stringify(value))
      } else {
        body.append(key, String(value))
      }
    })
    setUploading(true)
    setUploadProgress(0)
    try {
      await api.post('/videos/upload', body, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (event) => {
          if (event.total) setUploadProgress(Math.round(event.loaded * 100 / event.total))
        },
      })
      toast.success(t('Video uploaded'))
      setShowUpload(false)
      await load()
    } catch (error: any) {
      toast.error(error.response?.data?.detail || t('Video upload failed'))
    } finally {
      setUploading(false)
    }
  }

  const runAction = async (video: Camera, action: 'start' | 'stop' | 'test') => {
    setBusy(video.id)
    try {
      const response = await api.post(`/videos/${video.id}/${action}`)
      if (action === 'test') {
        if (!response.data.success) throw new Error(response.data.error || t('Video cannot be decoded'))
        toast.success(t('Video is readable: {width}x{height}', response.data))
      } else {
        toast.success(t(action === 'start' ? 'Started' : 'Stopped'))
      }
      await load()
    } catch (error: any) {
      toast.error(error.response?.data?.detail || error.message || t('Error'))
    } finally {
      setBusy(null)
    }
  }

  const deleteVideo = async (video: Camera) => {
    if (!confirm(t('Delete this video and its uploaded file?'))) return
    setBusy(video.id)
    try {
      await api.delete(`/videos/${video.id}`)
      toast.success(t('Deleted'))
      await load()
    } catch (error: any) {
      toast.error(error.response?.data?.detail || t('Delete failed'))
    } finally {
      setBusy(null)
    }
  }

  return (
    <AppShell>
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between pr-40">
          <div>
            <h1 className="text-2xl font-bold text-foreground">{t('Videos')}</h1>
            <p className="text-sm text-muted-foreground">{t('Test the full recognition pipeline on recorded traffic footage')}</p>
          </div>
          <button onClick={openUpload} className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:bg-primary/90">
            <Plus className="w-4 h-4" /> {t('Upload video')}
          </button>
        </div>

        {videos.length === 0 ? (
          <div className="bg-card border border-dashed border-border rounded-xl p-12 text-center">
            <FileVideo className="w-10 h-10 text-muted-foreground mx-auto mb-3" />
            <p className="font-medium text-foreground">{t('No uploaded videos yet')}</p>
            <p className="text-sm text-muted-foreground mt-1 mb-4">{t('Upload a recording to test detection, tracking and plate recognition')}</p>
            <button onClick={openUpload} className="px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm">{t('Upload video')}</button>
          </div>
        ) : (
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            {videos.map((video) => (
              <div key={video.id} className="bg-card border border-border rounded-xl overflow-hidden">
                {(video.is_active || video.status === 'completed') && (
                  <VideoPreview video={video} />
                )}
                <div className="p-5 space-y-4">
                  <div className="flex items-start justify-between gap-4">
                    <div className="min-w-0">
                      <h2 className="font-semibold text-foreground truncate">{video.name}</h2>
                      <p className="text-xs text-muted-foreground truncate">{video.source_file_name}</p>
                    </div>
                    <span className={`shrink-0 text-xs px-2 py-1 border rounded-full ${statusClasses(video.status)}`}>
                      {video.status === 'online' ? t('Processing') : t(video.status.charAt(0).toUpperCase() + video.status.slice(1))}
                    </span>
                  </div>

                  <div className="grid grid-cols-3 gap-3 text-xs">
                    <div><p className="text-muted-foreground">{t('Duration')}</p><p className="text-foreground font-medium">{video.source_duration_seconds ? formatDuration(video.source_duration_seconds, locale) : '—'}</p></div>
                    <div><p className="text-muted-foreground">{t('Source FPS')}</p><p className="text-foreground font-medium">{video.source_fps?.toFixed(1) || '—'}</p></div>
                    <div><p className="text-muted-foreground">{t('AI Mode')}</p><p className="text-foreground font-medium">{t(video.ai_mode.charAt(0).toUpperCase() + video.ai_mode.slice(1))}</p></div>
                  </div>

                  {(video.is_active || video.status === 'completed') && (
                    <div>
                      <div className="flex justify-between text-xs text-muted-foreground mb-1">
                        <span>{t('Processing progress')}</span>
                        <span>{Math.round(video.progress_percent || 0)}%</span>
                      </div>
                      <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                        <div className="h-full bg-primary transition-all" style={{ width: `${video.progress_percent || 0}%` }} />
                      </div>
                    </div>
                  )}

                  {video.status === 'completed' && (
                    <div className="flex items-center gap-2 text-xs text-cyan-400"><CheckCircle2 className="w-4 h-4" />{t('Processing completed. You can review events and vehicles or run it again.')}</div>
                  )}
                  {video.status === 'error' && (
                    <div className="flex items-center gap-2 text-xs text-red-400"><AlertCircle className="w-4 h-4" />{t('Processing failed. Test the file for decoder details.')}</div>
                  )}

                  <div className="flex gap-2">
                    {video.is_active ? (
                      <button onClick={() => runAction(video, 'stop')} disabled={busy === video.id} className="flex-1 flex items-center justify-center gap-1.5 py-2 bg-red-500/10 text-red-400 border border-red-500/20 rounded-lg text-xs font-medium disabled:opacity-50">
                        <Square className="w-3.5 h-3.5" /> {t('Stop')}
                      </button>
                    ) : (
                      <button onClick={() => runAction(video, 'start')} disabled={busy === video.id} className="flex-1 flex items-center justify-center gap-1.5 py-2 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded-lg text-xs font-medium disabled:opacity-50">
                        <Play className="w-3.5 h-3.5" /> {t(video.status === 'completed' ? 'Run again' : 'Start')}
                      </button>
                    )}
                    <button onClick={() => runAction(video, 'test')} disabled={busy === video.id} title={t('Test')} className="p-2 bg-muted text-muted-foreground rounded-lg hover:text-foreground disabled:opacity-50">
                      <RefreshCw className={`w-4 h-4 ${busy === video.id ? 'animate-spin' : ''}`} />
                    </button>
                    <button onClick={() => deleteVideo(video)} disabled={busy === video.id} title={t('Delete')} className="p-2 bg-red-500/10 text-red-400 rounded-lg disabled:opacity-50">
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {showUpload && (
          <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4">
            <div className="bg-card border border-border rounded-2xl w-full max-w-5xl shadow-2xl max-h-[92vh] overflow-y-auto">
              <div className="flex items-center justify-between p-6 border-b border-border">
                <div><h2 className="font-semibold text-foreground">{t('Upload video')}</h2><p className="text-xs text-muted-foreground mt-1">MP4, MOV, MKV, AVI, WEBM · {t('up to 2 GB')}</p></div>
                <button onClick={() => !uploading && setShowUpload(false)} aria-label={t('Close')} className="text-muted-foreground hover:text-foreground"><X className="w-5 h-5" /></button>
              </div>
              <div className="grid lg:grid-cols-[minmax(0,1fr)_minmax(390px,0.95fr)]">
              <div className="p-6 space-y-4">
                <input ref={inputRef} type="file" accept=".mp4,.mov,.mkv,.avi,.webm,video/*" className="hidden" onChange={(e) => chooseFile(e.target.files?.[0] || null)} />
                <button onClick={() => inputRef.current?.click()} className="w-full border border-dashed border-border rounded-xl p-6 hover:border-primary/50 transition-colors">
                  <Upload className="w-7 h-7 text-primary mx-auto mb-2" />
                  <p className="text-sm text-foreground font-medium truncate">{file?.name || t('Choose a video file')}</p>
                  <p className="text-xs text-muted-foreground mt-1">{file ? `${(file.size / 1024 / 1024).toFixed(1)} MB` : t('The file will be validated before it is added')}</p>
                </button>

                <div><label className="block text-sm text-muted-foreground mb-1">{t('Name')}</label><input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} className="w-full px-3 py-2 bg-background border border-input rounded-lg text-sm" /></div>
                <div><label className="block text-sm text-muted-foreground mb-1">{t('Location')}</label><input value={form.location} onChange={(e) => setForm({ ...form, location: e.target.value })} placeholder={t('Optional')} className="w-full px-3 py-2 bg-background border border-input rounded-lg text-sm" /></div>
                <div className="grid grid-cols-2 gap-3">
                  <div><label className="block text-sm text-muted-foreground mb-1">{t('AI Mode')}</label><select value={form.ai_mode} onChange={(e) => setForm({ ...form, ai_mode: e.target.value as AIMode })} className="w-full px-3 py-2 bg-background border border-input rounded-lg text-sm">{AI_MODE_OPTIONS.map((mode) => <option key={mode.id} value={mode.id}>{t(mode.label)}</option>)}</select></div>
                  <div><label className="block text-sm text-muted-foreground mb-1">{t('Max FPS')}</label><input type="number" min={1} max={60} value={form.max_fps} onChange={(e) => setForm({ ...form, max_fps: Number(e.target.value) })} className="w-full px-3 py-2 bg-background border border-input rounded-lg text-sm" /></div>
                </div>
                <div>
                  <label className="block text-sm text-muted-foreground mb-2">{t('Pipeline selection')}</label>
                  <div className="grid grid-cols-2 gap-2 rounded-lg border border-border bg-background/50 p-1">
                    {(['automatic', 'manual'] as PipelineMode[]).map((mode) => (
                      <button
                        key={mode}
                        type="button"
                        onClick={() => setForm({ ...form, pipeline_mode: mode })}
                        className={`rounded-md px-3 py-2 text-xs font-medium transition-colors ${form.pipeline_mode === mode ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground'}`}
                      >
                        {t(mode === 'automatic' ? 'Automatic' : 'Manual')}
                      </button>
                    ))}
                  </div>
                </div>
                {form.pipeline_mode === 'manual' && (
                  <div className="space-y-3 rounded-xl border border-border bg-muted/20 p-4">
                    <p className="text-xs leading-5 text-muted-foreground">
                      {t('Only initialized components are selectable. TensorRT choices appear after compatible engines are built on an NVIDIA runtime.')}
                    </p>
                    {([
                      ['vehicle_detector', 'Vehicle detector'],
                      ['tracker', 'Tracker'],
                      ['plate_detector', 'Plate detector'],
                      ['ocr_engine', 'OCR Engine'],
                    ] as const).map(([key, label]) => (
                      <div key={key}>
                        <label className="mb-1 block text-xs font-medium text-muted-foreground">{t(label)}</label>
                        <select
                          value={(form.pipeline_config as any)[key] || ''}
                          onChange={(e) => setForm({
                            ...form,
                            pipeline_config: { ...form.pipeline_config, [key]: e.target.value },
                          })}
                          className="w-full px-3 py-2 bg-background border border-input rounded-lg text-sm"
                        >
                          {(manualOptions as any)[key].filter((option: any) => option.available).map((option: any) => (
                            <option key={option.value} value={option.value}>
                              {t(option.label)}
                            </option>
                          ))}
                        </select>
                      </div>
                    ))}
                    <p className="text-xs text-muted-foreground">{t('Temporal voting and composite confidence scoring are always active. Embedding-based Vehicle ReID is not exposed until a validated model is integrated.')}</p>
                  </div>
                )}
                <LocalRecognitionSettings value={form.pipeline_config} onChange={(pipeline_config) => setForm({ ...form, pipeline_config: pipeline_config as typeof form.pipeline_config })} />
                <div className="flex gap-4">
                  <label className="flex items-center gap-2 text-sm text-muted-foreground"><input type="checkbox" checked={form.save_crops} onChange={(e) => setForm({ ...form, save_crops: e.target.checked })} />{t('Save crops')}</label>
                  <label className="flex items-center gap-2 text-sm text-muted-foreground"><input type="checkbox" checked={form.anonymization} onChange={(e) => setForm({ ...form, anonymization: e.target.checked })} />{t('Anonymize plates/faces')}</label>
                </div>
                <div className="rounded-xl border border-violet-500/25 bg-violet-500/5 p-4 space-y-3">
                  <label className="flex items-start gap-2"><input type="checkbox" checked={form.gemini_enabled} onChange={(e) => setForm({ ...form, gemini_enabled: e.target.checked })} className="mt-1" /><span><span className="block text-sm font-medium text-foreground">{t('Use selected frames to improve models with Gemini')}</span><span className="block text-xs text-muted-foreground mt-1">{t('Only bounded representative frames are uploaded. The full video stays local.')}</span></span></label>
                  {form.gemini_enabled && <div className="pl-6 space-y-3">
                    <label className="flex items-center gap-2 text-sm text-muted-foreground"><input type="checkbox" checked={form.gemini_verify_predictions} onChange={(e) => setForm({ ...form, gemini_verify_predictions: e.target.checked })} />{t('Verify model detections with Gemini')}</label>
                    <label className="flex items-center gap-2 text-sm text-muted-foreground"><input type="checkbox" checked={form.gemini_collect_training} onChange={(e) => setForm({ ...form, gemini_collect_training: e.target.checked })} />{t('Create mask candidates for training')}</label>
                    <div className="grid grid-cols-2 gap-3">
                      <label className="text-xs text-muted-foreground">{t('Minimum interval, seconds')}<input type="number" min={5} max={3600} value={form.gemini_sample_interval_seconds} onChange={(e) => setForm({ ...form, gemini_sample_interval_seconds: Number(e.target.value) })} className="mt-1 w-full px-3 py-2 bg-background border border-input rounded-lg text-sm" /></label>
                      <label className="text-xs text-muted-foreground">{t('Maximum candidates per run')}<input type="number" min={1} max={500} value={form.gemini_max_candidates_per_run} onChange={(e) => setForm({ ...form, gemini_max_candidates_per_run: Number(e.target.value) })} className="mt-1 w-full px-3 py-2 bg-background border border-input rounded-lg text-sm" /></label>
                    </div>
                    <p className="text-xs text-amber-400">{t('Frames may contain license plates or people. Enable this only when external processing is permitted.')}</p>
                  </div>}
                </div>
                {uploading && <div><div className="flex justify-between text-xs text-muted-foreground mb-1"><span>{t('Uploading...')}</span><span>{uploadProgress}%</span></div><div className="h-2 bg-muted rounded-full overflow-hidden"><div className="h-full bg-primary transition-all" style={{ width: `${uploadProgress}%` }} /></div></div>}
              </div>
              <div className="border-t border-border lg:border-l lg:border-t-0">
                <AiModeExplainer mode={form.ai_mode} />
              </div>
              </div>
              <div className="flex gap-3 p-6 border-t border-border">
                <button onClick={() => setShowUpload(false)} disabled={uploading} className="flex-1 py-2 border border-border rounded-lg text-sm text-muted-foreground disabled:opacity-50">{t('Cancel')}</button>
                <button onClick={uploadVideo} disabled={uploading || !file || !form.name.trim()} className="flex-1 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-medium disabled:opacity-50">{uploading ? t('Uploading...') : t('Upload video')}</button>
              </div>
            </div>
          </div>
        )}
      </div>
    </AppShell>
  )
}
