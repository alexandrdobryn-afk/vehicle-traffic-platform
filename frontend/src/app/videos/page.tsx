'use client'

import { useEffect, useRef, useState } from 'react'
import {
  AlertCircle, CheckCircle2, FileVideo, Play, Plus, RefreshCw,
  Square, Trash2, Upload, X,
} from 'lucide-react'
import toast from 'react-hot-toast'

import AppShell from '@/components/shared/AppShell'
import AiModeExplainer from '@/components/shared/AiModeExplainer'
import AerialSourceSettings, { AERIAL_SOURCE_DEFAULTS } from '@/components/shared/AerialSourceSettings'
import api, { apiErrorMessage } from '@/lib/api'
import { AI_MODE_OPTIONS, MANUAL_PIPELINE_OPTIONS, ManualPipelineOption } from '@/lib/aiModeProfiles'
import { useTranslation } from '@/lib/i18n'
import { formatDuration } from '@/lib/utils'
import { useStreamUrl } from '@/hooks/useStreamUrl'
import { AIMode, Camera, PipelineMode } from '@/types'

const EMPTY_FORM = {
  name: '',
  location: '',
  ai_mode: 'balanced' as AIMode,
  pipeline_mode: 'manual' as PipelineMode,
  pipeline_config: {
    ...AERIAL_SOURCE_DEFAULTS,
    object_detector: 'yolo11s_object',
    verifier_detector: 'rfdetr_medium_object',
    tracker: 'bytetrack',
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

const MANUAL_OPTIONS = MANUAL_PIPELINE_OPTIONS

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
    visible,
    resource,
  )
  return (
    <div ref={containerRef} className="aspect-video bg-black">
      {streamUrl ? (
        <img src={streamUrl} alt={video.name} className="w-full h-full object-contain" />
      ) : (
        <div className="flex h-full w-full items-center justify-center bg-muted/20 text-muted-foreground">
          <FileVideo className="h-8 w-8" />
        </div>
      )}
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
  const [showModeGuide, setShowModeGuide] = useState(true)
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
        object_detector: data.manual_options?.object_detectors || [],
        verifier_detector: data.manual_options?.verifier_detectors || [],
        tracker: data.manual_options?.trackers || [],
      }
      setManualOptions(Object.fromEntries(Object.entries(MANUAL_OPTIONS).map(([key, options]) => [key, options.map((option) => {
        if (!option.value) return option
        const match = groups[key]?.find((entry) => entry.name === option.value)
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
    setShowModeGuide(true)
    setShowUpload(true)
  }

  const updateAiMode = (ai_mode: AIMode) => {
    setForm((current) => {
      const nextConfig = { ...current.pipeline_config }
      if (ai_mode === 'hybrid') {
        if (!String(nextConfig.object_detector || '').startsWith('yolo')) {
          nextConfig.object_detector = 'yolo11s_object'
        }
        nextConfig.verifier_detector = nextConfig.verifier_detector || 'rfdetr_medium_object'
      }
      return { ...current, ai_mode, pipeline_mode: 'manual', pipeline_config: nextConfig }
    })
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
    body.append('task_profile', 'aerial_small_objects')
    setUploading(true)
    setUploadProgress(0)
    try {
      await api.post('/videos/upload', body, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 0,
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
        if (!response.data.success) {
          throw new Error(t(action === 'start' ? 'Failed to start' : 'Failed to stop'))
        }
        toast.success(t(action === 'start' ? 'Started' : 'Stopped'))
      }
      await load()
    } catch (error: any) {
      toast.error(apiErrorMessage(error, t('Error')))
      await load()
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
            <p className="text-sm text-muted-foreground">{t('Analyze recorded aerial footage with small-object detection and tracking')}</p>
          </div>
          <button onClick={openUpload} className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:bg-primary/90">
            <Plus className="w-4 h-4" /> {t('Upload video')}
          </button>
        </div>

        {videos.length === 0 ? (
          <div className="bg-card border border-dashed border-border rounded-xl p-12 text-center">
            <FileVideo className="w-10 h-10 text-muted-foreground mx-auto mb-3" />
            <p className="font-medium text-foreground">{t('No uploaded videos yet')}</p>
            <p className="text-sm text-muted-foreground mt-1 mb-4">{t('Upload drone footage to detect, classify and track small objects')}</p>
            <button onClick={openUpload} className="px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm">{t('Upload video')}</button>
          </div>
        ) : (
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            {videos.map((video) => (
              <div key={video.id} className="bg-card border border-border rounded-xl overflow-hidden">
                <VideoPreview video={video} />
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
                    <div><p className="text-muted-foreground">{t('Duration')}</p><p className="text-foreground font-medium">{video.source_duration_seconds ? formatDuration(video.source_duration_seconds, locale) : '-'}</p></div>
                    <div><p className="text-muted-foreground">{t('Source FPS')}</p><p className="text-foreground font-medium">{video.source_fps?.toFixed(1) || '-'}</p></div>
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
                    <div className="flex items-center gap-2 text-xs text-cyan-400"><CheckCircle2 className="w-4 h-4" />{t('Processing completed. You can review events and object analytics or run it again.')}</div>
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
            <div className="flex max-h-[92vh] w-full max-w-[min(96vw,1500px)] flex-col overflow-hidden rounded-2xl border border-border bg-card shadow-2xl">
              <div className="flex shrink-0 items-center justify-between gap-4 border-b border-border p-5">
                <div><h2 className="font-semibold text-foreground">{t('Upload video')}</h2><p className="text-xs text-muted-foreground mt-1">MP4, MOV, MKV, AVI, WEBM - {t('up to 2 GB')}</p></div>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setShowModeGuide((current) => !current)}
                    className="rounded-lg border border-border bg-muted/30 px-3 py-2 text-xs font-medium text-foreground hover:bg-muted"
                  >
                    {showModeGuide ? t('Hide mode explanation') : t('Show how this mode works')}
                  </button>
                  <button onClick={() => !uploading && setShowUpload(false)} aria-label={t('Close')} className="text-muted-foreground hover:text-foreground"><X className="w-5 h-5" /></button>
                </div>
              </div>
              <div className={`grid min-h-0 flex-1 overflow-hidden ${showModeGuide ? 'lg:grid-cols-[minmax(0,1fr)_380px] xl:grid-cols-[minmax(0,1fr)_420px]' : 'grid-cols-1'}`}>
              <div className="min-h-0 space-y-4 overflow-y-auto p-5 lg:p-6">
                <input ref={inputRef} type="file" accept=".mp4,.mov,.mkv,.avi,.webm,video/*" className="hidden" onChange={(e) => chooseFile(e.target.files?.[0] || null)} />

                <div className="grid gap-4 rounded-xl border border-border bg-muted/10 p-4 xl:grid-cols-[280px_minmax(0,1fr)]">
                  <button onClick={() => inputRef.current?.click()} className="flex min-h-[150px] flex-col items-center justify-center rounded-xl border border-dashed border-primary/35 bg-primary/5 p-4 text-center transition-colors hover:border-primary/70 hover:bg-primary/10">
                    <Upload className="mb-2 h-7 w-7 text-primary" />
                    <p className="max-w-full truncate text-sm font-semibold text-foreground">{file?.name || t('Choose a video file')}</p>
                    <p className="mt-1 text-xs leading-5 text-muted-foreground">{file ? `${(file.size / 1024 / 1024).toFixed(1)} MB` : t('MP4, MOV, MKV, AVI, WEBM')}</p>
                    <span className="mt-3 rounded-full border border-primary/25 px-3 py-1 text-[11px] font-medium text-primary">{t('Select file')}</span>
                  </button>

                  <div className="grid content-start gap-3 xl:grid-cols-2">
                    <div><label className="block text-sm text-muted-foreground mb-1">{t('Name')}</label><input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} className="w-full px-3 py-2 bg-background border border-input rounded-lg text-sm" /></div>
                    <div><label className="block text-sm text-muted-foreground mb-1">{t('Location')}</label><input value={form.location} onChange={(e) => setForm({ ...form, location: e.target.value })} placeholder={t('Optional')} className="w-full px-3 py-2 bg-background border border-input rounded-lg text-sm" /></div>
                    <div><label className="block text-sm text-muted-foreground mb-1">{t('Processing mode')}</label><select value={form.ai_mode} onChange={(e) => updateAiMode(e.target.value as AIMode)} className="w-full px-3 py-2 bg-background border border-input rounded-lg text-sm">{AI_MODE_OPTIONS.map((mode) => <option key={mode.id} value={mode.id}>{t(mode.label)}</option>)}</select></div>
                    <div><label className="block text-sm text-muted-foreground mb-1">{t('Max FPS')}</label><input type="number" min={1} max={60} value={form.max_fps} onChange={(e) => setForm({ ...form, max_fps: Number(e.target.value) })} className="w-full px-3 py-2 bg-background border border-input rounded-lg text-sm" /></div>
                  </div>
                </div>
                <div className="space-y-3 rounded-xl border border-border bg-muted/20 p-4">
                  <p className="text-xs leading-5 text-muted-foreground">
                    {t(form.ai_mode === 'hybrid'
                      ? 'Double verification uses YOLO as the first detector and RF-DETR as the verifier.'
                      : 'Standard mode runs one selected detector before tracking.')}
                  </p>
                  <div className={`grid gap-3 ${form.ai_mode === 'hybrid' ? 'xl:grid-cols-3' : 'xl:grid-cols-2'}`}>
                    <ManualSelect
                      label="Object detector"
                      options={(manualOptions as any).object_detector.filter((option: ManualPipelineOption) => form.ai_mode !== 'hybrid' || !option.value || option.value.startsWith('yolo'))}
                      value={(form.pipeline_config as any).object_detector || ''}
                      onChange={(next) => setForm({
                        ...form,
                        pipeline_config: { ...form.pipeline_config, object_detector: next },
                      })}
                    />
                    {form.ai_mode === 'hybrid' && (
                      <ManualSelect
                        label="Verifier detector"
                        options={(manualOptions as any).verifier_detector}
                        value={(form.pipeline_config as any).verifier_detector || 'rfdetr_medium_object'}
                        onChange={(next) => setForm({
                          ...form,
                          pipeline_config: { ...form.pipeline_config, verifier_detector: next },
                        })}
                      />
                    )}
                    <ManualSelect
                      label="Tracker"
                      options={(manualOptions as any).tracker}
                      value={(form.pipeline_config as any).tracker || ''}
                      onChange={(next) => setForm({
                        ...form,
                        pipeline_config: { ...form.pipeline_config, tracker: next },
                      })}
                    />
                  </div>
                  </div>
                <AerialSourceSettings value={form.pipeline_config} onChange={(pipeline_config) => setForm({ ...form, pipeline_config: pipeline_config as typeof form.pipeline_config })} />
                <div className="flex gap-4">
                  <label className="flex items-center gap-2 text-sm text-muted-foreground"><input type="checkbox" checked={form.save_crops} onChange={(e) => setForm({ ...form, save_crops: e.target.checked })} />{t('Save object crops')}</label>
                </div>
                <div className="rounded-xl border border-violet-500/25 bg-violet-500/5 p-4 space-y-3">
                  <label className="flex items-start gap-2"><input type="checkbox" checked={form.gemini_enabled} onChange={(e) => setForm({ ...form, gemini_enabled: e.target.checked })} className="mt-1" /><span><span className="block text-sm font-medium text-foreground">{t('Use selected frames to improve models with Gemini')}</span><span className="block text-xs text-muted-foreground mt-1">{t('Only bounded representative frames are uploaded. The full video stays local.')}</span></span></label>
                  {form.gemini_enabled && <div className="pl-6 space-y-3">
                    <label className="flex items-center gap-2 text-sm text-muted-foreground"><input type="checkbox" checked={form.gemini_verify_predictions} onChange={(e) => setForm({ ...form, gemini_verify_predictions: e.target.checked })} />{t('Verify model detections with Gemini')}</label>
                    <label className="flex items-center gap-2 text-sm text-muted-foreground"><input type="checkbox" checked={form.gemini_collect_training} onChange={(e) => setForm({ ...form, gemini_collect_training: e.target.checked })} />{t('Create mask candidates for training')}</label>
                    <div className="grid grid-cols-2 gap-3">
                      <label className="text-xs text-muted-foreground">{t('Send one selected frame every N seconds')}<input type="number" min={5} max={3600} value={form.gemini_sample_interval_seconds} onChange={(e) => setForm({ ...form, gemini_sample_interval_seconds: Number(e.target.value) })} className="mt-1 w-full px-3 py-2 bg-background border border-input rounded-lg text-sm" /></label>
                      <label className="text-xs text-muted-foreground">{t('Maximum candidates per run')}<input type="number" min={1} max={500} value={form.gemini_max_candidates_per_run} onChange={(e) => setForm({ ...form, gemini_max_candidates_per_run: Number(e.target.value) })} className="mt-1 w-full px-3 py-2 bg-background border border-input rounded-lg text-sm" /></label>
                    </div>
                    <p className="text-xs text-amber-400">{t('Frames may contain people or sensitive locations. Enable external processing only when permitted.')}</p>
                  </div>}
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
              {showModeGuide && <div className="min-h-0 border-t border-border lg:border-l lg:border-t-0">
                <AiModeExplainer mode={form.ai_mode} pipelineMode={form.pipeline_mode} pipelineConfig={form.pipeline_config} />
              </div>}
              </div>
              <div className="flex shrink-0 gap-3 border-t border-border p-5">
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

function ManualSelect({
  label,
  options,
  value,
  onChange,
}: {
  label: string
  options: ManualPipelineOption[]
  value: string
  onChange: (value: string) => void
}) {
  const { t } = useTranslation()
  const selected = options.find((option) => option.value === value) || options[0]
  return (
    <div>
      <label className="mb-1 block text-xs font-medium text-muted-foreground">{t(label)}</label>
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="w-full px-3 py-2 bg-background border border-input rounded-lg text-sm"
      >
        {options.map((option) => (
          <option key={option.value} value={option.value} disabled={!option.available}>
            {t(option.label)} - {t(option.family)}{option.available ? '' : ` - ${t('not installed')}`}
          </option>
        ))}
      </select>
      <p className="mt-1 text-[11px] leading-4 text-muted-foreground">{t(selected.detail)}</p>
    </div>
  )
}
