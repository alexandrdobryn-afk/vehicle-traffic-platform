'use client'
import { useEffect, useState } from 'react'
import AppShell from '@/components/shared/AppShell'
import AiModeExplainer from '@/components/shared/AiModeExplainer'
import AerialSourceSettings, { AERIAL_SOURCE_DEFAULTS } from '@/components/shared/AerialSourceSettings'
import api from '@/lib/api'
import { AI_MODE_OPTIONS, MANUAL_PIPELINE_OPTIONS, ManualPipelineOption } from '@/lib/aiModeProfiles'
import { Camera, AIMode, CameraSourceType, PipelineMode } from '@/types'
import { Plus, Play, Square, Trash2, Pencil, Wifi, WifiOff, RefreshCw, MapPin, X } from 'lucide-react'
import toast from 'react-hot-toast'
import { useTranslation } from '@/lib/i18n'

const SOURCE_TYPES: { value: CameraSourceType; label: string; desc: string }[] = [
  { value: 'rtsp', label: 'RTSP', desc: 'Direct camera stream' },
  { value: 'hls', label: 'HLS', desc: 'HTTP Live Streaming (.m3u8)' },
  { value: 'mjpeg', label: 'MJPEG', desc: 'Multipart HTTP video stream' },
  { value: 'jpeg', label: 'JPEG snapshots', desc: 'Periodically refreshed image URL' },
  { value: 'usb', label: 'USB camera', desc: 'Local device index such as usb://0' },
  { value: 'drone', label: 'Drone stream', desc: 'RTSP or UDP stream from an airborne platform' },
]

const EMPTY_FORM = {
  name: '',
  rtsp_url: '',
  source_type: 'rtsp' as CameraSourceType,
  snapshot_interval_seconds: 1,
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

export default function CamerasPage() {
  const { t } = useTranslation()
  const [cameras, setCameras] = useState<Camera[]>([])
  const [showModal, setShowModal] = useState(false)
  const [editing, setEditing] = useState<Camera | null>(null)
  const [form, setForm] = useState({ ...EMPTY_FORM })
  const [loading, setLoading] = useState(false)
  const [testing, setTesting] = useState<number | null>(null)
  const [manualOptions, setManualOptions] = useState(MANUAL_OPTIONS)
  const [showModeGuide, setShowModeGuide] = useState(true)
  const sourceUrlRequired = !editing || form.source_type !== (editing.source_type || 'rtsp')

  const load = async () => {
    try {
      const res = await api.get('/cameras')
      setCameras(res.data)
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
  }, [])

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

  const openAdd = () => { setEditing(null); setForm({ ...EMPTY_FORM }); setShowModeGuide(true); setShowModal(true) }
  const openEdit = (cam: Camera) => {
    setEditing(cam)
    setShowModeGuide(true)
    setForm({
      name: cam.name,
      rtsp_url: '',
      source_type: cam.source_type || 'rtsp',
      snapshot_interval_seconds: cam.snapshot_interval_seconds || 1,
      location: cam.location || '',
      ai_mode: cam.ai_mode,
      pipeline_mode: 'manual',
      pipeline_config: {
        ...EMPTY_FORM.pipeline_config,
        ...(cam.pipeline_config || {}),
      },
      max_fps: cam.max_fps,
      priority: cam.priority,
      save_crops: cam.save_crops,
      anonymization: cam.anonymization,
      gemini_enabled: cam.gemini_enabled,
      gemini_verify_predictions: cam.gemini_verify_predictions,
      gemini_collect_training: cam.gemini_collect_training,
      gemini_sample_interval_seconds: cam.gemini_sample_interval_seconds,
      gemini_max_candidates_per_run: cam.gemini_max_candidates_per_run,
    })
    setShowModal(true)
  }

  const handleSave = async () => {
    setLoading(true)
    try {
      const payload = {
        ...form,
        project_id: null,
        pipeline_id: null,
        task_profile: 'aerial_small_objects',
        pipeline_config: form.pipeline_config,
      }
      if (editing) {
        const { rtsp_url, ...unchangedUrlForm } = payload
        await api.patch(`/cameras/${editing.id}`, rtsp_url ? payload : unchangedUrlForm)
        toast.success(t('Camera updated'))
      } else {
        await api.post('/cameras', payload)
        toast.success(t('Camera added'))
      }
      setShowModal(false)
      load()
    } catch (e: any) {
      toast.error(e.response?.data?.detail || t('Error'))
    } finally { setLoading(false) }
  }

  const handleDelete = async (id: number) => {
    if (!confirm(t('Delete this camera?'))) return
    try { await api.delete(`/cameras/${id}`); toast.success(t('Deleted')); load() }
    catch { toast.error(t('Delete failed')) }
  }

  const handleStart = async (id: number) => {
    try { await api.post(`/cameras/${id}/start`); toast.success(t('Started')); load() }
    catch { toast.error(t('Failed to start')) }
  }

  const handleStop = async (id: number) => {
    try { await api.post(`/cameras/${id}/stop`); toast.success(t('Stopped')); load() }
    catch { toast.error(t('Failed to stop')) }
  }

  const handleTest = async (id: number) => {
    setTesting(id)
    try {
      const res = await api.post(`/cameras/${id}/test`)
      if (res.data.success) toast.success(t('Connected! {width}x{height} @ {latency}ms', { width: res.data.width, height: res.data.height, latency: res.data.latency_ms }))
      else toast.error(t('Failed: {error}', { error: res.data.error }))
    } catch { toast.error(t('Test failed')) }
    finally { setTesting(null) }
  }

  return (
    <AppShell>
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between pr-40">
          <div>
            <h1 className="text-2xl font-bold text-foreground">{t('Cameras')}</h1>
            <p className="text-muted-foreground text-sm">{t('{count} configured', { count: cameras.length })}</p>
          </div>
          <button onClick={openAdd} className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:bg-primary/90 transition-colors">
            <Plus className="w-4 h-4" /> {t('Add Camera')}
          </button>
        </div>

        {cameras.length === 0 ? (
          <div className="bg-card border border-dashed border-border rounded-xl p-12 text-center">
            <div className="w-12 h-12 bg-muted rounded-xl flex items-center justify-center mx-auto mb-4">
              <WifiOff className="w-6 h-6 text-muted-foreground" />
            </div>
            <p className="font-medium text-foreground mb-1">{t('No cameras yet')}</p>
            <p className="text-sm text-muted-foreground mb-4">{t('Add a camera source to start monitoring')}</p>
            <button onClick={openAdd} className="px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm">{t('Add Camera')}</button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {cameras.map((cam) => (
              <div key={cam.id} className="bg-card border border-border rounded-xl p-5 space-y-4">
                {/* Header */}
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-2.5">
                    <span className={`w-2.5 h-2.5 rounded-full ${cam.status === 'online' ? 'bg-emerald-400' : 'bg-red-400'}`} />
                    <div>
                      <h3 className="font-semibold text-foreground text-sm">{cam.name}</h3>
                      {cam.location && <p className="text-xs text-muted-foreground flex items-center gap-1"><MapPin className="w-3 h-3" />{cam.location}</p>}
                    </div>
                  </div>
                  <div className="flex gap-1">
                    <button onClick={() => openEdit(cam)} title={t('Edit')} aria-label={t('Edit')} className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-accent"><Pencil className="w-3.5 h-3.5" /></button>
                    <button onClick={() => handleDelete(cam.id)} title={t('Delete')} aria-label={t('Delete')} className="p-1.5 rounded-md text-muted-foreground hover:text-red-400 hover:bg-red-400/10"><Trash2 className="w-3.5 h-3.5" /></button>
                  </div>
                </div>

                {/* Badges */}
                <div className="flex flex-wrap gap-1.5">
                  <span className="text-xs px-2 py-0.5 bg-cyan-500/10 text-cyan-400 rounded-full font-medium uppercase">{cam.source_type || 'rtsp'}</span>
                  <span className="text-xs px-2 py-0.5 bg-primary/10 text-primary rounded-full font-medium uppercase">{t(cam.ai_mode.charAt(0).toUpperCase() + cam.ai_mode.slice(1))}</span>
                  <span className="text-xs px-2 py-0.5 bg-muted text-muted-foreground rounded-full">{cam.max_fps} FPS</span>
                  <span className="text-xs px-2 py-0.5 bg-muted text-muted-foreground rounded-full">P{cam.priority}</span>
                  {cam.anonymization && <span className="text-xs px-2 py-0.5 bg-amber-500/10 text-amber-400 rounded-full">{t('Anon')}</span>}
                </div>

                {/* Actions */}
                <div className="flex gap-2">
                  {cam.is_active ? (
                    <button onClick={() => handleStop(cam.id)} className="flex-1 flex items-center justify-center gap-1.5 py-1.5 bg-red-500/10 text-red-400 border border-red-500/20 rounded-lg text-xs font-medium hover:bg-red-500/20">
                      <Square className="w-3 h-3" /> {t('Stop')}
                    </button>
                  ) : (
                    <button onClick={() => handleStart(cam.id)} className="flex-1 flex items-center justify-center gap-1.5 py-1.5 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded-lg text-xs font-medium hover:bg-emerald-500/20">
                      <Play className="w-3 h-3" /> {t('Start')}
                    </button>
                  )}
                  <button onClick={() => handleTest(cam.id)} disabled={testing === cam.id} className="flex items-center gap-1.5 px-3 py-1.5 bg-muted text-muted-foreground rounded-lg text-xs hover:text-foreground disabled:opacity-50">
                    {testing === cam.id ? <RefreshCw className="w-3 h-3 animate-spin" /> : <Wifi className="w-3 h-3" />} {t('Test')}
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Modal */}
        {showModal && (
          <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4">
            <div className="flex max-h-[92vh] w-full max-w-[min(96vw,1500px)] flex-col overflow-hidden rounded-2xl border border-border bg-card shadow-2xl">
              <div className="flex shrink-0 items-center justify-between gap-4 border-b border-border p-5">
                <h2 className="font-semibold text-foreground">{editing ? t('Edit Camera') : t('Add Camera')}</h2>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setShowModeGuide((current) => !current)}
                    className="rounded-lg border border-border bg-muted/30 px-3 py-2 text-xs font-medium text-foreground hover:bg-muted"
                  >
                    {showModeGuide ? t('Hide mode explanation') : t('Show how this mode works')}
                  </button>
                  <button onClick={() => setShowModal(false)} className="text-muted-foreground hover:text-foreground" aria-label={t('Close')}><X className="h-5 w-5" /></button>
                </div>
              </div>
              <div className={`grid min-h-0 flex-1 overflow-hidden ${showModeGuide ? 'lg:grid-cols-[minmax(0,1fr)_380px] xl:grid-cols-[minmax(0,1fr)_420px]' : 'grid-cols-1'}`}>
              <div className="min-h-0 space-y-4 overflow-y-auto p-5 lg:p-6">
                <div className="grid gap-3 xl:grid-cols-2">
                  {[
                    { label: 'Name', key: 'name', placeholder: 'Drone camera' },
                    { label: 'Location', key: 'location', placeholder: 'Flight area or ground station' },
                  ].map(({ label, key, placeholder }) => (
                    <div key={key}>
                      <label className="block text-sm font-medium text-muted-foreground mb-1">{t(label)}{key === 'rtsp_url' && editing ? t(' (leave blank to keep)') : ''}</label>
                      <input
                        value={(form as any)[key]}
                        onChange={(e) => setForm({ ...form, [key]: e.target.value })}
                        placeholder={t(placeholder)}
                        className="w-full px-3 py-2 bg-background border border-input rounded-lg text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                      />
                    </div>
                  ))}
                </div>

                <div className="grid gap-3 xl:grid-cols-[minmax(220px,0.8fr)_minmax(0,1.2fr)]">
                  <div>
                    <label className="block text-sm font-medium text-muted-foreground mb-1">{t('Source Type')}</label>
                    <select
                      value={form.source_type}
                      onChange={(e) => setForm({ ...form, source_type: e.target.value as CameraSourceType })}
                      className="w-full px-3 py-2 bg-background border border-input rounded-lg text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                    >
                      {SOURCE_TYPES.map((source) => (
                        <option key={source.value} value={source.value}>{t(source.label)} - {t(source.desc)}</option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-muted-foreground mb-1">
                      {t('Source URL')}{editing ? t(' (leave blank to keep)') : ''}
                    </label>
                    <input
                      value={form.rtsp_url}
                      onChange={(e) => setForm({ ...form, rtsp_url: e.target.value })}
                      placeholder={
                        form.source_type === 'usb'
                          ? 'usb://0'
                          : form.source_type === 'drone'
                            ? 'rtsp://drone-or-ground-station/stream'
                        : form.source_type === 'rtsp'
                          ? 'rtsp://user:pass@192.168.1.100/stream'
                          : form.source_type === 'hls'
                            ? 'https://example.com/live/stream.m3u8'
                            : form.source_type === 'mjpeg'
                              ? 'https://example.com/camera/video.mjpg'
                              : 'https://example.com/camera/latest.jpg'
                      }
                      className="w-full px-3 py-2 bg-background border border-input rounded-lg text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                    />
                  </div>
                </div>

                {form.source_type === 'jpeg' && (
                  <div>
                    <label className="block text-sm font-medium text-muted-foreground mb-1">{t('Snapshot Interval (seconds)')}</label>
                    <input
                      type="number"
                      min={0.25}
                      max={300}
                      step={0.25}
                      value={form.snapshot_interval_seconds}
                      onChange={(e) => setForm({ ...form, snapshot_interval_seconds: Number(e.target.value) })}
                      className="w-full px-3 py-2 bg-background border border-input rounded-lg text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                    />
                  </div>
                )}

                <div className="grid gap-3 xl:grid-cols-2">
                <div>
                  <label className="block text-sm font-medium text-muted-foreground mb-1">{t('Processing mode')}</label>
                  <select value={form.ai_mode} onChange={(e) => updateAiMode(e.target.value as AIMode)}
                    className="w-full px-3 py-2 bg-background border border-input rounded-lg text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring">
                    {AI_MODE_OPTIONS.map((mode) => <option key={mode.id} value={mode.id}>{t(mode.label)}</option>)}
                  </select>
                </div>
                  <div>
                    <label className="block text-sm font-medium text-muted-foreground mb-1">{t('Max FPS')}</label>
                    <input type="number" min={1} max={60} value={form.max_fps} onChange={(e) => setForm({ ...form, max_fps: Number(e.target.value) })}
                      className="w-full px-3 py-2 bg-background border border-input rounded-lg text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring" />
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

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-sm font-medium text-muted-foreground mb-1">{t('Priority (1-10)')}</label>
                    <input type="number" min={1} max={10} value={form.priority} onChange={(e) => setForm({ ...form, priority: Number(e.target.value) })}
                      className="w-full px-3 py-2 bg-background border border-input rounded-lg text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring" />
                  </div>
                </div>

                <div className="flex gap-4">
                  {[{ key: 'save_crops', label: 'Save object crops' }].map(({ key, label }) => (
                    <label key={key} className="flex items-center gap-2 cursor-pointer">
                      <input type="checkbox" checked={(form as any)[key]} onChange={(e) => setForm({ ...form, [key]: e.target.checked })} className="rounded" />
                      <span className="text-sm text-muted-foreground">{t(label)}</span>
                    </label>
                  ))}
                </div>

                <div className="rounded-xl border border-violet-500/25 bg-violet-500/5 p-4 space-y-3">
                  <label className="flex items-start gap-2 cursor-pointer">
                    <input type="checkbox" checked={form.gemini_enabled} onChange={(e) => setForm({ ...form, gemini_enabled: e.target.checked })} className="mt-1 rounded" />
                    <span><span className="block text-sm font-medium text-foreground">{t('Use selected frames to improve models with Gemini')}</span><span className="block text-xs text-muted-foreground mt-1">{t('Only bounded representative frames are uploaded. The full stream stays local.')}</span></span>
                  </label>
                  {form.gemini_enabled && (
                    <div className="pl-6 space-y-3">
                      <label className="flex items-center gap-2 text-sm text-muted-foreground"><input type="checkbox" checked={form.gemini_verify_predictions} onChange={(e) => setForm({ ...form, gemini_verify_predictions: e.target.checked })} />{t('Verify model detections with Gemini')}</label>
                      <label className="flex items-center gap-2 text-sm text-muted-foreground"><input type="checkbox" checked={form.gemini_collect_training} onChange={(e) => setForm({ ...form, gemini_collect_training: e.target.checked })} />{t('Create mask candidates for training')}</label>
                      <div className="grid grid-cols-2 gap-3">
                        <label className="text-xs text-muted-foreground">{t('Send one selected frame every N seconds')}<input type="number" min={5} max={3600} value={form.gemini_sample_interval_seconds} onChange={(e) => setForm({ ...form, gemini_sample_interval_seconds: Number(e.target.value) })} className="mt-1 w-full px-3 py-2 bg-background border border-input rounded-lg text-sm" /></label>
                        <label className="text-xs text-muted-foreground">{t('Maximum candidates per run')}<input type="number" min={1} max={500} value={form.gemini_max_candidates_per_run} onChange={(e) => setForm({ ...form, gemini_max_candidates_per_run: Number(e.target.value) })} className="mt-1 w-full px-3 py-2 bg-background border border-input rounded-lg text-sm" /></label>
                      </div>
                      <p className="text-xs text-amber-400">{t('Frames may contain people or sensitive locations. Enable external processing only when permitted.')}</p>
                    </div>
                  )}
                </div>
              </div>
              {showModeGuide && <div className="min-h-0 border-t border-border lg:border-l lg:border-t-0">
                <AiModeExplainer mode={form.ai_mode} pipelineMode={form.pipeline_mode} pipelineConfig={form.pipeline_config} />
              </div>}
              </div>
              <div className="flex shrink-0 gap-3 border-t border-border p-5">
                <button onClick={() => setShowModal(false)} className="flex-1 py-2 border border-border rounded-lg text-sm text-muted-foreground hover:text-foreground">{t('Cancel')}</button>
                <button onClick={handleSave} disabled={loading || (sourceUrlRequired && !form.rtsp_url)} className="flex-1 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:bg-primary/90 disabled:opacity-50">
                  {loading ? t('Saving...') : t('Save')}
                </button>
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
        className="w-full px-3 py-2 bg-background border border-input rounded-lg text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
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
