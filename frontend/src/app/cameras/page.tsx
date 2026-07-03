'use client'
import { useEffect, useState } from 'react'
import AppShell from '@/components/shared/AppShell'
import AiModeExplainer from '@/components/shared/AiModeExplainer'
import LocalRecognitionSettings, { LOCAL_RECOGNITION_DEFAULTS } from '@/components/shared/LocalRecognitionSettings'
import api from '@/lib/api'
import { AI_MODE_OPTIONS } from '@/lib/aiModeProfiles'
import { Camera, AIMode, CameraSourceType, PipelineMode } from '@/types'
import { Plus, Play, Square, Trash2, Pencil, Wifi, WifiOff, RefreshCw, MapPin } from 'lucide-react'
import toast from 'react-hot-toast'
import { useTranslation } from '@/lib/i18n'

const SOURCE_TYPES: { value: CameraSourceType; label: string; desc: string }[] = [
  { value: 'rtsp', label: 'RTSP', desc: 'Direct camera stream' },
  { value: 'hls', label: 'HLS', desc: 'HTTP Live Streaming (.m3u8)' },
  { value: 'mjpeg', label: 'MJPEG', desc: 'Multipart HTTP video stream' },
  { value: 'jpeg', label: 'JPEG snapshots', desc: 'Periodically refreshed image URL' },
]

const EMPTY_FORM = {
  name: '',
  rtsp_url: '',
  source_type: 'rtsp' as CameraSourceType,
  snapshot_interval_seconds: 1,
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

export default function CamerasPage() {
  const { t } = useTranslation()
  const [cameras, setCameras] = useState<Camera[]>([])
  const [showModal, setShowModal] = useState(false)
  const [editing, setEditing] = useState<Camera | null>(null)
  const [form, setForm] = useState({ ...EMPTY_FORM })
  const [loading, setLoading] = useState(false)
  const [testing, setTesting] = useState<number | null>(null)
  const [manualOptions, setManualOptions] = useState(MANUAL_OPTIONS)
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
  }, [])

  const openAdd = () => { setEditing(null); setForm({ ...EMPTY_FORM }); setShowModal(true) }
  const openEdit = (cam: Camera) => {
    setEditing(cam)
    setForm({
      name: cam.name,
      rtsp_url: '',
      source_type: cam.source_type || 'rtsp',
      snapshot_interval_seconds: cam.snapshot_interval_seconds || 1,
      location: cam.location || '',
      ai_mode: cam.ai_mode,
      pipeline_mode: cam.pipeline_mode || 'automatic',
      pipeline_config: {
        ...EMPTY_FORM.pipeline_config,
        ...(cam.pipeline_config || {}),
      },
      max_fps: cam.max_fps,
      priority: cam.priority,
      save_crops: cam.save_crops,
      anonymization: cam.anonymization,
    })
    setShowModal(true)
  }

  const handleSave = async () => {
    setLoading(true)
    try {
      const payload = {
        ...form,
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
            <div className="bg-card border border-border rounded-2xl w-full max-w-5xl shadow-2xl max-h-[92vh] overflow-y-auto">
              <div className="flex items-center justify-between p-6 border-b border-border">
                <h2 className="font-semibold text-foreground">{editing ? t('Edit Camera') : t('Add Camera')}</h2>
                <button onClick={() => setShowModal(false)} className="text-muted-foreground hover:text-foreground" aria-label={t('Close')}>✕</button>
              </div>
              <div className="grid lg:grid-cols-[minmax(0,1fr)_minmax(390px,0.95fr)]">
              <div className="p-6 space-y-4">
                {[
                  { label: 'Name', key: 'name', placeholder: 'Entrance Camera' },
                  { label: 'Location', key: 'location', placeholder: 'Gate 1, North Entrance' },
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

                <div>
                  <label className="block text-sm font-medium text-muted-foreground mb-1">{t('Source Type')}</label>
                  <select
                    value={form.source_type}
                    onChange={(e) => setForm({ ...form, source_type: e.target.value as CameraSourceType })}
                    className="w-full px-3 py-2 bg-background border border-input rounded-lg text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                  >
                    {SOURCE_TYPES.map((source) => (
                      <option key={source.value} value={source.value}>{t(source.label)} — {t(source.desc)}</option>
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
                      form.source_type === 'rtsp'
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

                <div>
                  <label className="block text-sm font-medium text-muted-foreground mb-1">{t('AI Mode')}</label>
                  <select value={form.ai_mode} onChange={(e) => setForm({ ...form, ai_mode: e.target.value as AIMode })}
                    className="w-full px-3 py-2 bg-background border border-input rounded-lg text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring">
                    {AI_MODE_OPTIONS.map((mode) => <option key={mode.id} value={mode.id}>{t(mode.label)} — {t(mode.eyebrow)}</option>)}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-muted-foreground mb-2">{t('Pipeline selection')}</label>
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
                    <div>
                      <p className="text-sm font-medium text-foreground">{t('Manual pipeline')}</p>
                      <p className="mt-1 text-xs leading-5 text-muted-foreground">
                        {t('Only initialized components are selectable. TensorRT choices appear after compatible engines are built on an NVIDIA runtime.')}
                      </p>
                    </div>
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
                          className="w-full px-3 py-2 bg-background border border-input rounded-lg text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
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

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-sm font-medium text-muted-foreground mb-1">{t('Max FPS')}</label>
                    <input type="number" min={1} max={60} value={form.max_fps} onChange={(e) => setForm({ ...form, max_fps: Number(e.target.value) })}
                      className="w-full px-3 py-2 bg-background border border-input rounded-lg text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring" />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-muted-foreground mb-1">{t('Priority (1-10)')}</label>
                    <input type="number" min={1} max={10} value={form.priority} onChange={(e) => setForm({ ...form, priority: Number(e.target.value) })}
                      className="w-full px-3 py-2 bg-background border border-input rounded-lg text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring" />
                  </div>
                </div>

                <div className="flex gap-4">
                  {[{ key: 'save_crops', label: 'Save crops' }, { key: 'anonymization', label: 'Anonymize plates/faces' }].map(({ key, label }) => (
                    <label key={key} className="flex items-center gap-2 cursor-pointer">
                      <input type="checkbox" checked={(form as any)[key]} onChange={(e) => setForm({ ...form, [key]: e.target.checked })} className="rounded" />
                      <span className="text-sm text-muted-foreground">{t(label)}</span>
                    </label>
                  ))}
                </div>
              </div>
              <div className="border-t border-border lg:border-l lg:border-t-0">
                <AiModeExplainer mode={form.ai_mode} />
              </div>
              </div>
              <div className="flex gap-3 p-6 border-t border-border">
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
