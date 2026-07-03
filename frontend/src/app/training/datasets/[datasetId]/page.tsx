'use client'
import { useEffect, useRef, useState } from 'react'
import AppShell from '@/components/shared/AppShell'
import trainingApi from '@/lib/trainingApi'
import { TDataset, TDatasetImage, MODEL_TYPE_LABELS } from '@/types/training'
import { Upload, Video, Split, BarChart2, Image, Pencil } from 'lucide-react'
import Link from 'next/link'
import toast from 'react-hot-toast'
import { useParams } from 'next/navigation'
import { useTranslation } from '@/lib/i18n'

export default function DatasetDetailPage() {
  const { t } = useTranslation()
  const params = useParams<{ datasetId: string }>()
  const datasetId = parseInt(params.datasetId)
  const [dataset, setDataset] = useState<TDataset | null>(null)
  const [stats, setStats] = useState<any>(null)
  const [annStats, setAnnStats] = useState<any>(null)
  const [images, setImages] = useState<TDatasetImage[]>([])
  const [uploading, setUploading] = useState(false)
  const [splitting, setSplitting] = useState(false)
  const imgInputRef = useRef<HTMLInputElement>(null)
  const vidInputRef = useRef<HTMLInputElement>(null)
  const TRAINING_URL = process.env.NEXT_PUBLIC_TRAINING_API_URL || 'http://localhost:8001'

  const [splitConfig, setSplitConfig] = useState({ train_ratio: 0.7, val_ratio: 0.2, test_ratio: 0.1, seed: 42 })

  const load = async () => {
    const [ds, st, as_, imgs] = await Promise.all([
      trainingApi.get(`/datasets/${datasetId}`),
      trainingApi.get(`/datasets/${datasetId}/stats`),
      trainingApi.get(`/annotations/dataset/${datasetId}/stats`),
      trainingApi.get(`/datasets/${datasetId}/images?limit=30`),
    ])
    setDataset(ds.data); setStats(st.data); setAnnStats(as_.data); setImages(imgs.data)
  }
  useEffect(() => { load() }, [datasetId])

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    if (!files.length) return
    setUploading(true)
    try {
      const fd = new FormData()
      files.forEach(f => fd.append('files', f))
      await trainingApi.post(`/datasets/${datasetId}/images`, fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      toast.success(t('Uploaded {count} images', { count: files.length }))
      load()
    } catch { toast.error(t('Upload failed')) }
    finally { setUploading(false); if (imgInputRef.current) imgInputRef.current.value = '' }
  }

  const handleVideoUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setUploading(true)
    try {
      const fd = new FormData()
      fd.append('file', file)
      const r = await trainingApi.post(`/datasets/${datasetId}/videos`, fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      // Auto extract at 1fps
      await trainingApi.post(`/datasets/${datasetId}/videos/${r.data.id}/extract`, { fps: 1.0 })
      toast.success(t('Video uploaded — frame extraction queued'))
      load()
    } catch { toast.error(t('Video upload failed')) }
    finally { setUploading(false); if (vidInputRef.current) vidInputRef.current.value = '' }
  }

  const handleSplit = async () => {
    if (Math.abs(splitConfig.train_ratio + splitConfig.val_ratio + splitConfig.test_ratio - 1.0) > 0.01) {
      toast.error(t('Ratios must sum to 1.0'))
      return
    }
    setSplitting(true)
    try {
      const r = await trainingApi.post(`/datasets/${datasetId}/split`, splitConfig)
      toast.success(t('Split applied: {train} train, {val} val, {test} test', r.data.counts))
      load()
    } catch { toast.error(t('Split failed')) }
    finally { setSplitting(false) }
  }

  const handleExportYolo = async () => {
    try {
      await trainingApi.post(`/datasets/${datasetId}/export/yolo`)
      toast.success(t('YOLO export started'))
    } catch { toast.error(t('Export failed')) }
  }

  if (!dataset) return <AppShell><div className="p-6 text-muted-foreground">{t('Loading...')}</div></AppShell>

  return (
    <AppShell>
      <div className="p-6 space-y-6 max-w-5xl">
        {/* Header */}
        <div className="flex items-start justify-between pr-40">
          <div>
            <h1 className="text-2xl font-bold text-foreground">{dataset.name}</h1>
            <p className="text-sm text-muted-foreground mt-0.5">
              {t(MODEL_TYPE_LABELS[dataset.model_type])} · {t(dataset.annotation_type)} · v{dataset.version}
            </p>
          </div>
          <Link href={`/training/annotate/${datasetId}`}
            className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:bg-primary/90">
            <Pencil className="w-4 h-4" /> {t('Open Annotator')}
          </Link>
        </div>

        {/* Stats cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { label: 'Images', value: dataset.image_count, icon: Image },
            { label: 'Videos', value: dataset.video_count, icon: Video },
            { label: 'Annotations', value: annStats?.annotation_count ?? 0, icon: Pencil },
            { label: 'Annotated', value: `${annStats?.annotation_rate ?? 0}%`, icon: BarChart2 },
          ].map(({ label, value, icon: Icon }) => (
            <div key={label} className="bg-card border border-border rounded-xl p-4">
              <Icon className="w-4 h-4 text-muted-foreground mb-2" />
              <p className="text-2xl font-bold text-foreground">{value}</p>
              <p className="text-xs text-muted-foreground">{t(label)}</p>
            </div>
          ))}
        </div>

        {/* Upload section */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-card border border-dashed border-border rounded-xl p-6 text-center">
            <Upload className="w-8 h-8 text-muted-foreground mx-auto mb-3" />
            <p className="text-sm font-medium text-foreground mb-1">{t('Upload Images')}</p>
            <p className="text-xs text-muted-foreground mb-4">JPG, PNG, BMP, WEBP</p>
            <input ref={imgInputRef} type="file" multiple accept="image/*" className="hidden"
              onChange={handleImageUpload} />
            <button onClick={() => imgInputRef.current?.click()} disabled={uploading}
              className="px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:bg-primary/90 disabled:opacity-50">
              {uploading ? t('Uploading...') : t('Choose Images')}
            </button>
          </div>

          <div className="bg-card border border-dashed border-border rounded-xl p-6 text-center">
            <Video className="w-8 h-8 text-muted-foreground mx-auto mb-3" />
            <p className="text-sm font-medium text-foreground mb-1">{t('Upload Video')}</p>
            <p className="text-xs text-muted-foreground mb-4">{t('MP4, AVI, MOV — auto-extracts frames')}</p>
            <input ref={vidInputRef} type="file" accept="video/*" className="hidden"
              onChange={handleVideoUpload} />
            <button onClick={() => vidInputRef.current?.click()} disabled={uploading}
              className="px-4 py-2 bg-muted text-muted-foreground rounded-lg text-sm hover:text-foreground disabled:opacity-50">
              {uploading ? t('Uploading...') : t('Choose Video')}
            </button>
          </div>
        </div>

        {/* Split configuration */}
        <div className="bg-card border border-border rounded-xl p-5 space-y-4">
          <div className="flex items-center gap-2">
            <Split className="w-4 h-4 text-muted-foreground" />
            <h3 className="text-sm font-semibold text-foreground">{t('Train / Val / Test Split')}</h3>
          </div>
          <div className="grid grid-cols-3 gap-4">
            {[
              { label: 'Train', key: 'train_ratio', color: 'bg-blue-400' },
              { label: 'Validation', key: 'val_ratio', color: 'bg-amber-400' },
              { label: 'Test', key: 'test_ratio', color: 'bg-emerald-400' },
            ].map(({ label, key, color }) => (
              <div key={key}>
                <label className="block text-xs text-muted-foreground mb-1">{t(label)}</label>
                <div className="flex items-center gap-2">
                  <input type="range" min={0} max={1} step={0.05}
                    value={(splitConfig as any)[key]}
                    onChange={e => setSplitConfig(s => ({ ...s, [key]: parseFloat(e.target.value) }))}
                    className="flex-1" />
                  <span className="text-xs font-mono text-foreground w-8">
                    {Math.round((splitConfig as any)[key] * 100)}%
                  </span>
                </div>
                <div className={`w-full h-1.5 ${color} rounded-full mt-1 opacity-60`}
                  style={{ width: `${(splitConfig as any)[key] * 100}%` }} />
              </div>
            ))}
          </div>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <label className="text-xs text-muted-foreground">{t('Seed')}:</label>
              <input type="number" value={splitConfig.seed}
                onChange={e => setSplitConfig(s => ({ ...s, seed: parseInt(e.target.value) }))}
                className="w-20 px-2 py-1 bg-background border border-input rounded text-xs text-foreground focus:outline-none" />
            </div>
            <button onClick={handleSplit} disabled={splitting}
              className="px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:bg-primary/90 disabled:opacity-50">
              {splitting ? t('Splitting...') : t('Apply Split')}
            </button>
          </div>
        </div>

        {/* Class distribution */}
        {annStats?.class_distribution && Object.keys(annStats.class_distribution).length > 0 && (
          <div className="bg-card border border-border rounded-xl p-5">
            <h3 className="text-sm font-semibold text-foreground mb-4">{t('Class Distribution')}</h3>
            {Object.entries(annStats.class_distribution).map(([cls, count]) => {
              const total = Object.values(annStats.class_distribution as Record<string, number>).reduce((a, b) => a + b, 0)
              const pct = Math.round((count as number) / total * 100)
              return (
                <div key={cls} className="flex items-center gap-3 mb-2">
                  <span className="text-xs text-muted-foreground w-24 truncate capitalize">{t(cls)}</span>
                  <div className="flex-1 bg-muted rounded-full h-2">
                    <div className="bg-primary h-2 rounded-full" style={{ width: `${pct}%` }} />
                  </div>
                  <span className="text-xs text-foreground w-12 text-right">{count as number} ({pct}%)</span>
                </div>
              )
            })}
          </div>
        )}

        {/* Image preview grid */}
        {images.length > 0 && (
          <div className="bg-card border border-border rounded-xl p-5">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold text-foreground">{t('Recent Images')}</h3>
              <Link href={`/training/annotate/${datasetId}`}
                className="text-xs text-primary hover:underline">{t('Open Annotator')} →</Link>
            </div>
            <div className="grid grid-cols-5 md:grid-cols-8 gap-2">
              {images.map(img => (
                <div key={img.id} className="relative aspect-square">
                  <img
                    src={`${TRAINING_URL}/api/v1/training/datasets/${datasetId}/images/${img.id}/file`}
                    alt={img.filename}
                    className="w-full h-full object-cover rounded-lg border border-border"
                    onError={e => { (e.target as HTMLImageElement).style.display = 'none' }}
                  />
                  {img.is_annotated && (
                    <span className="absolute top-0.5 right-0.5 w-2 h-2 bg-emerald-400 rounded-full border border-card" />
                  )}
                  {img.split && (
                    <span className={`absolute bottom-0.5 left-0.5 text-[8px] px-1 rounded font-bold ${
                      img.split === 'train' ? 'bg-blue-500/80' :
                      img.split === 'val' ? 'bg-amber-500/80' : 'bg-emerald-500/80'
                    } text-white`}>{img.split[0].toUpperCase()}</span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Export */}
        <div className="flex gap-3">
          <button onClick={handleExportYolo}
            className="px-4 py-2 bg-muted text-muted-foreground rounded-lg text-sm hover:text-foreground">
            {t('Export YOLO Format')}
          </button>
        </div>
      </div>
    </AppShell>
  )
}
