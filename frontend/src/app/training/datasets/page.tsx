'use client'
import { useEffect, useState } from 'react'
import AppShell from '@/components/shared/AppShell'
import trainingApi from '@/lib/trainingApi'
import { TDataset, ModelType, AnnotationType, MODEL_TYPE_LABELS } from '@/types/training'
import { formatDateTime } from '@/lib/utils'
import Link from 'next/link'
import { Plus, Database, Trash2, Image, Video, Tag } from 'lucide-react'
import toast from 'react-hot-toast'
import { useTranslation } from '@/lib/i18n'

const MODEL_TYPES: { value: ModelType; label: string }[] = [
  { value: 'vehicle_detector', label: 'Vehicle Detector' },
  { value: 'plate_detector', label: 'Plate Detector' },
  { value: 'ocr', label: 'OCR' },
  { value: 'color_classifier', label: 'Color Classifier' },
]
const ANN_TYPES: { value: AnnotationType; label: string }[] = [
  { value: 'bbox', label: 'Bounding Box' },
  { value: 'classification', label: 'Classification' },
  { value: 'ocr', label: 'OCR Text' },
]
const DEFAULT_CLASSES: Record<ModelType, string[]> = {
  vehicle_detector: ['car', 'truck', 'bus', 'motorcycle', 'van'],
  plate_detector: ['license_plate'],
  color_classifier: ['black', 'white', 'gray', 'silver', 'red', 'blue', 'green', 'yellow', 'orange', 'brown', 'beige'],
  ocr: [],
}

export default function DatasetsPage() {
  const { t } = useTranslation()
  const [datasets, setDatasets] = useState<TDataset[]>([])
  const [showModal, setShowModal] = useState(false)
  const [filter, setFilter] = useState('')
  const [form, setForm] = useState({
    name: '', description: '', model_type: 'vehicle_detector' as ModelType,
    annotation_type: 'bbox' as AnnotationType,
    classes: [...DEFAULT_CLASSES.vehicle_detector], tags: [] as string[],
  })
  const [classInput, setClassInput] = useState('')
  const [loading, setLoading] = useState(false)

  const load = async () => {
    const res = await trainingApi.get('/datasets')
    setDatasets(res.data)
  }
  useEffect(() => { load() }, [])

  const handleModelTypeChange = (mt: ModelType) => {
    setForm(f => ({ ...f, model_type: mt, classes: DEFAULT_CLASSES[mt] || [] }))
  }

  const handleCreate = async () => {
    setLoading(true)
    try {
      await trainingApi.post('/datasets', form)
      toast.success(t('Dataset created'))
      setShowModal(false)
      setForm({
        name: '', description: '', model_type: 'vehicle_detector',
        annotation_type: 'bbox', classes: [...DEFAULT_CLASSES.vehicle_detector], tags: [],
      })
      load()
    } catch (e: any) { toast.error(e.response?.data?.detail || t('Error')) }
    finally { setLoading(false) }
  }

  const handleDelete = async (id: number) => {
    if (!confirm(t('Delete this dataset? All images and annotations will be lost.'))) return
    try { await trainingApi.delete(`/datasets/${id}`); toast.success(t('Deleted')); load() }
    catch { toast.error(t('Delete failed')) }
  }

  const addClass = () => {
    const c = classInput.trim().toLowerCase()
    if (c && !form.classes.includes(c)) {
      setForm(f => ({ ...f, classes: [...f.classes, c] }))
    }
    setClassInput('')
  }

  const filtered = datasets.filter(d =>
    d.name.toLowerCase().includes(filter.toLowerCase()) ||
    MODEL_TYPE_LABELS[d.model_type]?.toLowerCase().includes(filter.toLowerCase())
  )

  return (
    <AppShell>
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between pr-40">
          <div>
            <h1 className="text-2xl font-bold text-foreground">{t('Datasets')}</h1>
            <p className="text-muted-foreground text-sm">{t('{count} datasets', { count: datasets.length })}</p>
          </div>
          <button onClick={() => setShowModal(true)}
            className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:bg-primary/90">
            <Plus className="w-4 h-4" /> {t('New Dataset')}
          </button>
        </div>

        <input placeholder={t('Search datasets...')}
          value={filter} onChange={e => setFilter(e.target.value)}
          className="w-full max-w-sm px-4 py-2 bg-background border border-input rounded-lg text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring" />

        {filtered.length === 0 ? (
          <div className="border border-dashed border-border rounded-xl p-12 text-center">
            <Database className="w-10 h-10 text-muted-foreground mx-auto mb-3" />
            <p className="font-medium text-foreground mb-1">{t('No datasets yet')}</p>
            <p className="text-sm text-muted-foreground">{t('Create a dataset to start collecting training data')}</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {filtered.map((ds) => (
              <div key={ds.id} className="bg-card border border-border rounded-xl p-5 hover:border-primary/30 transition-colors">
                <div className="flex items-start justify-between mb-3">
                  <div className="min-w-0">
                    <h3 className="font-semibold text-foreground truncate">{ds.name}</h3>
                    <p className="text-xs text-muted-foreground mt-0.5">{t(MODEL_TYPE_LABELS[ds.model_type])}</p>
                  </div>
                  <button onClick={() => handleDelete(ds.id)}
                    className="p-1.5 text-muted-foreground hover:text-red-400 hover:bg-red-400/10 rounded-md ml-2 shrink-0">
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>

                {/* Stats */}
                <div className="grid grid-cols-3 gap-2 mb-3">
                  {[
                    { icon: Image, value: ds.image_count, label: 'Images' },
                    { icon: Video, value: ds.video_count, label: 'Videos' },
                    { icon: Tag, value: ds.classes.length, label: 'Classes' },
                  ].map(({ icon: Icon, value, label }) => (
                    <div key={label} className="text-center p-2 bg-muted/50 rounded-lg">
                      <p className="text-sm font-bold text-foreground">{value}</p>
                      <p className="text-xs text-muted-foreground">{t(label)}</p>
                    </div>
                  ))}
                </div>

                {/* Classes */}
                {ds.classes.length > 0 && (
                  <div className="flex flex-wrap gap-1 mb-3">
                    {ds.classes.slice(0, 4).map(c => (
                      <span key={c} className="text-xs px-1.5 py-0.5 bg-primary/10 text-primary rounded">{t(c)}</span>
                    ))}
                    {ds.classes.length > 4 && (
                      <span className="text-xs px-1.5 py-0.5 bg-muted text-muted-foreground rounded">
                        +{ds.classes.length - 4}
                      </span>
                    )}
                  </div>
                )}

                <div className="flex gap-2">
                  <Link href={`/training/datasets/${ds.id}`}
                    className="flex-1 text-center py-1.5 bg-muted text-muted-foreground rounded-lg text-xs hover:text-foreground">
                    {t('Manage')}
                  </Link>
                  <Link href={`/training/annotate/${ds.id}`}
                    className="flex-1 text-center py-1.5 bg-primary/10 text-primary rounded-lg text-xs hover:bg-primary/20">
                    {t('Annotate')}
                  </Link>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Modal */}
        {showModal && (
          <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4">
            <div className="bg-card border border-border rounded-2xl w-full max-w-lg shadow-2xl max-h-[90vh] overflow-y-auto">
              <div className="flex items-center justify-between p-6 border-b border-border sticky top-0 bg-card">
                <h2 className="font-semibold text-foreground">{t('Create Dataset')}</h2>
                <button onClick={() => setShowModal(false)} className="text-muted-foreground hover:text-foreground" aria-label={t('Close')}>✕</button>
              </div>
              <div className="p-6 space-y-4">
                <div>
                  <label className="block text-sm font-medium text-muted-foreground mb-1">{t('Name')}</label>
                  <input value={form.name} onChange={e => setForm(f => ({ ...f, name: e.target.value }))}
                    placeholder={t('UA Vehicles 2026')} className="w-full px-3 py-2 bg-background border border-input rounded-lg text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring" />
                </div>
                <div>
                  <label className="block text-sm font-medium text-muted-foreground mb-1">{t('Description')}</label>
                  <textarea value={form.description} onChange={e => setForm(f => ({ ...f, description: e.target.value }))}
                    rows={2} placeholder={t("What's in this dataset...")}
                    className="w-full px-3 py-2 bg-background border border-input rounded-lg text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring resize-none" />
                </div>
                <div>
                  <label className="block text-sm font-medium text-muted-foreground mb-1">{t('Model Type')}</label>
                  <select value={form.model_type} onChange={e => handleModelTypeChange(e.target.value as ModelType)}
                    className="w-full px-3 py-2 bg-background border border-input rounded-lg text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring">
                    {MODEL_TYPES.map(m => <option key={m.value} value={m.value}>{t(m.label)}</option>)}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-muted-foreground mb-1">{t('Annotation Type')}</label>
                  <select value={form.annotation_type} onChange={e => setForm(f => ({ ...f, annotation_type: e.target.value as AnnotationType }))}
                    className="w-full px-3 py-2 bg-background border border-input rounded-lg text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring">
                    {ANN_TYPES.map(a => <option key={a.value} value={a.value}>{t(a.label)}</option>)}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-muted-foreground mb-1">{t('Classes')}</label>
                  <div className="flex gap-2 mb-2">
                    <input value={classInput} onChange={e => setClassInput(e.target.value)}
                      onKeyDown={e => e.key === 'Enter' && addClass()}
                      placeholder={t('Add class...')} className="flex-1 px-3 py-2 bg-background border border-input rounded-lg text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring" />
                    <button onClick={addClass} className="px-3 py-2 bg-primary text-primary-foreground rounded-lg text-sm">{t('Add')}</button>
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {form.classes.map(c => (
                      <span key={c} className="flex items-center gap-1 text-xs px-2 py-1 bg-primary/10 text-primary rounded-full">
                        {c}
                        <button onClick={() => setForm(f => ({ ...f, classes: f.classes.filter(x => x !== c) }))}
                          className="hover:text-red-400">✕</button>
                      </span>
                    ))}
                  </div>
                </div>
              </div>
              <div className="flex gap-3 p-6 border-t border-border">
                <button onClick={() => setShowModal(false)} className="flex-1 py-2 border border-border rounded-lg text-sm text-muted-foreground">{t('Cancel')}</button>
                <button onClick={handleCreate} disabled={loading || !form.name}
                  className="flex-1 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-medium disabled:opacity-50">
                  {loading ? t('Creating...') : t('Create')}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </AppShell>
  )
}
