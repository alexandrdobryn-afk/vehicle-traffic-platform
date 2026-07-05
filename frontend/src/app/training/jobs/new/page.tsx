'use client'
import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import AppShell from '@/components/shared/AppShell'
import trainingApi from '@/lib/trainingApi'
import { TDataset, TArchitecture, ModelType, MODEL_TYPE_LABELS } from '@/types/training'
import toast from 'react-hot-toast'
import { ChevronRight, ChevronLeft } from 'lucide-react'
import { useTranslation } from '@/lib/i18n'

const STEPS = ['Model & Dataset', 'Architecture', 'Hyperparameters', 'Augmentation', 'Review']
const MODEL_TYPES: ModelType[] = ['vehicle_detector', 'plate_detector', 'vehicle_segmenter', 'plate_segmenter', 'color_classifier', 'ocr']
const OPTIMIZERS = ['SGD', 'Adam', 'AdamW']
const SCHEDULERS = ['cosine', 'linear', 'step', 'none']

export default function NewJobPage() {
  const router = useRouter()
  const { t } = useTranslation()
  const [step, setStep] = useState(0)
  const [datasets, setDatasets] = useState<TDataset[]>([])
  const [architectures, setArchitectures] = useState<TArchitecture[]>([])
  const [submitting, setSubmitting] = useState(false)

  const [form, setForm] = useState({
    name: '', description: '',
    model_type: 'vehicle_detector' as ModelType,
    dataset_id: 0,
    architecture: '',
    hyperparams: {
      epochs: 100, batch_size: 16, img_size: 640,
      learning_rate: 0.01, weight_decay: 0.0005,
      optimizer: 'SGD', scheduler: 'cosine',
      warmup_epochs: 3, patience: 50,
      device: 'auto', workers: 4,
      pretrained: true, half: false,
    },
    augmentation: {
      enabled: true, rotation: 10,
      scale_min: 0.8, scale_max: 1.2,
      brightness: 0.2, contrast: 0.2,
      saturation: 0.2, hue: 0.05,
      noise: 0.01, blur: 0.1,
      motion_blur: true, mosaic: 0.5,
      mixup: 0.1, random_crop: true,
      horizontal_flip: 0.5,
    },
  })

  useEffect(() => { trainingApi.get('/datasets').then(r => setDatasets(r.data)) }, [])

  useEffect(() => {
    if (form.model_type) {
      trainingApi.get(`/architectures/${form.model_type}`)
        .then(r => { setArchitectures(r.data); if (r.data[0]) setForm(f => ({ ...f, architecture: r.data[0].id })) })
    }
  }, [form.model_type])

  const filteredDatasets = datasets.filter(d => d.model_type === form.model_type)

  const handleSubmit = async () => {
    if (!form.name || !form.dataset_id || !form.architecture) {
      toast.error(t('Please fill all required fields'))
      return
    }
    setSubmitting(true)
    try {
      const r = await trainingApi.post('/jobs', {
        name: form.name,
        model_type: form.model_type,
        architecture: form.architecture,
        dataset_id: form.dataset_id,
        hyperparams: form.hyperparams,
        augmentation: form.augmentation,
      })
      toast.success(t('Training job created!'))
      router.push(`/training/jobs/${r.data.id}`)
    } catch (e: any) {
      toast.error(e.response?.data?.detail || t('Error creating job'))
    } finally { setSubmitting(false) }
  }

  const f = (key: string, val: any) => setForm(p => ({ ...p, [key]: val }))
  const hp = (key: string, val: any) => setForm(p => ({ ...p, hyperparams: { ...p.hyperparams, [key]: val } }))
  const aug = (key: string, val: any) => setForm(p => ({ ...p, augmentation: { ...p.augmentation, [key]: val } }))

  const inputCls = "w-full px-3 py-2 bg-background border border-input rounded-lg text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
  const selectCls = inputCls
  const labelCls = "block text-sm font-medium text-muted-foreground mb-1"

  const canNext = () => {
    if (step === 0) return !!form.name && !!form.dataset_id
    if (step === 1) return !!form.architecture
    return true
  }

  return (
    <AppShell>
      <div className="p-6 max-w-2xl">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-foreground">{t('New Training Job')}</h1>
          <p className="text-muted-foreground text-sm">{t('Configure and launch a model training run')}</p>
        </div>

        {/* Steps */}
        <div className="flex items-center gap-1 mb-8">
          {STEPS.map((s, i) => (
            <div key={s} className="flex items-center gap-1">
              <div className={`flex items-center justify-center w-6 h-6 rounded-full text-xs font-bold transition-colors ${
                i < step ? 'bg-emerald-500 text-white' :
                i === step ? 'bg-primary text-primary-foreground' :
                'bg-muted text-muted-foreground'
              }`}>{i < step ? '✓' : i + 1}</div>
              {i < STEPS.length - 1 && <div className={`w-6 h-0.5 ${i < step ? 'bg-emerald-500' : 'bg-muted'}`} />}
            </div>
          ))}
          <span className="ml-2 text-sm text-muted-foreground">{t(STEPS[step])}</span>
        </div>

        {/* Step content */}
        <div className="bg-card border border-border rounded-xl p-6 space-y-4">
          {/* Step 0: Model & Dataset */}
          {step === 0 && (
            <>
              <div>
                <label className={labelCls}>{t('Job Name')} *</label>
                <input value={form.name} onChange={e => f('name', e.target.value)}
                  placeholder="YOLO11n Vehicle Detector v2" className={inputCls} />
              </div>
              <div>
                <label className={labelCls}>{t('Model Type')}</label>
                <select value={form.model_type} onChange={e => f('model_type', e.target.value as ModelType)} className={selectCls}>
                  {MODEL_TYPES.map(modelType => <option key={modelType} value={modelType}>{t(MODEL_TYPE_LABELS[modelType])}</option>)}
                </select>
              </div>
              <div>
                <label className={labelCls}>{t('Dataset')} *</label>
                {filteredDatasets.length === 0 ? (
                  <p className="text-sm text-amber-400">
                    {t('No datasets for {type}. Create one first.', { type: t(MODEL_TYPE_LABELS[form.model_type]) })}
                  </p>
                ) : (
                  <select value={form.dataset_id} onChange={e => f('dataset_id', parseInt(e.target.value))} className={selectCls}>
                    <option value={0}>{t('Select dataset...')}</option>
                    {filteredDatasets.map(d => (
                      <option key={d.id} value={d.id}>
                        {d.name} ({t('{count} images', { count: d.image_count })})
                      </option>
                    ))}
                  </select>
                )}
              </div>
            </>
          )}

          {/* Step 1: Architecture */}
          {step === 1 && (
            <div className="space-y-3">
              {architectures.map(arch => (
                <label key={arch.id}
                  className={`flex items-start gap-3 p-4 rounded-xl border cursor-pointer transition-colors ${
                    form.architecture === arch.id ? 'border-primary bg-primary/5' : 'border-border hover:border-muted-foreground'
                  }`}>
                  <input type="radio" name="arch" value={arch.id}
                    checked={form.architecture === arch.id}
                    onChange={() => f('architecture', arch.id)} className="mt-1" />
                  <div>
                    <p className="font-medium text-foreground">{arch.name}</p>
                    <p className="text-xs text-muted-foreground">{t(arch.desc)}</p>
                    <p className="text-xs text-muted-foreground mt-0.5">{t('Params')}: {arch.params}</p>
                  </div>
                </label>
              ))}
            </div>
          )}

          {/* Step 2: Hyperparameters */}
          {step === 2 && (
            <div className="grid grid-cols-2 gap-4">
              {[
                { label: 'Epochs', key: 'epochs', type: 'number', min: 1, max: 1000 },
                { label: 'Batch Size', key: 'batch_size', type: 'number', min: 1, max: 128 },
                { label: 'Image Size', key: 'img_size', type: 'number', min: 320, max: 1280 },
                { label: 'Learning Rate', key: 'learning_rate', type: 'number', step: 0.001, min: 0 },
                { label: 'Weight Decay', key: 'weight_decay', type: 'number', step: 0.0001, min: 0 },
                { label: 'Warmup Epochs', key: 'warmup_epochs', type: 'number', min: 0, max: 20 },
                { label: 'Patience (early stop)', key: 'patience', type: 'number', min: 1 },
                { label: 'Workers', key: 'workers', type: 'number', min: 0, max: 16 },
              ].map(({ label, key, ...rest }) => (
                <div key={key}>
                  <label className={labelCls}>{t(label)}</label>
                  <input {...rest} value={(form.hyperparams as any)[key]}
                    onChange={e => hp(key, rest.type === 'number' ? parseFloat(e.target.value) : e.target.value)}
                    className={inputCls} />
                </div>
              ))}
              <div>
                <label className={labelCls}>{t('Optimizer')}</label>
                <select value={form.hyperparams.optimizer} onChange={e => hp('optimizer', e.target.value)} className={selectCls}>
                  {OPTIMIZERS.map(o => <option key={o}>{o}</option>)}
                </select>
              </div>
              <div>
                <label className={labelCls}>{t('Scheduler')}</label>
                <select value={form.hyperparams.scheduler} onChange={e => hp('scheduler', e.target.value)} className={selectCls}>
                  {SCHEDULERS.map(s => <option key={s}>{t(s)}</option>)}
                </select>
              </div>
              <div className="col-span-2 flex gap-6">
                {[{ k: 'pretrained', l: 'Use pretrained weights' }, { k: 'half', l: 'FP16 precision (GPU only)' }].map(({ k, l }) => (
                  <label key={k} className="flex items-center gap-2 cursor-pointer">
                    <input type="checkbox" checked={(form.hyperparams as any)[k]}
                      onChange={e => hp(k, e.target.checked)} className="rounded" />
                    <span className="text-sm text-muted-foreground">{t(l)}</span>
                  </label>
                ))}
              </div>
            </div>
          )}

          {/* Step 3: Augmentation */}
          {step === 3 && (
            <div className="space-y-4">
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" checked={form.augmentation.enabled}
                  onChange={e => aug('enabled', e.target.checked)} className="rounded" />
                <span className="text-sm font-medium text-foreground">{t('Enable Augmentation')}</span>
              </label>
              {form.augmentation.enabled && (
                <div className="grid grid-cols-2 gap-4 pt-2">
                  {[
                    { l: 'Rotation (°)', k: 'rotation', max: 180 },
                    { l: 'Brightness', k: 'brightness', max: 1, step: 0.05 },
                    { l: 'Contrast', k: 'contrast', max: 1, step: 0.05 },
                    { l: 'Saturation', k: 'saturation', max: 1, step: 0.05 },
                    { l: 'Hue', k: 'hue', max: 0.5, step: 0.01 },
                    { l: 'Blur prob', k: 'blur', max: 1, step: 0.05 },
                    { l: 'Mosaic prob', k: 'mosaic', max: 1, step: 0.1 },
                    { l: 'Mixup prob', k: 'mixup', max: 1, step: 0.05 },
                    { l: 'H-Flip prob', k: 'horizontal_flip', max: 1, step: 0.1 },
                  ].map(({ l, k, max, step: st }) => (
                    <div key={k}>
                      <label className={labelCls}>{t(l)}: {(form.augmentation as any)[k]}</label>
                      <input type="range" min={0} max={max || 360} step={st || 1}
                        value={(form.augmentation as any)[k]}
                        onChange={e => aug(k, parseFloat(e.target.value))}
                        className="w-full" />
                    </div>
                  ))}
                  {[{ k: 'motion_blur', l: 'Motion Blur' }, { k: 'random_crop', l: 'Random Crop' }].map(({ k, l }) => (
                    <label key={k} className="flex items-center gap-2">
                      <input type="checkbox" checked={(form.augmentation as any)[k]}
                        onChange={e => aug(k, e.target.checked)} className="rounded" />
                      <span className="text-sm text-muted-foreground">{t(l)}</span>
                    </label>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Step 4: Review */}
          {step === 4 && (
            <div className="space-y-3 text-sm">
              {[
                ['Job Name', form.name],
                ['Model Type', MODEL_TYPE_LABELS[form.model_type]],
                ['Architecture', form.architecture],
                ['Dataset', datasets.find(d => d.id === form.dataset_id)?.name || '—'],
                ['Epochs', form.hyperparams.epochs],
                ['Batch Size', form.hyperparams.batch_size],
                ['Image Size', form.hyperparams.img_size],
                ['Learning Rate', form.hyperparams.learning_rate],
                ['Optimizer', form.hyperparams.optimizer],
                ['Augmentation', form.augmentation.enabled ? 'Enabled' : 'Disabled'],
                ['Pretrained', form.hyperparams.pretrained ? 'Yes' : 'No'],
              ].map(([k, v]) => (
                <div key={String(k)} className="flex justify-between py-1.5 border-b border-border/50 last:border-0">
                  <span className="text-muted-foreground">{t(String(k))}</span>
                  <span className="font-medium text-foreground">{t(String(v))}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Navigation */}
        <div className="flex justify-between mt-6">
          <button onClick={() => setStep(s => s - 1)} disabled={step === 0}
            className="flex items-center gap-2 px-4 py-2 border border-border rounded-lg text-sm text-muted-foreground hover:text-foreground disabled:opacity-30">
            <ChevronLeft className="w-4 h-4" /> {t('Back')}
          </button>
          {step < STEPS.length - 1 ? (
            <button onClick={() => setStep(s => s + 1)} disabled={!canNext()}
              className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:bg-primary/90 disabled:opacity-50">
              {t('Next')} <ChevronRight className="w-4 h-4" />
            </button>
          ) : (
            <button onClick={handleSubmit} disabled={submitting}
              className="px-6 py-2 bg-emerald-600 text-white rounded-lg text-sm font-medium hover:bg-emerald-700 disabled:opacity-50">
              {submitting ? t('Launching...') : `🚀 ${t('Launch Training')}`}
            </button>
          )}
        </div>
      </div>
    </AppShell>
  )
}
