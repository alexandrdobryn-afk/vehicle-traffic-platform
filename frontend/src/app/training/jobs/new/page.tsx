'use client'
import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import AppShell from '@/components/shared/AppShell'
import trainingApi from '@/lib/trainingApi'
import { TDataset, TArchitecture, TBaseModel, ModelType, MODEL_TYPE_LABELS } from '@/types/training'
import toast from 'react-hot-toast'
import { ChevronRight, ChevronLeft, Info } from 'lucide-react'
import { useTranslation } from '@/lib/i18n'

const STEPS = ['Model & Dataset', 'Architecture', 'Hyperparameters', 'Augmentation', 'Review']
const MODEL_TYPES: ModelType[] = ['object_detector', 'object_segmenter', 'object_classifier']
const TRAINING_MODES = [
  {
    value: 'baseline_inference',
    label: 'Baseline inference',
    desc: 'Runs the selected model over frames and creates reviewable predictions without updating weights.',
    backbone: 'Locked',
    head: 'Locked',
    output: 'Annotations only',
  },
  {
    value: 'head_finetune',
    label: 'Head fine-tuning',
    desc: 'Freezes the backbone and trains only the task head. Best first step for small verified datasets.',
    backbone: 'Frozen',
    head: 'Trainable',
    output: 'New model candidate',
  },
  {
    value: 'full_finetune',
    label: 'Full fine-tuning',
    desc: 'Trains backbone and head from the selected base model. Use when the dataset is broad enough.',
    backbone: 'Trainable',
    head: 'Trainable',
    output: 'New model candidate',
  },
  {
    value: 'tiled_training',
    label: 'Sliced / tiled training',
    desc: 'Cuts high-resolution aerial frames into overlapping tiles so small objects occupy more pixels.',
    backbone: 'Trainable',
    head: 'Trainable',
    output: 'Small-object candidate',
  },
  {
    value: 'hard_negative_training',
    label: 'Hard negative training',
    desc: 'Adds reviewed empty or rejected frames to reduce false positives on confusing backgrounds.',
    backbone: 'Trainable',
    head: 'Trainable',
    output: 'Lower FP candidate',
  },
  {
    value: 'semi_supervised',
    label: 'Semi-supervised reviewed labels',
    desc: 'Uses reviewed pseudo-labels only. Unverified auto-labels remain outside the training export.',
    backbone: 'Trainable',
    head: 'Trainable',
    output: 'Reviewed-label candidate',
  },
  {
    value: 'continual_retraining',
    label: 'Continual retraining',
    desc: 'Starts a new controlled run from a previous registry model and increments cumulative epochs.',
    backbone: 'Trainable',
    head: 'Trainable',
    output: 'Next model version',
  },
]
const OPTIMIZERS = ['SGD', 'Adam', 'AdamW']
const SCHEDULERS = ['cosine', 'linear', 'step', 'none']

export default function NewJobPage() {
  const router = useRouter()
  const { t } = useTranslation()
  const [step, setStep] = useState(0)
  const [datasets, setDatasets] = useState<TDataset[]>([])
  const [architectures, setArchitectures] = useState<TArchitecture[]>([])
  const [baseModels, setBaseModels] = useState<TBaseModel[]>([])
  const [submitting, setSubmitting] = useState(false)
  const [nameTouched, setNameTouched] = useState(false)

  const [form, setForm] = useState({
    name: '', description: '',
    model_type: 'object_detector' as ModelType,
    dataset_id: 0,
    architecture: '',
    base_model_id: '',
    training_mode: 'full_finetune',
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
    tile_config: {
      enabled: true, tile_size: 1024, overlap: 0.2,
      include_empty_tiles: true, max_empty_tile_ratio: 0.25,
      min_bbox_area: 0.00001, min_visibility: 0.2,
    },
    evaluation_policy: {
      auto_validate_after_training: true,
      min_ap_small_delta: 0.05,
      min_recall_small: 0.65,
      max_recall_small_drop: 0,
      max_fp_per_frame: 1,
      max_fp_per_frame_increase_ratio: 0.1,
      max_old_holdout_drop: 0.02,
      small_object_area_threshold: 0.01,
      error_iou_threshold: 0.5,
      error_confidence_threshold: 0.25,
      max_error_items: 300,
      min_fps: 0,
      max_latency_p95_ms: '',
    },
  })

  useEffect(() => { trainingApi.get('/datasets').then(r => setDatasets(r.data)) }, [])

  useEffect(() => {
    if (form.model_type) {
      Promise.all([
        trainingApi.get(`/architectures/${form.model_type}`),
        trainingApi.get(`/architectures/${form.model_type}/base-models`),
      ]).then(([archResponse, baseResponse]) => {
        const archs = archResponse.data as TArchitecture[]
        const bases = baseResponse.data as TBaseModel[]
        const selectedBase = bases.find(model => model.available) || bases[0]
        setArchitectures(archs)
        setBaseModels(bases)
        setForm(f => ({
          ...f,
          architecture: selectedBase?.architecture || archs[0]?.id || '',
          base_model_id: selectedBase?.id || '',
          dataset_id: 0,
        }))
      })
    }
  }, [form.model_type])

  useEffect(() => {
    if (nameTouched) return
    const base = baseModels.find(model => model.id === form.base_model_id)
    if (!base || !form.architecture) return
    const date = new Date().toISOString().slice(0, 10)
    const nextVersion = Number(base.training?.cumulative_epochs || 0) > 0
      ? `epoch-${Number(base.training?.cumulative_epochs || 0) + Number(form.hyperparams.epochs || 0)}`
      : `epoch-${form.hyperparams.epochs}`
    const typeLabel = MODEL_TYPE_LABELS[form.model_type].replace('Object ', '').toLowerCase()
    f('name', `${typeLabel} ${form.architecture} ${nextVersion} ${date}`)
  }, [baseModels, form.base_model_id, form.architecture, form.hyperparams.epochs, form.model_type, nameTouched])

  const filteredDatasets = datasets.filter(d =>
    d.model_type === form.model_type &&
    (form.training_mode === 'baseline_inference' || (d.is_frozen && Boolean(d.content_hash)))
  )

  const handleSubmit = async () => {
    if (!form.name || !form.dataset_id || !form.architecture || !form.base_model_id) {
      toast.error(t('Please fill all required fields'))
      return
    }
    setSubmitting(true)
    try {
      const r = await trainingApi.post('/jobs', {
        name: form.name,
        model_type: form.model_type,
        architecture: form.architecture,
        base_model_id: form.base_model_id,
        dataset_id: form.dataset_id,
        training_mode: form.training_mode,
        hyperparams: form.hyperparams,
        augmentation: form.augmentation,
        tile_config: form.tile_config,
        evaluation_policy: {
          ...form.evaluation_policy,
          max_latency_p95_ms: form.evaluation_policy.max_latency_p95_ms === '' ? null : Number(form.evaluation_policy.max_latency_p95_ms),
        },
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
  const tile = (key: string, val: any) => setForm(p => ({ ...p, tile_config: { ...p.tile_config, [key]: val } }))
  const policy = (key: string, val: any) => setForm(p => ({ ...p, evaluation_policy: { ...p.evaluation_policy, [key]: val } }))

  const inputCls = "w-full px-3 py-2 bg-background border border-input rounded-lg text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
  const selectCls = inputCls
  const labelCls = "block text-sm font-medium text-muted-foreground mb-1"
  const selectedMode = TRAINING_MODES.find(mode => mode.value === form.training_mode) || TRAINING_MODES[0]
  const selectedBaseModel = baseModels.find(model => model.id === form.base_model_id)
  const selectedBaseEpochs = Number(selectedBaseModel?.training?.cumulative_epochs || 0)

  const canNext = () => {
    if (step === 0) return !!form.name && !!form.dataset_id && !!form.base_model_id
    if (step === 1) return !!form.architecture
    return true
  }

  return (
    <AppShell>
      <div className="p-6 max-w-6xl">
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
            <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
              <div className="space-y-4">
                <div>
                  <label className={labelCls}>{t('Job Name')} *</label>
                  <input
                    value={form.name}
                    onChange={e => { setNameTouched(true); f('name', e.target.value) }}
                    placeholder="aerial detector yolo11s epoch-100"
                    className={inputCls}
                  />
                </div>
                <div>
                  <label className={labelCls}>{t('Model Type')}</label>
                  <select value={form.model_type} onChange={e => f('model_type', e.target.value as ModelType)} className={selectCls}>
                    {MODEL_TYPES.map(modelType => <option key={modelType} value={modelType}>{t(MODEL_TYPE_LABELS[modelType])}</option>)}
                  </select>
                </div>
                <div>
                  <label className={labelCls}>{t('Base Model')} *</label>
                  {baseModels.length === 0 ? (
                    <p className="text-sm text-amber-400">{t('No base models are available for this type')}</p>
                  ) : (
                    <select
                      value={form.base_model_id}
                      onChange={e => {
                        const base = baseModels.find(model => model.id === e.target.value)
                        setForm(p => ({
                          ...p,
                          base_model_id: e.target.value,
                          architecture: base?.architecture || p.architecture,
                        }))
                      }}
                      className={selectCls}
                    >
                      <option value="">{t('Select base model...')}</option>
                      {baseModels.map(model => (
                        <option key={model.id} value={model.id} disabled={!model.available}>
                          {model.label} ({model.source})
                        </option>
                      ))}
                    </select>
                  )}
                </div>
                <div>
                  <label className={labelCls}>{t('Training Mode')}</label>
                  <select value={form.training_mode} onChange={e => f('training_mode', e.target.value)} className={selectCls}>
                    {TRAINING_MODES.map(mode => <option key={mode.value} value={mode.value}>{t(mode.label)}</option>)}
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
              </div>
              <aside className="rounded-lg border border-border bg-background/60 p-4">
                <div className="mb-3 flex items-center gap-2">
                  <Info className="h-4 w-4 text-primary" />
                  <p className="text-sm font-semibold text-foreground">{t(selectedMode.label)}</p>
                </div>
                <p className="text-sm text-muted-foreground">{t(selectedMode.desc)}</p>
                <div className="mt-4 space-y-3 text-sm">
                  {[
                    ['Backbone', selectedMode.backbone],
                    ['Head', selectedMode.head],
                    ['Output', selectedMode.output],
                    ['Architecture', form.architecture || '-'],
                    ['Base', selectedBaseModel?.label || '-'],
                    ['Previous epochs', selectedBaseEpochs],
                  ].map(([label, value]) => (
                    <div key={String(label)} className="flex justify-between gap-3 border-t border-border/50 pt-2">
                      <span className="text-muted-foreground">{t(String(label))}</span>
                      <span className="text-right font-medium text-foreground">{t(String(value))}</span>
                    </div>
                  ))}
                </div>
              </aside>
            </div>
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
                    onChange={() => {
                      const matchingBase = baseModels.find(model => model.architecture === arch.id && model.available)
                      setForm(p => ({
                        ...p,
                        architecture: arch.id,
                        base_model_id: matchingBase?.id || '',
                      }))
                    }} className="mt-1" />
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
              <div className="col-span-2 rounded-lg border border-border p-4">
                <div className="mb-3 flex items-center justify-between">
                  <p className="text-sm font-medium text-foreground">{t('Tiled Training')}</p>
                  <label className="flex items-center gap-2 text-sm text-muted-foreground">
                    <input type="checkbox" checked={form.tile_config.enabled}
                      onChange={e => tile('enabled', e.target.checked)} />
                    {t('Enabled')}
                  </label>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className={labelCls}>{t('Tile Size')}</label>
                    <input type="number" min={320} max={4096} value={form.tile_config.tile_size}
                      onChange={e => tile('tile_size', Number(e.target.value))} className={inputCls} />
                  </div>
                  <div>
                    <label className={labelCls}>{t('Overlap')}</label>
                    <input type="number" min={0} max={0.75} step={0.05} value={form.tile_config.overlap}
                      onChange={e => tile('overlap', Number(e.target.value))} className={inputCls} />
                  </div>
                  <div>
                    <label className={labelCls}>{t('Min Object Visibility')}</label>
                    <input type="number" min={0.05} max={1} step={0.05} value={form.tile_config.min_visibility}
                      onChange={e => tile('min_visibility', Number(e.target.value))} className={inputCls} />
                  </div>
                  <label className="flex items-center gap-2 text-sm text-muted-foreground">
                    <input type="checkbox" checked={form.tile_config.include_empty_tiles}
                      onChange={e => tile('include_empty_tiles', e.target.checked)} />
                    {t('Include empty hard-negative tiles')}
                  </label>
                  <div>
                    <label className={labelCls}>{t('Max Empty Tile Ratio')}</label>
                    <input type="number" min={0} max={1} step={0.05} value={form.tile_config.max_empty_tile_ratio}
                      onChange={e => tile('max_empty_tile_ratio', Number(e.target.value))} className={inputCls} />
                  </div>
                </div>
              </div>
              <label className="col-span-2 flex items-start gap-2 rounded-lg border border-border p-3 text-sm text-muted-foreground">
                <input
                  type="checkbox"
                  checked={Boolean(form.evaluation_policy.auto_validate_after_training)}
                  onChange={e => policy('auto_validate_after_training', e.target.checked)}
                  className="mt-1"
                />
                <span>
                  <span className="block font-medium text-foreground">{t('Create validation report after training')}</span>
                  <span className="block text-xs">{t('This does not deploy the model or start another training run.')}</span>
                </span>
              </label>
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
                ['Training Mode', form.training_mode],
                ['Architecture', form.architecture],
                ['Base Model', selectedBaseModel?.label || '-'],
                ['Previous epochs', selectedBaseEpochs],
                ['Dataset', datasets.find(d => d.id === form.dataset_id)?.name || '—'],
                ['Epochs', form.hyperparams.epochs],
                ['Batch Size', form.hyperparams.batch_size],
                ['Image Size', form.hyperparams.img_size],
                ['Learning Rate', form.hyperparams.learning_rate],
                ['Optimizer', form.hyperparams.optimizer],
                ['Augmentation', form.augmentation.enabled ? 'Enabled' : 'Disabled'],
                ['Pretrained', form.hyperparams.pretrained ? 'Yes' : 'No'],
                ['Tile Size', form.tile_config.tile_size],
                ['Tile Overlap', form.tile_config.overlap],
                ['Validation report', form.evaluation_policy.auto_validate_after_training ? 'Enabled' : 'Manual only'],
              ].map(([k, v]) => (
                <div key={String(k)} className="flex justify-between py-1.5 border-b border-border/50 last:border-0">
                  <span className="text-muted-foreground">{t(String(k))}</span>
                  <span className="font-medium text-foreground">{t(String(v))}</span>
                </div>
              ))}
              <div className="rounded-lg border border-border p-3">
                <p className="mb-2 text-xs font-semibold uppercase text-muted-foreground">{t('Decision Gate')}</p>
                <div className="grid grid-cols-2 gap-3">
                  {[
                    ['Min AP-small delta', 'min_ap_small_delta'],
                    ['Min Recall-small', 'min_recall_small'],
                    ['Max recall-small drop', 'max_recall_small_drop'],
                    ['Max FP/frame', 'max_fp_per_frame'],
                    ['Max FP/frame increase', 'max_fp_per_frame_increase_ratio'],
                    ['Max old holdout drop', 'max_old_holdout_drop'],
                    ['Small object area threshold', 'small_object_area_threshold'],
                    ['Error IoU threshold', 'error_iou_threshold'],
                    ['Error confidence threshold', 'error_confidence_threshold'],
                    ['Max error items', 'max_error_items'],
                    ['Min FPS', 'min_fps'],
                    ['Max P95 latency ms', 'max_latency_p95_ms'],
                  ].map(([label, key]) => (
                    <div key={key}>
                      <label className={labelCls}>{t(label)}</label>
                      <input
                        type="number"
                        step={key === 'min_fps' || key === 'max_latency_p95_ms' || key === 'max_error_items' ? 1 : 0.01}
                        value={(form.evaluation_policy as any)[key]}
                        onChange={e => policy(key, e.target.value === '' ? '' : Number(e.target.value))}
                        className={inputCls}
                      />
                    </div>
                  ))}
                </div>
              </div>
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
