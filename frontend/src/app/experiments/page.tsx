'use client'

import { useEffect, useMemo, useState } from 'react'
import AppShell from '@/components/shared/AppShell'
import api, { apiErrorMessage } from '@/lib/api'
import trainingApi from '@/lib/trainingApi'
import { formatDateTime } from '@/lib/utils'
import { useTranslation } from '@/lib/i18n'
import { Camera, EvaluationRun } from '@/types'
import { TDataset, TModelVersion } from '@/types/training'
import {
  Activity,
  AlertTriangle,
  CheckCircle,
  ChevronDown,
  ChevronRight,
  Cpu,
  Database,
  FlaskConical,
  GitCompare,
  History,
  Layers,
  Play,
  RefreshCw,
  Timer,
  Video,
  XCircle,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

type MainTab = 'test' | 'compare' | 'history'
type TestTarget = 'model' | 'pipeline'
type InputKind = 'live_source' | 'video_replay' | 'frozen_dataset'

type RuntimeStats = {
  camera_id: number
  fps?: number
  latency_ms?: number
  inference_latency_ms?: number
  peak_memory_mb?: number
  stage_counts?: Record<string, unknown>
  stage_latency_ms?: Record<string, unknown>
  resolved_pipeline?: Record<string, unknown>
}

const INPUT_OPTIONS: Array<{ id: InputKind; label: string; icon: LucideIcon }> = [
  { id: 'live_source', label: 'Live source', icon: Activity },
  { id: 'video_replay', label: 'Video replay', icon: Video },
  { id: 'frozen_dataset', label: 'Frozen dataset', icon: Database },
]

const TAB_OPTIONS: Array<{ id: MainTab; label: string; icon: LucideIcon }> = [
  { id: 'test', label: 'Test Run', icon: FlaskConical },
  { id: 'compare', label: 'Compare', icon: GitCompare },
  { id: 'history', label: 'History', icon: History },
]

const MODULES = [
  'Detection',
  'Tracking',
  'Object Memory',
  'Classification',
  'Segmentation',
  'Super Resolution',
  'Telemetry',
  'Performance',
]

const DEFAULT_CONFIG = {
  requested_pipeline: {
    mode: 'current_runtime',
    detector: 'registry_default',
    tracker: 'source_default',
    object_memory: true,
    segmentation: 'source_default',
    super_resolution: 'source_default',
    telemetry: 'metadata',
  },
  evaluation_protocol_version: 'experiment-ui-mvp-1',
  metric_implementation_version: 'runtime-operational-v1',
}

function valueText(value: unknown) {
  if (typeof value === 'number') {
    if (Math.abs(value) <= 1) return `${(value * 100).toFixed(1)}%`
    return Number.isInteger(value) ? String(value) : value.toFixed(2)
  }
  if (value == null || value === '') return '-'
  return String(value)
}

function statusTone(status: string) {
  if (status === 'completed') return 'border-emerald-500/20 bg-emerald-500/10 text-emerald-300'
  if (status === 'failed') return 'border-red-500/20 bg-red-500/10 text-red-300'
  if (status === 'running') return 'border-sky-500/20 bg-sky-500/10 text-sky-300'
  return 'border-amber-500/20 bg-amber-500/10 text-amber-300'
}

function sourceLabel(source: Camera | undefined) {
  if (!source) return 'Current runtime'
  return `${source.name}${source.source_type === 'file' && source.source_file_name ? ` (${source.source_file_name})` : ''}`
}

function runMetric(run: EvaluationRun, key: string) {
  const metrics = run.metrics || {}
  const runtime = (metrics.runtime || {}) as Record<string, unknown>
  return runtime[key] ?? metrics[key]
}

function comparability(left: EvaluationRun | null, right: EvaluationRun | null) {
  if (!left || !right) return { label: 'Select two runs', tone: 'muted', reasons: [] as string[] }
  const reasons = []
  if (left.task_type !== right.task_type) reasons.push('different test types')
  if (left.dataset_ref !== right.dataset_ref) reasons.push('different inputs')
  const leftProtocol = left.config?.['evaluation_protocol_version']
  const rightProtocol = right.config?.['evaluation_protocol_version']
  if (leftProtocol !== rightProtocol) reasons.push('different evaluator versions')
  if (reasons.length === 0) return { label: 'Comparable', tone: 'good', reasons }
  if (reasons.length <= 2) return { label: 'Partially comparable', tone: 'warn', reasons }
  return { label: 'Not comparable', tone: 'bad', reasons }
}

export default function ExperimentsPage() {
  const { t, locale } = useTranslation()
  const [tab, setTab] = useState<MainTab>('test')
  const [runs, setRuns] = useState<EvaluationRun[]>([])
  const [models, setModels] = useState<TModelVersion[]>([])
  const [datasets, setDatasets] = useState<TDataset[]>([])
  const [sources, setSources] = useState<Camera[]>([])
  const [performance, setPerformance] = useState<RuntimeStats[]>([])
  const [target, setTarget] = useState<TestTarget>('pipeline')
  const [inputKind, setInputKind] = useState<InputKind>('live_source')
  const [selectedModelId, setSelectedModelId] = useState<number | ''>('')
  const [selectedDatasetId, setSelectedDatasetId] = useState<number | ''>('')
  const [selectedSourceId, setSelectedSourceId] = useState<number | ''>('')
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null)
  const [compareA, setCompareA] = useState<number | ''>('')
  const [compareB, setCompareB] = useState<number | ''>('')
  const [expanded, setExpanded] = useState<Record<string, boolean>>({ Detection: true, Performance: true })
  const [loading, setLoading] = useState(false)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')

  const selectedModel = models.find(model => model.id === selectedModelId)
  const selectedDataset = datasets.find(dataset => dataset.id === selectedDatasetId)
  const selectedSource = sources.find(source => source.id === selectedSourceId)
  const selectedStats = performance.find(item => item.camera_id === selectedSourceId)
  const selectedRun = runs.find(run => run.id === selectedRunId) || runs[0] || null
  const leftRun = runs.find(run => run.id === compareA) || null
  const rightRun = runs.find(run => run.id === compareB) || null
  const compareState = comparability(leftRun, rightRun)

  const frozenDatasets = useMemo(
    () => datasets.filter(dataset => dataset.is_frozen || dataset.status === 'frozen'),
    [datasets],
  )
  const selectableSources = useMemo(
    () => sources.filter(source => inputKind !== 'video_replay' || source.source_type === 'file'),
    [sources, inputKind],
  )

  const load = async () => {
    setLoading(true)
    setError('')
    try {
      const [runResponse, modelResponse, datasetResponse, cameraResponse, videoResponse, perfResponse] = await Promise.allSettled([
        api.get('/evaluations'),
        trainingApi.get('/registry'),
        trainingApi.get('/datasets'),
        api.get('/cameras'),
        api.get('/videos'),
        api.get('/analytics/performance'),
      ])
      if (runResponse.status === 'fulfilled') setRuns(runResponse.value.data)
      if (modelResponse.status === 'fulfilled') setModels(modelResponse.value.data)
      if (datasetResponse.status === 'fulfilled') setDatasets(datasetResponse.value.data)
      const loadedSources: Camera[] = []
      if (cameraResponse.status === 'fulfilled') loadedSources.push(...cameraResponse.value.data)
      if (videoResponse.status === 'fulfilled') {
        const seen = new Set(loadedSources.map(source => source.id))
        loadedSources.push(...videoResponse.value.data.filter((source: Camera) => !seen.has(source.id)))
      }
      setSources(loadedSources)
      if (perfResponse.status === 'fulfilled') setPerformance(perfResponse.value.data)
      const rejected = [runResponse, modelResponse, datasetResponse, cameraResponse, videoResponse, perfResponse].find(result => result.status === 'rejected')
      if (rejected?.status === 'rejected') setError(apiErrorMessage(rejected.reason, 'Some experiment data could not be loaded.'))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  useEffect(() => {
    if (!selectedRunId && runs[0]) setSelectedRunId(runs[0].id)
    if (!compareA && runs[0]) setCompareA(runs[0].id)
    if (!compareB && runs[1]) setCompareB(runs[1].id)
  }, [runs, selectedRunId, compareA, compareB])

  const runTest = async () => {
    setRunning(true)
    setError('')
    try {
      const hasGroundTruth = inputKind === 'frozen_dataset' && Boolean(selectedDataset?.annotation_count)
      const datasetRef = inputKind === 'frozen_dataset'
        ? `${selectedDataset?.name || 'Frozen dataset'}:${selectedDataset?.content_hash || selectedDataset?.id || 'unselected'}`
        : sourceLabel(selectedSource)
      const modelName = target === 'model'
        ? (selectedModel?.name || 'Unselected model')
        : `Pipeline: ${sourceLabel(selectedSource)}`
      const warnings = []
      if (inputKind === 'frozen_dataset') warnings.push('benchmark_runner_not_connected_yet')
      if (inputKind !== 'frozen_dataset' && !selectedStats) warnings.push('source_runtime_stats_unavailable')
      if (target === 'model' && !selectedModel) warnings.push('model_not_selected')

      const createResponse = await api.post('/evaluations', {
        name: `${target === 'model' ? 'Model' : 'Pipeline'} test - ${new Date().toLocaleString()}`,
        task_type: inputKind === 'video_replay' ? 'replay' : target === 'pipeline' ? 'pipeline' : 'profiling',
        model_name: modelName,
        dataset_ref: datasetRef,
        config: {
          ...DEFAULT_CONFIG,
          test_target: target,
          input_kind: inputKind,
          model_version_id: selectedModel?.id ?? null,
          dataset_id: selectedDataset?.id ?? null,
          source_id: selectedSource?.id ?? null,
          requested_pipeline: {
            ...DEFAULT_CONFIG.requested_pipeline,
            source_pipeline_config: selectedSource?.pipeline_config || {},
            model_version_id: selectedModel?.id ?? null,
          },
        },
      })

      const run = createResponse.data as EvaluationRun
      const executeResponse = await api.post(`/evaluations/${run.id}/execute`, {
        data: {
          has_ground_truth: hasGroundTruth,
          quality_metrics_available: false,
          warnings,
          runtime: selectedStats || {},
          stage_counts: selectedStats?.stage_counts || {},
          stage_latency_ms: selectedStats?.stage_latency_ms || {},
          requested_pipeline: {
            ...DEFAULT_CONFIG.requested_pipeline,
            source_pipeline_config: selectedSource?.pipeline_config || {},
          },
          resolved_pipeline: selectedStats?.resolved_pipeline || {},
        },
        runtime: {
          fps: Number(selectedStats?.fps || 0),
          latency_ms: Number(selectedStats?.inference_latency_ms || selectedStats?.latency_ms || 0),
          peak_memory_mb: Number(selectedStats?.peak_memory_mb || 0),
        },
      })
      const nextRun = executeResponse.data as EvaluationRun
      setRuns(current => [nextRun, ...current.filter(item => item.id !== nextRun.id)])
      setSelectedRunId(nextRun.id)
      setTab('history')
    } catch (err) {
      setError(apiErrorMessage(err, 'Could not start experiment test.'))
    } finally {
      setRunning(false)
    }
  }

  return (
    <AppShell>
      <div className="space-y-6 p-6">
        <div className="flex flex-col gap-4 pr-0 lg:flex-row lg:items-center lg:justify-between lg:pr-40">
          <div>
            <h1 className="text-2xl font-bold text-foreground">{t('Experiments')}</h1>
            <p className="text-sm text-muted-foreground">{t('Run model and pipeline tests, then compare saved results')}</p>
          </div>
          <button onClick={load} className="inline-flex items-center gap-2 rounded-lg border border-border px-3 py-2 text-sm text-muted-foreground hover:text-foreground">
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} /> {t('Refresh')}
          </button>
        </div>

        {error && <div className="rounded-lg border border-amber-500/20 bg-amber-500/10 p-3 text-sm text-amber-200">{t(error)}</div>}

        <div className="flex flex-wrap gap-2">
          {TAB_OPTIONS.map(({ id, label, icon: Icon }) => (
            <button key={id} onClick={() => setTab(id)} className={`inline-flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium ${tab === id ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground hover:text-foreground'}`}>
              <Icon className="h-4 w-4" /> {t(label)}
            </button>
          ))}
        </div>

        {tab === 'test' && (
          <div className="grid gap-6 xl:grid-cols-[minmax(0,520px)_1fr]">
            <section className="rounded-xl border border-border bg-card p-5">
              <h2 className="mb-4 font-semibold text-foreground">{t('New test')}</h2>
              <div className="space-y-5">
                <div>
                  <p className="mb-2 text-xs font-medium uppercase text-muted-foreground">{t('What to test')}</p>
                  <div className="grid grid-cols-2 gap-2">
                    {(['pipeline', 'model'] as TestTarget[]).map(option => (
                      <button key={option} onClick={() => setTarget(option)} className={`rounded-lg border px-3 py-2 text-sm capitalize ${target === option ? 'border-primary bg-primary/10 text-primary' : 'border-border bg-background text-muted-foreground hover:text-foreground'}`}>
                        {t(option)}
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <p className="mb-2 text-xs font-medium uppercase text-muted-foreground">{t('Input')}</p>
                  <div className="grid gap-2 sm:grid-cols-3">
                    {INPUT_OPTIONS.map(option => {
                      const Icon = option.icon
                      return (
                        <button key={option.id} onClick={() => setInputKind(option.id)} className={`rounded-lg border p-3 text-left text-xs ${inputKind === option.id ? 'border-primary bg-primary/10 text-primary' : 'border-border bg-background text-muted-foreground hover:text-foreground'}`}>
                          <Icon className="mb-2 h-4 w-4" />
                          {t(option.label)}
                        </button>
                      )
                    })}
                  </div>
                </div>

                {target === 'model' && (
                  <label className="block text-sm text-muted-foreground">
                    {t('Model')}
                    <select value={selectedModelId} onChange={event => setSelectedModelId(event.target.value ? Number(event.target.value) : '')} className="mt-1 w-full rounded-lg border border-input bg-background px-3 py-2 text-sm text-foreground">
                      <option value="">{t('Select model')}</option>
                      {models.map(model => <option key={model.id} value={model.id}>{model.name} ({model.model_type})</option>)}
                    </select>
                  </label>
                )}

                {inputKind === 'frozen_dataset' ? (
                  <label className="block text-sm text-muted-foreground">
                    {t('Frozen dataset')}
                    <select value={selectedDatasetId} onChange={event => setSelectedDatasetId(event.target.value ? Number(event.target.value) : '')} className="mt-1 w-full rounded-lg border border-input bg-background px-3 py-2 text-sm text-foreground">
                      <option value="">{t('Select dataset')}</option>
                      {frozenDatasets.map(dataset => <option key={dataset.id} value={dataset.id}>{dataset.name} v{dataset.version}</option>)}
                    </select>
                  </label>
                ) : (
                  <label className="block text-sm text-muted-foreground">
                    {t(inputKind === 'video_replay' ? 'Video source' : 'Live source')}
                    <select value={selectedSourceId} onChange={event => setSelectedSourceId(event.target.value ? Number(event.target.value) : '')} className="mt-1 w-full rounded-lg border border-input bg-background px-3 py-2 text-sm text-foreground">
                      <option value="">{t('Current runtime / no source')}</option>
                      {selectableSources.map(source => <option key={source.id} value={source.id}>{sourceLabel(source)}</option>)}
                    </select>
                  </label>
                )}

                <button onClick={runTest} disabled={running} className="inline-flex w-full items-center justify-center gap-2 rounded-lg bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground disabled:opacity-50">
                  <Play className="h-4 w-4" /> {running ? t('Running...') : t('Run test')}
                </button>
              </div>
            </section>

            <section className="rounded-xl border border-border bg-card p-5">
              <h2 className="mb-4 font-semibold text-foreground">{t('Test preview')}</h2>
              <div className="grid gap-3 md:grid-cols-2">
                <InfoTile icon={Layers} label={t('Target')} value={target === 'model' ? selectedModel?.name || t('No model selected') : t('Pipeline snapshot')} />
                <InfoTile icon={Database} label={t('Input')} value={inputKind === 'frozen_dataset' ? selectedDataset?.name || t('No dataset selected') : sourceLabel(selectedSource)} />
                <InfoTile icon={Cpu} label={t('Runtime')} value={selectedStats ? `${(selectedStats.fps || 0).toFixed(1)} FPS` : t('No active stats')} />
                <InfoTile icon={Timer} label={t('Metric type')} value={inputKind === 'frozen_dataset' ? t('GT quality request') : t('Runtime indicators')} />
              </div>
              <div className="mt-5 rounded-lg border border-border bg-background p-4 text-sm text-muted-foreground">
                {inputKind === 'frozen_dataset'
                  ? t('This will save a benchmark request. Full quality metrics need the backend dataset runner to execute predictions against Ground Truth.')
                  : t('This records real runtime indicators from the selected source when it is running: FPS, latency, stage counters and resolved pipeline.')}
              </div>
            </section>
          </div>
        )}

        {tab === 'compare' && (
          <section className="rounded-xl border border-border bg-card p-5">
            <div className="mb-4 flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
              <h2 className="font-semibold text-foreground">{t('Compare saved runs')}</h2>
              <span className={`rounded-lg border px-3 py-1.5 text-xs font-medium ${compareState.tone === 'good' ? 'border-emerald-500/20 bg-emerald-500/10 text-emerald-300' : compareState.tone === 'bad' ? 'border-red-500/20 bg-red-500/10 text-red-300' : compareState.tone === 'warn' ? 'border-amber-500/20 bg-amber-500/10 text-amber-300' : 'border-border bg-muted text-muted-foreground'}`}>
                {t(compareState.label)}
              </span>
            </div>
            <div className="mb-5 grid gap-3 md:grid-cols-2">
              <RunSelect label="Run A" value={compareA} runs={runs} onChange={setCompareA} />
              <RunSelect label="Run B" value={compareB} runs={runs} onChange={setCompareB} />
            </div>
            {compareState.reasons.length > 0 && (
              <div className="mb-5 rounded-lg border border-amber-500/20 bg-amber-500/10 p-3 text-sm text-amber-200">
                {t('Comparison warning')}: {compareState.reasons.map(reason => t(reason)).join(', ')}
              </div>
            )}
            <ComparisonTable left={leftRun} right={rightRun} />
          </section>
        )}

        {tab === 'history' && (
          <div className="grid gap-6 xl:grid-cols-[360px_1fr]">
            <section className="space-y-3">
              {runs.length ? runs.map(run => (
                <button key={run.id} onClick={() => setSelectedRunId(run.id)} className={`w-full rounded-xl border p-4 text-left transition-colors ${selectedRun?.id === run.id ? 'border-primary bg-primary/5' : 'border-border bg-card hover:border-primary/30'}`}>
                  <div className="mb-2 flex items-center justify-between gap-3">
                    <span className={`rounded border px-2 py-1 text-xs font-medium ${statusTone(run.status)}`}>{t(run.status)}</span>
                    <span className="text-xs text-muted-foreground">#{run.id}</span>
                  </div>
                  <p className="truncate text-sm font-medium text-foreground">{run.name}</p>
                  <p className="mt-1 text-xs text-muted-foreground">{formatDateTime(run.created_at, locale)}</p>
                </button>
              )) : (
                <div className="rounded-xl border border-dashed border-border bg-card p-8 text-center text-sm text-muted-foreground">
                  {t('No experiment runs yet')}
                </div>
              )}
            </section>

            <section className="rounded-xl border border-border bg-card p-5">
              {selectedRun ? (
                <RunDetails run={selectedRun} expanded={expanded} setExpanded={setExpanded} />
              ) : (
                <p className="py-20 text-center text-sm text-muted-foreground">{t('Select a run')}</p>
              )}
            </section>
          </div>
        )}
      </div>
    </AppShell>
  )
}

function InfoTile({ icon: Icon, label, value }: { icon: LucideIcon; label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border bg-background p-3">
      <div className="mb-2 flex items-center gap-2 text-xs text-muted-foreground">
        <Icon className="h-4 w-4" /> {label}
      </div>
      <p className="truncate text-sm font-medium text-foreground">{value}</p>
    </div>
  )
}

function RunSelect({ label, value, runs, onChange }: { label: string; value: number | ''; runs: EvaluationRun[]; onChange: (value: number | '') => void }) {
  const { t, locale } = useTranslation()
  return (
    <label className="block text-sm text-muted-foreground">
      {t(label)}
      <select value={value} onChange={event => onChange(event.target.value ? Number(event.target.value) : '')} className="mt-1 w-full rounded-lg border border-input bg-background px-3 py-2 text-sm text-foreground">
        <option value="">{t('Select run')}</option>
        {runs.map(run => <option key={run.id} value={run.id}>{run.name} - {formatDateTime(run.created_at, locale)}</option>)}
      </select>
    </label>
  )
}

function ComparisonTable({ left, right }: { left: EvaluationRun | null; right: EvaluationRun | null }) {
  const { t } = useTranslation()
  const rows = [
    ['Status', left?.status, right?.status],
    ['Input', left?.dataset_ref, right?.dataset_ref],
    ['FPS', left ? runMetric(left, 'fps') : null, right ? runMetric(right, 'fps') : null],
    ['Latency', left ? runMetric(left, 'latency_ms') : null, right ? runMetric(right, 'latency_ms') : null],
    ['Frames', left ? runMetric(left, 'frames_processed') : null, right ? runMetric(right, 'frames_processed') : null],
    ['Objects', left ? runMetric(left, 'objects_detected') : null, right ? runMetric(right, 'objects_detected') : null],
    ['Segments', left ? runMetric(left, 'segments_detected') : null, right ? runMetric(right, 'segments_detected') : null],
  ]
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="text-xs text-muted-foreground">
          <tr>
            <th className="pb-2 text-left font-medium">{t('Metric')}</th>
            <th className="pb-2 text-left font-medium">{t('Run A')}</th>
            <th className="pb-2 text-left font-medium">{t('Run B')}</th>
            <th className="pb-2 text-left font-medium">{t('Delta')}</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {rows.map(([label, a, b]) => {
            const delta = typeof a === 'number' && typeof b === 'number' ? b - a : null
            return (
              <tr key={label as string}>
                <td className="py-3 text-foreground">{t(label as string)}</td>
                <td className="py-3 text-muted-foreground">{valueText(a)}</td>
                <td className="py-3 text-muted-foreground">{valueText(b)}</td>
                <td className={delta == null ? 'py-3 text-muted-foreground' : `py-3 ${delta >= 0 ? 'text-emerald-300' : 'text-red-300'}`}>{delta == null ? '-' : valueText(delta)}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function RunDetails({ run, expanded, setExpanded }: { run: EvaluationRun; expanded: Record<string, boolean>; setExpanded: (value: Record<string, boolean>) => void }) {
  const { t } = useTranslation()
  const metrics = run.metrics || {}
  const warnings = (metrics.warnings || []) as string[]
  const hasQuality = metrics.quality_metrics_available === true
  return (
    <div className="space-y-5">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <div className="mb-2 flex items-center gap-2">
            {run.status === 'completed' ? <CheckCircle className="h-4 w-4 text-emerald-400" /> : run.status === 'failed' ? <XCircle className="h-4 w-4 text-red-400" /> : <AlertTriangle className="h-4 w-4 text-amber-400" />}
            <span className={`rounded border px-2 py-1 text-xs font-medium ${statusTone(run.status)}`}>{t(run.status)}</span>
          </div>
          <h2 className="text-lg font-semibold text-foreground">{run.name}</h2>
          <p className="mt-1 text-sm text-muted-foreground">{run.model_name}</p>
        </div>
        <span className={`rounded-lg border px-3 py-1.5 text-xs font-medium ${hasQuality ? 'border-emerald-500/20 bg-emerald-500/10 text-emerald-300' : 'border-sky-500/20 bg-sky-500/10 text-sky-300'}`}>
          {t(hasQuality ? 'GT quality metrics' : 'Runtime indicators')}
        </span>
      </div>

      {warnings.length > 0 && (
        <div className="rounded-lg border border-amber-500/20 bg-amber-500/10 p-3 text-sm text-amber-200">
          {warnings.map(item => t(item)).join(', ')}
        </div>
      )}

      <div className="grid gap-3 md:grid-cols-4">
        <InfoTile icon={Database} label={t('Input')} value={run.dataset_ref} />
        <InfoTile icon={Activity} label={t('FPS')} value={valueText(runMetric(run, 'fps'))} />
        <InfoTile icon={Timer} label={t('Latency')} value={valueText(runMetric(run, 'latency_ms'))} />
        <InfoTile icon={Layers} label={t('Objects')} value={valueText(metrics.objects_detected)} />
      </div>

      <div className="space-y-3">
        {MODULES.map(module => (
          <section key={module} className="rounded-lg border border-border bg-background">
            <button onClick={() => setExpanded({ ...expanded, [module]: !expanded[module] })} className="flex w-full items-center justify-between px-4 py-3 text-left text-sm font-medium text-foreground">
              <span>{t(module)}</span>
              {expanded[module] ? <ChevronDown className="h-4 w-4 text-muted-foreground" /> : <ChevronRight className="h-4 w-4 text-muted-foreground" />}
            </button>
            {expanded[module] && <ModuleMetrics module={module} metrics={metrics} config={run.config} />}
          </section>
        ))}
      </div>
    </div>
  )
}

function ModuleMetrics({ module, metrics, config }: { module: string; metrics: Record<string, unknown>; config: Record<string, unknown> }) {
  const { t } = useTranslation()
  const runtime = (metrics.runtime || {}) as Record<string, unknown>
  const stageLatency = (metrics.stage_latency_ms || {}) as Record<string, unknown>
  const stageCounts = (metrics.stage_counts || {}) as Record<string, unknown>
  const rows: Array<[string, unknown]> = module === 'Performance'
    ? [['FPS', runtime.fps], ['Latency', runtime.latency_ms], ['Peak memory MB', runtime.peak_memory_mb]]
    : module === 'Detection'
      ? [['Objects detected', metrics.objects_detected], ['Frames processed', metrics.frames_processed], ['Object detection latency', stageLatency.object_detection]]
      : module === 'Tracking'
        ? [['Tracking latency', stageLatency.tracking], ['Kalman predicted objects', stageCounts.kalman_predicted_objects]]
        : module === 'Object Memory'
          ? [['Merged tracks', metrics.tracks_merged], ['Duplicates suppressed', metrics.duplicates_suppressed], ['Object memory latency', stageLatency.object_memory]]
          : module === 'Segmentation'
            ? [['Segments detected', metrics.segments_detected], ['Segmentation latency', stageLatency.segmentation]]
            : module === 'Classification'
              ? [['Objects classified', metrics.objects_classified], ['Classification latency', stageLatency.classification]]
              : module === 'Super Resolution'
                ? [['Enhanced crops', stageCounts.super_resolution_enhanced], ['SR latency', stageLatency.super_resolution]]
                : [['Resolved pipeline', JSON.stringify(metrics.resolved_pipeline || config.requested_pipeline || {}, null, 2)]]
  return (
    <div className="border-t border-border px-4 py-3">
      <div className="grid gap-2 md:grid-cols-2">
        {rows.map(([label, value]) => (
          <div key={label} className="rounded-lg bg-muted/40 p-3">
            <p className="text-xs text-muted-foreground">{t(label)}</p>
            <p className="mt-1 break-words text-sm font-medium text-foreground">{valueText(value)}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
