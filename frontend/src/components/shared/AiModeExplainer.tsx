'use client'

import { useEffect, useMemo, useState } from 'react'
import {
  ArrowDown, Boxes, GitBranch, ScanSearch,
  SlidersHorizontal, Sparkles, SplitSquareVertical, Waypoints,
} from 'lucide-react'

import api from '@/lib/api'
import { AI_MODE_PROFILES, AiModelRole, MODEL_CHOICE_META, RuntimeMode } from '@/lib/aiModeProfiles'
import { useTranslation } from '@/lib/i18n'
import { AIMode, PipelineMode } from '@/types'

const ROLE_META: Record<AiModelRole, { label: string; icon: typeof Boxes }> = {
  detection: { label: 'Object detector', icon: ScanSearch },
  tracking: { label: 'Tracker', icon: GitBranch },
  prediction: { label: 'Kalman prediction', icon: Waypoints },
  classification: { label: 'Object classification', icon: Boxes },
  segmentation: { label: 'Instance segmentation', icon: SplitSquareVertical },
}

type Config = Record<string, any>
type RuntimeOption = { name: string; format: string; available: boolean }
type RuntimeOptions = Record<string, RuntimeOption[]>

function enabledLabel(value: unknown) {
  return value ? 'Enabled' : 'Disabled'
}

function selectedModelName(config: Config | undefined, key: string, fallback: string) {
  const selected = config?.[key]
  return typeof selected === 'string' && selected ? selected : fallback
}

export default function AiModeExplainer({
  mode,
  pipelineMode = 'automatic',
  pipelineConfig = {},
}: {
  mode: AIMode
  pipelineMode?: PipelineMode
  pipelineConfig?: Config
}) {
  const { t } = useTranslation()
  const [runtimeModes, setRuntimeModes] = useState<RuntimeMode[]>([])
  const [runtimeOptions, setRuntimeOptions] = useState<RuntimeOptions>({})
  const profile = AI_MODE_PROFILES[mode]
  const selectedDetector = selectedModelName(pipelineConfig, 'object_detector', profile.detectorChoice)
  const selectedVerifier = selectedModelName(pipelineConfig, 'verifier_detector', 'rfdetr_medium_object')
  const selectedTracker = selectedModelName(pipelineConfig, 'tracker', profile.trackerChoice)
  const detectorMeta = MODEL_CHOICE_META[selectedDetector]
  const verifierMeta = MODEL_CHOICE_META[selectedVerifier]
  const trackerMeta = MODEL_CHOICE_META[selectedTracker]
  const targetClasses = Array.isArray(pipelineConfig.target_classes) ? pipelineConfig.target_classes : []
  const aerial = pipelineConfig.aerial || {}
  const kalman = pipelineConfig.kalman_prediction || {}
  const objectMemory = pipelineConfig.object_memory || {}
  const reid = pipelineConfig.reid || {}
  const classification = pipelineConfig.classification || {}
  const segmentation = pipelineConfig.segmentation || {}
  const ocr = pipelineConfig.ocr || {}
  const geo = pipelineConfig.geo || {}
  const superResolution = pipelineConfig.super_resolution || {}

  useEffect(() => {
    let active = true
    api.get('/health/ai-modes')
      .then((response) => {
        if (!active) return
        setRuntimeModes(response.data.modes || [])
        setRuntimeOptions({
          object_detector: response.data.manual_options?.object_detectors || [],
          verifier_detector: response.data.manual_options?.verifier_detectors || [],
          tracker: response.data.manual_options?.trackers || [],
        })
      })
      .catch(() => {})
    return () => { active = false }
  }, [])

  const runtime = useMemo(
    () => runtimeModes.find((item) => item.id === mode),
    [mode, runtimeModes],
  )
  const missingOptionalModels = useMemo(
    () => runtime?.models.filter((model) => ['classification', 'segmentation'].includes(model.role) && !model.available) || [],
    [runtime],
  )
  const detectorRuntime = runtimeOptions.object_detector?.find((item) => item.name === selectedDetector)
  const verifierRuntime = runtimeOptions.verifier_detector?.find((item) => item.name === selectedVerifier)
  const trackerRuntime = runtimeOptions.tracker?.find((item) => item.name === selectedTracker)
  const selectedRuntimeModels = [
    {
      role: 'detection' as AiModelRole,
      name: detectorMeta?.label || selectedDetector,
      format: detectorRuntime?.format || detectorMeta?.family || profile.architecture,
      available: detectorRuntime?.available ?? Boolean(detectorMeta),
    },
    {
      role: 'tracking' as AiModelRole,
      name: trackerMeta?.label || selectedTracker,
      format: trackerRuntime?.format || trackerMeta?.family || 'algorithm',
      available: trackerRuntime?.available ?? Boolean(trackerMeta),
    },
    ...(runtime?.models.filter((model) => !['detection', 'tracking'].includes(model.role)) || []),
  ]
  const enabledModules = [
    objectMemory.enabled && 'Object Memory',
    reid.enabled && 'Embedding / ReID',
    kalman.enabled !== false && 'Kalman prediction',
    classification.enabled && 'Object classification',
    segmentation.enabled && 'Instance segmentation',
    ocr.enabled && 'OCR text recognition',
    geo.enabled && 'Geo telemetry',
    superResolution.enabled && 'Super resolution',
  ].filter(Boolean) as string[]
  const runtimeWarnings = [
    detectorRuntime && !detectorRuntime.available ? 'Selected detector is not installed' : null,
    verifierRuntime && mode === 'hybrid' && !verifierRuntime.available ? 'Selected verifier is not installed' : null,
    trackerRuntime && !trackerRuntime.available ? 'Selected tracker is not installed' : null,
  ].filter(Boolean) as string[]
  const livePipeline = [
    {
      title: 'Prepare input',
      detail: aerial.enabled
        ? t('The frame is split into overlapping tiles so small aerial objects stay visible.')
        : t('The frame is processed without tiling; this is faster but can lose small distant objects.'),
    },
    {
      title: 'Run selected detector',
      detail: `${t(detectorMeta?.label || selectedDetector)}${detectorRuntime && !detectorRuntime.available ? ` - ${t('not installed')}` : ''}.`,
    },
    ...(mode === 'hybrid' ? [{
      title: 'Verify difficult detections',
      detail: `${t(verifierMeta?.label || selectedVerifier)}${verifierRuntime && !verifierRuntime.available ? ` - ${t('not installed')}` : ''}.`,
    }] : []),
    {
      title: 'Track detections',
      detail: t('{tracker} links boxes between frames.', { tracker: t(trackerMeta?.label || selectedTracker) }),
    },
    ...(kalman.enabled !== false ? [{
      title: 'Smooth short gaps',
      detail: t('Kalman prediction keeps object IDs stable when the detector misses one or two frames.'),
    }] : []),
    ...(objectMemory.enabled ? [{
      title: 'Merge duplicate object IDs',
      detail: reid.enabled
        ? t('Object Memory compares geometry, motion and embedding similarity before writing one canonical object card.')
        : t('Object Memory compares geometry and motion to suppress duplicated report rows after tracker ID resets.'),
    }] : []),
    ...(enabledModules.some((name) => !['Object Memory', 'Embedding / ReID', 'Kalman prediction'].includes(name)) ? [{
      title: 'Run enabled optional modules',
      detail: enabledModules.filter((name) => !['Object Memory', 'Embedding / ReID', 'Kalman prediction'].includes(name)).map((name) => t(name)).join(', '),
    }] : []),
    {
      title: 'Write events and report data',
      detail: t('The system saves tracks, object cards, confidence, crops and enabled module outputs for review.'),
    },
  ]

  return (
    <aside className="relative h-full min-h-0 overflow-y-auto bg-muted/20 p-5 lg:p-6">
      <div className={`pointer-events-none absolute inset-x-0 top-0 h-40 bg-gradient-to-b ${profile.accent}`} />
      <div className="relative space-y-5">
        <div>
          <div className="mb-2 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
            <Sparkles className="h-3.5 w-3.5" /> {t('How this AI mode works')}
          </div>
          <div className="flex items-center gap-2">
            <span className={`h-2.5 w-2.5 rounded-full ${profile.dot}`} />
            <h3 className="text-xl font-semibold text-foreground">{t(profile.label)}</h3>
            <span className="rounded-full border border-border bg-background/60 px-2 py-0.5 text-[10px] font-medium uppercase text-muted-foreground">
              {t(profile.eyebrow)}
            </span>
          </div>
          <p className="mt-2 text-sm leading-6 text-muted-foreground">{t(profile.summary)}</p>
          <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2">
            <div className="rounded-lg border border-border/70 bg-background/50 p-2.5">
              <p className="text-[10px] uppercase tracking-wide text-muted-foreground">{t('Architecture')}</p>
              <p className="mt-1 text-xs font-semibold text-foreground">{t(profile.architecture)}</p>
            </div>
            <div className="rounded-lg border border-border/70 bg-background/50 p-2.5">
              <p className="text-[10px] uppercase tracking-wide text-muted-foreground">{t('Selection mode')}</p>
              <p className="mt-1 text-xs font-semibold text-foreground">{t(pipelineMode === 'manual' ? 'Manual' : 'Automatic')}</p>
            </div>
          </div>
          <p className="mt-2 text-xs text-foreground"><span className="text-muted-foreground">{t('Best for')}:</span> {t(profile.bestFor)}</p>
        </div>

        {runtimeWarnings.length > 0 && (
          <div className="rounded-lg border border-amber-500/25 bg-amber-500/5 p-3 text-[11px] leading-5 text-amber-100/90">
            <p className="font-medium text-amber-100">{t('Runtime warning')}</p>
            <ul className="mt-1 list-disc space-y-1 pl-4">
              {runtimeWarnings.map((warning) => <li key={warning}>{t(warning)}</li>)}
            </ul>
          </div>
        )}

        <div>
          <div className="mb-3 flex items-center justify-between">
            <p className="text-xs font-semibold uppercase tracking-wider text-foreground">{t('Pipeline')}</p>
            <span className="text-[10px] text-muted-foreground">{t('updates with selection')}</span>
          </div>
          <div className="grid grid-cols-[22px_1fr] gap-x-2">
            {livePipeline.map((step, index) => (
              <div className="contents" key={step.title}>
                <div className="flex flex-col items-center">
                  <span className="flex h-5 w-5 items-center justify-center rounded-full border border-primary/30 bg-primary/10 text-[10px] font-semibold text-primary">{index + 1}</span>
                  {index < livePipeline.length - 1 && <ArrowDown className="my-1 h-4 w-4 text-border" />}
                </div>
                <div className="pb-3">
                  <p className="text-xs font-medium text-foreground">{t(step.title)}</p>
                  <p className="mt-0.5 text-[11px] leading-4 text-muted-foreground">{step.detail}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div>
          <div className="mb-2 flex items-center gap-2">
            <p className="text-xs font-semibold uppercase tracking-wider text-foreground">{t('Selected components')}</p>
            {runtime && <span className="rounded-full bg-emerald-500/10 px-2 py-0.5 text-[9px] uppercase text-emerald-400">{t('live registry')}</span>}
          </div>
          <div className="mb-3 space-y-2">
            <div className="rounded-lg border border-border/70 bg-background/60 p-3">
              <p className="text-[10px] uppercase tracking-wide text-muted-foreground">{t('Object detector')}</p>
              <p className="mt-1 text-sm font-semibold text-foreground">{t(detectorMeta?.label || selectedDetector)}</p>
              <p className="mt-1 text-[11px] text-primary">
                {t(detectorMeta?.family || profile.architecture)}
                {detectorRuntime && (
                  <span className={`ml-2 ${detectorRuntime.available ? 'text-emerald-400' : 'text-amber-400'}`}>
                    {detectorRuntime.available ? t('available') : t('not installed')}
                  </span>
                )}
              </p>
              <p className="mt-1 text-[11px] leading-4 text-muted-foreground">{t(detectorMeta?.detail || 'The backend resolves the detector from the selected preset.')}</p>
            </div>
            {mode === 'hybrid' && (
              <div className="rounded-lg border border-border/70 bg-background/60 p-3">
                <p className="text-[10px] uppercase tracking-wide text-muted-foreground">{t('Verifier detector')}</p>
                <p className="mt-1 text-sm font-semibold text-foreground">{t(verifierMeta?.label || selectedVerifier)}</p>
                <p className="mt-1 text-[11px] text-primary">
                  {t(verifierMeta?.family || 'Transformer / DETR')}
                  {verifierRuntime && (
                    <span className={`ml-2 ${verifierRuntime.available ? 'text-emerald-400' : 'text-amber-400'}`}>
                      {verifierRuntime.available ? t('available') : t('not installed')}
                    </span>
                  )}
                </p>
                <p className="mt-1 text-[11px] leading-4 text-muted-foreground">{t(verifierMeta?.detail || 'The verifier re-checks difficult detections before tracking.')}</p>
              </div>
            )}
            <div className="rounded-lg border border-border/70 bg-background/60 p-3">
              <p className="text-[10px] uppercase tracking-wide text-muted-foreground">{t('Tracker')}</p>
              <p className="mt-1 text-sm font-semibold text-foreground">{t(trackerMeta?.label || selectedTracker)}</p>
              <p className="mt-1 text-[11px] text-primary">
                {t(trackerMeta?.family || 'Algorithm')}
                {trackerRuntime && (
                  <span className={`ml-2 ${trackerRuntime.available ? 'text-emerald-400' : 'text-amber-400'}`}>
                    {trackerRuntime.available ? t('available') : t('not installed')}
                  </span>
                )}
              </p>
              <p className="mt-1 text-[11px] leading-4 text-muted-foreground">{t(trackerMeta?.detail || 'The tracker links detections into object trajectories.')}</p>
            </div>
          </div>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">
            {(Object.keys(ROLE_META) as AiModelRole[]).map((role) => {
              const meta = ROLE_META[role]
              const Icon = meta.icon
              const resolved = selectedRuntimeModels.find((model) => model.role === role)
              return (
                <div key={role} className="rounded-lg border border-border/70 bg-background/60 p-2.5">
                  <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-muted-foreground">
                    <Icon className="h-3.5 w-3.5" /> {t(meta.label)}
                  </div>
                  <p className="mt-1 truncate text-xs font-medium text-foreground" title={resolved?.name || profile.modelPreferences[role]}>
                    {resolved?.name || t(profile.modelPreferences[role])}
                  </p>
                  {resolved && <p className={`mt-0.5 text-[10px] uppercase ${resolved.available ? 'text-emerald-400' : 'text-amber-400'}`}>{resolved.available ? t('available') : t('not installed')} - {t(resolved.format)}</p>}
                </div>
              )
            })}
          </div>
          {missingOptionalModels.length > 0 && (
            <div className="mt-3 rounded-lg border border-amber-500/25 bg-amber-500/5 p-3 text-[11px] leading-5 text-amber-100/90">
              <p className="font-medium text-amber-100">{t('Optional models are not installed')}</p>
              <p className="mt-1 text-amber-100/75">{t('Detection and tracking keep working. Classification and segmentation start only after their trained artifacts are deployed.')}</p>
              <p className="mt-2 rounded-md bg-background/60 px-2 py-1 font-mono text-[10px] text-amber-50">
                object_classifier/production.onnx + labels.json<br />
                object_segmenter/production.pt
              </p>
            </div>
          )}
        </div>

        <div className="rounded-lg border border-border/70 bg-background/50 p-3 text-[11px] leading-4 text-muted-foreground">
          <div className="mb-1 flex items-center gap-1.5 font-medium text-foreground"><SlidersHorizontal className="h-3.5 w-3.5" />{t('Frame policy')}</div>
          {t(profile.framePolicy)}
          {runtime && <span className="mt-1 block">{t('Resolved default')}: {t('frame skip: {count}', { count: runtime.frame_skip_default })}</span>}
          <span className="mt-1 block">{t('Source settings control thresholds, aerial tiling, target classes, tracking and optional analysis modules.')}</span>
        </div>

        <div className="rounded-lg border border-border/70 bg-background/50 p-3 text-[11px] leading-4 text-muted-foreground">
          <div className="mb-2 flex items-center gap-1.5 font-medium text-foreground"><SlidersHorizontal className="h-3.5 w-3.5" />{t('Current source settings')}</div>
          <div className="grid grid-cols-2 gap-x-3 gap-y-1">
            <span>{t('Detection confidence')}</span><span className="text-foreground">{pipelineConfig.object_confidence_threshold ?? 0.55}</span>
            <span>{t('Tile size')}</span><span className="text-foreground">{aerial.tile_size ?? 1024}</span>
            <span>{t('Tile overlap')}</span><span className="text-foreground">{aerial.tile_overlap ?? 0.2}</span>
            <span>{t('Target object classes')}</span><span className="text-foreground">{targetClasses.length ? targetClasses.join(', ') : t('All model classes')}</span>
            {mode === 'hybrid' && <><span>{t('Verifier detector')}</span><span className="text-foreground">{t(verifierMeta?.label || selectedVerifier)}</span></>}
            <span>{t('Kalman prediction')}</span><span className="text-foreground">{t(enabledLabel(kalman.enabled ?? true))}</span>
            <span>{t('Object Memory')}</span><span className="text-foreground">{t(enabledLabel(objectMemory.enabled))}</span>
            <span>{t('Embedding / ReID')}</span><span className="text-foreground">{t(enabledLabel(reid.enabled))}</span>
            <span>{t('Object classification')}</span><span className="text-foreground">{t(enabledLabel(classification.enabled))}</span>
            <span>{t('Instance segmentation')}</span><span className="text-foreground">{t(enabledLabel(segmentation.enabled))}</span>
            <span>{t('OCR text recognition')}</span><span className="text-foreground">{t(enabledLabel(ocr.enabled))}</span>
            <span>{t('Geo telemetry')}</span><span className="text-foreground">{t(enabledLabel(geo.enabled))}</span>
          </div>
        </div>

        {profile.warning && (
          <div className="rounded-lg border border-amber-500/20 bg-amber-500/5 p-3 text-[11px] leading-4 text-amber-200/80">
            {t(profile.warning)}
          </div>
        )}
      </div>
    </aside>
  )
}
