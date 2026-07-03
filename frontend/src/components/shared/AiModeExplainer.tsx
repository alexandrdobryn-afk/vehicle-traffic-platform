'use client'

import { useEffect, useMemo, useState } from 'react'
import {
  ArrowDown, CarFront, Cpu, Gauge, GitBranch, Hash, Palette,
  ScanLine, ShieldCheck, SlidersHorizontal, Sparkles,
} from 'lucide-react'

import api from '@/lib/api'
import { AI_MODE_PROFILES, AiModelRole, RuntimeMode } from '@/lib/aiModeProfiles'
import { useTranslation } from '@/lib/i18n'
import { AIMode } from '@/types'

const ROLE_META: Record<AiModelRole, { label: string; icon: typeof CarFront }> = {
  vehicle: { label: 'Vehicle detector', icon: CarFront },
  tracking: { label: 'Tracker', icon: GitBranch },
  plate: { label: 'Plate detector', icon: ScanLine },
  ocr: { label: 'OCR / LPR', icon: Hash },
  color: { label: 'Color', icon: Palette },
  brand: { label: 'Vehicle brand', icon: CarFront },
}

function Meter({ label, value, icon: Icon }: { label: string; value: number; icon: typeof Gauge }) {
  const { t } = useTranslation()
  return (
    <div className="rounded-lg border border-border/70 bg-background/50 p-2.5">
      <div className="mb-2 flex items-center gap-1.5 text-[11px] text-muted-foreground">
        <Icon className="h-3.5 w-3.5" /> {t(label)}
      </div>
      <div className="flex gap-1">
        {[1, 2, 3, 4, 5].map((step) => (
          <span key={step} className={`h-1.5 flex-1 rounded-full ${step <= value ? 'bg-primary' : 'bg-muted'}`} />
        ))}
      </div>
    </div>
  )
}

export default function AiModeExplainer({ mode }: { mode: AIMode }) {
  const { t } = useTranslation()
  const [runtimeModes, setRuntimeModes] = useState<RuntimeMode[]>([])
  const profile = AI_MODE_PROFILES[mode]

  useEffect(() => {
    let active = true
    api.get('/health/ai-modes')
      .then((response) => {
        if (active) setRuntimeModes(response.data.modes || [])
      })
      .catch(() => {})
    return () => { active = false }
  }, [])

  const runtime = useMemo(
    () => runtimeModes.find((item) => item.id === mode),
    [mode, runtimeModes],
  )

  return (
    <aside className="relative h-full overflow-hidden bg-muted/20 p-5 lg:p-6">
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
          <p className="mt-2 text-xs text-foreground"><span className="text-muted-foreground">{t('Best for')}:</span> {t(profile.bestFor)}</p>
        </div>

        <div className="grid grid-cols-3 gap-2">
          <Meter label="Speed" value={profile.speed} icon={Gauge} />
          <Meter label="Accuracy" value={profile.accuracy} icon={ShieldCheck} />
          <Meter label="Compute" value={profile.compute} icon={Cpu} />
        </div>

        <div>
          <div className="mb-3 flex items-center justify-between">
            <p className="text-xs font-semibold uppercase tracking-wider text-foreground">{t('Pipeline')}</p>
            <span className="text-[10px] text-muted-foreground">{t('updates with selection')}</span>
          </div>
          <div className="grid grid-cols-[22px_1fr] gap-x-2">
            {profile.pipeline.map((step, index) => (
              <div className="contents" key={step.title}>
                <div className="flex flex-col items-center">
                  <span className="flex h-5 w-5 items-center justify-center rounded-full border border-primary/30 bg-primary/10 text-[10px] font-semibold text-primary">{index + 1}</span>
                  {index < profile.pipeline.length - 1 && <ArrowDown className="my-1 h-4 w-4 text-border" />}
                </div>
                <div className="pb-3">
                  <p className="text-xs font-medium text-foreground">{t(step.title)}</p>
                  <p className="mt-0.5 text-[11px] leading-4 text-muted-foreground">{t(step.detail)}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div>
          <div className="mb-2 flex items-center gap-2">
            <p className="text-xs font-semibold uppercase tracking-wider text-foreground">{t('Resolved models')}</p>
            {runtime && <span className="rounded-full bg-emerald-500/10 px-2 py-0.5 text-[9px] uppercase text-emerald-400">{t('live registry')}</span>}
          </div>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">
            {(Object.keys(ROLE_META) as AiModelRole[]).map((role) => {
              const meta = ROLE_META[role]
              const Icon = meta.icon
              const resolved = runtime?.models.find((model) => model.role === role)
              return (
                <div key={role} className="rounded-lg border border-border/70 bg-background/60 p-2.5">
                  <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-muted-foreground">
                    <Icon className="h-3.5 w-3.5" /> {t(meta.label)}
                  </div>
                  <p className="mt-1 truncate text-xs font-medium text-foreground" title={resolved?.name || profile.modelPreferences[role]}>
                    {resolved?.name || t(profile.modelPreferences[role])}
                  </p>
                  {resolved && <p className="mt-0.5 text-[10px] uppercase text-muted-foreground">{resolved.format}</p>}
                </div>
              )
            })}
          </div>
        </div>

        <div className="rounded-lg border border-border/70 bg-background/50 p-3 text-[11px] leading-4 text-muted-foreground">
          <div className="mb-1 flex items-center gap-1.5 font-medium text-foreground"><SlidersHorizontal className="h-3.5 w-3.5" />{t('Frame policy')}</div>
          {t(profile.framePolicy)}
          {runtime && <span className="mt-1 block">{t('Resolved default')}: {t('frame skip: {count}', { count: runtime.frame_skip_default })}</span>}
          <span className="mt-1 block">{t('Global Settings apply thresholds, resolution, FPS caps, storage and privacy. Mode or Manual selection owns frame skip, tracker and OCR.')}</span>
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
