'use client'

import { useEffect, useMemo, useState } from 'react'
import {
  AlertTriangle, Boxes, CheckCircle2, CircleDashed, Cpu, Database,
  Layers3, Loader2, RefreshCw, ServerCog, SlidersHorizontal, Wrench,
  XCircle,
} from 'lucide-react'

import AppShell from '@/components/shared/AppShell'
import api, { apiErrorMessage } from '@/lib/api'
import { cn } from '@/lib/utils'
import { useTranslation } from '@/lib/i18n'
import type { ModuleCapability, ModuleCapabilityStatus, ModuleCapabilitySummary } from '@/types'

const STATUS_META: Record<ModuleCapabilityStatus, { label: string; icon: any; className: string }> = {
  ready: { label: 'Ready', icon: CheckCircle2, className: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20' },
  partial: { label: 'Partial', icon: AlertTriangle, className: 'text-amber-300 bg-amber-500/10 border-amber-500/20' },
  planned: { label: 'Planned', icon: CircleDashed, className: 'text-sky-300 bg-sky-500/10 border-sky-500/20' },
  blocked: { label: 'Blocked', icon: XCircle, className: 'text-red-300 bg-red-500/10 border-red-500/20' },
}

const GROUP_LABELS: Record<string, string> = {
  input: 'Input',
  vision: 'Vision',
  motion: 'Motion',
  image_processing: 'Image processing',
  geo: 'Geo',
  training: 'Training',
  quality: 'Quality',
  deployment: 'Deployment',
  integration: 'Integration',
  assistant: 'LLM/VLM',
  simulation: 'Simulation',
}

export default function ModulesPage() {
  const { t } = useTranslation()
  const [data, setData] = useState<ModuleCapabilitySummary | null>(null)
  const [loading, setLoading] = useState(false)
  const [filter, setFilter] = useState<'all' | ModuleCapabilityStatus>('all')
  const [group, setGroup] = useState('all')
  const [error, setError] = useState('')

  const load = async () => {
    setLoading(true)
    setError('')
    try {
      setData((await api.get('/modules')).data)
    } catch (err: any) {
      setError(apiErrorMessage(err, t('Failed to load modules')))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const groups = useMemo(() => {
    const source = data?.modules || []
    return Array.from(new Set(source.map((item) => item.group))).sort()
  }, [data])

  const modules = useMemo(() => {
    return (data?.modules || []).filter((item) => (
      (filter === 'all' || item.status === filter)
      && (group === 'all' || item.group === group)
    ))
  }, [data, filter, group])

  return (
    <AppShell>
      <div className="p-6 space-y-6 max-w-7xl">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <h1 className="text-2xl font-bold text-foreground">{t('Modules')}</h1>
            <p className="text-sm text-muted-foreground mt-1">
              {t('Full platform capability map for aerial computer vision')}
            </p>
          </div>
          <button
            onClick={load}
            className="inline-flex items-center gap-2 rounded-lg border border-border px-3 py-2 text-sm text-muted-foreground hover:text-foreground"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
            {t('Refresh')}
          </button>
        </div>

        {error && (
          <div className="rounded-lg border border-red-500/20 bg-red-500/10 p-3 text-sm text-red-300">
            {error}
          </div>
        )}

        {!data && loading ? (
          <div className="flex items-center gap-2 text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" /> {t('Loading...')}
          </div>
        ) : !data ? (
          <div className="rounded-xl border border-border bg-card p-6 text-sm text-muted-foreground">
            {t('No module data is available. Check that the backend API is running, then refresh.')}
          </div>
        ) : (
          <>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
              <SummaryCard icon={Boxes} label={t('Total modules')} value={data.total} />
              <SummaryCard icon={CheckCircle2} label={t('Ready')} value={data.counts.ready} tone="ready" />
              <SummaryCard icon={AlertTriangle} label={t('Partial')} value={data.counts.partial} tone="partial" />
              <SummaryCard icon={CircleDashed} label={t('Planned')} value={data.counts.planned} tone="planned" />
              <SummaryCard icon={XCircle} label={t('Blocked')} value={data.counts.blocked} tone="blocked" />
            </div>

            <section className="rounded-xl border border-border bg-card p-4">
              <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                <div className="flex items-center gap-2 text-sm font-medium text-foreground">
                  <SlidersHorizontal className="h-4 w-4 text-primary" />
                  {t('Module filters')}
                </div>
                <div className="flex flex-wrap gap-2">
                  {(['all', 'ready', 'partial', 'planned', 'blocked'] as const).map((item) => (
                    <button
                      key={item}
                      onClick={() => setFilter(item)}
                      className={cn(
                        'rounded-lg border px-3 py-1.5 text-xs font-medium',
                        filter === item ? 'border-primary bg-primary/10 text-primary' : 'border-border text-muted-foreground hover:text-foreground',
                      )}
                    >
                      {t(item === 'all' ? 'All statuses' : STATUS_META[item].label)}
                    </button>
                  ))}
                  <select
                    value={group}
                    onChange={(event) => setGroup(event.target.value)}
                    className="rounded-lg border border-border bg-background px-3 py-1.5 text-xs text-foreground"
                  >
                    <option value="all">{t('All groups')}</option>
                    {groups.map((item) => (
                      <option key={item} value={item}>{t(GROUP_LABELS[item] || item)}</option>
                    ))}
                  </select>
                </div>
              </div>
            </section>

            <div className="grid gap-4 xl:grid-cols-2">
              {modules.map((module) => (
                <ModuleCard key={module.id} module={module} />
              ))}
            </div>
          </>
        )}
      </div>
    </AppShell>
  )
}

function SummaryCard({ icon: Icon, label, value, tone }: { icon: any; label: string; value: number; tone?: ModuleCapabilityStatus }) {
  const toneClass = tone ? STATUS_META[tone].className : 'text-primary bg-primary/10 border-primary/20'
  return (
    <div className="rounded-xl border border-border bg-card p-4">
      <div className={cn('mb-3 inline-flex rounded-lg border p-2', toneClass)}>
        <Icon className="h-4 w-4" />
      </div>
      <div className="text-2xl font-semibold text-foreground">{value}</div>
      <div className="text-xs text-muted-foreground">{label}</div>
    </div>
  )
}

function ModuleCard({ module }: { module: ModuleCapability }) {
  const { t } = useTranslation()
  const meta = STATUS_META[module.status]
  const StatusIcon = meta.icon
  return (
    <section className="rounded-xl border border-border bg-card p-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="text-xs font-medium text-muted-foreground">
            {String(module.order).padStart(2, '0')} / {t(GROUP_LABELS[module.group] || module.group)}
          </div>
          <h2 className="mt-1 text-lg font-semibold text-foreground">{module.name}</h2>
        </div>
        <div className={cn('inline-flex shrink-0 items-center gap-2 rounded-lg border px-2.5 py-1.5 text-xs font-medium', meta.className)}>
          <StatusIcon className="h-3.5 w-3.5" />
          {t(meta.label)}
        </div>
      </div>

      <div className="mt-4 grid grid-cols-4 gap-2">
        <Coverage icon={Layers3} label="UI" ready={module.ui_ready} />
        <Coverage icon={ServerCog} label="API" ready={module.api_ready} />
        <Coverage icon={Cpu} label="Runtime" ready={module.runtime_ready} />
        <Coverage icon={Database} label="Training" ready={module.training_ready} />
      </div>

      <div className="mt-5 grid gap-4 md:grid-cols-2">
        <ListBlock title={t('Implemented')} items={module.implemented} empty={t('No production implementation yet')} />
        <ListBlock title={t('Missing')} items={module.missing} empty={t('Nothing listed')} />
      </div>
      <div className="mt-4 grid gap-4 md:grid-cols-2">
        <ListBlock title={t('Dependencies')} items={module.dependencies} empty={t('No special dependencies')} compact />
        <ListBlock title={t('Next steps')} items={module.next_steps} empty={t('No next steps')} compact icon={Wrench} />
      </div>
    </section>
  )
}

function Coverage({ icon: Icon, label, ready }: { icon: any; label: string; ready: boolean }) {
  const { t } = useTranslation()
  return (
    <div className="rounded-lg border border-border bg-background p-2">
      <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
        <Icon className="h-3.5 w-3.5" /> {label}
      </div>
      <div className={cn('mt-1 text-xs font-semibold', ready ? 'text-emerald-400' : 'text-slate-500')}>
        {ready ? t('Ready') : t('Not ready')}
      </div>
    </div>
  )
}

function ListBlock({ title, items, empty, compact, icon: Icon }: { title: string; items: string[]; empty: string; compact?: boolean; icon?: any }) {
  return (
    <div>
      <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase text-muted-foreground">
        {Icon && <Icon className="h-3.5 w-3.5" />}
        {title}
      </div>
      {items.length === 0 ? (
        <p className="text-xs text-muted-foreground">{empty}</p>
      ) : (
        <ul className={cn('space-y-1.5 text-sm text-foreground', compact && 'text-xs')}>
          {items.map((item) => (
            <li key={item} className="flex gap-2">
              <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-primary" />
              <span>{item}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
