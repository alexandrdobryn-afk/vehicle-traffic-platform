'use client'
import { useEffect, useState } from 'react'
import AppShell from '@/components/shared/AppShell'
import trainingApi from '@/lib/trainingApi'
import { TTrainingJob, TDataset, TModelVersion, MODEL_TYPE_LABELS, JOB_STATUS_COLORS } from '@/types/training'
import { formatDateTime } from '@/lib/utils'
import Link from 'next/link'
import {
  Database, Play, Archive, Cpu, CheckCircle,
  AlertCircle, Clock, TrendingUp, Plus
} from 'lucide-react'
import { useTranslation } from '@/lib/i18n'

export default function TrainingHubPage() {
  const { t } = useTranslation()
  const [jobs, setJobs] = useState<TTrainingJob[]>([])
  const [datasets, setDatasets] = useState<TDataset[]>([])
  const [models, setModels] = useState<TModelVersion[]>([])
  const [gpuInfo, setGpuInfo] = useState<any>(null)

  useEffect(() => {
    const load = async () => {
      try {
        const [j, d, m, g] = await Promise.all([
          trainingApi.get('/jobs?limit=5'),
          trainingApi.get('/datasets'),
          trainingApi.get('/registry?limit=5'),
          trainingApi.get('/gpu'),
        ])
        setJobs(j.data)
        setDatasets(d.data)
        setModels(m.data)
        setGpuInfo(g.data)
      } catch {}
    }
    load()
    const t = setInterval(load, 10000)
    return () => clearInterval(t)
  }, [])

  const activeJobs = jobs.filter(j => ['queued','preparing','training','validation','export'].includes(j.status))
  const prodModels = models.filter(m => m.is_production)

  const cards = [
    { label: 'Datasets', value: datasets.length, icon: Database, href: '/training/datasets', color: 'text-blue-400' },
    { label: 'Active Jobs', value: activeJobs.length, icon: Play, href: '/training/jobs', color: 'text-cyan-400' },
    { label: 'Model Versions', value: models.length, icon: Archive, href: '/training/registry', color: 'text-violet-400' },
    { label: 'Production Models', value: prodModels.length, icon: CheckCircle, href: '/training/registry', color: 'text-emerald-400' },
  ]

  return (
    <AppShell>
      <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between pr-40">
          <div>
            <h1 className="text-2xl font-bold text-foreground">{t('AI Training Platform')}</h1>
            <p className="text-muted-foreground text-sm mt-0.5">{t('Model lifecycle management')}</p>
          </div>
          <Link href="/training/jobs/new"
            className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:bg-primary/90">
            <Plus className="w-4 h-4" /> {t('New Training Job')}
          </Link>
        </div>

        {/* GPU status */}
        {gpuInfo?.gpus?.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {gpuInfo.gpus.map((gpu: any) => (
              <div key={gpu.id} className="bg-card border border-border rounded-xl p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Cpu className="w-4 h-4 text-violet-400" />
                    <span className="text-sm font-medium text-foreground truncate max-w-[160px]">{gpu.name}</span>
                  </div>
                  <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                    gpu.is_available ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'
                  }`}>{gpu.is_available ? t('Available') : t('Busy')}</span>
                </div>
                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>{t('GPU Util')}</span>
                    <span className="font-medium text-foreground">{gpu.utilization_pct?.toFixed(0)}%</span>
                  </div>
                  <div className="w-full bg-muted rounded-full h-1.5">
                    <div className="bg-violet-400 h-1.5 rounded-full" style={{ width: `${gpu.utilization_pct}%` }} />
                  </div>
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>VRAM</span>
                    <span className="font-medium text-foreground">
                      {gpu.memory_used_mb?.toFixed(0)} / {gpu.memory_total_mb?.toFixed(0)} MB
                    </span>
                  </div>
                  <div className="w-full bg-muted rounded-full h-1.5">
                    <div className="bg-cyan-400 h-1.5 rounded-full"
                      style={{ width: `${(gpu.memory_used_mb / gpu.memory_total_mb) * 100}%` }} />
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {cards.map(({ label, value, icon: Icon, href, color }) => (
            <Link key={label} href={href}
              className="bg-card border border-border rounded-xl p-4 hover:border-primary/30 transition-colors group">
              <Icon className={`w-5 h-5 mb-3 ${color}`} />
              <p className="text-2xl font-bold text-foreground">{value}</p>
              <p className="text-xs text-muted-foreground mt-0.5 group-hover:text-foreground transition-colors">{t(label)}</p>
            </Link>
          ))}
        </div>

        {/* Active jobs + recent models */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          {/* Active jobs */}
          <div className="bg-card border border-border rounded-xl">
            <div className="flex items-center justify-between p-4 border-b border-border">
              <h3 className="text-sm font-semibold text-foreground">{t('Recent Jobs')}</h3>
              <Link href="/training/jobs" className="text-xs text-primary hover:underline">{t('View all')}</Link>
            </div>
            <div className="divide-y divide-border">
              {jobs.length === 0 ? (
                <p className="text-center py-8 text-muted-foreground text-sm">{t('No training jobs yet')}</p>
              ) : jobs.slice(0, 5).map((job) => (
                <Link key={job.id} href={`/training/jobs/${job.id}`}
                  className="flex items-center justify-between px-4 py-3 hover:bg-accent/30 transition-colors">
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-foreground truncate">{job.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {t(MODEL_TYPE_LABELS[job.model_type])} · {job.architecture}
                    </p>
                  </div>
                  <div className="text-right shrink-0 ml-4">
                    <p className={`text-xs font-medium capitalize ${JOB_STATUS_COLORS[job.status]}`}>{t(job.status)}</p>
                    {job.status === 'training' && (
                      <p className="text-xs text-muted-foreground">{job.progress_pct.toFixed(0)}%</p>
                    )}
                  </div>
                </Link>
              ))}
            </div>
          </div>

          {/* Model registry */}
          <div className="bg-card border border-border rounded-xl">
            <div className="flex items-center justify-between p-4 border-b border-border">
              <h3 className="text-sm font-semibold text-foreground">{t('Model Registry')}</h3>
              <Link href="/training/registry" className="text-xs text-primary hover:underline">{t('View all')}</Link>
            </div>
            <div className="divide-y divide-border">
              {models.length === 0 ? (
                <p className="text-center py-8 text-muted-foreground text-sm">{t('No models yet')}</p>
              ) : models.slice(0, 5).map((mv) => (
                <div key={mv.id} className="flex items-center justify-between px-4 py-3">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <p className="text-sm font-medium text-foreground truncate">{mv.name}</p>
                      {mv.is_production && (
                        <span className="text-xs px-1.5 py-0.5 bg-emerald-500/10 text-emerald-400 rounded font-medium">{t('PROD')}</span>
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {t(MODEL_TYPE_LABELS[mv.model_type])} · {mv.version}
                    </p>
                  </div>
                  <div className="text-right shrink-0 ml-4">
                    {mv.metrics?.map50 && (
                      <p className="text-xs font-medium text-foreground">
                        mAP50: {(mv.metrics.map50 * 100).toFixed(1)}%
                      </p>
                    )}
                    {mv.metrics?.accuracy && (
                      <p className="text-xs font-medium text-foreground">
                        {t('Acc')}: {(mv.metrics.accuracy * 100).toFixed(1)}%
                      </p>
                    )}
                    <p className="text-xs text-muted-foreground capitalize">{t(mv.deploy_status)}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </AppShell>
  )
}
