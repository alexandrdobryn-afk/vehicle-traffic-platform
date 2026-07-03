'use client'
import { useEffect, useState } from 'react'
import AppShell from '@/components/shared/AppShell'
import trainingApi from '@/lib/trainingApi'
import { useTrainingWebSocket } from '@/hooks/useTrainingWebSocket'
import { TTrainingJob, TMetrics, JOB_STATUS_COLORS, MODEL_TYPE_LABELS } from '@/types/training'
import { formatDateTime, formatDuration } from '@/lib/utils'
import { Square, RefreshCw, CheckCircle, XCircle, Clock, Cpu, MemoryStick } from 'lucide-react'
import Link from 'next/link'
import toast from 'react-hot-toast'
import { useParams } from 'next/navigation'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend,
} from 'recharts'
import { useTranslation } from '@/lib/i18n'

export default function JobDetailPage() {
  const { t, locale } = useTranslation()
  const params = useParams<{ jobId: string }>()
  const jobId = parseInt(params.jobId)
  const [job, setJob] = useState<TTrainingJob | null>(null)
  const [metrics, setMetrics] = useState<TMetrics[]>([])
  const [cancelling, setCancelling] = useState(false)

  const { connected, progress } = useTrainingWebSocket(
    job?.status && !['completed', 'failed', 'cancelled'].includes(job.status) ? jobId : null
  )

  const load = async () => {
    const [j, m] = await Promise.all([
      trainingApi.get(`/jobs/${jobId}`),
      trainingApi.get(`/jobs/${jobId}/metrics`),
    ])
    setJob(j.data)
    setMetrics(m.data)
  }

  useEffect(() => { load() }, [jobId])

  // Refresh DB state periodically for terminal states
  useEffect(() => {
    if (!job) return
    if (['completed', 'failed', 'cancelled'].includes(job.status)) return
    const t = setInterval(load, 8000)
    return () => clearInterval(t)
  }, [job?.status])

  // Merge live progress into display
  const displayJob = job ? {
    ...job,
    status: progress?.status || job.status,
    current_epoch: progress?.current_epoch ?? job.current_epoch,
    progress_pct: progress?.progress_pct ?? job.progress_pct,
    eta_seconds: progress?.eta_seconds ?? job.eta_seconds,
  } : null

  // Merge live metrics into chart data
  const chartData = metrics.map(m => ({
    epoch: m.epoch,
    train_loss: m.train_loss,
    val_loss: m.val_loss,
    precision: m.precision ? +(m.precision * 100).toFixed(2) : null,
    recall: m.recall ? +(m.recall * 100).toFixed(2) : null,
    map50: m.map50 ? +(m.map50 * 100).toFixed(2) : null,
    map50_95: m.map50_95 ? +(m.map50_95 * 100).toFixed(2) : null,
    accuracy: m.accuracy ? +(m.accuracy * 100).toFixed(2) : null,
    lr: m.lr,
  }))

  const formatProgressMessage = (message: string) => {
    let match = message.match(/^Training (.+) for (\d+) epochs\.\.\.$/)
    if (match) return t('Training {architecture} for {epochs} epochs...', { architecture: match[1], epochs: match[2] })
    match = message.match(/^Epoch (\d+)\/(\d+) — accuracy: (.+)$/)
    if (match) return t('Epoch {current}/{total} — accuracy: {accuracy}', { current: match[1], total: match[2], accuracy: match[3] })
    match = message.match(/^Epoch (\d+)\/(\d+) — mAP50: (.+)$/)
    if (match) return t('Epoch {current}/{total} — mAP50: {value}', { current: match[1], total: match[2], value: match[3] })
    match = message.match(/^LPRNet epoch (\d+)\/(\d+) — plate accuracy: (.+)$/)
    if (match) return t('LPRNet epoch {current}/{total} — plate accuracy: {accuracy}', { current: match[1], total: match[2], accuracy: match[3] })
    return t(message)
  }

  const handleCancel = async () => {
    if (!confirm(t('Cancel this training job?'))) return
    setCancelling(true)
    try {
      await trainingApi.post(`/jobs/${jobId}/cancel`)
      toast.success(t('Job cancelled'))
      load()
    } catch { toast.error(t('Cancel failed')) }
    finally { setCancelling(false) }
  }

  const isRunning = displayJob?.status && ['queued', 'preparing', 'training', 'validation', 'export'].includes(displayJob.status)
  const isCompleted = displayJob?.status === 'completed'

  if (!displayJob) {
    return <AppShell><div className="p-6 text-muted-foreground">{t('Loading...')}</div></AppShell>
  }

  const CHART_COLORS = {
    train_loss: '#ef4444', val_loss: '#f97316',
    map50: '#22c55e', map50_95: '#3b82f6',
    precision: '#8b5cf6', recall: '#ec4899',
    accuracy: '#22c55e',
  }

  const tooltipStyle = {
    contentStyle: { background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: 8 },
    labelStyle: { color: 'hsl(var(--foreground))' },
  }

  return (
    <AppShell>
      <div className="p-6 space-y-6 max-w-6xl">
        {/* Header */}
        <div className="flex items-start justify-between pr-40">
          <div>
            <div className="flex items-center gap-3 mb-1">
              <h1 className="text-2xl font-bold text-foreground">{displayJob.name}</h1>
              <span className={`text-sm font-medium capitalize ${JOB_STATUS_COLORS[displayJob.status]}`}>
                {t(displayJob.status)}
              </span>
              {isRunning && connected && (
                <span className="flex items-center gap-1.5 text-xs text-emerald-400">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                  {t('Live')}
                </span>
              )}
            </div>
            <p className="text-sm text-muted-foreground">
              {t(MODEL_TYPE_LABELS[displayJob.model_type])} · {displayJob.architecture} · {t('Dataset')} #{displayJob.dataset_id}
            </p>
          </div>
          <div className="flex gap-2">
            <button onClick={load} className="p-2 bg-muted rounded-lg text-muted-foreground hover:text-foreground" title={t('Refresh')} aria-label={t('Refresh')}>
              <RefreshCw className="w-4 h-4" />
            </button>
            {isRunning && (
              <button onClick={handleCancel} disabled={cancelling}
                className="flex items-center gap-2 px-3 py-2 bg-red-500/10 text-red-400 border border-red-500/20 rounded-lg text-sm hover:bg-red-500/20 disabled:opacity-50">
                <Square className="w-4 h-4" /> {t('Cancel')}
              </button>
            )}
            {isCompleted && displayJob.final_metrics && (
              <Link href={`/training/registry`}
                className="px-3 py-2 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded-lg text-sm hover:bg-emerald-500/20">
                {t('View in Registry')} →
              </Link>
            )}
          </div>
        </div>

        {/* Progress bar */}
        {isRunning && (
          <div className="bg-card border border-border rounded-xl p-5 space-y-3">
            <div className="flex justify-between items-center">
              <div className="flex items-center gap-3">
                <span className="text-sm font-medium text-foreground">
                  {t('Epoch')} {displayJob.current_epoch} / {displayJob.total_epochs}
                </span>
                {displayJob.eta_seconds && (
                  <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
                    <Clock className="w-3.5 h-3.5" />
                    {t('ETA')}: {formatDuration(displayJob.eta_seconds, locale)}
                  </span>
                )}
              </div>
              <span className="text-sm font-bold text-foreground">
                {displayJob.progress_pct.toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-muted rounded-full h-3 overflow-hidden">
              <div
                className="h-3 rounded-full bg-gradient-to-r from-primary to-cyan-400 transition-all duration-500"
                style={{ width: `${displayJob.progress_pct}%` }}
              />
            </div>

            {/* Live message */}
            {progress?.message && (
              <p className="text-xs text-muted-foreground font-mono">{formatProgressMessage(progress.message)}</p>
            )}

            {/* GPU stats */}
            {(progress?.gpu_utilization !== null || progress?.gpu_memory_mb !== null) && (
              <div className="flex gap-6 pt-1">
                {progress?.gpu_utilization !== null && progress?.gpu_utilization !== undefined && (
                  <div className="flex items-center gap-2">
                    <Cpu className="w-3.5 h-3.5 text-violet-400" />
                    <span className="text-xs text-muted-foreground">GPU</span>
                    <span className="text-xs font-medium text-foreground">{progress.gpu_utilization?.toFixed(0)}%</span>
                  </div>
                )}
                {progress?.gpu_memory_mb !== null && progress?.gpu_memory_mb !== undefined && (
                  <div className="flex items-center gap-2">
                    <MemoryStick className="w-3.5 h-3.5 text-cyan-400" />
                    <span className="text-xs text-muted-foreground">VRAM</span>
                    <span className="text-xs font-medium text-foreground">{(progress.gpu_memory_mb / 1024).toFixed(1)} GB</span>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Best metrics */}
        {displayJob.best_metrics && Object.keys(displayJob.best_metrics).length > 0 && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(displayJob.best_metrics)
              .filter(([k]) => k !== 'epoch')
              .map(([key, val]) => (
                <div key={key} className="bg-card border border-border rounded-xl p-4">
                  <p className="text-xs text-muted-foreground uppercase mb-1">{t(key.replace(/_/g, ' '))}</p>
                  <p className="text-2xl font-bold text-foreground">
                    {typeof val === 'number'
                      ? val < 2 ? `${(val * 100).toFixed(1)}%` : val.toFixed(4)
                      : String(val)}
                  </p>
                  {displayJob.best_metrics?.epoch && (
                    <p className="text-xs text-muted-foreground mt-0.5">{t('at epoch {epoch}', { epoch: displayJob.best_metrics.epoch })}</p>
                  )}
                </div>
              ))}
          </div>
        )}

        {/* Charts */}
        {chartData.length > 0 && (
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            {/* Loss chart */}
            <div className="bg-card border border-border rounded-xl p-5">
              <h3 className="text-sm font-semibold text-foreground mb-4">{t('Loss')}</h3>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="epoch" tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }} />
                  <YAxis tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }} />
                  <Tooltip {...tooltipStyle} />
                  <Legend />
                  <Line type="monotone" dataKey="train_loss" stroke={CHART_COLORS.train_loss}
                    strokeWidth={2} dot={false} name={t('Train Loss')} />
                  <Line type="monotone" dataKey="val_loss" stroke={CHART_COLORS.val_loss}
                    strokeWidth={2} dot={false} name={t('Val Loss')} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Detection metrics */}
            {chartData[0]?.map50 !== null && chartData[0]?.map50 !== undefined ? (
              <div className="bg-card border border-border rounded-xl p-5">
                <h3 className="text-sm font-semibold text-foreground mb-4">{t('Detection Metrics (%)')}</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="epoch" tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }} />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }} />
                    <Tooltip {...tooltipStyle} />
                    <Legend />
                    <Line type="monotone" dataKey="map50" stroke={CHART_COLORS.map50}
                      strokeWidth={2} dot={false} name="mAP50" />
                    <Line type="monotone" dataKey="map50_95" stroke={CHART_COLORS.map50_95}
                      strokeWidth={2} dot={false} name="mAP50-95" />
                    <Line type="monotone" dataKey="precision" stroke={CHART_COLORS.precision}
                      strokeWidth={1.5} dot={false} name={t('Precision')} strokeDasharray="4 2" />
                    <Line type="monotone" dataKey="recall" stroke={CHART_COLORS.recall}
                      strokeWidth={1.5} dot={false} name={t('Recall')} strokeDasharray="4 2" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ) : chartData[0]?.accuracy !== null && chartData[0]?.accuracy !== undefined ? (
              <div className="bg-card border border-border rounded-xl p-5">
                <h3 className="text-sm font-semibold text-foreground mb-4">{t('Accuracy (%)')}</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="epoch" tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }} />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }} />
                    <Tooltip {...tooltipStyle} />
                    <Line type="monotone" dataKey="accuracy" stroke={CHART_COLORS.accuracy}
                      strokeWidth={2} dot={false} name={t('Accuracy')} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ) : null}

            {/* LR chart */}
            {chartData.some(d => d.lr !== null) && (
              <div className="bg-card border border-border rounded-xl p-5">
                <h3 className="text-sm font-semibold text-foreground mb-4">{t('Learning Rate')}</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="epoch" tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }} />
                    <YAxis tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }} tickFormatter={v => v.toExponential(1)} />
                    <Tooltip {...tooltipStyle} />
                    <Line type="monotone" dataKey="lr" stroke="#f59e0b"
                      strokeWidth={2} dot={false} name="LR" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        )}

        {/* Error message */}
        {displayJob.error_message && (
          <div className="flex items-start gap-3 p-4 bg-red-500/10 border border-red-500/20 rounded-xl">
            <XCircle className="w-5 h-5 text-red-400 mt-0.5 shrink-0" />
            <div>
              <p className="text-sm font-medium text-red-400">{t('Training Failed')}</p>
              <p className="text-sm text-red-300 font-mono mt-1">{displayJob.error_message}</p>
            </div>
          </div>
        )}

        {/* Job details */}
        <div className="bg-card border border-border rounded-xl p-5">
          <h3 className="text-sm font-semibold text-foreground mb-4">{t('Job Details')}</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
            {[
              ['Job ID', `#${displayJob.id}`],
              ['Status', displayJob.status],
              ['Author', displayJob.author_email || '—'],
              ['Created', formatDateTime(displayJob.created_at, locale)],
              ['Started', displayJob.started_at ? formatDateTime(displayJob.started_at, locale) : '—'],
              ['Finished', displayJob.finished_at ? formatDateTime(displayJob.finished_at, locale) : '—'],
              ['Celery Task', displayJob.celery_task_id ? displayJob.celery_task_id.slice(0, 16) + '...' : '—'],
              ['Device', displayJob.gpu_device || 'auto'],
            ].map(([k, v]) => (
              <div key={k}>
                <p className="text-xs text-muted-foreground">{t(k)}</p>
                <p className="font-medium text-foreground mt-0.5">{t(String(v))}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Hyperparams */}
        {displayJob.hyperparams && (
          <div className="bg-card border border-border rounded-xl p-5">
            <h3 className="text-sm font-semibold text-foreground mb-4">{t('Hyperparameters')}</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
              {Object.entries(displayJob.hyperparams).map(([k, v]) => (
                <div key={k}>
                  <p className="text-xs text-muted-foreground">{t(k.replace(/_/g, ' '))}</p>
                  <p className="font-medium text-foreground mt-0.5">{t(String(v))}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </AppShell>
  )
}
