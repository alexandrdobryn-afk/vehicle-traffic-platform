'use client'
import { useEffect, useState } from 'react'
import AppShell from '@/components/shared/AppShell'
import trainingApi from '@/lib/trainingApi'
import { TTrainingJob, MODEL_TYPE_LABELS, JOB_STATUS_COLORS } from '@/types/training'
import { formatDateTime } from '@/lib/utils'
import Link from 'next/link'
import { Plus, RefreshCw, Square } from 'lucide-react'
import toast from 'react-hot-toast'
import { useTranslation } from '@/lib/i18n'

const STATUS_ORDER = ['training', 'preparing', 'queued', 'validation', 'export', 'completed', 'failed', 'cancelled']

export default function JobsPage() {
  const { t: tr, locale } = useTranslation()
  const [jobs, setJobs] = useState<TTrainingJob[]>([])
  const [loading, setLoading] = useState(false)
  const [filter, setFilter] = useState('')

  const load = async () => {
    setLoading(true)
    try {
      const r = await trainingApi.get('/jobs?limit=100')
      setJobs(r.data)
    } finally { setLoading(false) }
  }

  useEffect(() => {
    load()
    const t = setInterval(load, 5000)
    return () => clearInterval(t)
  }, [])

  const handleCancel = async (id: number) => {
    try { await trainingApi.post(`/jobs/${id}/cancel`); toast.success(tr('Cancelled')); load() }
    catch { toast.error(tr('Cancel failed')) }
  }

  const filtered = jobs.filter(j =>
    j.name.toLowerCase().includes(filter.toLowerCase()) ||
    j.status.includes(filter.toLowerCase())
  )

  const sorted = [...filtered].sort((a, b) =>
    STATUS_ORDER.indexOf(a.status) - STATUS_ORDER.indexOf(b.status)
  )

  return (
    <AppShell>
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between pr-40">
          <div>
            <h1 className="text-2xl font-bold text-foreground">{tr('Training Jobs')}</h1>
            <p className="text-muted-foreground text-sm">{tr('{count} jobs total', { count: jobs.length })}</p>
          </div>
          <div className="flex gap-2">
            <button onClick={load} className="p-2 bg-muted rounded-lg text-muted-foreground hover:text-foreground" title={tr('Refresh')} aria-label={tr('Refresh')}>
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            </button>
            <Link href="/training/jobs/new"
              className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:bg-primary/90">
              <Plus className="w-4 h-4" /> {tr('New Job')}
            </Link>
          </div>
        </div>

        <input placeholder={tr('Filter jobs...')} value={filter} onChange={e => setFilter(e.target.value)}
          className="w-full max-w-sm px-4 py-2 bg-background border border-input rounded-lg text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring" />

        <div className="bg-card border border-border rounded-xl overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                {['Name', 'Type', 'Architecture', 'Status', 'Progress', 'Started', 'Actions'].map(h => (
                  <th key={h} className="text-left px-4 py-3 text-xs font-medium text-muted-foreground uppercase">{tr(h)}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {sorted.length === 0 ? (
                <tr><td colSpan={7} className="text-center py-12 text-muted-foreground">{tr('No jobs yet')}</td></tr>
              ) : sorted.map(job => (
                <tr key={job.id} className="hover:bg-accent/30 transition-colors">
                  <td className="px-4 py-3">
                    <Link href={`/training/jobs/${job.id}`} className="font-medium text-foreground hover:text-primary">
                      {job.name}
                    </Link>
                    {job.error_message && (
                      <p className="text-xs text-red-400 mt-0.5 truncate max-w-[200px]">{job.error_message}</p>
                    )}
                  </td>
                  <td className="px-4 py-3 text-xs text-muted-foreground">{tr(MODEL_TYPE_LABELS[job.model_type])}</td>
                  <td className="px-4 py-3 text-xs font-mono text-muted-foreground">{job.architecture}</td>
                  <td className="px-4 py-3">
                    <span className={`text-xs font-medium capitalize ${JOB_STATUS_COLORS[job.status]}`}>
                      {tr(job.status)}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <div className="w-24 bg-muted rounded-full h-1.5">
                        <div className="bg-primary h-1.5 rounded-full transition-all"
                          style={{ width: `${job.progress_pct}%` }} />
                      </div>
                      <span className="text-xs text-muted-foreground">
                        {job.current_epoch}/{job.total_epochs}
                      </span>
                    </div>
                    {job.best_metrics?.map50 !== undefined && (
                      <p className="text-xs text-emerald-400 mt-0.5">
                        mAP50: {(job.best_metrics.map50 * 100).toFixed(1)}%
                      </p>
                    )}
                  </td>
                  <td className="px-4 py-3 text-xs text-muted-foreground">
                    {job.started_at ? formatDateTime(job.started_at, locale) : '—'}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex gap-1">
                      <Link href={`/training/jobs/${job.id}`}
                        className="px-2 py-1 bg-muted text-muted-foreground rounded text-xs hover:text-foreground">
                        {tr('View')}
                      </Link>
                      {['queued', 'preparing', 'training'].includes(job.status) && (
                        <button onClick={() => handleCancel(job.id)} title={tr('Cancel')} aria-label={tr('Cancel')}
                          className="px-2 py-1 bg-red-500/10 text-red-400 rounded text-xs hover:bg-red-500/20">
                          <Square className="w-3 h-3" />
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </AppShell>
  )
}
