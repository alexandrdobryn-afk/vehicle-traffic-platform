'use client'
import { useEffect, useMemo, useState } from 'react'
import AppShell from '@/components/shared/AppShell'
import trainingApi from '@/lib/trainingApi'
import { TActiveLearningItem, TDataset, TDatasetImage } from '@/types/training'
import { formatDateTime } from '@/lib/utils'
import Link from 'next/link'
import { CheckCircle, Clock, Filter, SkipForward, Target, XCircle } from 'lucide-react'
import toast from 'react-hot-toast'
import { useTranslation } from '@/lib/i18n'

const STATUSES = ['', 'open', 'in_review', 'annotated', 'approved', 'rejected', 'skipped']

export default function ActiveLearningPage() {
  const { t, locale } = useTranslation()
  const [items, setItems] = useState<TActiveLearningItem[]>([])
  const [datasets, setDatasets] = useState<TDataset[]>([])
  const [images, setImages] = useState<Record<number, TDatasetImage>>({})
  const [status, setStatus] = useState('')
  const [loading, setLoading] = useState(false)

  const datasetById = useMemo(() => Object.fromEntries(datasets.map(item => [item.id, item])), [datasets])

  const load = async () => {
    setLoading(true)
    try {
      const [queue, datasetResponse] = await Promise.all([
        trainingApi.get(`/active-learning${status ? `?status=${status}` : ''}`),
        trainingApi.get('/datasets'),
      ])
      setItems(queue.data)
      setDatasets(datasetResponse.data)

      const needed = queue.data.slice(0, 80) as TActiveLearningItem[]
      const imagePairs = await Promise.all(
        needed.map(async item => {
          try {
            const response = await trainingApi.get(`/datasets/${item.dataset_id}/images?limit=500`)
            const image = (response.data as TDatasetImage[]).find(candidate => candidate.id === item.image_id)
            return image ? [image.id, image] as const : null
          } catch {
            return null
          }
        })
      )
      setImages(Object.fromEntries(imagePairs.filter(Boolean) as Array<readonly [number, TDatasetImage]>))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [status])

  const updateStatus = async (item: TActiveLearningItem, nextStatus: TActiveLearningItem['status']) => {
    try {
      await trainingApi.patch(`/active-learning/${item.id}/status`, { status: nextStatus })
      toast.success(t('Queue item updated'))
      load()
    } catch {
      toast.error(t('Update failed'))
    }
  }

  return (
    <AppShell>
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between pr-40">
          <div>
            <h1 className="text-2xl font-bold text-foreground">{t('Active Learning')}</h1>
            <p className="text-sm text-muted-foreground">{t('{count} frames need attention', { count: items.length })}</p>
          </div>
          <button onClick={load} className="px-3 py-2 bg-muted rounded-lg text-sm text-muted-foreground hover:text-foreground">
            {loading ? t('Refreshing...') : t('Refresh')}
          </button>
        </div>

        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-muted-foreground" />
          {STATUSES.map(value => (
            <button
              key={value || 'all'}
              onClick={() => setStatus(value)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium ${status === value ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground hover:text-foreground'}`}
            >
              {t(value || 'All')}
            </button>
          ))}
        </div>

        <div className="space-y-3">
          {items.length === 0 ? (
            <div className="rounded-xl border border-dashed border-border p-12 text-center">
              <Target className="mx-auto mb-3 h-10 w-10 text-muted-foreground" />
              <p className="font-medium text-foreground">{t('No active learning items')}</p>
              <p className="text-sm text-muted-foreground">{t('Rejected predictions, low confidence frames and error report items will appear here')}</p>
            </div>
          ) : items.map(item => {
            const dataset = datasetById[item.dataset_id]
            const image = images[item.image_id]
            return (
              <div key={item.id} className="rounded-xl border border-border bg-card p-4">
                <div className="flex items-start justify-between gap-4">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="rounded bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">{t(item.reason)}</span>
                      <span className="rounded bg-muted px-2 py-0.5 text-xs text-muted-foreground">{t(item.status)}</span>
                      <span className="text-xs text-muted-foreground">{t('Priority')}: {item.priority_score.toFixed(0)}</span>
                    </div>
                    <p className="text-sm font-medium text-foreground truncate">{dataset?.name || `${t('Dataset')} #${item.dataset_id}`}</p>
                    <p className="text-xs text-muted-foreground">
                      {image?.filename || `${t('Image')} #${item.image_id}`} · {item.source} · {formatDateTime(item.created_at, locale)}
                    </p>
                    {item.suggested_class && <p className="mt-1 text-xs text-muted-foreground">{t('Suggested class')}: {item.suggested_class}</p>}
                  </div>
                  <div className="flex shrink-0 flex-wrap justify-end gap-2">
                    <Link href={`/training/annotate/${item.dataset_id}`}
                      className="flex items-center gap-1.5 rounded-lg bg-primary/10 px-3 py-1.5 text-xs text-primary hover:bg-primary/20">
                      <Clock className="h-3.5 w-3.5" /> {t('Review')}
                    </Link>
                    <button onClick={() => updateStatus(item, 'approved')}
                      className="flex items-center gap-1.5 rounded-lg bg-emerald-500/10 px-3 py-1.5 text-xs text-emerald-400 hover:bg-emerald-500/20">
                      <CheckCircle className="h-3.5 w-3.5" /> {t('Approve')}
                    </button>
                    <button onClick={() => updateStatus(item, 'skipped')}
                      className="flex items-center gap-1.5 rounded-lg bg-muted px-3 py-1.5 text-xs text-muted-foreground hover:text-foreground">
                      <SkipForward className="h-3.5 w-3.5" /> {t('Skip')}
                    </button>
                    <button onClick={() => updateStatus(item, 'rejected')}
                      className="flex items-center gap-1.5 rounded-lg bg-red-500/10 px-3 py-1.5 text-xs text-red-400 hover:bg-red-500/20">
                      <XCircle className="h-3.5 w-3.5" /> {t('Reject')}
                    </button>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </AppShell>
  )
}
