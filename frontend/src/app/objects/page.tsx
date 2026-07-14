'use client'

import { useCallback, useEffect, useMemo, useState } from 'react'
import { AlertTriangle, Box, Filter, ImageOff, Maximize2, RefreshCw } from 'lucide-react'
import toast from 'react-hot-toast'

import AppShell from '@/components/shared/AppShell'
import api, { apiErrorMessage, storageUrl } from '@/lib/api'
import { ObjectTrack } from '@/types'
import { formatConfidence, formatDateTime } from '@/lib/utils'
import { useTranslation } from '@/lib/i18n'

const EMPTY_VALUE = '-'

function formatVideoTime(seconds: number | null | undefined) {
  const numeric = Number(seconds)
  if (!Number.isFinite(numeric)) return EMPTY_VALUE
  const whole = Math.max(0, Math.floor(numeric))
  const hours = Math.floor(whole / 3600)
  const minutes = Math.floor((whole % 3600) / 60)
  const remaining = whole % 60
  return [hours, minutes, remaining].map((part) => String(part).padStart(2, '0')).join(':')
}

function formatCoordinates(bbox: unknown) {
  if (!Array.isArray(bbox) || bbox.length !== 4) return EMPTY_VALUE
  const values = bbox.map((item) => Number(item))
  if (values.some((item) => !Number.isFinite(item))) return EMPTY_VALUE
  return `${Math.round(values[0])}, ${Math.round(values[1])} -> ${Math.round(values[2])}, ${Math.round(values[3])}`
}

function safeNumber(value: unknown): number | null {
  const numeric = Number(value)
  return Number.isFinite(numeric) ? numeric : null
}

function safeText(value: unknown, fallback = EMPTY_VALUE) {
  return typeof value === 'string' && value.trim() ? value : fallback
}

export default function ObjectsPage() {
  const { t, locale } = useTranslation()
  const [objects, setObjects] = useState<ObjectTrack[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [classFilter, setClassFilter] = useState('')
  const [sourceFilter, setSourceFilter] = useState('')
  const [selectedImage, setSelectedImage] = useState<{ url: string; label: string } | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const params = Object.fromEntries(Object.entries({
        limit: 500,
        object_class: classFilter || undefined,
        source_id: sourceFilter || undefined,
      }).filter(([, value]) => value !== undefined))
      const response = await api.get('/objects', { params })
      setObjects(Array.isArray(response.data) ? response.data : [])
    } catch (loadError: any) {
      const message = apiErrorMessage(loadError, t('Failed to load objects'))
      setError(message)
      toast.error(message)
    } finally {
      setLoading(false)
    }
  }, [classFilter, sourceFilter, t])

  useEffect(() => { load() }, [load])

  const sourceIds = useMemo(() => {
    const ids = objects.map((item) => safeNumber(item.source_id)).filter((item): item is number => item != null)
    return [...new Set(ids)].sort((a, b) => a - b)
  }, [objects])

  return (
    <AppShell>
      <div className="space-y-6 p-6">
        <div className="flex items-start justify-between gap-4 pr-40">
          <div>
            <h1 className="text-2xl font-bold">{t('Objects')}</h1>
            <p className="mt-1 text-sm text-muted-foreground">{t('{count} detected objects', { count: objects.length })}</p>
          </div>
          <button onClick={load} className="rounded-lg bg-muted p-2 text-muted-foreground transition-colors hover:text-foreground" title={t('Refresh')} aria-label={t('Refresh')}>
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>

        <section className="rounded-xl border border-border bg-card p-4">
          <div className="mb-3 flex items-center gap-2">
            <Filter className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">{t('Filters')}</span>
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <input value={classFilter} onChange={(event) => setClassFilter(event.target.value)} placeholder={t('Object class')} className="rounded-lg border border-input bg-background px-3 py-2 text-sm" />
            <select value={sourceFilter} onChange={(event) => setSourceFilter(event.target.value)} className="rounded-lg border border-input bg-background px-3 py-2 text-sm">
              <option value="">{t('All sources')}</option>
              {sourceIds.map((sourceId) => <option key={sourceId} value={sourceId}>{t('Source')} #{sourceId}</option>)}
            </select>
          </div>
        </section>

        {error && (
          <div className="flex items-start gap-2 rounded-xl border border-red-500/20 bg-red-500/10 p-4 text-sm text-red-200">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            <div>
              <div className="font-medium">{t('Failed to load objects')}</div>
              <div className="mt-1 text-red-100/80">{error}</div>
            </div>
          </div>
        )}

        <section className="overflow-x-auto rounded-xl border border-border bg-card">
          <table className="min-w-[980px] w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                {['Screenshot', 'Object', 'Class', 'Video time', 'Coordinates', 'Confidence', 'First detected', 'Source'].map((heading) => (
                  <th key={heading} className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wide text-muted-foreground">{t(heading)}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {!objects.length ? (
                <tr><td colSpan={8} className="py-12 text-center text-muted-foreground">{t('No objects found')}</td></tr>
              ) : objects.map((object, index) => {
                const cropUrl = storageUrl(safeText(object.best_crop_path, ''))
                const trackId = safeNumber(object.track_id)
                const sourceId = safeNumber(object.source_id)
                const objectClass = safeText(object.object_class, 'unknown')
                const rowKey = safeNumber(object.id) ?? `${sourceId ?? 'source'}-${trackId ?? 'track'}-${index}`
                const objectLabel = `${t('Object')} #${trackId ?? EMPTY_VALUE}`
                return (
                  <tr key={rowKey} className="hover:bg-accent/30">
                    <td className="px-4 py-3">
                      {cropUrl ? (
                        <button onClick={() => setSelectedImage({ url: cropUrl, label: objectLabel })} className="group relative block h-16 w-24 overflow-hidden rounded-md border border-border bg-muted" title={t('Open enlarged fragment')}>
                          <img src={cropUrl} alt={objectLabel} className="h-full w-full object-cover" />
                          <span className="absolute inset-0 hidden items-center justify-center bg-black/55 text-white group-hover:flex"><Maximize2 className="h-4 w-4" /></span>
                        </button>
                      ) : (
                        <span className="flex h-16 w-24 flex-col items-center justify-center rounded-md border border-dashed border-border text-xs text-muted-foreground"><ImageOff className="mb-1 h-4 w-4" />{t('No crop')}</span>
                      )}
                    </td>
                    <td className="px-4 py-3 font-mono text-xs">#{trackId ?? EMPTY_VALUE}</td>
                    <td className="px-4 py-3"><span className="inline-flex items-center gap-1.5 rounded-full bg-primary/10 px-2 py-1 text-xs font-medium text-primary"><Box className="h-3.5 w-3.5" />{t(objectClass)}</span></td>
                    <td className="px-4 py-3 font-mono text-xs text-muted-foreground">{formatVideoTime(object.first_video_timestamp_seconds)}</td>
                    <td className="px-4 py-3 font-mono text-xs text-muted-foreground">{formatCoordinates(object.last_bbox)}</td>
                    <td className="px-4 py-3 text-xs">{formatConfidence(object.confidence)}</td>
                    <td className="whitespace-nowrap px-4 py-3 text-xs text-muted-foreground">{formatDateTime(object.first_seen, locale)}</td>
                    <td className="px-4 py-3 text-xs text-muted-foreground">#{sourceId ?? EMPTY_VALUE}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </section>
      </div>

      {selectedImage && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-6" role="dialog" aria-modal="true" aria-label={selectedImage.label} onClick={() => setSelectedImage(null)}>
          <button className="max-h-full max-w-full" onClick={(event) => event.stopPropagation()}><img src={selectedImage.url} alt={selectedImage.label} className="max-h-[85vh] max-w-[90vw] rounded-lg" /></button>
        </div>
      )}
    </AppShell>
  )
}
