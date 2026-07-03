'use client'
import { useEffect, useState, useCallback, useRef, PointerEvent as ReactPointerEvent, WheelEvent as ReactWheelEvent } from 'react'
import AppShell from '@/components/shared/AppShell'
import api from '@/lib/api'
import { VehicleTrack, COLOR_HEX } from '@/types'
import { formatDateTime, formatDuration, formatConfidence } from '@/lib/utils'
import { CheckCircle, Maximize2, Minus, Plus, RotateCcw, Search, X } from 'lucide-react'
import { useTranslation } from '@/lib/i18n'

type EvidencePreview = { src: string; label: string }

function useRetryingImage(src: string) {
  const [attempt, setAttempt] = useState(0)
  const [failed, setFailed] = useState(false)
  const retryTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    setAttempt(0)
    setFailed(false)
    return () => {
      if (retryTimer.current) clearTimeout(retryTimer.current)
    }
  }, [src])

  const onError = () => {
    setFailed(true)
    if (retryTimer.current) clearTimeout(retryTimer.current)
    retryTimer.current = setTimeout(() => setAttempt((value) => value + 1), 2500)
  }
  const onLoad = () => setFailed(false)
  const separator = src.includes('?') ? '&' : '?'
  return { url: `${src}${separator}preview_attempt=${attempt}`, failed, onError, onLoad }
}

function EvidenceImage({ src, alt, className }: { src: string; alt: string; className: string }) {
  const image = useRetryingImage(src)

  return (
    <span className="relative block overflow-hidden rounded border border-border bg-[linear-gradient(45deg,#182235_25%,transparent_25%),linear-gradient(-45deg,#182235_25%,transparent_25%),linear-gradient(45deg,transparent_75%,#182235_75%),linear-gradient(-45deg,transparent_75%,#182235_75%)] bg-[length:12px_12px] bg-[position:0_0,0_6px,6px_-6px,-6px_0px]">
      <img
        key={image.url}
        src={image.url}
        alt={alt}
        className={`${className} ${image.failed ? 'invisible' : ''}`}
        onLoad={image.onLoad}
        onError={image.onError}
      />
      {image.failed && <span className="absolute inset-0 grid place-items-center px-1 text-[9px] text-amber-300">retrying…</span>}
    </span>
  )
}

function EnlargedEvidenceImage({ preview, zoom, offset }: {
  preview: EvidencePreview
  zoom: number
  offset: { x: number; y: number }
}) {
  const image = useRetryingImage(preview.src)
  return (
    <>
      <img
        key={image.url}
        src={image.url}
        alt={preview.label}
        draggable={false}
        onLoad={image.onLoad}
        onError={image.onError}
        className={`max-h-[82vh] max-w-[92vw] select-none object-contain transition-transform duration-100 ${image.failed ? 'invisible' : ''}`}
        style={{ transform: `translate3d(${offset.x}px, ${offset.y}px, 0) scale(${zoom})` }}
      />
      {image.failed && <div className="absolute text-sm text-amber-300">{`Preview unavailable — retrying…`}</div>}
    </>
  )
}

export default function VehiclesPage() {
  const { t: tr, locale, language } = useTranslation()
  const [tracks, setTracks] = useState<VehicleTrack[]>([])
  const [filters, setFilters] = useState({ plate: '', color: '', camera_id: '' })
  const [loading, setLoading] = useState(false)
  const [preview, setPreview] = useState<EvidencePreview | null>(null)
  const [zoom, setZoom] = useState(1)
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const dragRef = useRef<{ x: number; y: number; offsetX: number; offsetY: number } | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const params: any = { limit: 200 }
      if (filters.plate) params.plate = filters.plate
      if (filters.color) params.color = filters.color
      if (filters.camera_id) params.camera_id = filters.camera_id
      const res = await api.get('/tracks', { params })
      setTracks(res.data)
    } finally { setLoading(false) }
  }, [filters])

  useEffect(() => { load() }, [load])

  useEffect(() => {
    if (!preview) return
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setPreview(null)
    }
    window.addEventListener('keydown', closeOnEscape)
    return () => window.removeEventListener('keydown', closeOnEscape)
  }, [preview])

  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  const storageUrl = (path: string, version?: string | number | null) => {
    const normalized = path.replace(/\\/g, '/')
    const relative = normalized.includes('/storage/')
      ? normalized.split('/storage/')[1]
      : normalized.replace(/^\/?storage\//, '')
    const url = `${API_URL}/storage/${relative.split('/').map(encodeURIComponent).join('/')}`
    return version == null ? url : `${url}?evidence_version=${encodeURIComponent(String(version))}`
  }
  const openPreview = (src: string, label: string) => {
    setPreview({ src, label })
    setZoom(1)
    setOffset({ x: 0, y: 0 })
  }
  const changeZoom = (next: number) => {
    const clamped = Math.min(8, Math.max(0.5, next))
    setZoom(clamped)
    if (clamped <= 1) setOffset({ x: 0, y: 0 })
  }
  const startDrag = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (zoom <= 1) return
    event.currentTarget.setPointerCapture(event.pointerId)
    dragRef.current = { x: event.clientX, y: event.clientY, offsetX: offset.x, offsetY: offset.y }
  }
  const drag = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!dragRef.current) return
    setOffset({
      x: dragRef.current.offsetX + event.clientX - dragRef.current.x,
      y: dragRef.current.offsetY + event.clientY - dragRef.current.y,
    })
  }
  const stopDrag = () => { dragRef.current = null }
  const zoomWithWheel = (event: ReactWheelEvent<HTMLDivElement>) => {
    event.preventDefault()
    changeZoom(zoom + (event.deltaY < 0 ? 0.25 : -0.25))
  }

  return (
    <AppShell>
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between pr-40">
          <div>
            <h1 className="text-2xl font-bold text-foreground">{tr('Vehicles')}</h1>
            <p className="text-muted-foreground text-sm">{tr('{count} tracks', { count: tracks.length })}</p>
          </div>
        </div>

        {/* Filters */}
        <div className="flex flex-wrap gap-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <input
              placeholder={tr('Search plate...')}
              value={filters.plate}
              onChange={(e) => setFilters({ ...filters, plate: e.target.value })}
              className="pl-9 pr-4 py-2 bg-background border border-input rounded-lg text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
            />
          </div>
          <select value={filters.color} onChange={(e) => setFilters({ ...filters, color: e.target.value })}
            className="px-3 py-2 bg-background border border-input rounded-lg text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring">
            <option value="">{tr('All colors')}</option>
            {Object.keys(COLOR_HEX).map((c) => <option key={c} value={c}>{tr(c)}</option>)}
          </select>
          <input placeholder={tr('Camera ID')} value={filters.camera_id} onChange={(e) => setFilters({ ...filters, camera_id: e.target.value })}
            className="w-28 px-3 py-2 bg-background border border-input rounded-lg text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring" />
        </div>

        {/* Table */}
        <div className="bg-card border border-border rounded-xl overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                {['ID', 'Plate', 'Color', 'Vehicle', 'Camera', 'Duration', 'First Seen', 'Last Seen', 'Recognition', 'Evidence'].map((h) => (
                  <th key={h} className="text-left px-4 py-3 text-xs font-medium text-muted-foreground uppercase">{tr(h)}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {tracks.length === 0 ? (
                <tr><td colSpan={10} className="text-center py-12 text-muted-foreground">{tr('No vehicles found')}</td></tr>
              ) : tracks.map((t) => (
                <tr key={t.id} className="hover:bg-accent/30 transition-colors">
                  <td className="px-4 py-3 text-xs font-mono text-muted-foreground">
                    <div>#{t.track_id}</div>
                    <div className="text-[10px] opacity-60" title={t.processing_run_id}>{t.processing_run_id.slice(0, 6)}</div>
                  </td>
                  <td className="px-4 py-3">
                    {t.final_plate ? (
                      <div className="flex items-center gap-1.5">
                        <span className="plate-badge bg-primary/10 text-primary">{t.final_plate}</span>
                        {t.plate_status === 'verified' && <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />}
                        <span className="text-xs text-muted-foreground">{formatConfidence(t.final_plate_confidence)}</span>
                      </div>
                    ) : (
                      <div>
                        <span className="text-xs text-muted-foreground italic">{tr(t.plate_status)}</span>
                        {t.recognition_diagnostics?.plate?.reason && (
                          <div className="text-[10px] text-amber-400/80 mt-1">{tr(t.recognition_diagnostics.plate.reason)}</div>
                        )}
                      </div>
                    )}
                  </td>
                  <td className="px-4 py-2.5">
                    <div className="min-w-28">
                      <div className="flex items-center gap-2">
                        <span className="w-3 h-3 rounded-full border border-white/10" style={{ backgroundColor: t.recognition_diagnostics?.color?.html_hex || COLOR_HEX[t.color] }} />
                        <span className="text-xs text-muted-foreground">{tr(t.color)}</span>
                        <span className="text-[10px] text-muted-foreground/70">{formatConfidence(t.color_confidence)}</span>
                      </div>
                      <div className="mt-0.5 text-[10px] text-muted-foreground/70">
                        <code className="text-foreground/70">
                          {(t.recognition_diagnostics?.color?.html_hex || COLOR_HEX[t.color] || COLOR_HEX.unknown).toUpperCase()}
                        </code>
                      </div>
                    </div>
                  </td>
                  <td className="px-4 py-2.5">
                    <div className="whitespace-nowrap text-xs font-medium text-foreground">
                      {t.vehicle_make && t.vehicle_make !== 'unknown' ? t.vehicle_make : '—'}
                      {t.vehicle_make && t.vehicle_make !== 'unknown' && <span className="ml-1 text-[10px] font-normal text-muted-foreground">{formatConfidence(t.make_confidence)}</span>}
                    </div>
                    <div className="mt-0.5 text-[10px] text-muted-foreground">{tr(t.vehicle_class)}</div>
                  </td>
                  <td className="px-4 py-3 text-xs text-muted-foreground">#{t.camera_id}</td>
                  <td className="px-4 py-3 text-xs text-muted-foreground">{formatDuration(t.duration_seconds, language)}</td>
                  <td className="px-4 py-3 text-xs text-muted-foreground whitespace-nowrap">{formatDateTime(t.first_seen, locale)}</td>
                  <td className="px-4 py-3 text-xs text-muted-foreground whitespace-nowrap">{formatDateTime(t.last_seen, locale)}</td>
                  <td className="px-4 py-2.5">
                    <div className="min-w-28 text-[10px] text-muted-foreground">
                      <div className="truncate">{tr(t.recognition_diagnostics?.plate?.reason || t.plate_status)}</div>
                      <div className="mt-1 flex flex-wrap gap-x-2 gap-y-0.5">
                        {t.recognition_diagnostics?.plate?.crop_size && <span>{t.recognition_diagnostics.plate.crop_size.join('×')} px</span>}
                        {t.recognition_diagnostics?.plate?.detector_confidence != null && <span>DET {formatConfidence(t.recognition_diagnostics.plate.detector_confidence)}</span>}
                        {t.recognition_diagnostics?.plate?.ocr_confidence != null && <span>OCR {formatConfidence(t.recognition_diagnostics.plate.ocr_confidence)}</span>}
                      </div>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    {(t.best_vehicle_crop_path || t.best_plate_crop_path) ? (
                      <div className="flex items-center gap-1">
                        {t.best_vehicle_crop_path && <button
                          type="button"
                          className="group relative rounded focus:outline-none focus:ring-2 focus:ring-primary"
                          onClick={() => openPreview(storageUrl(t.best_vehicle_crop_path!, t.last_seen), tr('vehicle crop'))}
                          aria-label={tr('Open enlarged fragment')}
                        >
                          <EvidenceImage
                            src={storageUrl(t.best_vehicle_crop_path, t.last_seen)}
                            alt={tr('vehicle crop')}
                            className="w-16 h-10 object-contain"
                          />
                          <span className="absolute inset-0 grid place-items-center rounded bg-black/55 opacity-0 transition-opacity group-hover:opacity-100 group-focus:opacity-100">
                            <Maximize2 className="h-4 w-4 text-white" />
                          </span>
                        </button>}
                        {t.best_plate_crop_path && <button
                          type="button"
                          className="group relative rounded focus:outline-none focus:ring-2 focus:ring-primary"
                          onClick={() => openPreview(storageUrl(
                            t.best_plate_crop_path!,
                            t.recognition_diagnostics?.plate?.best_crop_score ?? t.last_seen,
                          ), tr('plate crop'))}
                          aria-label={tr('Open enlarged fragment')}
                        >
                          <EvidenceImage
                            src={storageUrl(
                              t.best_plate_crop_path,
                              t.recognition_diagnostics?.plate?.best_crop_score ?? t.last_seen,
                            )}
                            alt={tr('plate crop')}
                            className="w-20 h-10 object-contain bg-white"
                          />
                          <span className="absolute inset-0 grid place-items-center rounded bg-black/55 opacity-0 transition-opacity group-hover:opacity-100 group-focus:opacity-100">
                            <Maximize2 className="h-4 w-4 text-white" />
                          </span>
                        </button>}
                      </div>
                    ) : <span className="text-xs text-muted-foreground">—</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {preview && (
        <div
          className="fixed inset-0 z-[100] flex flex-col bg-black/95 backdrop-blur-sm"
          role="dialog"
          aria-modal="true"
          aria-label={preview.label}
        >
          <div className="flex h-16 shrink-0 items-center justify-between border-b border-white/10 px-4 text-white">
            <div>
              <div className="text-sm font-medium">{preview.label}</div>
              <div className="text-xs text-white/55">{Math.round(zoom * 100)}%</div>
            </div>
            <div className="flex items-center gap-2">
              <button type="button" onClick={() => changeZoom(zoom - 0.25)} className="rounded-lg border border-white/15 p-2 hover:bg-white/10" aria-label={tr('Zoom out')}>
                <Minus className="h-5 w-5" />
              </button>
              <button type="button" onClick={() => changeZoom(zoom + 0.25)} className="rounded-lg border border-white/15 p-2 hover:bg-white/10" aria-label={tr('Zoom in')}>
                <Plus className="h-5 w-5" />
              </button>
              <button type="button" onClick={() => { setZoom(1); setOffset({ x: 0, y: 0 }) }} className="rounded-lg border border-white/15 p-2 hover:bg-white/10" aria-label={tr('Reset zoom')}>
                <RotateCcw className="h-5 w-5" />
              </button>
              <button type="button" onClick={() => setPreview(null)} className="ml-2 rounded-lg border border-white/15 p-2 hover:bg-white/10" aria-label={tr('Close')}>
                <X className="h-5 w-5" />
              </button>
            </div>
          </div>
          <div
            className={`relative flex flex-1 touch-none items-center justify-center overflow-hidden ${zoom > 1 ? 'cursor-grab active:cursor-grabbing' : 'cursor-zoom-in'}`}
            onPointerDown={startDrag}
            onPointerMove={drag}
            onPointerUp={stopDrag}
            onPointerCancel={stopDrag}
            onWheel={zoomWithWheel}
            onDoubleClick={() => changeZoom(zoom === 1 ? 2 : 1)}
          >
            <EnlargedEvidenceImage preview={preview} zoom={zoom} offset={offset} />
          </div>
          <div className="shrink-0 py-3 text-center text-xs text-white/45">{tr('Use wheel or buttons to zoom; drag to inspect')}</div>
        </div>
      )}
    </AppShell>
  )
}
