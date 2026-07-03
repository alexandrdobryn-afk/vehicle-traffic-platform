'use client'
import { useEffect, useState, useCallback } from 'react'
import AppShell from '@/components/shared/AppShell'
import api from '@/lib/api'
import { useWebSocket } from '@/hooks/useWebSocket'
import { useStreamUrl } from '@/hooks/useStreamUrl'
import { AnalyticsSummary, WSFrame, ActiveTrack, Camera, VehicleTrack, COLOR_HEX, EVENT_LABELS, EVENT_COLORS } from '@/types'
import { formatConfidence, formatDateTime } from '@/lib/utils'
import { Camera as CameraIcon, Car, CheckCircle, AlertTriangle, Zap, Activity, Clock } from 'lucide-react'
import { useTranslation } from '@/lib/i18n'

export default function DashboardPage() {
  const { t: tr, locale } = useTranslation()
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null)
  const [cameras, setCameras] = useState<Camera[]>([])
  const [activeTracks, setActiveTracks] = useState<ActiveTrack[]>([])
  const [recentEvents, setRecentEvents] = useState<any[]>([])
  const [alerts, setAlerts] = useState<any[]>([])
  const [selectedCam, setSelectedCam] = useState<number | null>(null)
  const selectedSource = cameras.find((source) => source.id === selectedCam)
  const streamUrl = useStreamUrl(
    selectedCam,
    !!selectedSource && (selectedSource.is_active || selectedSource.status === 'completed'),
  )

  const { latestFrame } = useWebSocket({
    cameraId: selectedCam || 0,
    enabled: !!selectedSource?.is_active,
    onFrame: (frame: WSFrame) => {
      setActiveTracks(frame.objects as any)
    },
    onAlert: (alert) => {
      setAlerts((prev) => [alert, ...prev].slice(0, 5))
    },
  })

  const load = useCallback(async () => {
    try {
      const [sumRes, camRes, videoRes, evRes] = await Promise.all([
        api.get('/analytics/summary'),
        api.get('/cameras'),
        api.get('/videos'),
        api.get('/events?limit=10'),
      ])
      setSummary(sumRes.data)
      const sources: Camera[] = [...camRes.data, ...videoRes.data]
      setCameras(sources)
      setRecentEvents(evRes.data)
      setSelectedCam((current) => {
        if (current && sources.some((source) => source.id === current)) return current
        const saved = Number(localStorage.getItem('vtp_dashboard_source_id'))
        if (saved && sources.some((source) => source.id === saved)) return saved
        return sources.find((source) => source.is_active)?.id
          ?? sources.find((source) => source.status === 'completed')?.id
          ?? sources[0]?.id
          ?? null
      })
    } catch {}
  }, [])

  useEffect(() => {
    load()
    const t = setInterval(load, 10000)
    return () => clearInterval(t)
  }, [load])

  useEffect(() => {
    if (!selectedCam) {
      setActiveTracks([])
      return
    }
    localStorage.setItem('vtp_dashboard_source_id', String(selectedCam))
    let cancelled = false
    const loadSelectedTracks = async () => {
      try {
        if (selectedSource?.is_active) {
          const response = await api.get('/tracks/active', { params: { camera_id: selectedCam } })
          if (!cancelled) setActiveTracks(response.data)
          return
        }
        const response = await api.get('/tracks', { params: { camera_id: selectedCam, limit: 200 } })
        const saved: VehicleTrack[] = response.data
        const latestRun = saved[0]?.processing_run_id
        const latestTracks = latestRun ? saved.filter((track) => track.processing_run_id === latestRun) : []
        if (!cancelled) setActiveTracks(latestTracks.map((track) => ({
          track_id: track.track_id,
          camera_id: track.camera_id,
          vehicle_class: track.vehicle_class,
          bbox: [0, 0, 0, 0],
          color: track.color,
          color_confidence: track.color_confidence,
          vehicle_make: track.vehicle_make,
          make_confidence: track.make_confidence,
          plate: track.final_plate,
          plate_status: track.plate_status,
          plate_confidence: track.final_plate_confidence,
          first_seen: track.first_seen,
          last_seen: track.last_seen,
        })))
      } catch {
        if (!cancelled) setActiveTracks([])
      }
    }
    loadSelectedTracks()
    const timer = setInterval(loadSelectedTracks, selectedSource?.is_active ? 2000 : 10000)
    return () => { cancelled = true; clearInterval(timer) }
  }, [selectedCam, selectedSource?.is_active])

  const stats = [
    { label: 'Vehicles Today', value: summary?.total_vehicles_today ?? 0, icon: Car, color: 'text-blue-400' },
    { label: 'Plates Recognized', value: summary?.total_plates_recognized ?? 0, icon: CheckCircle, color: 'text-emerald-400' },
    { label: 'OCR Success Rate', value: `${summary?.ocr_success_rate ?? 0}%`, icon: Activity, color: 'text-violet-400' },
    { label: 'Active Cameras', value: summary?.active_cameras ?? 0, icon: CameraIcon, color: 'text-amber-400' },
    { label: 'Avg FPS', value: summary?.avg_fps ?? 0, icon: Zap, color: 'text-cyan-400' },
    { label: 'Watchlist Hits', value: summary?.watchlist_matches_today ?? 0, icon: AlertTriangle, color: 'text-red-400' },
  ]

  return (
    <AppShell>
      <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between pr-40">
          <div>
            <h1 className="text-2xl font-bold text-foreground">{tr('Dashboard')}</h1>
            <p className="text-muted-foreground text-sm mt-0.5">{tr('Real-time traffic monitoring')}</p>
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 bg-emerald-500/10 border border-emerald-500/20 rounded-full">
            <span className="live-dot" />
            <span className="text-xs font-medium text-emerald-400">{tr('LIVE')}</span>
          </div>
        </div>

        {/* Alerts */}
        {alerts.length > 0 && (
          <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-xl flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-red-400 mt-0.5 shrink-0" />
            <div>
              <p className="font-semibold text-red-400 text-sm">{tr('Watchlist Alert')}</p>
              <p className="text-sm text-red-300">{tr('{plate} detected on Camera {camera}', { plate: alerts[0]?.plate || '—', camera: alerts[0]?.camera_id })}</p>
            </div>
          </div>
        )}

        {/* Stats grid */}
        <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
          {stats.map(({ label, value, icon: Icon, color }) => (
            <div key={label} className="bg-card border border-border rounded-xl p-4">
              <Icon className={`w-5 h-5 mb-3 ${color}`} />
              <p className="text-2xl font-bold text-foreground">{value}</p>
              <p className="text-xs text-muted-foreground mt-0.5">{tr(label)}</p>
            </div>
          ))}
        </div>

        {/* Main grid */}
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          {/* Live Feed */}
          <div className="xl:col-span-2 bg-card border border-border rounded-xl overflow-hidden">
            <div className="flex items-center justify-between p-4 border-b border-border">
              <div className="flex items-center gap-2">
                <span className="live-dot" />
                <span className="text-sm font-semibold text-foreground">{tr('Live Feed')}</span>
                {latestFrame && (
                  <span className="text-xs text-muted-foreground">
                    {latestFrame.fps.toFixed(1)} FPS · {latestFrame.latency_ms} ms
                  </span>
                )}
              </div>
              {/* Camera selector */}
              <select
                value={selectedCam || ''}
                onChange={(e) => setSelectedCam(Number(e.target.value))}
                className="text-xs bg-background border border-input rounded-md px-2 py-1 text-foreground"
              >
                {cameras.map((c) => (
                  <option key={c.id} value={c.id}>{c.name}</option>
                ))}
              </select>
            </div>

            {selectedSource && streamUrl ? (
              <div className="relative bg-black aspect-video">
                <img
                  src={streamUrl}
                  alt={tr('Live stream')}
                  className="w-full h-full object-contain"
                  onError={(e) => { (e.target as HTMLImageElement).style.display = 'none' }}
                />
                {/* Overlay info */}
                {latestFrame && (
                  <div className="absolute top-2 left-2 bg-black/70 rounded-lg px-2 py-1">
                    <p className="text-xs text-green-400 font-mono">
                      {latestFrame.fps.toFixed(1)} FPS · {tr('{count} vehicles', { count: latestFrame.objects.length })}
                    </p>
                  </div>
                )}
              </div>
            ) : (
              <div className="aspect-video bg-muted flex items-center justify-center">
                <p className="text-muted-foreground text-sm">{tr('No camera selected')}</p>
              </div>
            )}
          </div>

          {/* Active Vehicles */}
          <div className="bg-card border border-border rounded-xl flex flex-col">
            <div className="p-4 border-b border-border">
              <h3 className="text-sm font-semibold text-foreground">
                {tr(selectedSource?.is_active ? 'Active Vehicles' : 'Vehicles')} <span className="text-muted-foreground font-normal">({activeTracks.length})</span>
              </h3>
            </div>
            <div className="flex-1 overflow-y-auto divide-y divide-border">
              {activeTracks.length === 0 ? (
                <div className="p-6 text-center text-muted-foreground text-sm">{tr('No vehicles detected')}</div>
              ) : (
                activeTracks.map((t) => (
                  <div key={`${t.camera_id}-${t.track_id}`} className="p-3 hover:bg-accent/50 transition-colors">
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex items-center gap-2 min-w-0">
                        {/* Color dot */}
                        <span
                          className="w-3 h-3 rounded-full shrink-0 border border-white/10"
                          style={{ backgroundColor: COLOR_HEX[t.color] || '#6b7280' }}
                        />
                        <div className="min-w-0">
                          <div className="flex items-center gap-1.5">
                            <span className="text-xs font-bold text-foreground">#{t.track_id}</span>
                            <span className="text-xs text-muted-foreground">{tr(t.vehicle_class)}</span>
                          </div>
                          {t.plate ? (
                            <div className="flex items-center gap-1.5 mt-0.5">
                              <span className="plate-badge bg-primary/10 text-primary">{t.plate}</span>
                              {t.plate_status === 'verified' && (
                                <CheckCircle className="w-3 h-3 text-emerald-400" />
                              )}
                            </div>
                          ) : (
                            <span className="text-xs text-muted-foreground italic">{tr('searching plate...')}</span>
                          )}
                        </div>
                      </div>
                      <span className="text-xs text-muted-foreground shrink-0">
                        {formatConfidence(t.plate_confidence)}
                      </span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Bottom row: Cameras status + Recent events */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          {/* Camera status */}
          <div className="bg-card border border-border rounded-xl">
            <div className="p-4 border-b border-border">
              <h3 className="text-sm font-semibold text-foreground">{tr('Cameras')}</h3>
            </div>
            <div className="divide-y divide-border">
              {cameras.length === 0 ? (
                <div className="p-4 text-center text-muted-foreground text-sm">{tr('No cameras configured')}</div>
              ) : (
                cameras.map((cam) => (
                  <div key={cam.id} className="flex items-center justify-between px-4 py-3">
                    <div className="flex items-center gap-3">
                      <span className={`w-2 h-2 rounded-full ${cam.status === 'online' ? 'bg-emerald-400' : 'bg-red-400'}`} />
                      <div>
                        <p className="text-sm font-medium text-foreground">{cam.name}</p>
                        <p className="text-xs text-muted-foreground">{cam.location || tr('No location')}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <span className="text-xs font-medium text-muted-foreground">{tr(cam.ai_mode.charAt(0).toUpperCase() + cam.ai_mode.slice(1))}</span>
                      <p className="text-xs text-muted-foreground">{tr('{count} fps max', { count: cam.max_fps })}</p>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Recent events */}
          <div className="bg-card border border-border rounded-xl">
            <div className="p-4 border-b border-border">
              <h3 className="text-sm font-semibold text-foreground">{tr('Recent Events')}</h3>
            </div>
            <div className="divide-y divide-border">
              {recentEvents.length === 0 ? (
                <div className="p-4 text-center text-muted-foreground text-sm">{tr('No events')}</div>
              ) : (
                recentEvents.slice(0, 8).map((ev) => (
                  <div key={ev.id} className="flex items-center gap-3 px-4 py-2.5">
                    <Clock className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
                    <div className="min-w-0 flex-1">
                      <span className={`text-xs font-medium ${EVENT_COLORS[ev.event_type] || 'text-foreground'}`}>
                        {tr(EVENT_LABELS[ev.event_type] || ev.event_type)}
                      </span>
                      {ev.payload_json?.plate && (
                        <span className="ml-2 text-xs plate-badge bg-muted text-foreground">
                          {ev.payload_json.plate}
                        </span>
                      )}
                    </div>
                    <span className="text-xs text-muted-foreground shrink-0">
                      {formatDateTime(ev.created_at, locale)}
                    </span>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </AppShell>
  )
}
