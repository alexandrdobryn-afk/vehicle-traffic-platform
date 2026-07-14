'use client'

import { useCallback, useEffect, useState } from 'react'
import { Activity, Box, Camera as CameraIcon, Clock, FileVideo, Route, Zap } from 'lucide-react'

import AppShell from '@/components/shared/AppShell'
import api from '@/lib/api'
import { useStreamUrl } from '@/hooks/useStreamUrl'
import { useWebSocket } from '@/hooks/useWebSocket'
import { Camera, ObjectTrack, WSFrame } from '@/types'
import { formatDateTime } from '@/lib/utils'
import { useTranslation } from '@/lib/i18n'

type DisplayObject = {
  track_id: number
  object_class: string
  confidence: number
  state?: string
  speed?: number
}

export default function DashboardPage() {
  const { t, locale } = useTranslation()
  const [sources, setSources] = useState<Camera[]>([])
  const [objects, setObjects] = useState<DisplayObject[]>([])
  const [recentObjects, setRecentObjects] = useState<ObjectTrack[]>([])
  const [totalObjects, setTotalObjects] = useState(0)
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const selected = sources.find((source) => source.id === selectedId)
  const streamUrl = useStreamUrl(selectedId, Boolean(selected && (selected.is_active || selected.status === 'completed')))

  const { latestFrame } = useWebSocket({
    cameraId: selectedId || 0,
    enabled: Boolean(selected?.is_active),
    onFrame: (frame: WSFrame) => setObjects(frame.objects.map((object) => ({
      track_id: object.track_id,
      object_class: object.object_class || 'object',
      confidence: object.detection_confidence ?? 0,
      state: object.state,
      speed: object.speed_pixels_per_second,
    }))),
  })

  const load = useCallback(async () => {
    try {
      const [cameraResponse, videoResponse, objectResponse, recentObjectResponse] = await Promise.all([
        api.get('/cameras'), api.get('/videos'), api.get('/objects?limit=500'), api.get('/objects?limit=10'),
      ])
      const nextSources: Camera[] = [...cameraResponse.data, ...videoResponse.data]
      setSources(nextSources)
      setRecentObjects(recentObjectResponse.data)
      setTotalObjects(objectResponse.data.length)
      setSelectedId((current) => current && nextSources.some((item) => item.id === current)
        ? current
        : nextSources.find((item) => item.is_active)?.id ?? nextSources.find((item) => item.status === 'completed')?.id ?? nextSources[0]?.id ?? null)
    } catch {}
  }, [])

  useEffect(() => {
    load()
    const timer = setInterval(load, 10000)
    return () => clearInterval(timer)
  }, [load])

  useEffect(() => {
    if (!selectedId || selected?.is_active) return
    api.get('/objects', { params: { source_id: selectedId, limit: 300 } }).then(({ data }) => {
      const tracks: ObjectTrack[] = data
      const run = tracks[0]?.processing_run_id
      setObjects(tracks.filter((item) => !run || item.processing_run_id === run).map((item) => ({
        track_id: item.track_id,
        object_class: item.object_class,
        confidence: item.confidence,
        state: item.state,
        speed: item.speed_pixels_per_second,
      })))
    }).catch(() => setObjects([]))
  }, [selectedId, selected?.is_active])

  const activeSources = sources.filter((source) => source.is_active).length
  const videoCount = sources.filter((source) => source.source_type === 'file').length
  const cameraCount = sources.length - videoCount
  const stats = [
    { label: 'Detected objects', value: totalObjects, icon: Box, color: 'text-blue-400' },
    { label: 'Active sources', value: activeSources, icon: Activity, color: 'text-emerald-400' },
    { label: 'Cameras', value: cameraCount, icon: CameraIcon, color: 'text-cyan-400' },
    { label: 'Drone videos', value: videoCount, icon: FileVideo, color: 'text-violet-400' },
    { label: 'Current tracks', value: objects.length, icon: Route, color: 'text-amber-400' },
    { label: 'Current FPS', value: latestFrame?.fps.toFixed(1) || '—', icon: Zap, color: 'text-rose-400' },
  ]

  return (
    <AppShell>
      <div className="p-6 space-y-6">
        <div className="pr-40">
          <h1 className="text-2xl font-bold">{t('Aerial analysis dashboard')}</h1>
          <p className="mt-1 text-sm text-muted-foreground">{t('Small-object detection, tracking and trajectory analysis for drone footage')}</p>
        </div>

        <div className="grid grid-cols-2 gap-4 lg:grid-cols-3 xl:grid-cols-6">
          {stats.map(({ label, value, icon: Icon, color }) => <div key={label} className="rounded-xl border border-border bg-card p-4"><Icon className={`mb-3 h-5 w-5 ${color}`} /><p className="text-2xl font-bold">{value}</p><p className="mt-0.5 text-xs text-muted-foreground">{t(label)}</p></div>)}
        </div>

        <div className="grid gap-6 xl:grid-cols-3">
          <section className="overflow-hidden rounded-xl border border-border bg-card xl:col-span-2">
            <div className="flex items-center justify-between border-b border-border p-4">
              <div><p className="text-sm font-semibold">{t('Source preview')}</p>{latestFrame && <p className="text-xs text-muted-foreground">{latestFrame.fps.toFixed(1)} FPS · {latestFrame.latency_ms} ms</p>}</div>
              <select value={selectedId || ''} onChange={(event) => setSelectedId(Number(event.target.value))} className="rounded-md border border-input bg-background px-2 py-1 text-xs">
                {sources.map((source) => <option key={source.id} value={source.id}>{source.name}</option>)}
              </select>
            </div>
            {selected && streamUrl ? <div className="aspect-video bg-black"><img src={streamUrl} alt={selected.name} className="h-full w-full object-contain" /></div> : <div className="flex aspect-video items-center justify-center bg-muted text-sm text-muted-foreground">{t('Select or start an aerial source')}</div>}
          </section>

          <section className="rounded-xl border border-border bg-card">
            <div className="border-b border-border p-4"><h2 className="text-sm font-semibold">{t('Tracked objects')} <span className="font-normal text-muted-foreground">({objects.length})</span></h2></div>
            <div className="max-h-[420px] divide-y divide-border overflow-y-auto">
              {!objects.length ? <p className="p-6 text-center text-sm text-muted-foreground">{t('No objects detected')}</p> : objects.map((object) => <div key={object.track_id} className="flex items-center justify-between p-3"><div><p className="text-sm font-medium">#{object.track_id} · {t(object.object_class)}</p><p className="text-xs text-muted-foreground">{t(object.state || 'tracked')}{object.speed != null ? ` · ${object.speed.toFixed(1)} px/s` : ''}</p></div><span className="text-xs text-muted-foreground">{Math.round(object.confidence * 100)}%</span></div>)}
            </div>
          </section>
        </div>

        <section className="rounded-xl border border-border bg-card">
          <div className="border-b border-border p-4"><h2 className="text-sm font-semibold">{t('Recent detected objects')}</h2></div>
          <div className="divide-y divide-border">
            {!recentObjects.length ? <p className="p-4 text-center text-sm text-muted-foreground">{t('No objects detected')}</p> : recentObjects.map((object) => <div key={object.id} className="flex items-center gap-3 px-4 py-3"><Clock className="h-4 w-4 text-muted-foreground" /><span className="flex-1 text-xs font-medium">#{object.track_id} · {t(object.object_class)}</span><span className="text-xs text-muted-foreground">{formatDateTime(object.first_seen, locale)}</span></div>)}
          </div>
        </section>
      </div>
    </AppShell>
  )
}
