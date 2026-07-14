'use client'

import { useEffect, useMemo, useState } from 'react'
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import AppShell from '@/components/shared/AppShell'
import api from '@/lib/api'
import { Camera, ObjectTrack } from '@/types'
import { useTranslation } from '@/lib/i18n'

export default function AnalyticsPage() {
  const { t } = useTranslation()
  const [objects, setObjects] = useState<ObjectTrack[]>([])
  const [sources, setSources] = useState<Camera[]>([])
  const [performance, setPerformance] = useState<any[]>([])
  const [hours, setHours] = useState(24)

  useEffect(() => {
    Promise.all([api.get('/objects?limit=500'), api.get('/cameras'), api.get('/videos'), api.get('/analytics/performance')])
      .then(([objectResponse, cameraResponse, videoResponse, performanceResponse]) => {
        setObjects(objectResponse.data)
        setSources([...cameraResponse.data, ...videoResponse.data])
        setPerformance(performanceResponse.data)
      }).catch(() => {})
  }, [])

  const filtered = useMemo(() => {
    const cutoff = Date.now() - hours * 3600_000
    return objects.filter((object) => new Date(object.first_seen).getTime() >= cutoff)
  }, [objects, hours])

  const classCounts = useMemo(() => Object.entries(filtered.reduce<Record<string, number>>((result, object) => {
    result[object.object_class] = (result[object.object_class] || 0) + 1
    return result
  }, {})).sort((a, b) => b[1] - a[1]), [filtered])

  const timeline = useMemo(() => {
    const buckets = new Map<string, number>()
    filtered.forEach((object) => {
      const date = new Date(object.first_seen)
      date.setMinutes(0, 0, 0)
      const key = date.toISOString()
      buckets.set(key, (buckets.get(key) || 0) + 1)
    })
    return [...buckets.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([timestamp, count]) => ({ timestamp, count }))
  }, [filtered])

  const averageConfidence = filtered.length ? Math.round(filtered.reduce((sum, object) => sum + object.confidence, 0) / filtered.length * 100) : 0
  const averageDuration = filtered.length ? filtered.reduce((sum, object) => sum + object.duration_seconds, 0) / filtered.length : 0

  return (
    <AppShell>
      <div className="space-y-6 p-6">
        <div className="flex items-center justify-between pr-40">
          <div><h1 className="text-2xl font-bold">{t('Analytics')}</h1><p className="text-sm text-muted-foreground">{t('Aerial object statistics and inference performance')}</p></div>
          <select value={hours} onChange={(event) => setHours(Number(event.target.value))} className="rounded-lg border border-input bg-background px-3 py-2 text-sm">{[6, 12, 24, 48, 168].map((value) => <option key={value} value={value}>{t('Last {value}', { value: value < 24 ? `${value}h` : `${value / 24}d` })}</option>)}</select>
        </div>

        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          {[
            ['Detected objects', filtered.length],
            ['Object classes', classCounts.length],
            ['Average confidence', `${averageConfidence}%`],
            ['Average track duration', `${averageDuration.toFixed(1)} s`],
          ].map(([label, value]) => <div key={label} className="rounded-xl border border-border bg-card p-4"><p className="text-3xl font-bold">{value}</p><p className="mt-1 text-xs text-muted-foreground">{t(String(label))}</p></div>)}
        </div>

        <section className="rounded-xl border border-border bg-card p-6">
          <h2 className="mb-4 font-semibold">{t('Objects over time')}</h2>
          {timeline.length ? <ResponsiveContainer width="100%" height={240}><AreaChart data={timeline}><CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" /><XAxis dataKey="timestamp" tickFormatter={(value) => new Date(value).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }} /><YAxis tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }} /><Tooltip labelFormatter={(value) => new Date(value).toLocaleString()} /><Area type="monotone" dataKey="count" name={t('Objects')} stroke="hsl(221 83% 53%)" fill="hsl(221 83% 53% / .2)" /></AreaChart></ResponsiveContainer> : <div className="flex h-60 items-center justify-center text-sm text-muted-foreground">{t('No object data for this period')}</div>}
        </section>

        <div className="grid gap-6 md:grid-cols-2">
          <section className="rounded-xl border border-border bg-card p-6"><h2 className="mb-4 font-semibold">{t('Objects by class')}</h2>{classCounts.length ? <div className="space-y-3">{classCounts.slice(0, 12).map(([name, count]) => <div key={name}><div className="mb-1 flex justify-between text-xs"><span>{t(name)}</span><span>{count}</span></div><div className="h-2 rounded-full bg-muted"><div className="h-2 rounded-full bg-primary" style={{ width: `${count / classCounts[0][1] * 100}%` }} /></div></div>)}</div> : <p className="py-16 text-center text-sm text-muted-foreground">{t('No classified objects')}</p>}</section>
          <section className="rounded-xl border border-border bg-card p-6"><h2 className="mb-4 font-semibold">{t('Source performance')}</h2>{performance.length ? <div className="space-y-4">{performance.map((item) => <div key={item.camera_id}><div className="mb-1 flex justify-between text-xs"><span>{sources.find((source) => source.id === item.camera_id)?.name || `${t('Source')} #${item.camera_id}`}</span><span>{(item.fps || 0).toFixed(1)} FPS · {(item.latency_ms || 0).toFixed(0)} ms</span></div><div className="h-2 rounded-full bg-muted"><div className="h-2 rounded-full bg-emerald-500" style={{ width: `${Math.min((item.fps || 0) / 30 * 100, 100)}%` }} /></div></div>)}</div> : <p className="py-16 text-center text-sm text-muted-foreground">{t('No sources running')}</p>}</section>
        </div>
      </div>
    </AppShell>
  )
}
