'use client'
import { useEffect, useState } from 'react'
import AppShell from '@/components/shared/AppShell'
import api from '@/lib/api'
import { AnalyticsSummary, TrafficPoint, ColorDist, COLOR_HEX } from '@/types'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
} from 'recharts'
import { format } from 'date-fns'
import { useTranslation } from '@/lib/i18n'

export default function AnalyticsPage() {
  const { t } = useTranslation()
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null)
  const [traffic, setTraffic] = useState<TrafficPoint[]>([])
  const [colors, setColors] = useState<ColorDist[]>([])
  const [perf, setPerf] = useState<any[]>([])
  const [hours, setHours] = useState(24)

  useEffect(() => {
    const load = async () => {
      const [s, t, c, p] = await Promise.all([
        api.get('/analytics/summary'),
        api.get(`/analytics/traffic-volume?hours=${hours}`),
        api.get('/analytics/colors'),
        api.get('/analytics/performance'),
      ])
      setSummary(s.data)
      setTraffic(t.data)
      setColors(c.data)
      setPerf(p.data)
    }
    load()
  }, [hours])

  const formatHour = (ts: string) => {
    try { return format(new Date(ts), 'HH:mm') } catch { return ts }
  }

  return (
    <AppShell>
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between pr-40">
          <div>
            <h1 className="text-2xl font-bold text-foreground">{t('Analytics')}</h1>
            <p className="text-muted-foreground text-sm">{t('Traffic insights and AI performance')}</p>
          </div>
          <select
            value={hours}
            onChange={(e) => setHours(Number(e.target.value))}
            className="px-3 py-2 bg-background border border-input rounded-lg text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
          >
            {[6, 12, 24, 48, 168].map((h) => (
              <option key={h} value={h}>{t('Last {value}', { value: h < 24 ? `${h}h` : `${h / 24}d` })}</option>
            ))}
          </select>
        </div>

        {/* Summary KPIs */}
        {summary && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: 'Total Vehicles', value: summary.total_vehicles_today },
              { label: 'Plates Recognized', value: summary.total_plates_recognized },
              { label: 'OCR Success', value: `${summary.ocr_success_rate}%` },
              { label: 'Watchlist Hits', value: summary.watchlist_matches_today },
            ].map(({ label, value }) => (
              <div key={label} className="bg-card border border-border rounded-xl p-4">
                <p className="text-3xl font-bold text-foreground">{value}</p>
                <p className="text-xs text-muted-foreground mt-1">{t(label)}</p>
              </div>
            ))}
          </div>
        )}

        {/* Traffic volume chart */}
        <div className="bg-card border border-border rounded-xl p-6">
          <h3 className="font-semibold text-foreground mb-4">{t('Traffic Volume')}</h3>
          {traffic.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={traffic}>
                <defs>
                  <linearGradient id="trafficGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="hsl(221 83% 53%)" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="hsl(221 83% 53%)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="timestamp" tickFormatter={formatHour} tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }} />
                <YAxis tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }} />
                <Tooltip
                  contentStyle={{ background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: 8 }}
                  labelStyle={{ color: 'hsl(var(--foreground))' }}
                  labelFormatter={formatHour}
                />
                <Area type="monotone" dataKey="count" name={t('Vehicles')} stroke="hsl(221 83% 53%)" fill="url(#trafficGrad)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-56 flex items-center justify-center text-muted-foreground text-sm">{t('No traffic data')}</div>
          )}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Color distribution */}
          <div className="bg-card border border-border rounded-xl p-6">
            <h3 className="font-semibold text-foreground mb-4">{t('Color Distribution')}</h3>
            {colors.length > 0 ? (
              <div className="flex gap-6">
                <ResponsiveContainer width="50%" height={180}>
                  <PieChart>
                    <Pie data={colors} dataKey="count" nameKey="color" cx="50%" cy="50%" outerRadius={70} innerRadius={40}>
                      {colors.map((c) => (
                        <Cell key={c.color} fill={COLOR_HEX[c.color] || '#6b7280'} stroke="hsl(var(--card))" strokeWidth={2} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value: any, name: any) => [value, t(String(name))]} contentStyle={{ background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: 8 }} />
                  </PieChart>
                </ResponsiveContainer>
                <div className="flex-1 space-y-1.5 overflow-auto">
                  {colors.slice(0, 8).map((c) => (
                    <div key={c.color} className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-2">
                        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: COLOR_HEX[c.color] || '#6b7280' }} />
                        <span className="text-muted-foreground">{t(c.color)}</span>
                      </div>
                      <span className="font-medium text-foreground">{c.percentage}%</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="h-44 flex items-center justify-center text-muted-foreground text-sm">{t('No color data')}</div>
            )}
          </div>

          {/* Camera performance */}
          <div className="bg-card border border-border rounded-xl p-6">
            <h3 className="font-semibold text-foreground mb-4">{t('Camera Performance')}</h3>
            {perf.length > 0 ? (
              <div className="space-y-3">
                {perf.map((cam: any) => (
                  <div key={cam.camera_id} className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-muted-foreground">{t('Camera')} #{cam.camera_id}</span>
                      <span className="font-medium text-foreground">{(cam.fps || 0).toFixed(1)} FPS</span>
                    </div>
                    <div className="w-full bg-muted rounded-full h-1.5">
                      <div
                        className="bg-primary h-1.5 rounded-full transition-all"
                        style={{ width: `${Math.min((cam.fps || 0) / 30 * 100, 100)}%` }}
                      />
                    </div>
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>{cam.is_running ? `🟢 ${t('Running')}` : `🔴 ${t('Stopped')}`}</span>
                      <span>{t('{count} dropped', { count: cam.frames_dropped || 0 })}</span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="h-44 flex items-center justify-center text-muted-foreground text-sm">{t('No cameras running')}</div>
            )}
          </div>
        </div>
      </div>
    </AppShell>
  )
}
