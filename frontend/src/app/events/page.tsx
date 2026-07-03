'use client'
import { useEffect, useState, useCallback } from 'react'
import AppShell from '@/components/shared/AppShell'
import api from '@/lib/api'
import { Event, EVENT_LABELS, EVENT_COLORS } from '@/types'
import { formatDateTime } from '@/lib/utils'
import { Download, Filter, RefreshCw, AlertTriangle, CheckCircle, Car, Camera, Clock } from 'lucide-react'
import toast from 'react-hot-toast'
import { useTranslation } from '@/lib/i18n'

const EVENT_ICONS: Record<string, any> = {
  vehicle_entered: Car,
  plate_verified: CheckCircle,
  watchlist_match: AlertTriangle,
  camera_offline: Camera,
  default: Clock,
}
export default function EventsPage() {
  const { t, locale } = useTranslation()
  const [events, setEvents] = useState<Event[]>([])
  const [loading, setLoading] = useState(false)
  const [filters, setFilters] = useState({
    event_type: '',
    camera_id: '',
    alert_only: false,
    date_from: '',
    date_to: '',
  })

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const params: Record<string, any> = { limit: 200 }
      if (filters.event_type) params.event_type = filters.event_type
      if (filters.camera_id) params.camera_id = filters.camera_id
      if (filters.alert_only) params.alert_only = 'true'
      if (filters.date_from) params.date_from = filters.date_from
      if (filters.date_to) params.date_to = filters.date_to
      const res = await api.get('/events', { params })
      setEvents(res.data)
    } catch { toast.error(t('Failed to load events')) }
    finally { setLoading(false) }
  }, [filters])

  useEffect(() => { load() }, [load])

  const handleExport = async (format: string) => {
    try {
      const res = await api.get(`/events/export?format=${format}`, { responseType: 'blob' })
      const url = URL.createObjectURL(res.data)
      const a = document.createElement('a')
      a.href = url; a.download = `events.${format}`; a.click()
      URL.revokeObjectURL(url)
      toast.success(t('Exported as {format}', { format: format.toUpperCase() }))
    } catch { toast.error(t('Export failed')) }
  }

  return (
    <AppShell>
      <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between pr-40">
          <div>
            <h1 className="text-2xl font-bold text-foreground">{t('Events')}</h1>
            <p className="text-muted-foreground text-sm">{t('{count} events', { count: events.length })}</p>
          </div>
          <div className="flex gap-2">
            <button onClick={load} aria-label={t('Refresh')} title={t('Refresh')} className="p-2 bg-muted rounded-lg text-muted-foreground hover:text-foreground">
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            </button>
            <div className="relative group">
              <button className="flex items-center gap-2 px-3 py-2 bg-muted text-muted-foreground rounded-lg text-sm hover:text-foreground">
                <Download className="w-4 h-4" /> {t('Export')}
              </button>
              <div className="absolute right-0 top-full mt-1 bg-card border border-border rounded-lg shadow-xl z-10 hidden group-hover:block">
                {['csv', 'excel', 'json'].map((f) => (
                  <button key={f} onClick={() => handleExport(f)} className="block w-full text-left px-4 py-2 text-sm text-muted-foreground hover:text-foreground hover:bg-accent first:rounded-t-lg last:rounded-b-lg">
                    {f.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Filters */}
        <div className="bg-card border border-border rounded-xl p-4">
          <div className="flex items-center gap-2 mb-3">
            <Filter className="w-4 h-4 text-muted-foreground" />
            <span className="text-sm font-medium text-foreground">{t('Filters')}</span>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <select
              value={filters.event_type}
              onChange={(e) => setFilters({ ...filters, event_type: e.target.value })}
              className="px-3 py-2 bg-background border border-input rounded-lg text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
            >
              <option value="">{t('All event types')}</option>
              {Object.entries(EVENT_LABELS).map(([k, v]) => (
                <option key={k} value={k}>{t(v)}</option>
              ))}
            </select>
            <input
              type="number"
              placeholder={t('Camera ID')}
              value={filters.camera_id}
              onChange={(e) => setFilters({ ...filters, camera_id: e.target.value })}
              className="px-3 py-2 bg-background border border-input rounded-lg text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
            />
            <input
              type="datetime-local"
              value={filters.date_from}
              onChange={(e) => setFilters({ ...filters, date_from: e.target.value })}
              className="px-3 py-2 bg-background border border-input rounded-lg text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
            />
            <input
              type="datetime-local"
              value={filters.date_to}
              onChange={(e) => setFilters({ ...filters, date_to: e.target.value })}
              className="px-3 py-2 bg-background border border-input rounded-lg text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
            />
          </div>
          <label className="flex items-center gap-2 mt-3 cursor-pointer">
            <input type="checkbox" checked={filters.alert_only} onChange={(e) => setFilters({ ...filters, alert_only: e.target.checked })} className="rounded" />
            <span className="text-sm text-muted-foreground">{t('Watchlist alerts only')}</span>
          </label>
        </div>

        {/* Events table */}
        <div className="bg-card border border-border rounded-xl overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                {['Time', 'Type', 'Camera', 'Plate', 'Color', 'Vehicle'].map((h) => (
                  <th key={h} className="text-left px-4 py-3 text-xs font-medium text-muted-foreground uppercase tracking-wide">{t(h)}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {events.length === 0 ? (
                <tr><td colSpan={6} className="text-center py-12 text-muted-foreground">{t('No events found')}</td></tr>
              ) : events.map((ev) => {
                const Icon = EVENT_ICONS[ev.event_type] || EVENT_ICONS.default
                const payload = ev.payload_json as any || {}
                return (
                  <tr key={ev.id} className="hover:bg-accent/30 transition-colors">
                    <td className="px-4 py-3 text-xs text-muted-foreground whitespace-nowrap">{formatDateTime(ev.created_at, locale)}</td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <Icon className={`w-4 h-4 ${EVENT_COLORS[ev.event_type] || 'text-muted-foreground'}`} />
                        <span className={`text-xs font-medium ${EVENT_COLORS[ev.event_type] || 'text-foreground'}`}>
                          {t(EVENT_LABELS[ev.event_type] || ev.event_type)}
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-xs text-muted-foreground">#{ev.camera_id}</td>
                    <td className="px-4 py-3">
                      {payload.plate ? (
                        <span className="plate-badge bg-primary/10 text-primary">{payload.plate}</span>
                      ) : (
                        <span className="text-xs text-muted-foreground">—</span>
                      )}
                    </td>
                    <td className="px-4 py-3">
                      {payload.color ? (
                        <span className="text-xs text-muted-foreground">{t(payload.color)}</span>
                      ) : '—'}
                    </td>
                    <td className="px-4 py-3 text-xs text-muted-foreground">{payload.vehicle_class ? t(payload.vehicle_class) : '—'}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </AppShell>
  )
}
