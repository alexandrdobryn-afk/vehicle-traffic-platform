'use client'
import { useEffect, useState } from 'react'
import AppShell from '@/components/shared/AppShell'
import api from '@/lib/api'
import { WatchlistEntry } from '@/types'
import { Plus, Trash2, AlertTriangle, Shield } from 'lucide-react'
import { formatDateTime } from '@/lib/utils'
import toast from 'react-hot-toast'
import { useTranslation } from '@/lib/i18n'

export default function WatchlistPage() {
  const { t, locale } = useTranslation()
  const [entries, setEntries] = useState<WatchlistEntry[]>([])
  const [showModal, setShowModal] = useState(false)
  const [form, setForm] = useState({ plate_number: '', description: '', alert_channels: ['frontend'] })
  const [loading, setLoading] = useState(false)

  const load = async () => {
    const res = await api.get('/watchlist')
    setEntries(res.data)
  }
  useEffect(() => { load() }, [])

  const handleAdd = async () => {
    setLoading(true)
    try {
      await api.post('/watchlist', form)
      toast.success(t('Added to watchlist'))
      setShowModal(false)
      setForm({ plate_number: '', description: '', alert_channels: ['frontend'] })
      load()
    } catch (e: any) { toast.error(t('Error')) }
    finally { setLoading(false) }
  }

  const handleDelete = async (id: number) => {
    if (!confirm(t('Remove from watchlist?'))) return
    try { await api.delete(`/watchlist/${id}`); toast.success(t('Removed')); load() }
    catch { toast.error(t('Error')) }
  }

  const toggleChannel = (ch: string) => {
    const channels = form.alert_channels.includes(ch)
      ? form.alert_channels.filter((c) => c !== ch)
      : [...form.alert_channels, ch]
    setForm({ ...form, alert_channels: channels })
  }

  return (
    <AppShell>
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between pr-40">
          <div>
            <h1 className="text-2xl font-bold text-foreground">{t('Watchlist')}</h1>
            <p className="text-muted-foreground text-sm">{t('{count} plates monitored', { count: entries.length })}</p>
          </div>
          <button onClick={() => setShowModal(true)} className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:bg-primary/90">
            <Plus className="w-4 h-4" /> {t('Add Plate')}
          </button>
        </div>

        {/* Info banner */}
        <div className="flex items-start gap-3 p-4 bg-amber-500/10 border border-amber-500/20 rounded-xl">
          <AlertTriangle className="w-5 h-5 text-amber-400 mt-0.5 shrink-0" />
          <p className="text-sm text-amber-300">
            {t('When a plate on this list is detected, an alert is triggered via the configured channels. Alerts appear in real-time on the dashboard.')}
          </p>
        </div>

        {entries.length === 0 ? (
          <div className="bg-card border border-dashed border-border rounded-xl p-12 text-center">
            <Shield className="w-10 h-10 text-muted-foreground mx-auto mb-3" />
            <p className="font-medium text-foreground mb-1">{t('Watchlist is empty')}</p>
            <p className="text-sm text-muted-foreground">{t('Add plates you want to be alerted about')}</p>
          </div>
        ) : (
          <div className="bg-card border border-border rounded-xl overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  {['Plate', 'Description', 'Alert Channels', 'Added', 'Actions'].map((h) => (
                    <th key={h} className="text-left px-4 py-3 text-xs font-medium text-muted-foreground uppercase">{t(h)}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {entries.map((e) => (
                  <tr key={e.id} className="hover:bg-accent/30">
                    <td className="px-4 py-3">
                      <span className="plate-badge bg-red-500/10 text-red-400">{e.plate_number}</span>
                    </td>
                    <td className="px-4 py-3 text-sm text-muted-foreground">{e.description || '—'}</td>
                    <td className="px-4 py-3">
                      <div className="flex gap-1 flex-wrap">
                        {(e.alert_channels || []).map((ch) => (
                          <span key={ch} className="text-xs px-2 py-0.5 bg-muted text-muted-foreground rounded-full">{t(ch)}</span>
                        ))}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-xs text-muted-foreground">{formatDateTime(e.created_at, locale)}</td>
                    <td className="px-4 py-3">
                      <button onClick={() => handleDelete(e.id)} aria-label={t('Delete')} title={t('Delete')} className="p-1.5 rounded text-muted-foreground hover:text-red-400 hover:bg-red-400/10">
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Modal */}
        {showModal && (
          <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4">
            <div className="bg-card border border-border rounded-2xl w-full max-w-md shadow-2xl">
              <div className="flex items-center justify-between p-6 border-b border-border">
                <h2 className="font-semibold text-foreground">{t('Add to Watchlist')}</h2>
                <button onClick={() => setShowModal(false)} aria-label={t('Close')} title={t('Close')} className="text-muted-foreground hover:text-foreground">✕</button>
              </div>
              <div className="p-6 space-y-4">
                <div>
                  <label className="block text-sm font-medium text-muted-foreground mb-1">{t('Plate Number')}</label>
                  <input
                    value={form.plate_number}
                    onChange={(e) => setForm({ ...form, plate_number: e.target.value.toUpperCase() })}
                    placeholder="AA1234BB"
                    className="w-full px-3 py-2 bg-background border border-input rounded-lg text-sm font-mono tracking-widest text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-muted-foreground mb-1">{t('Description')}</label>
                  <input
                    value={form.description}
                    onChange={(e) => setForm({ ...form, description: e.target.value })}
                    placeholder={t('Stolen vehicle, VIP, etc.')}
                    className="w-full px-3 py-2 bg-background border border-input rounded-lg text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-muted-foreground mb-2">{t('Alert Channels')}</label>
                  <div className="flex flex-wrap gap-2">
                    {['frontend', 'webhook', 'telegram', 'email'].map((ch) => (
                      <button
                        key={ch}
                        onClick={() => toggleChannel(ch)}
                        className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                          form.alert_channels.includes(ch)
                            ? 'bg-primary text-primary-foreground'
                            : 'bg-muted text-muted-foreground hover:text-foreground'
                        }`}
                      >
                        {t(ch)}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
              <div className="flex gap-3 p-6 border-t border-border">
                <button onClick={() => setShowModal(false)} className="flex-1 py-2 border border-border rounded-lg text-sm text-muted-foreground">{t('Cancel')}</button>
                <button onClick={handleAdd} disabled={loading || !form.plate_number} className="flex-1 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-medium disabled:opacity-50">
                  {loading ? t('Adding...') : t('Add')}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </AppShell>
  )
}
