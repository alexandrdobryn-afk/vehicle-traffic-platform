'use client'
import { useEffect, useState } from 'react'
import AppShell from '@/components/shared/AppShell'
import api from '@/lib/api'
import { CheckCircle, XCircle, AlertTriangle, RefreshCw, Database, Wifi, Cpu, Camera } from 'lucide-react'
import { useTranslation } from '@/lib/i18n'

export default function HealthPage() {
  const { t, locale } = useTranslation()
  const [health, setHealth] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  const load = async () => {
    setLoading(true)
    try { setHealth((await api.get('/health/full')).data) }
    catch (e: any) { setHealth({ status: 'error', error: e.message }) }
    finally { setLoading(false) }
  }

  useEffect(() => { load(); const t = setInterval(load, 10000); return () => clearInterval(t) }, [])

  const StatusIcon = ({ ok }: { ok: boolean }) =>
    ok ? <CheckCircle className="w-5 h-5 text-emerald-400" /> : <XCircle className="w-5 h-5 text-red-400" />

  return (
    <AppShell>
      <div className="p-6 space-y-6 max-w-3xl">
        <div className="flex items-center justify-between pr-40">
          <div>
            <h1 className="text-2xl font-bold text-foreground">{t('System Health')}</h1>
            <p className="text-muted-foreground text-sm">{t('Real-time system diagnostics')}</p>
          </div>
          <button onClick={load} aria-label={t('Refresh')} title={t('Refresh')} className="p-2 bg-muted rounded-lg text-muted-foreground hover:text-foreground">
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>

        {!health ? (
          <div className="text-muted-foreground">{t('Loading...')}</div>
        ) : (
          <>
            {/* Overall status */}
            <div className={`flex items-center gap-3 p-4 rounded-xl border ${
              health.status === 'ok' ? 'bg-emerald-500/10 border-emerald-500/20' : 'bg-amber-500/10 border-amber-500/20'
            }`}>
              {health.status === 'ok'
                ? <CheckCircle className="w-6 h-6 text-emerald-400" />
                : <AlertTriangle className="w-6 h-6 text-amber-400" />}
              <div>
                <p className="font-semibold text-foreground">
                  {health.status === 'ok' ? t('All systems operational') : t('Degraded — some services unavailable')}
                </p>
                <p className="text-xs text-muted-foreground">{new Date().toLocaleString(locale)}</p>
              </div>
            </div>

            {/* Services */}
            <div className="bg-card border border-border rounded-xl divide-y divide-border">
              {[
                { label: 'PostgreSQL Database', icon: Database, ok: health.database === 'ok', detail: t(health.database) },
                { label: 'Redis Cache', icon: Wifi, ok: health.redis === 'ok', detail: t(health.redis) },
                { label: 'WebSocket Connections', icon: Cpu, ok: true, detail: t('{count} active', { count: health.total_ws_connections || 0 }) },
              ].map(({ label, icon: Icon, ok, detail }) => (
                <div key={label} className="flex items-center justify-between px-5 py-4">
                  <div className="flex items-center gap-3">
                    <Icon className="w-4 h-4 text-muted-foreground" />
                    <span className="text-sm font-medium text-foreground">{t(label)}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-muted-foreground">{detail}</span>
                    <StatusIcon ok={ok} />
                  </div>
                </div>
              ))}
            </div>

            {/* Camera pipeline stats */}
            {health.cameras?.length > 0 && (
              <div className="bg-card border border-border rounded-xl">
                <div className="px-5 py-4 border-b border-border">
                  <h3 className="text-sm font-semibold text-foreground flex items-center gap-2">
                    <Camera className="w-4 h-4" /> {t('Source runtimes')}
                  </h3>
                </div>
                <div className="divide-y divide-border">
                  {health.cameras.map((cam: any) => (
                    <div key={cam.camera_id} className="px-5 py-3">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-foreground">{t('Camera')} #{cam.camera_id}</span>
                        <StatusIcon ok={cam.is_running && cam.connected} />
                      </div>
                      <div className="grid grid-cols-4 gap-3 text-xs text-muted-foreground">
                        <div><p className="font-medium text-foreground">{(cam.fps || 0).toFixed(1)}</p><p>FPS</p></div>
                        <div><p className="font-medium text-foreground">{cam.active_tracks || 0}</p><p>{t('Tracks')}</p></div>
                        <div><p className="font-medium text-foreground">{cam.frames_read || 0}</p><p>{t('Frames')}</p></div>
                        <div><p className="font-medium text-foreground">{cam.reconnect_count || 0}</p><p>{t('Reconnects')}</p></div>
                      </div>
                      {(cam.source_resolution || cam.analysis_resolution) && (
                        <p className="text-[11px] text-muted-foreground mt-2">
                          {t('Source')} {cam.source_resolution?.join('×') || '—'} → {t('Analysis')} {cam.analysis_resolution?.join('×') || '—'}
                        </p>
                      )}
                      {cam.stage_counts && (
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mt-3">
                          {Object.entries(cam.stage_counts).map(([stage, value]) => (
                            <div key={stage} className="rounded bg-muted/50 px-2 py-1 text-[10px]">
                              <span className="text-muted-foreground">{t(stage)}</span>{' '}
                              <span className="font-medium text-foreground">{String(value)}</span>
                            </div>
                          ))}
                        </div>
                      )}
                      {cam.last_error && (
                        <p className="text-xs text-red-400 mt-2">⚠ {cam.last_error}</p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </AppShell>
  )
}
