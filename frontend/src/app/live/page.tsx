'use client'
import { useEffect, useState } from 'react'
import AppShell from '@/components/shared/AppShell'
import api from '@/lib/api'
import { Camera, WSFrame, COLOR_HEX } from '@/types'
import { useWebSocket } from '@/hooks/useWebSocket'
import { useStreamUrl } from '@/hooks/useStreamUrl'
import { CheckCircle } from 'lucide-react'
import { useTranslation } from '@/lib/i18n'

function CameraFeed({ camera }: { camera: Camera }) {
  const { t } = useTranslation()
  const [frame, setFrame] = useState<WSFrame | null>(null)
  const { connected } = useWebSocket({
    cameraId: camera.id,
    enabled: camera.is_active,
    onFrame: setFrame,
  })

  const streamUrl = useStreamUrl(camera.id, camera.is_active)

  return (
    <div className="bg-card border border-border rounded-xl overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2 border-b border-border">
        <div className="flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${camera.is_active && connected ? 'bg-emerald-400 animate-pulse' : 'bg-red-400'}`} />
          <span className="text-xs font-semibold text-foreground">{camera.name}</span>
        </div>
        {frame && (
          <span className="text-xs text-muted-foreground font-mono">{frame.fps.toFixed(1)} fps · {frame.latency_ms}ms</span>
        )}
      </div>

      <div className="relative bg-black aspect-video">
        {camera.is_active && streamUrl ? (
          <img
            src={streamUrl}
            alt={camera.name}
            className="w-full h-full object-contain"
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center">
            <p className="text-muted-foreground text-xs">{t('Camera offline')}</p>
          </div>
        )}

        {/* Vehicle count overlay */}
        {frame && frame.objects.length > 0 && (
          <div className="absolute bottom-2 left-2 bg-black/70 rounded-md px-2 py-1">
            <p className="text-xs text-green-400 font-mono">{t('{count} vehicles', { count: frame.objects.length })}</p>
          </div>
        )}
      </div>

      {/* Active tracks for this camera */}
      {frame && frame.objects.length > 0 && (
        <div className="p-2 space-y-1 max-h-28 overflow-y-auto">
          {frame.objects.map((obj) => (
            <div key={obj.track_id} className="flex items-center justify-between text-xs px-1">
              <div className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: COLOR_HEX[obj.color] || '#6b7280' }} />
                <span className="font-mono font-bold text-foreground">#{obj.track_id}</span>
                <span className="text-muted-foreground">{t(obj.vehicle_class)}</span>
              </div>
              <div className="flex items-center gap-1">
                {obj.plate ? (
                  <>
                    <span className="plate-badge bg-primary/10 text-primary">{obj.plate}</span>
                    {obj.plate_status === 'verified' && <CheckCircle className="w-3 h-3 text-emerald-400" />}
                  </>
                ) : (
                  <span className="text-muted-foreground italic">{t('searching...')}</span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default function LivePage() {
  const { t } = useTranslation()
  const [cameras, setCameras] = useState<Camera[]>([])

  useEffect(() => {
    api.get('/cameras').then((r) => setCameras(r.data)).catch(() => {})
    const t = setInterval(() => api.get('/cameras').then((r) => setCameras(r.data)).catch(() => {}), 15000)
    return () => clearInterval(t)
  }, [])

  const activeCams = cameras.filter((c) => c.is_active)
  const gridCols = activeCams.length <= 1 ? 'grid-cols-1' : activeCams.length <= 2 ? 'grid-cols-2' : 'grid-cols-2 xl:grid-cols-3'

  return (
    <AppShell>
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between pr-40">
          <div>
            <h1 className="text-2xl font-bold text-foreground">{t('Live View')}</h1>
            <p className="text-muted-foreground text-sm">{t('{count} cameras active', { count: activeCams.length })}</p>
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 bg-emerald-500/10 border border-emerald-500/20 rounded-full">
            <span className="live-dot" />
            <span className="text-xs font-medium text-emerald-400">{t('LIVE')}</span>
          </div>
        </div>

        {cameras.length === 0 ? (
          <div className="text-center py-16 text-muted-foreground">
            <p className="font-medium text-foreground mb-1">{t('No cameras configured')}</p>
            <p className="text-sm">{t('Add cameras in the Cameras section')}</p>
          </div>
        ) : (
          <div className={`grid ${gridCols} gap-4`}>
            {cameras.map((cam) => (
              <CameraFeed key={cam.id} camera={cam} />
            ))}
          </div>
        )}
      </div>
    </AppShell>
  )
}
