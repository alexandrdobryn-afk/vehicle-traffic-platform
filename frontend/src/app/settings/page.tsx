'use client'

import { useEffect, useState } from 'react'
import { AlertTriangle, CheckCircle2, Cpu, Gauge, KeyRound, Loader2, Save, Sparkles } from 'lucide-react'

import AppShell from '@/components/shared/AppShell'
import api from '@/lib/api'
import { useTranslation } from '@/lib/i18n'
import type { AppSettings, GeminiSettings, RuntimeStatus } from '@/types'

export default function SettingsPage() {
  const { t } = useTranslation()
  const [settings, setSettings] = useState<AppSettings | null>(null)
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null)
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState('')
  const [gemini, setGemini] = useState<GeminiSettings | null>(null)
  const [geminiKey, setGeminiKey] = useState('')
  const [geminiBusy, setGeminiBusy] = useState(false)
  const [geminiMessage, setGeminiMessage] = useState('')

  const load = async () => {
    const [settingsResponse, runtimeResponse, geminiResponse] = await Promise.all([
      api.get('/settings'),
      api.get('/settings/runtime-status'),
      api.get('/settings/gemini'),
    ])
    setSettings(settingsResponse.data)
    setRuntime(runtimeResponse.data)
    setGemini(geminiResponse.data)
  }

  const saveGemini = async () => {
    if (!gemini) return
    setGeminiBusy(true)
    setGeminiMessage('')
    try {
      const response = await api.put('/settings/gemini', {
        enabled: gemini.enabled,
        model: gemini.model,
        request_timeout_seconds: gemini.request_timeout_seconds,
        max_concurrent_requests: gemini.max_concurrent_requests,
        ...(geminiKey.trim() ? { api_key: geminiKey.trim() } : {}),
      })
      setGemini(response.data)
      setGeminiKey('')
      setGeminiMessage(t('Gemini settings saved'))
    } catch (error: any) {
      setGeminiMessage(error.response?.data?.detail || t('Failed to save'))
    } finally {
      setGeminiBusy(false)
    }
  }

  const testGemini = async () => {
    setGeminiBusy(true)
    setGeminiMessage('')
    try {
      const response = await api.post('/settings/gemini/test')
      setGeminiMessage(`${t('Connection successful')}: ${response.data.model}`)
    } catch (error: any) {
      setGeminiMessage(error.response?.data?.detail || t('Connection failed'))
    } finally {
      setGeminiBusy(false)
    }
  }

  useEffect(() => { load().catch(() => setMessage(t('Failed to load settings'))) }, [])

  const save = async () => {
    if (!settings) return
    setSaving(true)
    setMessage('')
    try {
      const response = await api.patch('/settings', settings)
      setRuntime(response.data.runtime)
      setMessage(t('Settings saved'))
    } catch (error: any) {
      setMessage(error.response?.data?.detail || t('Failed to save'))
    } finally {
      setSaving(false)
    }
  }

  return (
    <AppShell>
      <div className="p-6 space-y-6 max-w-4xl">
        <div>
          <h1 className="text-2xl font-bold text-foreground">{t('Settings')}</h1>
          <p className="text-muted-foreground text-sm">{t('Global compute configuration')}</p>
        </div>

        {!settings || !runtime ? (
          <div className="flex items-center gap-2 text-muted-foreground"><Loader2 className="w-4 h-4 animate-spin" />{t('Loading...')}</div>
        ) : (
          <>
            <section className="bg-card border border-border rounded-xl p-6 space-y-5">
              <div className="flex items-start gap-3">
                <Cpu className="w-5 h-5 text-primary mt-0.5" />
                <div>
                  <h2 className="font-semibold text-foreground">{t('Compute runtime')}</h2>
                  <p className="text-sm text-muted-foreground mt-1">{t('This global policy decides where every camera and video runs. Recognition models remain source-local.')}</p>
                </div>
              </div>

              <div className="grid md:grid-cols-3 gap-3">
                {([
                  ['auto', 'Automatic', 'Use CUDA when available, otherwise CPU'],
                  ['cpu', 'CPU', 'Compatible with every installation'],
                  ['cuda', 'NVIDIA CUDA', 'Requires NVIDIA GPU and CUDA runtime'],
                ] as const).map(([value, label, description]) => {
                  const disabled = value === 'cuda' && !runtime.capabilities.cuda.available
                  return (
                    <button
                      key={value}
                      type="button"
                      disabled={disabled}
                      onClick={() => setSettings({ ...settings, execution_provider: value })}
                      className={`text-left rounded-lg border p-4 transition ${settings.execution_provider === value ? 'border-primary bg-primary/10' : 'border-border bg-background'} ${disabled ? 'opacity-45 cursor-not-allowed' : 'hover:border-primary/60'}`}
                    >
                      <div className="font-medium text-foreground">{t(label)}</div>
                      <div className="text-xs text-muted-foreground mt-1">{t(description)}</div>
                    </button>
                  )
                })}
              </div>

              {runtime.capabilities.cuda.devices.length > 0 && (
                <label className="block text-sm text-muted-foreground">
                  {t('GPU device')}
                  <select
                    value={settings.gpu_device_index}
                    onChange={(event) => setSettings({ ...settings, gpu_device_index: Number(event.target.value) })}
                    className="mt-2 w-full bg-background border border-border rounded-lg px-3 py-2 text-foreground"
                  >
                    {runtime.capabilities.cuda.devices.map((device) => (
                      <option key={device.index} value={device.index}>
                        GPU {device.index}: {device.name} ({device.memory_mb} MB)
                      </option>
                    ))}
                  </select>
                </label>
              )}

              <div className="grid md:grid-cols-2 gap-4">
                <label className="text-sm text-muted-foreground">
                  {t('If requested GPU is unavailable')}
                  <select
                    value={settings.runtime_fallback}
                    onChange={(event) => setSettings({ ...settings, runtime_fallback: event.target.value as AppSettings['runtime_fallback'] })}
                    className="mt-2 w-full bg-background border border-border rounded-lg px-3 py-2 text-foreground"
                  >
                    <option value="fail_closed">{t('Stop and show an error')}</option>
                    <option value="allow_cpu">{t('Allow explicit CPU fallback')}</option>
                  </select>
                </label>
                <label className="text-sm text-muted-foreground">
                  {t('Inference precision')}
                  <input value="FP32" disabled className="mt-2 w-full bg-muted border border-border rounded-lg px-3 py-2 text-foreground" />
                  <span className="block text-xs mt-1">{t('FP32 is locked to preserve recognition quality.')}</span>
                </label>
              </div>
            </section>

            <section className="bg-card border border-border rounded-xl p-6 space-y-4">
              <div className="flex items-center gap-3">
                <Gauge className="w-5 h-5 text-primary" />
                <h2 className="font-semibold text-foreground">{t('Detected hardware and active runtime')}</h2>
              </div>
              <div className="grid sm:grid-cols-3 gap-3 text-sm">
                <Status label={t('Requested')} value={settings.execution_provider.toUpperCase()} />
                <Status label={t('Currently resolved')} value={(runtime.resolved || 'UNAVAILABLE').toUpperCase()} />
                <Status label={t('Container runtime')} value={runtime.capabilities.container_runtime} />
              </div>
              <div className={`flex items-start gap-2 rounded-lg p-3 text-sm ${runtime.available ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'}`}>
                {runtime.available ? <CheckCircle2 className="w-4 h-4 mt-0.5" /> : <AlertTriangle className="w-4 h-4 mt-0.5" />}
                <span>{runtime.available ? t('Selected runtime is available.') : t('Selected runtime is not available on this installation.')}</span>
              </div>
              <div className="text-xs text-muted-foreground space-y-1">
                <div>PyTorch: {runtime.capabilities.torch_version || '—'}</div>
                <div>ONNX Runtime: {runtime.capabilities.onnxruntime_version || '—'}</div>
                <div>ONNX providers: {runtime.capabilities.onnxruntime_providers.join(', ') || '—'}</div>
              </div>
            </section>

            {gemini && (
              <section className="bg-card border border-border rounded-xl p-6 space-y-5">
                <div className="flex items-start gap-3">
                  <Sparkles className="w-5 h-5 text-violet-400 mt-0.5" />
                  <div>
                    <h2 className="font-semibold text-foreground">Gemini API</h2>
                    <p className="text-sm text-muted-foreground mt-1">{t('Gemini verifies selected detections and proposes segmentation masks. Camera frames remain local unless a source explicitly opts in.')}</p>
                  </div>
                </div>
                <div className="rounded-lg border border-border bg-background p-3 text-sm flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2"><KeyRound className="w-4 h-4" /><span>{gemini.configured ? t('API key configured') : t('API key not configured')}</span></div>
                  <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={gemini.enabled} onChange={(e) => setGemini({ ...gemini, enabled: e.target.checked })} />{t('Enabled')}</label>
                </div>
                <label className="block text-sm text-muted-foreground">
                  {t('Gemini API key')}
                  <input type="password" autoComplete="new-password" value={geminiKey} onChange={(e) => setGeminiKey(e.target.value)} placeholder={gemini.configured ? '******** (leave blank to keep)' : 'AIza...'} className="mt-2 w-full bg-background border border-border rounded-lg px-3 py-2 text-foreground" />
                  <span className="block text-xs mt-1">{t('The key is encrypted by the backend and is never returned to the browser.')}</span>
                </label>
                <div className="grid md:grid-cols-3 gap-4">
                  <label className="text-sm text-muted-foreground">{t('Model')}<input value={gemini.model} onChange={(e) => setGemini({ ...gemini, model: e.target.value })} className="mt-2 w-full bg-background border border-border rounded-lg px-3 py-2 text-foreground" /></label>
                  <label className="text-sm text-muted-foreground">{t('Timeout, seconds')}<input type="number" min={10} max={120} value={gemini.request_timeout_seconds} onChange={(e) => setGemini({ ...gemini, request_timeout_seconds: Number(e.target.value) })} className="mt-2 w-full bg-background border border-border rounded-lg px-3 py-2 text-foreground" /></label>
                  <label className="text-sm text-muted-foreground">{t('Concurrent requests')}<input type="number" min={1} max={4} value={gemini.max_concurrent_requests} onChange={(e) => setGemini({ ...gemini, max_concurrent_requests: Number(e.target.value) })} className="mt-2 w-full bg-background border border-border rounded-lg px-3 py-2 text-foreground" /></label>
                </div>
                <div className="flex flex-wrap items-center gap-3">
                  <button onClick={saveGemini} disabled={geminiBusy} className="btn-primary inline-flex items-center gap-2 px-4 py-2 rounded-lg disabled:opacity-50"><Save className="w-4 h-4" />{t('Save Gemini settings')}</button>
                  <button onClick={testGemini} disabled={geminiBusy || !gemini.configured} className="px-4 py-2 rounded-lg border border-border text-sm disabled:opacity-50">{t('Test connection')}</button>
                  {geminiMessage && <span className="text-sm text-muted-foreground">{geminiMessage}</span>}
                </div>
              </section>
            )}

            <div className="bg-card border border-border rounded-xl p-4 text-sm text-muted-foreground">
              {t('Detector thresholds, OCR, voting, resolution, FPS, storage and privacy remain configured separately for each camera or video.')}
            </div>

            <div className="flex items-center gap-3">
              <button onClick={save} disabled={saving} className="btn-primary inline-flex items-center gap-2 px-4 py-2 rounded-lg disabled:opacity-50">
                {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
                {t('Save Changes')}
              </button>
              {message && <span className="text-sm text-muted-foreground">{message}</span>}
            </div>
          </>
        )}
      </div>
    </AppShell>
  )
}

function Status({ label, value }: { label: string; value: string }) {
  return <div className="rounded-lg border border-border bg-background p-3"><div className="text-xs text-muted-foreground">{label}</div><div className="font-medium text-foreground mt-1">{value}</div></div>
}
