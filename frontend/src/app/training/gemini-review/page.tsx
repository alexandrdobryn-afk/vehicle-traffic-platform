'use client'

import { useEffect, useMemo, useState } from 'react'
import Link from 'next/link'
import { Check, Database, Loader2, RefreshCw, RotateCcw, Sparkles, X } from 'lucide-react'
import toast from 'react-hot-toast'

import AppShell from '@/components/shared/AppShell'
import api from '@/lib/api'
import trainingApi from '@/lib/trainingApi'
import { useTranslation } from '@/lib/i18n'
import type { GeminiCandidate } from '@/types'

function SecureImage({ path, className }: { path: string | null; className?: string }) {
  const [url, setUrl] = useState<string | null>(null)
  useEffect(() => {
    let active = true
    let objectUrl: string | null = null
    if (!path) { setUrl(null); return }
    api.get(path.replace(/^\/api\/v1/, ''), { responseType: 'blob' }).then((response) => {
      objectUrl = URL.createObjectURL(response.data)
      if (active) setUrl(objectUrl)
    }).catch(() => setUrl(null))
    return () => { active = false; if (objectUrl) URL.revokeObjectURL(objectUrl) }
  }, [path])
  return url ? <img src={url} alt="" className={className} /> : <div className={`${className || ''} flex items-center justify-center bg-muted`}><Loader2 className="w-5 h-5 animate-spin text-muted-foreground" /></div>
}

export default function GeminiReviewPage() {
  const { t } = useTranslation()
  const [items, setItems] = useState<GeminiCandidate[]>([])
  const [selected, setSelected] = useState<number | null>(null)
  const [checked, setChecked] = useState<number[]>([])
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState<number | null>(null)
  const [datasetName, setDatasetName] = useState('Gemini aerial object masks')
  const targetModelType = 'object_segmenter'
  const [datasetId, setDatasetId] = useState<number | null>(null)

  const load = async () => {
    const response = await api.get('/gemini/candidates?limit=200')
    setItems(response.data)
    setSelected((current) => current ?? response.data[0]?.id ?? null)
    setLoading(false)
  }

  useEffect(() => {
    load().catch(() => { setLoading(false); toast.error(t('Failed to load Gemini review queue')) })
    const timer = window.setInterval(() => load().catch(() => {}), 5000)
    return () => window.clearInterval(timer)
  }, [])

  const candidate = useMemo(() => items.find((item) => item.id === selected) || null, [items, selected])
  const approvedIds = items.filter((item) => checked.includes(item.id) && item.status === 'approved' && item.target_model_type !== 'verification_only').map((item) => item.id)

  const review = async (id: number, status: 'approved' | 'rejected') => {
    setBusy(id)
    try {
      await api.patch(`/gemini/candidates/${id}/review`, { status, annotation_index: 0 })
      if (status === 'approved') setChecked((value) => Array.from(new Set([...value, id])))
      else setChecked((value) => value.filter((candidateId) => candidateId !== id))
      await load()
      toast.success(t(status === 'approved' ? 'Candidate approved' : 'Candidate rejected'))
    } catch (error: any) {
      toast.error(error.response?.data?.detail || t('Review failed'))
    } finally { setBusy(null) }
  }

  const retry = async (id: number) => {
    setBusy(id)
    try { await api.post(`/gemini/candidates/${id}/retry`); await load() }
    catch (error: any) { toast.error(error.response?.data?.detail || t('Retry failed')) }
    finally { setBusy(null) }
  }

  const importDataset = async () => {
    if (!approvedIds.length || !datasetName.trim()) return
    setBusy(-1)
    try {
      const response = await trainingApi.post('/datasets/import-gemini', {
        candidate_ids: approvedIds,
        dataset_name: datasetName.trim(),
        target_model_type: targetModelType,
      })
      setDatasetId(response.data.dataset_id)
      setChecked([])
      await load()
      toast.success(t('Approved masks imported into a training dataset'))
    } catch (error: any) {
      toast.error(error.response?.data?.detail || t('Dataset import failed'))
    } finally { setBusy(null) }
  }

  return (
    <AppShell>
      <div className="p-6 space-y-6">
        <div className="flex items-start justify-between gap-4">
          <div><h1 className="text-2xl font-bold flex items-center gap-2"><Sparkles className="w-6 h-6 text-violet-400" />{t('Gemini Review')}</h1><p className="text-sm text-muted-foreground mt-1">{t('Human approval is required before any mask enters a training dataset.')}</p></div>
          <button onClick={() => load()} className="p-2 border border-border rounded-lg" title={t('Refresh')}><RefreshCw className="w-4 h-4" /></button>
        </div>

        {loading ? <div className="flex gap-2 text-muted-foreground"><Loader2 className="w-4 h-4 animate-spin" />{t('Loading...')}</div> : (
          <div className="grid xl:grid-cols-[340px_minmax(0,1fr)] gap-5">
            <section className="bg-card border border-border rounded-xl overflow-hidden">
              <div className="p-4 border-b border-border text-sm font-medium">{t('Review queue')} ({items.length})</div>
              <div className="max-h-[68vh] overflow-y-auto divide-y divide-border">
                {items.map((item) => <button key={item.id} onClick={() => setSelected(item.id)} className={`w-full text-left p-3 hover:bg-accent/40 ${selected === item.id ? 'bg-primary/10' : ''}`}>
                  <div className="flex items-center justify-between gap-2"><span className="text-sm font-medium">#{item.id} · {t('Camera')} {item.camera_id}</span><Status value={item.status} /></div>
                  <div className="text-xs text-muted-foreground mt-1">track {item.track_id ?? '—'} · {item.proposed_annotations.length} {t('masks')}</div>
                </button>)}
                {!items.length && <div className="p-6 text-sm text-muted-foreground">{t('No Gemini candidates yet. Enable capture for a camera or video.')}</div>}
              </div>
            </section>

            <div className="space-y-5">
              {candidate ? <section className="bg-card border border-border rounded-xl p-5 space-y-4">
                <div className="flex flex-wrap justify-between gap-3"><div><h2 className="font-semibold">{t('Candidate')} #{candidate.id}</h2><p className="text-xs text-muted-foreground mt-1">{candidate.provider_model || 'Gemini'} · {candidate.selection_reason}</p></div><Status value={candidate.status} /></div>
                <div className="relative bg-black rounded-xl overflow-hidden aspect-video">
                  <SecureImage path={candidate.image_url} className="absolute inset-0 w-full h-full object-contain" />
                  {candidate.mask_url && <SecureImage path={candidate.mask_url} className="absolute inset-0 w-full h-full object-contain opacity-45 mix-blend-screen" />}
                </div>
                {candidate.gemini_verification && <div className={`rounded-lg p-3 text-sm ${candidate.gemini_verification.local_predictions_correct ? 'bg-emerald-500/10 text-emerald-300' : 'bg-amber-500/10 text-amber-300'}`}><strong>{candidate.gemini_verification.local_predictions_correct ? t('Local prediction confirmed') : t('Gemini found a disagreement')}</strong><p className="mt-1 opacity-90">{candidate.gemini_verification.summary}</p><p className="text-xs mt-1">{t('Confidence')}: {Math.round(candidate.gemini_verification.confidence * 100)}%</p></div>}
                <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-2">{candidate.proposed_annotations.map((annotation, index) => <div key={index} className="border border-border rounded-lg p-3 text-sm"><div className="font-medium">{annotation.class_name}</div><div className="text-xs text-muted-foreground mt-1">{Math.round((annotation.confidence || 0) * 100)}% · {annotation.polygon.length} points</div></div>)}</div>
                {candidate.error_message && <div className="bg-red-500/10 text-red-300 rounded-lg p-3 text-sm">{candidate.error_message}</div>}
                <div className="flex flex-wrap gap-3">
                  <button onClick={() => review(candidate.id, 'approved')} disabled={busy === candidate.id || candidate.status === 'processing'} className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-500/15 text-emerald-400 disabled:opacity-50"><Check className="w-4 h-4" />{t('Approve masks')}</button>
                  <button onClick={() => review(candidate.id, 'rejected')} disabled={busy === candidate.id || candidate.status === 'processing'} className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-red-500/15 text-red-400 disabled:opacity-50"><X className="w-4 h-4" />{t('Reject')}</button>
                  {candidate.status === 'error' && <button onClick={() => retry(candidate.id)} disabled={busy === candidate.id} className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-border"><RotateCcw className="w-4 h-4" />{t('Retry')}</button>}
                </div>
              </section> : null}

              <section className="bg-card border border-border rounded-xl p-5 space-y-3">
                <div className="flex items-center gap-2 font-semibold"><Database className="w-4 h-4" />{t('Create segmentation dataset')}</div>
                <p className="text-xs text-muted-foreground">{t('Select approved candidates in the queue, then import them. Imported data still requires an explicit split and Training Job.')}</p>
                <div className="grid md:grid-cols-[180px_1fr_auto] gap-3"><div className="bg-background border border-border rounded-lg px-3 py-2 text-sm">{t('Object Segmenter')}</div><input value={datasetName} onChange={(e) => setDatasetName(e.target.value)} className="bg-background border border-border rounded-lg px-3 py-2 text-sm" /><button onClick={importDataset} disabled={!approvedIds.length || busy === -1} className="btn-primary px-4 py-2 rounded-lg text-sm disabled:opacity-50">{t('Import selected')} ({approvedIds.length})</button></div>
                <div className="flex flex-wrap gap-2">{items.filter((item) => item.status === 'approved' && item.target_model_type !== 'verification_only').map((item) => <label key={item.id} className="flex items-center gap-2 border border-border rounded-lg px-3 py-2 text-xs"><input type="checkbox" checked={checked.includes(item.id)} onChange={(e) => setChecked((value) => e.target.checked ? [...value, item.id] : value.filter((id) => id !== item.id))} />#{item.id}</label>)}</div>
                {datasetId && <Link href={`/training/datasets/${datasetId}`} className="text-sm text-primary hover:underline">{t('Open created dataset')} #{datasetId}</Link>}
              </section>
            </div>
          </div>
        )}
      </div>
    </AppShell>
  )
}

function Status({ value }: { value: string }) {
  const colors: Record<string, string> = { ready: 'text-blue-400', approved: 'text-emerald-400', rejected: 'text-red-400', imported: 'text-violet-400', error: 'text-red-400', processing: 'text-amber-400' }
  return <span className={`text-xs font-medium uppercase ${colors[value] || 'text-muted-foreground'}`}>{value}</span>
}
