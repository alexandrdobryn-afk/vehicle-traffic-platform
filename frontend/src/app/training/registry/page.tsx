'use client'
import { useEffect, useState } from 'react'
import AppShell from '@/components/shared/AppShell'
import trainingApi from '@/lib/trainingApi'
import { TEvaluationReport, TModelVersion, ModelType, MODEL_TYPE_LABELS, DEPLOY_STATUS_COLORS } from '@/types/training'
import { formatDateTime } from '@/lib/utils'
import { CheckCircle, XCircle, RefreshCw, Upload, RotateCcw, Download, Shield, ChevronDown, ChevronRight } from 'lucide-react'
import toast from 'react-hot-toast'
import { useTranslation } from '@/lib/i18n'

const TYPE_FILTERS: { value: string; label: string }[] = [
  { value: '', label: 'All Types' },
  { value: 'object_detector', label: 'Object Detector' },
  { value: 'object_segmenter', label: 'Object Segmenter' },
  { value: 'object_classifier', label: 'Object Classifier' },
]

function metricValue(value: unknown) {
  if (typeof value === 'number') return value <= 1 ? `${(value * 100).toFixed(1)}%` : value.toFixed(3)
  if (value == null) return '-'
  return String(value)
}

function compactMetricLabel(key: string) {
  return key.replace(/_/g, ' ')
}

export default function RegistryPage() {
  const { t, locale } = useTranslation()
  const [models, setModels] = useState<TModelVersion[]>([])
  const [reports, setReports] = useState<TEvaluationReport[]>([])
  const [filter, setFilter] = useState('')
  const [loading, setLoading] = useState(false)
  const [expandedId, setExpandedId] = useState<number | null>(null)
  const [approveModal, setApproveModal] = useState<TModelVersion | null>(null)
  const [approveComment, setApproveComment] = useState('')
  const [approving, setApproving] = useState(false)

  const load = async () => {
    setLoading(true)
    try {
      const [modelResponse, reportResponse] = await Promise.all([
        trainingApi.get(`/registry${filter ? `?model_type=${filter}` : ''}`),
        trainingApi.get('/evaluation/reports'),
      ])
      setModels(modelResponse.data)
      setReports(reportResponse.data)
    } finally { setLoading(false) }
  }

  useEffect(() => { load() }, [filter])

  const handleApprove = async () => {
    if (!approveModal) return
    setApproving(true)
    try {
      const r = await trainingApi.post(`/registry/${approveModal.id}/approve`, {
        comment: approveComment,
        run_auto_tests: true,
      })
      if (r.data.success) {
        toast.success(t('Model deployed: {name}', { name: approveModal.name }))
        setApproveModal(null)
        setApproveComment('')
      } else {
        toast.error(t('Deploy failed: {error}', { error: r.data.error }))
      }
      load()
    } catch (e: any) {
      toast.error(e.response?.data?.detail || t('Deploy failed'))
    } finally { setApproving(false) }
  }

  const handleReject = async (mv: TModelVersion) => {
    if (!confirm(t('Reject model {name}?', { name: mv.name }))) return
    await trainingApi.post(`/registry/${mv.id}/reject`, null, { params: { comment: 'Manually rejected' } })
    toast.success(t('Model rejected'))
    load()
  }

  const handleRollback = async (mv: TModelVersion) => {
    if (!confirm(t('Roll back to {name}? This will replace the current production model.', { name: mv.name }))) return
    const r = await trainingApi.post('/registry/rollback', {
      model_version_id: mv.id,
      comment: 'Manual rollback via UI',
    })
    if (r.data.success) toast.success(t('Rolled back to {name}', { name: mv.name }))
    else toast.error(r.data.error)
    load()
  }

  const handleExport = async (mv: TModelVersion, format: string) => {
    const r = await trainingApi.post(`/registry/${mv.id}/export/${format}`)
    if (r.data.success) toast.success(t('{format} export queued', { format: format.toUpperCase() }))
    else toast.error(r.data.error)
  }

  const handleValidate = async (mv: TModelVersion) => {
    const r = await trainingApi.post(`/registry/${mv.id}/validate`)
    if (r.data.success) toast.success(t('Validation queued'))
    load()
  }

  return (
    <AppShell>
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between pr-40">
          <div>
            <h1 className="text-2xl font-bold text-foreground">{t('Model Registry')}</h1>
            <p className="text-muted-foreground text-sm">{t('{count} model versions', { count: models.length })}</p>
          </div>
          <button onClick={load} className="p-2 bg-muted rounded-lg text-muted-foreground hover:text-foreground" title={t('Refresh')} aria-label={t('Refresh')}>
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>

        {/* Filter */}
        <div className="flex gap-2 flex-wrap">
          {TYPE_FILTERS.map(f => (
            <button key={f.value} onClick={() => setFilter(f.value)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                filter === f.value
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted text-muted-foreground hover:text-foreground'
              }`}>
              {t(f.label)}
            </button>
          ))}
        </div>

        {models.length === 0 ? (
          <div className="border border-dashed border-border rounded-xl p-12 text-center">
            <Shield className="w-10 h-10 text-muted-foreground mx-auto mb-3" />
            <p className="font-medium text-foreground mb-1">{t('No models yet')}</p>
            <p className="text-sm text-muted-foreground">{t('Complete a training job to register a model')}</p>
          </div>
        ) : (
          <div className="space-y-4">
            {models.map((mv) => {
              const report = reports.find(item => item.id === mv.evaluation_report_id)
                || reports.find(item => item.model_version_id === mv.id)
              const metadata = mv.artifact_metadata || {}
              const training = (metadata.training || {}) as Record<string, any>
              const baseModel = (metadata.base_model || {}) as Record<string, any>
              const expanded = expandedId === mv.id
              return (
              <div key={mv.id} className={`bg-card border rounded-xl p-5 ${
                mv.is_production ? 'border-emerald-500/30' : 'border-border'
              }`}>
                <div className="flex items-start justify-between gap-4">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2 flex-wrap mb-1">
                      <h3 className="font-semibold text-foreground">{mv.name}</h3>
                      <span className="text-xs px-2 py-0.5 bg-muted text-muted-foreground rounded">{mv.version}</span>
                      {mv.is_production && (
                        <span className="text-xs px-2 py-0.5 bg-emerald-500/10 text-emerald-400 rounded font-medium">
                          ✓ {t('PRODUCTION')}
                        </span>
                      )}
                      <span className={`text-xs font-medium capitalize ${DEPLOY_STATUS_COLORS[mv.deploy_status]}`}>
                        {t(mv.deploy_status)}
                      </span>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {t(MODEL_TYPE_LABELS[mv.model_type])} · {mv.architecture}
                      {mv.dataset_name && ` · ${mv.dataset_name}`}
                      {mv.author_email && ` · ${t('by')} ${mv.author_email}`}
                    </p>
                  </div>

                  {/* Actions */}
                  <div className="flex gap-2 shrink-0 flex-wrap justify-end">
                    <button onClick={() => setExpandedId(expanded ? null : mv.id)}
                      className="flex items-center gap-1.5 px-3 py-1.5 bg-muted text-muted-foreground rounded-lg text-xs hover:text-foreground">
                      {expanded ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
                      {t('Analysis')}
                    </button>
                    {(mv.deploy_status === 'pending' || mv.deploy_status === 'candidate') && (
                      <>
                        <button onClick={() => setApproveModal(mv)}
                          className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded-lg text-xs hover:bg-emerald-500/20">
                          <Upload className="w-3.5 h-3.5" /> {t('Deploy')}
                        </button>
                        <button onClick={() => handleReject(mv)}
                          className="flex items-center gap-1.5 px-3 py-1.5 bg-red-500/10 text-red-400 border border-red-500/20 rounded-lg text-xs hover:bg-red-500/20">
                          <XCircle className="w-3.5 h-3.5" /> {t('Reject')}
                        </button>
                      </>
                    )}
                    {mv.deploy_status === 'deployed' && !mv.is_production && (
                      <button onClick={() => handleRollback(mv)}
                        className="flex items-center gap-1.5 px-3 py-1.5 bg-amber-500/10 text-amber-400 border border-amber-500/20 rounded-lg text-xs hover:bg-amber-500/20">
                        <RotateCcw className="w-3.5 h-3.5" /> {t('Rollback to this')}
                      </button>
                    )}
                    <button onClick={() => handleValidate(mv)}
                      className="flex items-center gap-1.5 px-3 py-1.5 bg-muted text-muted-foreground rounded-lg text-xs hover:text-foreground">
                      <RefreshCw className="w-3.5 h-3.5" /> {t('Validate')}
                    </button>
                    {!mv.onnx_path && (
                      <button onClick={() => handleExport(mv, 'onnx')}
                        className="flex items-center gap-1.5 px-3 py-1.5 bg-muted text-muted-foreground rounded-lg text-xs hover:text-foreground">
                        <Download className="w-3.5 h-3.5" /> ONNX
                      </button>
                    )}
                    {!mv.trt_path && (
                      <button onClick={() => handleExport(mv, 'tensorrt')}
                        className="flex items-center gap-1.5 px-3 py-1.5 bg-muted text-muted-foreground rounded-lg text-xs hover:text-foreground">
                        <Download className="w-3.5 h-3.5" /> TRT
                      </button>
                    )}
                  </div>
                </div>

                {/* Metrics */}
                {mv.metrics && Object.keys(mv.metrics).length > 0 && (
                  <div className="flex gap-4 mt-3 pt-3 border-t border-border/50 flex-wrap">
                    {Object.entries(mv.metrics).map(([k, v]) => (
                      <div key={k} className="text-center">
                        <p className="text-xs text-muted-foreground uppercase">{t(k.replace(/_/g, ' '))}</p>
                        <p className="text-sm font-bold text-foreground">
                          {typeof v === 'number' ? (v < 2 ? `${(v * 100).toFixed(1)}%` : v.toFixed(3)) : String(v)}
                        </p>
                      </div>
                    ))}
                  </div>
                )}

                {/* Benchmark vs prev */}
                {mv.benchmark_vs_prev && !mv.benchmark_vs_prev.is_first && (
                  <div className="mt-3 pt-3 border-t border-border/50">
                    <p className="text-xs text-muted-foreground mb-1">{t('vs Production')} ({String(mv.benchmark_vs_prev.production_model_name ?? t('unknown'))})</p>
                    <div className="flex gap-4 flex-wrap">
                      {Object.entries(mv.benchmark_vs_prev)
                        .filter(([k]) => k.endsWith('_delta'))
                        .map(([k, v]) => {
                          const val = v as number
                          return (
                            <span key={k} className={`text-xs font-medium ${val >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                              {k.replace('_delta', '').toUpperCase()}: {val >= 0 ? '+' : ''}{(val * 100).toFixed(2)}%
                            </span>
                          )
                        })}
                      <span className={`text-xs font-medium ${mv.benchmark_vs_prev.is_improvement ? 'text-emerald-400' : 'text-amber-400'}`}>
                        {mv.benchmark_vs_prev.is_improvement ? `✓ ${t('Improvement')}` : `⚠ ${t('Regression')}`}
                      </span>
                    </div>
                  </div>
                )}

                {mv.gate_result && (
                  <div className="mt-3 pt-3 border-t border-border/50">
                    <p className={`text-xs font-medium ${
                      mv.gate_result === 'approved' ? 'text-emerald-400' :
                      mv.gate_result === 'rejected' ? 'text-red-400' :
                      'text-amber-400'
                    }`}>
                      {t('Decision gate')}: {t(mv.gate_result)}
                    </p>
                    {mv.gate_reasons?.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-2">
                        {mv.gate_reasons.map(reason => (
                          <span key={reason} className="rounded bg-muted px-2 py-1 text-xs text-muted-foreground">
                            {t(reason)}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* Auto test results */}
                {mv.auto_test_results && (
                  <div className="mt-2 flex gap-3 flex-wrap">
                    {Object.entries(mv.auto_test_results.tests as Record<string, any> || {}).map(([k, v]) => {
                      if (typeof v === 'boolean') {
                        return (
                          <span key={k} className={`flex items-center gap-1 text-xs ${v ? 'text-emerald-400' : 'text-red-400'}`}>
                            {v ? <CheckCircle className="w-3 h-3" /> : <XCircle className="w-3 h-3" />}
                            {t(k.replace(/_/g, ' '))}
                          </span>
                        )
                      }
                      if (typeof v === 'number') {
                        return (
                          <span key={k} className="text-xs text-muted-foreground">
                            {t(k.replace(/_/g, ' '))}: {v}
                          </span>
                        )
                      }
                      return null
                    })}
                  </div>
                )}

                {expanded && (
                  <div className="mt-4 space-y-4 border-t border-border/50 pt-4">
                    <div className="grid gap-3 md:grid-cols-3">
                      <div className="rounded-lg bg-muted/40 p-3">
                        <p className="mb-2 text-xs font-semibold uppercase text-muted-foreground">{t('Training provenance')}</p>
                        {[
                          ['Base model', baseModel.label || baseModel.id || '-'],
                          ['Base source', baseModel.source || '-'],
                          ['Mode', training.mode || '-'],
                          ['Epochs', training.epochs_completed ?? mv.metrics?.epoch ?? '-'],
                          ['Cumulative epochs', training.cumulative_epochs ?? mv.metrics?.cumulative_epochs ?? '-'],
                        ].map(([label, value]) => (
                          <div key={String(label)} className="flex justify-between gap-3 py-1 text-xs">
                            <span className="text-muted-foreground">{t(String(label))}</span>
                            <span className="text-right text-foreground">{t(String(value))}</span>
                          </div>
                        ))}
                      </div>
                      <div className="rounded-lg bg-muted/40 p-3">
                        <p className="mb-2 text-xs font-semibold uppercase text-muted-foreground">{t('Backbone / Head')}</p>
                        {[
                          ['Backbone trainable', training.backbone?.trainable === false ? 'No' : 'Yes'],
                          ['Frozen layers', training.backbone?.freeze_layers ?? 0],
                          ['Head trainable', training.head?.trainable === false ? 'No' : 'Yes'],
                          ['Head architecture', training.head?.architecture || mv.architecture],
                          ['Metric type', mv.metrics?.metric_type || report?.summary?.metric_type || '-'],
                        ].map(([label, value]) => (
                          <div key={String(label)} className="flex justify-between gap-3 py-1 text-xs">
                            <span className="text-muted-foreground">{t(String(label))}</span>
                            <span className="text-right text-foreground">{t(String(value))}</span>
                          </div>
                        ))}
                      </div>
                      <div className="rounded-lg bg-muted/40 p-3">
                        <p className="mb-2 text-xs font-semibold uppercase text-muted-foreground">{t('Artifacts')}</p>
                        {[
                          ['Weights', mv.weights_path ? 'PT saved' : '-'],
                          ['ONNX', mv.onnx_path ? 'Exported' : '-'],
                          ['TensorRT', mv.trt_path ? 'Exported' : '-'],
                          ['Path', mv.weights_path || '-'],
                        ].map(([label, value]) => (
                          <div key={String(label)} className="flex justify-between gap-3 py-1 text-xs">
                            <span className="text-muted-foreground">{t(String(label))}</span>
                            <span className="max-w-[190px] truncate text-right text-foreground" title={String(value)}>{t(String(value))}</span>
                          </div>
                        ))}
                      </div>
                    </div>

                    {report && (
                      <div className="grid gap-4 lg:grid-cols-[1fr_1fr]">
                        <div className="rounded-lg bg-muted/40 p-3">
                          <p className="mb-3 text-xs font-semibold uppercase text-muted-foreground">{t('Validation metrics')}</p>
                          <div className="grid grid-cols-2 gap-2">
                            {Object.entries(report.summary).map(([key, value]) => (
                              <div key={key} className="rounded bg-background/60 p-2">
                                <p className="text-[10px] uppercase text-muted-foreground">{t(compactMetricLabel(key))}</p>
                                <p className="text-sm font-semibold text-foreground">{metricValue(value)}</p>
                              </div>
                            ))}
                          </div>
                        </div>
                        <div className="rounded-lg bg-muted/40 p-3">
                          <p className="mb-3 text-xs font-semibold uppercase text-muted-foreground">{t('Slice and class analysis')}</p>
                          <div className="space-y-2">
                            {Object.entries(report.per_class_metrics || {}).slice(0, 5).map(([name, value]: [string, any]) => (
                              <div key={name} className="flex justify-between gap-3 text-xs">
                                <span className="text-foreground">{name}</span>
                                <span className="text-muted-foreground">
                                  P {metricValue(value.precision)} / R {metricValue(value.recall)} / F1 {metricValue(value.f1)}
                                </span>
                              </div>
                            ))}
                            {Object.entries(report.slice_metrics || {}).slice(0, 4).map(([name, value]: [string, any]) => (
                              <div key={name} className="flex justify-between gap-3 border-t border-border/50 pt-2 text-xs">
                                <span className="text-foreground">{t(name)}</span>
                                <span className="text-muted-foreground">
                                  {t('Recall')} {metricValue(value.recall)} / {t('Samples')} {value.samples ?? 0}
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Footer */}
                <div className="flex justify-between items-center mt-3 pt-3 border-t border-border/50">
                  <div className="flex gap-3 text-xs text-muted-foreground">
                    {mv.weights_path && <span>✓ PT</span>}
                    {mv.onnx_path && <span>✓ ONNX</span>}
                    {mv.trt_path && <span>✓ TRT</span>}
                    {mv.validation_passed === true && <span className="text-emerald-400">✓ {t('Validated')}</span>}
                    {mv.validation_passed === false && <span className="text-red-400">✗ {t('Validation failed')}</span>}
                  </div>
                  <span className="text-xs text-muted-foreground">{formatDateTime(mv.created_at, locale)}</span>
                </div>
              </div>
              )
            })}
          </div>
        )}

        {/* Deploy approval modal */}
        {approveModal && (
          <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4">
            <div className="bg-card border border-border rounded-2xl w-full max-w-md shadow-2xl">
              <div className="p-6 border-b border-border">
                <h2 className="font-semibold text-foreground">{t('Deploy Model')}</h2>
                <p className="text-sm text-muted-foreground mt-1">{approveModal.name}</p>
              </div>
              <div className="p-6 space-y-4">
                <div className="p-3 bg-amber-500/10 border border-amber-500/20 rounded-lg">
                  <p className="text-sm text-amber-400">
                    ⚠️ {t('This will replace the current production model and copy weights to the inference backend. Auto tests will run first.')}
                  </p>
                </div>
                {approveModal.gate_result && (
                  <div className={`p-3 rounded-lg border ${
                    approveModal.gate_result === 'approved'
                      ? 'bg-emerald-500/10 border-emerald-500/20'
                      : 'bg-amber-500/10 border-amber-500/20'
                  }`}>
                    <p className={`text-sm font-medium ${approveModal.gate_result === 'approved' ? 'text-emerald-400' : 'text-amber-400'}`}>
                      {t('Decision gate')}: {t(approveModal.gate_result)}
                    </p>
                  </div>
                )}
                {approveModal.benchmark_vs_prev && !approveModal.benchmark_vs_prev.is_first && (
                  <div className={`p-3 rounded-lg border ${
                    approveModal.benchmark_vs_prev.is_improvement
                      ? 'bg-emerald-500/10 border-emerald-500/20'
                      : 'bg-red-500/10 border-red-500/20'
                  }`}>
                    <p className={`text-sm font-medium ${approveModal.benchmark_vs_prev.is_improvement ? 'text-emerald-400' : 'text-red-400'}`}>
                      {approveModal.benchmark_vs_prev.is_improvement ? `✓ ${t('Performance improvement detected')}` : `⚠ ${t('Performance regression detected')}`}
                    </p>
                  </div>
                )}
                <div>
                  <label className="block text-sm font-medium text-muted-foreground mb-1">{t('Comment (optional)')}</label>
                  <textarea value={approveComment} onChange={e => setApproveComment(e.target.value)}
                    rows={3} placeholder={t('Reason for deployment...')}
                    className="w-full px-3 py-2 bg-background border border-input rounded-lg text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring resize-none" />
                </div>
              </div>
              <div className="flex gap-3 p-6 border-t border-border">
                <button onClick={() => setApproveModal(null)}
                  className="flex-1 py-2 border border-border rounded-lg text-sm text-muted-foreground">
                  {t('Cancel')}
                </button>
                <button onClick={handleApprove} disabled={approving}
                  className="flex-1 py-2 bg-emerald-600 text-white rounded-lg text-sm font-medium hover:bg-emerald-700 disabled:opacity-50">
                  {approving ? t('Deploying...') : `🚀 ${t('Deploy to Production')}`}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </AppShell>
  )
}
