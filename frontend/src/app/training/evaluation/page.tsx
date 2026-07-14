'use client'
import { useEffect, useMemo, useState } from 'react'
import AppShell from '@/components/shared/AppShell'
import trainingApi from '@/lib/trainingApi'
import { TEvaluationError, TEvaluationReport, TModelVersion } from '@/types/training'
import { formatDateTime } from '@/lib/utils'
import { AlertTriangle, BarChart3, CheckCircle, RefreshCw, ShieldAlert, XCircle } from 'lucide-react'
import { useTranslation } from '@/lib/i18n'

const ERROR_TABS = [
  { key: '', label: 'All' },
  { key: 'false_positive', label: 'False Positives' },
  { key: 'false_negative', label: 'False Negatives' },
  { key: 'small_object_miss', label: 'Small Object Misses' },
  { key: 'wrong_class', label: 'Class Confusion' },
  { key: 'low_confidence', label: 'Low Confidence' },
  { key: 'hard_negative', label: 'Hard Negatives' },
  { key: 'needs_annotation', label: 'Needs Annotation' },
]

function metricValue(value: unknown) {
  if (typeof value === 'number') return value <= 1 ? `${(value * 100).toFixed(1)}%` : value.toFixed(2)
  if (value == null) return '—'
  return String(value)
}

export default function EvaluationPage() {
  const { t, locale } = useTranslation()
  const [reports, setReports] = useState<TEvaluationReport[]>([])
  const [errors, setErrors] = useState<TEvaluationError[]>([])
  const [models, setModels] = useState<TModelVersion[]>([])
  const [selectedReportId, setSelectedReportId] = useState<number | null>(null)
  const [errorTab, setErrorTab] = useState('')
  const [loading, setLoading] = useState(false)

  const modelById = useMemo(() => Object.fromEntries(models.map(model => [model.id, model])), [models])
  const selected = reports.find(report => report.id === selectedReportId) || reports[0] || null
  const selectedErrors = selected ? errors.filter(error => error.report_id === selected.id) : []
  const visibleErrors = errorTab ? selectedErrors.filter(error => error.error_type === errorTab) : selectedErrors

  const load = async () => {
    setLoading(true)
    try {
      const [reportResponse, errorResponse, modelResponse] = await Promise.all([
        trainingApi.get('/evaluation/reports'),
        trainingApi.get('/evaluation/errors?limit=300'),
        trainingApi.get('/registry'),
      ])
      setReports(reportResponse.data)
      setErrors(errorResponse.data)
      setModels(modelResponse.data)
      if (!selectedReportId && reportResponse.data[0]) setSelectedReportId(reportResponse.data[0].id)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  return (
    <AppShell>
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between pr-40">
          <div>
            <h1 className="text-2xl font-bold text-foreground">{t('Evaluation')}</h1>
            <p className="text-sm text-muted-foreground">{t('Quality reports, model comparison and decision gates')}</p>
          </div>
          <button onClick={load} className="flex items-center gap-2 rounded-lg bg-muted px-3 py-2 text-sm text-muted-foreground hover:text-foreground">
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} /> {t('Refresh')}
          </button>
        </div>

        {reports.length === 0 ? (
          <div className="rounded-xl border border-dashed border-border p-12 text-center">
            <BarChart3 className="mx-auto mb-3 h-10 w-10 text-muted-foreground" />
            <p className="font-medium text-foreground">{t('No evaluation reports yet')}</p>
            <p className="text-sm text-muted-foreground">{t('Validate a model from the registry to create the first report')}</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-6 xl:grid-cols-[360px_1fr]">
            <div className="space-y-3">
              {reports.map(report => {
                const model = modelById[report.model_version_id]
                const active = selected?.id === report.id
                return (
                  <button
                    key={report.id}
                    onClick={() => setSelectedReportId(report.id)}
                    className={`w-full rounded-xl border p-4 text-left transition-colors ${active ? 'border-primary bg-primary/5' : 'border-border bg-card hover:border-primary/30'}`}
                  >
                    <div className="mb-2 flex items-center gap-2">
                      {report.gate_result === 'approved' && <CheckCircle className="h-4 w-4 text-emerald-400" />}
                      {report.gate_result === 'candidate' && <AlertTriangle className="h-4 w-4 text-amber-400" />}
                      {report.gate_result === 'rejected' && <XCircle className="h-4 w-4 text-red-400" />}
                      <span className="text-xs font-medium uppercase text-muted-foreground">{t(report.gate_result)}</span>
                    </div>
                    <p className="truncate text-sm font-medium text-foreground">{model?.name || `${t('Model')} #${report.model_version_id}`}</p>
                    <p className="text-xs text-muted-foreground">{formatDateTime(report.created_at, locale)}</p>
                  </button>
                )
              })}
            </div>

            {selected && (
              <div className="space-y-6">
                <section className="rounded-xl border border-border bg-card p-5">
                  <div className="mb-4 flex items-center justify-between">
                    <div>
                      <h2 className="font-semibold text-foreground">{modelById[selected.model_version_id]?.name || `${t('Model')} #${selected.model_version_id}`}</h2>
                      <p className="text-xs text-muted-foreground">{t('Report')} #{selected.id}</p>
                    </div>
                    <span className={`rounded px-2 py-1 text-xs font-medium ${
                      selected.gate_result === 'approved' ? 'bg-emerald-500/10 text-emerald-400' :
                      selected.gate_result === 'rejected' ? 'bg-red-500/10 text-red-400' :
                      'bg-amber-500/10 text-amber-400'
                    }`}>
                      {t(selected.gate_result)}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                    {Object.entries(selected.summary).slice(0, 8).map(([key, value]) => (
                      <div key={key} className="rounded-lg bg-muted/50 p-3">
                        <p className="text-[10px] uppercase text-muted-foreground">{t(key.replace(/_/g, ' '))}</p>
                        <p className="text-lg font-bold text-foreground">{metricValue(value)}</p>
                      </div>
                    ))}
                  </div>
                </section>

                {Object.keys(selected.per_class_metrics || {}).length > 0 && (
                  <section className="rounded-xl border border-border bg-card p-5">
                    <h3 className="mb-4 font-semibold text-foreground">{t('Per-class quality')}</h3>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead className="text-xs text-muted-foreground">
                          <tr>
                            {['Class', 'Precision', 'Recall', 'F1', 'FP', 'FN'].map(label => (
                              <th key={label} className="pb-2 text-left font-medium">{t(label)}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-border">
                          {Object.entries(selected.per_class_metrics).map(([name, value]: [string, any]) => (
                            <tr key={name}>
                              <td className="py-2 text-foreground">{name}</td>
                              <td className="py-2 text-muted-foreground">{metricValue(value.precision)}</td>
                              <td className="py-2 text-muted-foreground">{metricValue(value.recall)}</td>
                              <td className="py-2 text-muted-foreground">{metricValue(value.f1)}</td>
                              <td className="py-2 text-muted-foreground">{value.false_positives ?? value.fp ?? 0}</td>
                              <td className="py-2 text-muted-foreground">{value.false_negatives ?? value.fn ?? 0}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </section>
                )}

                {Object.keys(selected.slice_metrics || {}).length > 0 && (
                  <section className="rounded-xl border border-border bg-card p-5">
                    <h3 className="mb-4 font-semibold text-foreground">{t('Condition slices')}</h3>
                    <div className="grid gap-3 md:grid-cols-2">
                      {Object.entries(selected.slice_metrics).map(([name, value]: [string, any]) => (
                        <div key={name} className="rounded-lg bg-muted/50 p-3">
                          <p className="text-xs text-muted-foreground">{t(name)}</p>
                          <p className="mt-1 text-sm text-foreground">
                            {t('Precision')}: {metricValue(value.precision)} · {t('Recall')}: {metricValue(value.recall)} · {t('F1')}: {metricValue(value.f1)}
                          </p>
                          <p className="mt-1 text-xs text-muted-foreground">{t('Samples')}: {value.samples ?? 0}</p>
                        </div>
                      ))}
                    </div>
                  </section>
                )}

                <section className="rounded-xl border border-border bg-card p-5">
                  <div className="mb-3 flex items-center gap-2">
                    <ShieldAlert className="h-4 w-4 text-amber-400" />
                    <h3 className="font-semibold text-foreground">{t('Decision Gate')}</h3>
                  </div>
                  <div className="space-y-2">
                    {selected.gate_reasons.length ? selected.gate_reasons.map(reason => (
                      <p key={reason} className="rounded-lg bg-muted/50 px-3 py-2 text-sm text-muted-foreground">{t(reason)}</p>
                    )) : (
                      <p className="text-sm text-muted-foreground">{t('No gate reasons recorded')}</p>
                    )}
                  </div>
                </section>

                <section className="rounded-xl border border-border bg-card p-5">
                  <h3 className="mb-4 font-semibold text-foreground">{t('Error Analysis')}</h3>
                  <div className="mb-4 flex flex-wrap gap-2">
                    {ERROR_TABS.map(tab => (
                      <button
                        key={tab.key || 'all'}
                        onClick={() => setErrorTab(tab.key)}
                        className={`rounded-lg px-3 py-1.5 text-xs font-medium ${errorTab === tab.key ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground hover:text-foreground'}`}
                      >
                        {t(tab.label)}
                      </button>
                    ))}
                  </div>
                  {visibleErrors.length === 0 ? (
                    <p className="py-10 text-center text-sm text-muted-foreground">{t('No error items recorded for this report')}</p>
                  ) : (
                    <div className="divide-y divide-border">
                      {visibleErrors.map(error => (
                        <div key={error.id} className="flex items-center justify-between py-3">
                          <div>
                            <p className="text-sm font-medium text-foreground">{t(error.error_type)}</p>
                            <p className="text-xs text-muted-foreground">
                              {error.class_name || t('unknown class')} · {t('Priority')}: {error.priority_score.toFixed(0)}
                            </p>
                          </div>
                          {error.confidence != null && (
                            <span className="text-xs text-muted-foreground">{Math.round(error.confidence * 100)}%</span>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </section>
              </div>
            )}
          </div>
        )}
      </div>
    </AppShell>
  )
}
