import { useState, useEffect } from 'react'
import { usePipeline } from '../context/PipelineContext'
import { usePoll } from '../hooks/usePoll'
import { API } from '../lib/api'
import { AlertCircle, ChevronRight } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

function StatCard({ label, value, sub, accent }) {
  return (
    <div className="rounded-lg border border-border bg-card p-4 space-y-1">
      <p className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: '#64748B' }}>{label}</p>
      <p className="text-[28px] font-bold leading-tight" style={{ color: accent ? '#A50044' : '#FFFFFF' }}>{value ?? '—'}</p>
      {sub && <p className="text-[11px]" style={{ color: '#475569' }}>{sub}</p>}
    </div>
  )
}

export default function Stage5BestModel() {
  const { onStage5Done, stage1Result, stage4Result, stage4League } = usePipeline()
  const { polling, progress, message, error, poll, reset } = usePoll()

  const best   = stage4Result?.best_model
  const models = stage4Result?.models || []
  const match  = models.find(m => m.name === best) || models[0]

  const [modelPath, setModelPath]     = useState('')
  const [scalerPath, setScalerPath]   = useState('')
  const [featuresPath, setFeaturesPath] = useState('')
  const [targetCol] = useState('bypasses_per_halftime')
  const [result, setResult] = useState(null)

  useEffect(() => {
    if (match?.model_path)          setModelPath(match.model_path)
    if (stage4Result?.scaler_path)  setScalerPath(stage4Result.scaler_path)
    // Use the league-specific features path if available
    const lg = stage4League || Object.keys(stage1Result?.features_paths || {})[0]
    setFeaturesPath(lg ? (stage1Result?.features_paths?.[lg] || '') : Object.values(stage1Result?.features_paths || {})[0] || '')
  }, [match, stage4Result, stage1Result, stage4League])

  // Stage 4 returns the scout features actually used in the per-team model
  const selectedFeatures = stage4Result?.available_features || []

  async function run() {
    if (!modelPath || !scalerPath || !featuresPath) return alert('Model, scaler and features paths required.')
    if (!selectedFeatures.length) return alert('Complete Stage 4 (Model Training) first.')
    reset()
    try {
      const r = await fetch(`${API}/stage5/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_path: modelPath, scaler_path: scalerPath, features_path: featuresPath, selected_features: selectedFeatures, target_col: targetCol }),
      })
      const { job_id } = await r.json()
      poll(`${API}/stage5/status/${job_id}`, res => {
        setResult(res)
        onStage5Done(res)
      })
    } catch (e) {
      alert(`Failed: ${e}`)
    }
  }

  const coefs = result?.coefficients || []
  const coefChartData = coefs.slice(0, 10).map(c => ({ feature: c.feature, value: c.coefficient }))

  const stats = [
    { label: 'Model',          value: result?.model_name ?? best ?? '—',          sub: 'selected model' },
    { label: 'Features Used',  value: coefs.length || selectedFeatures.length || '—', sub: 'feature coefficients' },
    { label: 'Top Feature',    value: coefs[0]?.feature?.split('_')[0] ?? '—',    sub: coefs[0]?.feature ?? '', accent: true },
    { label: 'Gradient Sens.', value: result?.gradient_sensitivity?.length ?? '—', sub: 'sensitivity scores' },
  ]

  return (
    <div className="space-y-5 pt-6 border-t border-border">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-[22px] font-bold text-white">Best Model Analysis</h1>
          <div className="flex items-center gap-1.5 mt-0.5">
            <span className="text-xs font-semibold" style={{ color: '#A50044' }}>Stage 5</span>
            <span className="text-xs" style={{ color: '#475569' }}>·</span>
            <span className="text-xs" style={{ color: '#64748B' }}>Feature coefficients and gradient sensitivity from the best linear model</span>
          </div>
        </div>
        <button
          onClick={run}
          disabled={polling}
          className="flex items-center gap-2 text-white disabled:opacity-50 transition-opacity hover:opacity-90"
          style={{ background: '#A50044', borderRadius: 8, padding: '8px 16px', fontSize: 13, fontWeight: 600, border: 'none', cursor: polling ? 'not-allowed' : 'pointer' }}
        >
          {polling ? 'Running…' : 'Analyse Model'} <ChevronRight className="h-3.5 w-3.5" />
        </button>
      </div>

      {/* Divider */}
      <div style={{ height: 1, background: '#1A1A1A' }} />

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4">
        {stats.map(s => <StatCard key={s.label} {...s} />)}
      </div>

      {/* Progress */}
      {polling && (
        <div className="rounded-lg border border-border bg-card p-5 space-y-2">
          <div className="flex justify-between text-xs">
            <span className="text-muted-foreground">{message}</span>
            <span className="font-mono" style={{ color: '#A50044' }}>{progress}%</span>
          </div>
          <div className="h-1.5 rounded-full bg-secondary overflow-hidden">
            <div className="h-full rounded-full transition-all" style={{ width: `${progress}%`, background: '#A50044' }} />
          </div>
        </div>
      )}

      {error && (
        <div className="flex items-start gap-2 rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm text-red-400">
          <AlertCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />{error}
        </div>
      )}

      {result && coefChartData.length > 0 && (
        <div className="grid grid-cols-2 gap-4">
          {/* Coefficient chart */}
          <div className="rounded-lg border border-border bg-card p-5 space-y-4">
            <h2 className="text-sm font-semibold text-white">Feature Importance (Lasso Coefficients)</h2>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={coefChartData} layout="vertical" margin={{ top: 4, right: 40, left: 8, bottom: 4 }}>
                <XAxis type="number" tick={{ fill: '#64748B', fontSize: 10 }} tickLine={false} axisLine={false} />
                <YAxis type="category" dataKey="feature" tick={{ fill: '#64748B', fontSize: 10 }} tickLine={false} axisLine={false} width={130} />
                <Tooltip
                  contentStyle={{ background: '#111111', border: '1px solid #1A1A1A', borderRadius: 8, fontSize: 12 }}
                  formatter={v => v.toFixed(4)}
                />
                <Bar dataKey="value" radius={[0, 2, 2, 0]}>
                  {coefChartData.map((e, i) => (
                    <Cell key={i} fill={e.value >= 0 ? '#A50044' : '#22C55E'} opacity={0.85} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Coefficients table */}
          <div className="rounded-lg border border-border bg-card p-5 space-y-4">
            <h2 className="text-sm font-semibold text-white">Feature Groups</h2>
            <div className="overflow-y-auto max-h-72 space-y-1">
              {coefs.map((c) => (
                <div key={c.feature} className="flex items-center justify-between py-2 border-b border-border/50 last:border-0">
                  <span className="text-xs font-mono truncate pr-4" style={{ color: '#64748B' }}>{c.feature}</span>
                  <div className="flex items-center gap-4 flex-shrink-0">
                    <span className="text-xs font-mono" style={{ color: c.coefficient >= 0 ? '#A50044' : '#22C55E' }}>
                      {c.coefficient > 0 ? '+' : ''}{parseFloat(c.coefficient).toFixed(3)}
                    </span>
                    <span className="text-xs w-12 text-right" style={{ color: '#64748B' }}>{c.relative_importance}%</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
