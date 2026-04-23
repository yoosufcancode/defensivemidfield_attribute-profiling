import { usePipeline } from '../context/PipelineContext'
import { usePoll } from '../hooks/usePoll'
import { API } from '../lib/api'
import { useState } from 'react'
import { AlertCircle, ChevronRight } from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Cell,
} from 'recharts'

function StatCard({ label, value, sub, accent }) {
  return (
    <div className="rounded-lg border border-border bg-card p-4 space-y-1">
      <p className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: '#64748B' }}>{label}</p>
      <p className="text-[28px] font-bold leading-tight" style={{ color: accent ? '#A50044' : '#FFFFFF' }}>{value ?? '—'}</p>
      {sub && <p className="text-[11px]" style={{ color: '#475569' }}>{sub}</p>}
    </div>
  )
}

const TOOLTIP_STYLE = {
  contentStyle: { background: '#111111', border: '1px solid #1A1A1A', borderRadius: 8, fontSize: 12 },
  labelStyle: { color: '#FFFFFF' },
  itemStyle: { color: '#A50044' },
}

export default function Stage2Eda() {
  const { onStage2Done, stage1Result } = usePipeline()
  const { polling, progress, message, error, poll, reset } = usePoll()
  const [result, setResult] = useState(null)

  const featuresPath = Object.values(stage1Result?.features_paths || {})[0] || null

  async function run() {
    if (!featuresPath) return alert('Complete Stage 1 first.')
    reset()
    try {
      const r = await fetch(`${API}/stage2/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features_path: featuresPath }),
      })
      const { job_id } = await r.json()
      poll(`${API}/stage2/status/${job_id}`, res => {
        setResult(res)
        onStage2Done(res)
      })
    } catch (e) {
      alert(`Failed: ${e}`)
    }
  }

  const corrMatrix = result?.correlation_matrix || {}
  const corrCols = Object.keys(corrMatrix)

  // All features sorted by |r| with target — top 15 for the ranking chart
  const allCorrelations = corrCols
    .map(col => ({
      feature: col,
      r: parseFloat(Math.abs((corrMatrix[col] || {})['bypasses_per_halftime'] ?? 0).toFixed(4)),
      raw: parseFloat(((corrMatrix[col] || {})['bypasses_per_halftime'] ?? 0).toFixed(4)),
    }))
    .filter(x => x.r > 0.01)
    .sort((a, b) => b.r - a.r)
    .slice(0, 15)

  const distData = result?.bypass_distribution?.counts
    ? result.bypass_distribution.bin_edges.slice(0, -1).map((edge, i) => ({
        label: edge.toFixed(1),
        count: result.bypass_distribution.counts[i],
      }))
    : []

  const distMean   = result?.bypass_distribution?.mean
  const distMedian = result?.bypass_distribution?.median

  // Descriptive stats for top 6 correlated features
  const topFeatureStats = allCorrelations.slice(0, 6).map(({ feature, r }) => ({
    feature,
    r,
    mean: result?.descriptive_stats?.mean?.[feature] != null
      ? Number(result.descriptive_stats.mean[feature]).toFixed(3) : '—',
    std: result?.descriptive_stats?.std?.[feature] != null
      ? Number(result.descriptive_stats.std[feature]).toFixed(3) : '—',
    max: result?.descriptive_stats?.max?.[feature] != null
      ? Number(result.descriptive_stats.max[feature]).toFixed(3) : '—',
  }))

  const missingCount = Object.keys(result?.missing_values || {}).length

  const stats = [
    { label: 'Total Rows',         value: result?.row_count?.toLocaleString() ?? '—', sub: 'half-match records' },
    { label: 'Features Analysed', value: result?.column_count ?? '—',              sub: 'engineered features' },
    { label: 'Missing Columns',   value: missingCount || '—',                       sub: missingCount ? 'cols w/ nulls' : 'no missing cols' },
    { label: 'Top |r| w/ Target', value: allCorrelations[0]?.r.toFixed(3) ?? '—',  sub: allCorrelations[0]?.feature ?? '', accent: true },
  ]

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-[22px] font-bold text-white">Exploratory Data Analysis</h1>
          <div className="flex items-center gap-1.5 mt-0.5">
            <span className="text-xs font-semibold" style={{ color: '#A50044' }}>Stage 2</span>
            <span className="text-xs" style={{ color: '#475569' }}>·</span>
            <span className="text-xs" style={{ color: '#64748B' }}>Feature distributions and correlations</span>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {result && (
            <div
              className="flex items-center gap-2 text-xs font-semibold"
              style={{ background: '#1A1A1A', borderRadius: 6, padding: '6px 14px', color: '#22C55E' }}
            >
              Analysis Complete
            </div>
          )}
          <button
            onClick={run}
            disabled={polling}
            className="flex items-center gap-2 text-white disabled:opacity-50 transition-opacity hover:opacity-90"
            style={{ background: '#A50044', borderRadius: 8, padding: '8px 16px', fontSize: 13, fontWeight: 600, border: 'none', cursor: polling ? 'not-allowed' : 'pointer' }}
          >
            {polling ? 'Running…' : 'Run EDA'} <ChevronRight className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

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
            <span className="font-mono text-xs" style={{ color: '#A50044' }}>{progress}%</span>
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

      {result && (
        <>
          {/* Row 1: Distribution + summary stats */}
          <div className="grid grid-cols-5 gap-4">
            {/* Bypass distribution */}
            <div className="col-span-3 rounded-lg border border-border bg-card p-5 space-y-3">
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-semibold text-white">Bypass Distribution</h2>
                <div className="flex items-center gap-4 text-xs font-mono" style={{ color: '#64748B' }}>
                  {distMean != null && <span>mean <strong style={{ color: '#A50044' }}>{distMean.toFixed(2)}</strong></span>}
                  {distMedian != null && <span>median <strong style={{ color: '#94A3B8' }}>{distMedian.toFixed(2)}</strong></span>}
                </div>
              </div>
              {distData.length > 0 ? (
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={distData} margin={{ top: 4, right: 4, left: -20, bottom: 4 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1A1A1A" />
                    <XAxis dataKey="label" tick={{ fill: '#64748B', fontSize: 10 }} tickLine={false} />
                    <YAxis tick={{ fill: '#64748B', fontSize: 10 }} tickLine={false} axisLine={false} />
                    <Tooltip {...TOOLTIP_STYLE} formatter={(v) => [v, 'Count']} />
                    {distMean != null && (
                      <ReferenceLine x={distMean.toFixed(1)} stroke="#A50044" strokeDasharray="4 2" strokeWidth={1.5} />
                    )}
                    <Bar dataKey="count" fill="#A50044" opacity={0.85} radius={[2, 2, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-sm text-muted-foreground">No distribution data available.</p>
              )}
              <p className="text-[10px]" style={{ color: '#2D3748' }}>Dashed line = mean · bypasses per half-match</p>
            </div>

            {/* Top correlations list */}
            <div className="col-span-2 rounded-lg border border-border bg-card p-5 space-y-3">
              <h2 className="text-sm font-semibold text-white">Top Correlations with Target</h2>
              <div className="space-y-2.5 max-h-56 overflow-y-auto pr-1">
                {allCorrelations.slice(0, 10).map(({ feature, r, raw }) => (
                  <div key={feature} className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="truncate pr-2 font-mono" style={{ color: '#64748B', fontSize: 10 }}>{feature}</span>
                      <span className="font-mono flex-shrink-0 text-xs" style={{ color: raw >= 0 ? '#A50044' : '#64748B' }}>
                        {raw >= 0 ? '+' : ''}{raw.toFixed(3)}
                      </span>
                    </div>
                    <div className="h-1 rounded-full bg-secondary overflow-hidden">
                      <div
                        className="h-full rounded-full"
                        style={{ width: `${(r * 100).toFixed(0)}%`, background: raw >= 0 ? '#A50044' : '#475569' }}
                      />
                    </div>
                  </div>
                ))}
                {!allCorrelations.length && (
                  <p className="text-xs text-muted-foreground">No correlation data available.</p>
                )}
              </div>
            </div>
          </div>

          {/* Row 2: Horizontal correlation ranking chart */}
          {allCorrelations.length > 0 && (
            <div className="rounded-lg border border-border bg-card p-5 space-y-3">
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-semibold text-white">Feature Correlation Ranking</h2>
                <span className="text-xs" style={{ color: '#64748B' }}>|r| with bypasses_per_halftime · top {allCorrelations.length}</span>
              </div>
              <ResponsiveContainer width="100%" height={Math.max(240, allCorrelations.length * 22)}>
                <BarChart layout="vertical" data={allCorrelations} margin={{ top: 4, right: 24, left: 8, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1A1A1A" vertical={true} horizontal={false} />
                  <XAxis
                    type="number"
                    domain={[0, Math.ceil(allCorrelations[0]?.r * 10) / 10 + 0.05]}
                    tick={{ fill: '#64748B', fontSize: 10 }}
                    tickLine={false}
                    tickFormatter={v => v.toFixed(2)}
                  />
                  <YAxis
                    type="category"
                    dataKey="feature"
                    width={210}
                    tick={{ fill: '#94A3B8', fontSize: 10, fontFamily: 'monospace' }}
                    tickLine={false}
                    axisLine={false}
                  />
                  <Tooltip
                    {...TOOLTIP_STYLE}
                    formatter={(v, _, p) => [`|r| = ${v.toFixed(4)}  (raw: ${p.payload.raw >= 0 ? '+' : ''}${p.payload.raw.toFixed(4)})`, p.payload.feature]}
                  />
                  <Bar dataKey="r" radius={[0, 3, 3, 0]}>
                    {allCorrelations.map((entry, i) => (
                      <Cell key={entry.feature} fill={i === 0 ? '#A50044' : `rgba(165,0,68,${0.9 - i * 0.05})`} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Row 3: Descriptive stats table */}
          {topFeatureStats.length > 0 && (
            <div className="rounded-lg border border-border bg-card p-5 space-y-3">
              <h2 className="text-sm font-semibold text-white">Feature Summary — Top 6 by Correlation</h2>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr style={{ borderBottom: '1px solid #1A1A1A' }}>
                      {['Feature', 'Mean', 'Std Dev', 'Max', '|r| w/ Target'].map(h => (
                        <th key={h} className="text-left pb-2 pr-6 text-[10px] font-semibold uppercase tracking-widest" style={{ color: '#64748B' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {topFeatureStats.map((s, i) => (
                      <tr key={s.feature} style={{ borderBottom: '1px solid #1A1A1A' }}>
                        <td className="py-2.5 pr-6">
                          <span className="text-xs font-mono" style={{ color: i === 0 ? '#FFFFFF' : '#94A3B8' }}>{s.feature}</span>
                        </td>
                        <td className="py-2.5 pr-6 text-xs font-mono" style={{ color: '#64748B' }}>{s.mean}</td>
                        <td className="py-2.5 pr-6 text-xs font-mono" style={{ color: '#64748B' }}>{s.std}</td>
                        <td className="py-2.5 pr-6 text-xs font-mono" style={{ color: '#64748B' }}>{s.max}</td>
                        <td className="py-2.5 text-xs font-mono font-semibold" style={{ color: '#A50044' }}>{s.r.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Missing data indicator */}
          {missingCount > 0 && (
            <div className="rounded-lg border border-border bg-card p-5 space-y-3">
              <h2 className="text-sm font-semibold text-white">Columns with Missing Values</h2>
              <div className="flex flex-wrap gap-2">
                {Object.entries(result.missing_values).map(([col, count]) => (
                  <div
                    key={col}
                    className="flex items-center gap-2 rounded px-2.5 py-1.5"
                    style={{ background: '#1A1A1A', border: '1px solid #2D3748' }}
                  >
                    <span className="text-xs font-mono" style={{ color: '#94A3B8' }}>{col}</span>
                    <span className="text-xs font-mono" style={{ color: '#475569' }}>{count}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
