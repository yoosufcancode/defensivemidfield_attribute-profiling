import { useState, useEffect } from 'react'
import { usePipeline } from '../context/PipelineContext'
import { usePoll } from '../hooks/usePoll'
import { API } from '../lib/api'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Select, SelectGroup, SelectLabel, SelectTrigger, SelectValue, SelectContent, SelectItem } from './ui/select'
import { AlertCircle, ChevronRight } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

function StatCard({ label, value, sub, accent }) {
  return (
    <div className="rounded-lg border border-border bg-card p-4 space-y-1">
      <p className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: '#64748B' }}>{label}</p>
      <p className="text-[28px] font-bold leading-tight" style={{ color: accent ? '#A50044' : '#FFFFFF' }}>{value ?? '—'}</p>
      {sub && <p className="text-[11px]" style={{ color: '#475569' }}>{sub}</p>}
    </div>
  )
}

export default function Stage4Models() {
  const {
    onStage1Done, onStage3Done, onStage4Done,
    stage1Result, stage3Result,
  } = usePipeline()
  const { polling, progress, message, error, setError, poll, reset } = usePoll()

  const [team, setTeam] = useState('')
  const [teamsByLeague, setTeamsByLeague] = useState({})
  const [target, setTarget] = useState('bypasses_per_halftime')
  const [testSize, setTestSize] = useState(0.15)
  const [seed, setSeed] = useState(42)
  const [result, setResult] = useState(null)
  const [submitting, setSubmitting] = useState(false)
  const [done, setDone] = useState(false)

  const league = Object.entries(teamsByLeague).find(([, ts]) => ts.includes(team))?.[0] || ''
  const selectedFeatures = stage3Result?.selected_features || []
  const featuresPath = league ? (stage1Result?.features_paths?.[league] || null) : null

  // Load teams from existing CSVs
  useEffect(() => {
    fetch(`${API}/stage1/teams`)
      .then(r => r.json())
      .then(d => setTeamsByLeague(d.teams_by_league || {}))
      .catch(() => {})
  }, [])

  // Pre-fill team from Stage 3 selection
  useEffect(() => {
    if (stage3Result?.team && !team) setTeam(stage3Result.team)
  }, [stage3Result?.team]) // eslint-disable-line react-hooks/exhaustive-deps

  async function run() {
    reset()
    setDone(false)
    setSubmitting(true)

    try {
      let path = featuresPath
      let features = selectedFeatures
      const lg = league
      const tm = team

      if (!path || !features.length) {
        const stateRes = await fetch(`${API}/pipeline/state`)
        if (!stateRes.ok) throw new Error('Could not reach the pipeline API.')
        const state = await stateRes.json()
        if (!path) path = lg ? (state.features_paths?.[lg] || null) : Object.values(state.features_paths || {})[0] || null
        if (!features.length) features = state.selected_features || []
      }

      if (!tm)   throw new Error('Select a team first.')
      if (!lg)   throw new Error('Could not determine league for selected team.')
      if (!path) throw new Error('Stage 1 (Data Ingestion) is not complete for this league.')
      if (!features.length) throw new Error('Stage 3 (Feature Selection) is not complete — run selection and confirm first.')

      const r = await fetch(`${API}/stage4/build`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          features_path:     path,
          league:            lg,
          team:              tm,
          selected_features: features,
          target_col:        target,
          test_size:         testSize,
          random_state:      seed,
        }),
      })
      if (!r.ok) {
        const text = await r.text()
        throw new Error(`Server returned ${r.status}: ${text}`)
      }
      const { job_id } = await r.json()
      setSubmitting(false)
      poll(`${API}/stage4/status/${job_id}`, res => {
        setResult(res)
        setDone(true)
        onStage4Done(res)
      })
    } catch (e) {
      setSubmitting(false)
      setError(String(e))
    }
  }

  // Hydrate context from disk if cold-loaded
  useEffect(() => {
    const hasStage1 = Object.keys(stage1Result?.features_paths || {}).length > 0
    const hasStage3 = (stage3Result?.selected_features?.length ?? 0) > 0
    if (hasStage1 && hasStage3) return
    fetch(`${API}/pipeline/state`)
      .then(r => r.json())
      .then(data => {
        if (data.stages?.['1'] && !hasStage1) {
          onStage1Done({
            features_paths:    data.features_paths    || {},
            row_counts:        data.row_counts        || {},
            leagues_processed: data.leagues_processed || [],
          })
        }
        if (data.stages?.['3'] && !hasStage3) {
          onStage3Done({ selected_features: data.selected_features || [] })
        }
      })
      .catch(() => {})
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const bestModel  = result?.models?.find(m => m.name === result.best_model)
  // Chart uses LOOCV Spearman (primary selection metric) and test Spearman
  const chartData  = result?.models?.map(m => ({
    name: m.name,
    'LOOCV ρ': parseFloat(m.loocv?.spearman) || 0,
    'Test ρ':  parseFloat(m.test?.spearman)  || 0,
  })) ?? []

  const scoutingFeatures = result?.scouting_features || []
  const signStableCount  = scoutingFeatures.filter(f => f.sign_stable).length
  const pSigCount        = scoutingFeatures.filter(f => f.p_value < 0.05).length
  const pModCount        = scoutingFeatures.filter(f => f.p_value >= 0.05 && f.p_value < 0.10).length

  const stats = [
    { label: 'Models Trained',   value: result?.models?.length ?? '—',     sub: 'Ridge, Lasso, MLR' },
    { label: 'Best Model',       value: result?.best_model ?? '—',          sub: 'by LOOCV Spearman ρ' },
    { label: 'LOOCV Spearman ρ', value: bestModel ? parseFloat(bestModel.loocv.spearman).toFixed(3) : '—', sub: 'training-set LOOCV', accent: true },
    { label: 'Test Spearman ρ',  value: bestModel ? parseFloat(bestModel.test.spearman).toFixed(3)  : '—', sub: 'held-out test set' },
  ]

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-[22px] font-bold text-white">Per-Team Model Training</h1>
          <div className="flex items-center gap-1.5 mt-0.5">
            <span className="text-xs font-semibold" style={{ color: '#A50044' }}>Stage 4</span>
            <span className="text-xs" style={{ color: '#475569' }}>·</span>
            <span className="text-xs" style={{ color: '#64748B' }}>Train MLR, Ridge, Lasso on team data — select best by LOOCV Spearman ρ, extract scouting gradients</span>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {result && (
            <div
              className="flex items-center gap-2 text-xs font-semibold"
              style={{ background: '#1A1A1A', borderRadius: 6, padding: '6px 14px', color: '#22C55E' }}
            >
              Model ready — Stage 6 will use these gradients →
            </div>
          )}
          <button
            onClick={run}
            disabled={polling || submitting || !team}
            className="flex items-center gap-2 text-white disabled:opacity-50 transition-opacity hover:opacity-90"
            style={{ background: '#A50044', borderRadius: 8, padding: '8px 16px', fontSize: 13, fontWeight: 600, border: 'none', cursor: (polling || submitting || !team) ? 'not-allowed' : 'pointer' }}
          >
            <ChevronRight className="h-3.5 w-3.5" />
            {submitting ? 'Submitting…' : polling ? 'Running…' : 'Train Models'}
          </button>
        </div>
      </div>

      <div style={{ height: 1, background: '#1A1A1A' }} />

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4">
        {stats.map(s => <StatCard key={s.label} {...s} />)}
      </div>

      {/* Post-model filter summary */}
      {result && (
        <div className="rounded-lg border border-border bg-card px-5 py-3 flex items-center gap-6 flex-wrap">
          <span className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: '#64748B' }}>Post-model filters</span>
          <div className="flex items-center gap-1.5">
            <span className="text-[11px] font-mono font-bold" style={{ color: '#22C55E' }}>{signStableCount}</span>
            <span className="text-[11px]" style={{ color: '#475569' }}>sign-stable across 5-fold CV</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-[11px] font-mono font-bold" style={{ color: '#22C55E' }}>{pSigCount}</span>
            <span className="text-[11px]" style={{ color: '#475569' }}>OLS p &lt; 0.05 (high confidence)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-[11px] font-mono font-bold" style={{ color: '#F59E0B' }}>{pModCount}</span>
            <span className="text-[11px]" style={{ color: '#475569' }}>OLS p 0.05–0.10 (moderate)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-[11px] font-mono font-bold" style={{ color: '#FFFFFF' }}>{scoutingFeatures.length}</span>
            <span className="text-[11px]" style={{ color: '#475569' }}>final scouting features (sign-stable + p &lt; 0.15)</span>
          </div>
        </div>
      )}

      {/* Config */}
      <div className="rounded-lg border border-border bg-card p-5 flex items-end gap-6 flex-wrap">
        {/* Team */}
        <div className="space-y-1.5 flex-1 min-w-44">
          <Label className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: '#64748B' }}>Team</Label>
          <Select value={team} onValueChange={setTeam}>
            <SelectTrigger className="bg-secondary/50 border-border h-9 text-sm">
              <SelectValue placeholder="— select team —" />
            </SelectTrigger>
            <SelectContent>
              {Object.entries(teamsByLeague).map(([lg, ts]) => (
                <SelectGroup key={lg}>
                  <SelectLabel>{lg}</SelectLabel>
                  {ts.map(t => <SelectItem key={t} value={t}>{t}</SelectItem>)}
                </SelectGroup>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Target col */}
        <div className="space-y-1.5 flex-1 max-w-xs">
          <Label className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: '#64748B' }}>Target column</Label>
          <Input value={target} onChange={e => setTarget(e.target.value)} className="bg-secondary/50 border-border h-9 text-sm" />
        </div>

        <div className="space-y-1.5 w-28">
          <Label className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: '#64748B' }}>Test size</Label>
          <Input type="number" value={testSize} step={0.05} min={0.05} max={0.4} onChange={e => setTestSize(+e.target.value)} className="bg-secondary/50 border-border h-9 text-sm" />
        </div>
        <div className="space-y-1.5 w-28">
          <Label className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: '#64748B' }}>Random state</Label>
          <Input type="number" value={seed} onChange={e => setSeed(+e.target.value)} className="bg-secondary/50 border-border h-9 text-sm" />
        </div>
      </div>

      {/* Progress */}
      {(submitting || polling) && (
        <div className="rounded-lg border border-border bg-card p-5 space-y-3">
          <div className="flex justify-between items-center text-xs">
            <div className="flex items-center gap-2">
              <div className="h-1.5 w-1.5 rounded-full animate-pulse" style={{ background: '#A50044' }} />
              <span className="text-muted-foreground">
                {submitting ? 'Submitting job…' : message || 'Starting…'}
              </span>
            </div>
            <span className="font-mono tabular-nums" style={{ color: '#A50044' }}>
              {submitting ? '—' : `${progress}%`}
            </span>
          </div>
          <div className="relative h-2 rounded-full overflow-hidden" style={{ background: '#1A1A1A' }}>
            <div
              className="absolute inset-y-0 left-0 rounded-full transition-all duration-700"
              style={{ width: submitting ? '4%' : `${progress}%`, background: '#A50044' }}
            />
            <div
              className="absolute inset-0 rounded-full"
              style={{
                background: 'linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.12) 50%, transparent 100%)',
                animation: 'shimmer 1.6s linear infinite',
              }}
            />
          </div>
          {polling && progress < 90 && (
            <p className="text-[10px]" style={{ color: '#475569' }}>
              Per-team LOOCV and scouting gradient extraction — this can take 2–4 minutes.
            </p>
          )}
        </div>
      )}

      {done && !polling && result && (
        <div className="rounded-lg border p-4 flex items-center justify-between" style={{ borderColor: 'rgba(34,197,94,0.35)', background: 'rgba(34,197,94,0.07)' }}>
          <div className="flex items-center gap-3">
            <div className="h-2 w-2 rounded-full" style={{ background: '#22C55E' }} />
            <span className="text-sm font-semibold" style={{ color: '#22C55E' }}>Training complete</span>
            <span className="text-xs" style={{ color: '#64748B' }}>
              Best model: <strong style={{ color: '#FFFFFF' }}>{result.best_model}</strong>
              {' '}· Team: <strong style={{ color: '#FFFFFF' }}>{result.team}</strong>
              {' '}· {result.scouting_features?.length ?? 0} scouting features
            </span>
          </div>
          <span className="text-xs font-mono" style={{ color: '#475569' }}>{result.feature_count} features · 3 models</span>
        </div>
      )}

      {error && (
        <div className="flex items-start gap-2 rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm text-red-400">
          <AlertCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />{error}
        </div>
      )}

      {result && (
        <div className="grid grid-cols-5 gap-4">
          {/* Model comparison table */}
          <div className="col-span-3 rounded-lg border border-border bg-card p-5 space-y-4">
            <h2 className="text-sm font-semibold text-white">Model Comparison</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border">
                    {['Model', 'LOOCV ρ', 'Test ρ', 'R²', 'RMSE', 'Status'].map(h => (
                      <th key={h} className="text-[10px] font-semibold uppercase tracking-wider text-left py-2 pr-4 last:text-center" style={{ color: '#64748B' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/50">
                  {result.models?.map(m => {
                    const isBest = m.name === result.best_model
                    return (
                      <tr key={m.name} style={{ background: isBest ? 'rgba(165,0,68,0.05)' : 'transparent' }}>
                        <td className="py-3 pr-4 font-medium" style={{ color: isBest ? '#A50044' : '#FFFFFF' }}>{m.name}</td>
                        <td className="py-3 pr-4 font-mono text-xs" style={{ color: isBest ? '#A50044' : '#64748B' }}>{parseFloat(m.loocv.spearman).toFixed(3)}</td>
                        <td className="py-3 pr-4 font-mono text-xs" style={{ color: '#64748B' }}>{parseFloat(m.test.spearman).toFixed(3)}</td>
                        <td className="py-3 pr-4 font-mono text-xs" style={{ color: '#64748B' }}>{parseFloat(m.test.r2).toFixed(3)}</td>
                        <td className="py-3 pr-4 font-mono text-xs" style={{ color: '#64748B' }}>{parseFloat(m.test.rmse).toFixed(2)}</td>
                        <td className="py-3 text-center">
                          {isBest
                            ? <span className="text-xs font-semibold text-white rounded px-2 py-0.5" style={{ background: '#A50044' }}>Best</span>
                            : <span className="text-xs" style={{ color: '#64748B' }}>—</span>
                          }
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>

            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={chartData} margin={{ top: 4, right: 4, left: -20, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1A1A1A" />
                <XAxis dataKey="name" tick={{ fill: '#64748B', fontSize: 11 }} tickLine={false} />
                <YAxis tick={{ fill: '#64748B', fontSize: 10 }} tickLine={false} axisLine={false} domain={[-0.2, 1]} />
                <Tooltip contentStyle={{ background: '#111111', border: '1px solid #1A1A1A', borderRadius: 8, fontSize: 12 }} />
                <Legend wrapperStyle={{ color: '#64748B', fontSize: 12 }} />
                <Bar dataKey="LOOCV ρ" fill="#A50044" opacity={0.85} radius={[2, 2, 0, 0]} />
                <Bar dataKey="Test ρ"  fill="#22C55E" opacity={0.7}  radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Scouting gradients */}
          <div className="col-span-2 rounded-lg border border-border bg-card p-5 space-y-4">
            <h2 className="text-sm font-semibold text-white">Scouting Dimensions</h2>
            <div className="overflow-y-auto max-h-80 space-y-1">
              {(result.scouting_features || []).map(sf => {
                const tierColor = sf.p_value < 0.05 ? '#22C55E' : sf.p_value < 0.10 ? '#F59E0B' : '#94A3B8'
                return (
                  <div
                    key={sf.feature}
                    className="flex items-center justify-between p-2.5 rounded-lg"
                    style={{ background: '#111111', border: '1px solid #1A1A1A' }}
                  >
                    <div className="min-w-0 flex-1">
                      <p className="text-xs font-mono text-white truncate">{sf.feature}</p>
                      <div className="flex items-center gap-2 mt-0.5">
                        <span
                          className="text-[10px] font-semibold"
                          style={{ color: sf.direction === 'look for LOW' ? '#22C55E' : '#A50044' }}
                        >
                          {sf.direction === 'look for LOW' ? '↓ lower' : '↑ higher'}
                        </span>
                        <span
                          className="text-[9px] font-semibold rounded px-1 py-px"
                          style={{ background: sf.sign_stable ? 'rgba(34,197,94,0.12)' : 'rgba(239,68,68,0.12)', color: sf.sign_stable ? '#22C55E' : '#EF4444' }}
                        >
                          {sf.sign_stable ? 'stable' : 'flips'}
                        </span>
                      </div>
                    </div>
                    <div className="text-right flex-shrink-0 ml-2">
                      <p className="text-xs font-mono font-semibold" style={{ color: tierColor }}>
                        p={sf.p_value < 0.001 ? '<0.001' : sf.p_value?.toFixed(3)}
                      </p>
                      <p className="text-[10px]" style={{ color: tierColor }}>{sf.confidence_tier?.split('—')[0].trim()}</p>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
