import { useState, useEffect } from 'react'
import { usePipeline } from '../context/PipelineContext'
import { usePoll } from '../hooks/usePoll'
import { API } from '../lib/api'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from './ui/select'
import { AlertCircle, ChevronRight, CheckCircle2, XCircle } from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
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

const METHODS = [
  { key: 'f_reg', label: 'F-Reg' },
  { key: 'mi',    label: 'MI'    },
  { key: 'rf',    label: 'RF'    },
  { key: 'rfe',   label: 'RFE'   },
]

export default function Stage3Features({ onConfirmed }) {
  const { onStage3Done, stage1Result } = usePipeline()
  const { polling, progress, message, error, poll, reset } = usePoll()

  const [team, setTeam] = useState('')
  const [teamsByLeague, setTeamsByLeague] = useState({})
  const [target, setTarget] = useState('bypasses_per_halftime')
  const [nTop, setNTop] = useState(15)
  const [result, setResult] = useState(null)

  const teamToLeague = Object.fromEntries(
    Object.entries(teamsByLeague).flatMap(([lg, ts]) => ts.map(t => [t, lg]))
  )
  const allTeams = Object.keys(teamToLeague).sort()
  const league = teamToLeague[team] || ''
  const featuresPath = league ? (stage1Result?.features_paths?.[league] || null) : null

  useEffect(() => {
    fetch(`${API}/stage1/teams`)
      .then(r => r.json())
      .then(d => setTeamsByLeague(d.teams_by_league || {}))
      .catch(() => {})
  }, [])

  async function run() {
    if (!team)   return alert('Select a team first.')
    if (!featuresPath) return alert('Stage 1 ingestion not complete for this league.')
    reset()
    try {
      const r = await fetch(`${API}/stage3/select`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          features_path: featuresPath,
          team,
          target_col: target,
          n_top: nTop,
        }),
      })
      const { job_id } = await r.json()
      poll(`${API}/stage3/status/${job_id}`, res => setResult(res))
    } catch (e) {
      alert(`Failed: ${e}`)
    }
  }

  function removeFeature(f) {
    setResult(prev => ({ ...prev, selected_features: prev.selected_features.filter(x => x !== f) }))
  }

  function confirm() {
    if (!result) return
    onStage3Done({ ...result, league, team })
    onConfirmed?.()
  }

  const selected = result?.selected_features || []
  const dropped = result?.consensus
    ? result.consensus.filter(f => !selected.includes(f.feature)).slice(0, 16)
    : []

  const totalInitial = result ? (selected.length + dropped.length) : null

  const methodAgreement = (result?.consensus || []).map(item => {
    const f = item.feature
    const isSelected = selected.includes(f)
    return {
      feature: f,
      score: item.avg_rank,
      selected: isSelected,
      f_reg: result.univariate?.some(x => x.feature === f) ? 1 : 0,
      mi:    result.mutual_info?.some(x => x.feature === f) ? 1 : 0,
      rf:    result.random_forest?.some(x => x.feature === f) ? 1 : 0,
      rfe:   result.rfe?.some(x => x.feature === f && x.score > 0.5) ? 1 : 0,
    }
  })

  const consensusChartData = methodAgreement.map(d => ({ ...d, displayScore: parseFloat((d.score * 100).toFixed(2)) }))

  const stats = [
    { label: 'Match Rows',    value: result?.n_match_rows ?? '—',  sub: `${team || 'team'} full matches` },
    { label: 'Selected',      value: selected.length || '—',       sub: 'consensus features' },
    { label: 'Dropped',       value: dropped.length || '—',        sub: 'low-importance' },
    { label: 'Method',        value: '4-Method',                   sub: 'F-Reg, MI, RF, RFE', accent: true },
  ]

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-[22px] font-bold text-white">Feature Selection</h1>
          <div className="flex items-center gap-1.5 mt-0.5">
            <span className="text-xs font-semibold" style={{ color: '#A50044' }}>Stage 3</span>
            <span className="text-xs" style={{ color: '#475569' }}>·</span>
            <span className="text-xs" style={{ color: '#64748B' }}>Per-team F-regression, Mutual Information, Random Forest and RFE consensus</span>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {result && (
            <div
              className="flex items-center gap-2 text-xs font-semibold"
              style={{ background: '#1A1A1A', borderRadius: 6, padding: '6px 14px', color: '#22C55E' }}
            >
              {selected.length} Features Selected
            </div>
          )}
          <button
            onClick={run}
            disabled={polling || !team}
            className="flex items-center gap-2 text-white disabled:opacity-50 transition-opacity hover:opacity-90"
            style={{ background: '#A50044', borderRadius: 8, padding: '8px 16px', fontSize: 13, fontWeight: 600, border: 'none', cursor: (polling || !team) ? 'not-allowed' : 'pointer' }}
          >
            {polling ? 'Running…' : 'Run Selection'} <ChevronRight className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      <div style={{ height: 1, background: '#1A1A1A' }} />

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4">
        {stats.map(s => <StatCard key={s.label} {...s} />)}
      </div>

      {/* Config */}
      <div className="rounded-lg border border-border bg-card p-5 flex items-end gap-6 flex-wrap">
        {/* Team */}
        <div className="space-y-1.5 flex-1 min-w-44">
          <Label className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: '#64748B' }}>Team</Label>
          <Select value={team} onValueChange={v => { setTeam(v); setResult(null) }}>
            <SelectTrigger className="bg-secondary/50 border-border h-9 text-sm">
              <SelectValue placeholder="— select team —" />
            </SelectTrigger>
            <SelectContent>
              {allTeams.map(t => <SelectItem key={t} value={t}>{t}</SelectItem>)}
            </SelectContent>
          </Select>
        </div>

        {/* Target col */}
        <div className="space-y-1.5 flex-1 max-w-xs">
          <Label className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: '#64748B' }}>Target column</Label>
          <Input value={target} onChange={e => setTarget(e.target.value)} className="bg-secondary/50 border-border h-9 text-sm" />
        </div>

        <div className="space-y-1.5 w-32">
          <Label className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: '#64748B' }}>Top N features</Label>
          <Input type="number" value={nTop} min={3} max={30} onChange={e => setNTop(+e.target.value)} className="bg-secondary/50 border-border h-9 text-sm" />
        </div>
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

      {result && (
        <>
          {/* Consensus ranking + method agreement chart */}
          <div className="rounded-lg border border-border bg-card p-5 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-sm font-semibold text-white">Consensus Ranking</h2>
                <p className="text-[11px] mt-0.5" style={{ color: '#64748B' }}>
                  {team} · {result.n_match_rows} full-match rows · 4 methods · higher = stronger signal
                </p>
              </div>
              <div className="flex items-center gap-3">
                {METHODS.map(m => (
                  <div key={m.key} className="flex items-center gap-1.5">
                    <div className="w-3 h-3 rounded-sm" style={{ background: 'rgba(165,0,68,0.3)', border: '1px solid rgba(165,0,68,0.6)' }} />
                    <span className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: '#64748B' }}>{m.label}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="space-y-1.5">
              {methodAgreement.map((item, i) => (
                <div key={item.feature} className="flex items-center gap-3">
                  <span className="text-[10px] font-mono w-5 text-right flex-shrink-0" style={{ color: '#2D3748' }}>{i + 1}</span>
                  <div className="flex-1 relative h-6 rounded flex items-center overflow-hidden" style={{ background: '#1A1A1A' }}>
                    <div
                      className="h-full rounded transition-all duration-500"
                      style={{
                        width: `${Math.min(100, item.displayScore)}%`,
                        background: item.selected ? '#A50044' : '#374151',
                        opacity: 0.85,
                      }}
                    />
                    <span
                      className="absolute left-2.5 text-[11px] font-mono z-10 truncate"
                      style={{ color: item.selected ? '#FFFFFF' : '#64748B' }}
                    >
                      {item.feature}
                    </span>
                  </div>
                  <span className="text-[10px] font-mono w-10 text-right flex-shrink-0" style={{ color: '#475569' }}>
                    {item.score.toFixed(3)}
                  </span>
                  <div className="flex items-center gap-1 flex-shrink-0">
                    {[item.f_reg, item.mi, item.rf, item.rfe].map((v, mi) => (
                      <div
                        key={mi}
                        className="w-5 h-5 rounded flex items-center justify-center"
                        style={{
                          background: v ? 'rgba(165,0,68,0.15)' : '#111111',
                          border: `1px solid ${v ? 'rgba(165,0,68,0.5)' : '#2D3748'}`,
                        }}
                      >
                        {v ? <span style={{ color: '#A50044', fontSize: 10, lineHeight: 1 }}>✓</span> : null}
                      </div>
                    ))}
                  </div>
                  {item.selected ? (
                    <CheckCircle2 className="h-3.5 w-3.5 flex-shrink-0" style={{ color: '#22C55E' }} />
                  ) : (
                    <XCircle className="h-3.5 w-3.5 flex-shrink-0" style={{ color: 'rgba(239,68,68,0.5)' }} />
                  )}
                </div>
              ))}
            </div>

            {consensusChartData.length > 0 && (
              <div className="pt-2" style={{ borderTop: '1px solid #1A1A1A' }}>
                <p className="text-[10px] font-semibold uppercase tracking-widest mb-3" style={{ color: '#64748B' }}>Score Distribution</p>
                <ResponsiveContainer width="100%" height={120}>
                  <BarChart data={consensusChartData} margin={{ top: 2, right: 8, left: -28, bottom: 2 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1A1A1A" />
                    <XAxis dataKey="feature" tick={false} tickLine={false} axisLine={{ stroke: '#1A1A1A' }} />
                    <YAxis tick={{ fill: '#64748B', fontSize: 9 }} tickLine={false} axisLine={false} />
                    <Tooltip
                      contentStyle={{ background: '#111111', border: '1px solid #1A1A1A', borderRadius: 8, fontSize: 11 }}
                      labelFormatter={(_, payload) => payload?.[0]?.payload?.feature || ''}
                      formatter={(v) => [`${(v / 100).toFixed(3)}`, 'Score']}
                    />
                    <Bar dataKey="displayScore" radius={[2, 2, 0, 0]}>
                      {consensusChartData.map((entry) => (
                        <Cell key={entry.feature} fill={entry.selected ? '#A50044' : '#374151'} opacity={0.9} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          {/* Selected / Dropped */}
          <div className="grid grid-cols-2 gap-4">
            <div className="rounded-lg border border-border bg-card p-5 space-y-3">
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-semibold text-white">Selected Features</h2>
                <span className="text-xs" style={{ color: '#64748B' }}>{selected.length} features</span>
              </div>
              <div className="space-y-1 max-h-72 overflow-y-auto pr-1">
                {selected.map((f) => {
                  const consensus = result.consensus?.find(c => c.feature === f)
                  const methods = methodAgreement.find(m => m.feature === f)
                  const methodCount = methods ? [methods.f_reg, methods.mi, methods.rf, methods.rfe].filter(Boolean).length : 0
                  return (
                    <div key={f} className="flex items-center justify-between py-2 border-b border-border/50 last:border-0 group">
                      <div className="flex items-center gap-2.5">
                        <CheckCircle2 className="h-3.5 w-3.5 flex-shrink-0" style={{ color: '#22C55E' }} />
                        <span className="text-xs text-white font-mono">{f}</span>
                      </div>
                      <div className="flex items-center gap-3">
                        {methodCount > 0 && (
                          <span
                            className="text-[10px] font-semibold px-1.5 py-0.5 rounded"
                            style={{
                              color: methodCount === 4 ? '#22C55E' : methodCount >= 2 ? '#F59E0B' : '#64748B',
                              background: 'rgba(255,255,255,0.04)',
                            }}
                          >
                            {methodCount}/4
                          </span>
                        )}
                        {consensus && (
                          <span className="text-xs font-mono" style={{ color: '#64748B' }}>{consensus.avg_rank?.toFixed(3)}</span>
                        )}
                        <button
                          onClick={() => removeFeature(f)}
                          className="opacity-0 group-hover:opacity-100 transition-all"
                          style={{ color: '#64748B' }}
                          onMouseEnter={e => e.currentTarget.style.color = '#EF4444'}
                          onMouseLeave={e => e.currentTarget.style.color = '#64748B'}
                        >
                          <XCircle className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    </div>
                  )
                })}
              </div>
              <button
                onClick={confirm}
                className="w-full flex items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-sm font-semibold text-white hover:opacity-90 transition-opacity"
                style={{ background: '#A50044' }}
              >
                Train Models <ChevronRight className="h-4 w-4" />
              </button>
            </div>

            <div className="rounded-lg border border-border bg-card p-5 space-y-3">
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-semibold text-white">Removed from Selection</h2>
                <span className="text-xs" style={{ color: '#64748B' }}>{dropped.length} dropped</span>
              </div>
              {dropped.length === 0 ? (
                <p className="text-xs" style={{ color: '#475569' }}>No features manually removed. Use the × button to exclude features.</p>
              ) : (
                <div className="space-y-1 max-h-72 overflow-y-auto pr-1">
                  {dropped.map(f => (
                    <div key={f.feature} className="flex items-center justify-between py-2 border-b border-border/50 last:border-0">
                      <div className="flex items-center gap-2.5">
                        <XCircle className="h-3.5 w-3.5 flex-shrink-0" style={{ color: 'rgba(239,68,68,0.6)' }} />
                        <span className="text-xs font-mono text-muted-foreground">{f.feature}</span>
                      </div>
                      <span className="text-xs font-mono" style={{ color: '#64748B' }}>{f.avg_rank?.toFixed(3)}</span>
                    </div>
                  ))}
                </div>
              )}
              <div className="pt-2 space-y-2" style={{ borderTop: '1px solid #1A1A1A' }}>
                <p className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: '#64748B' }}>Method Legend</p>
                <div className="grid grid-cols-2 gap-2">
                  {[
                    { key: 'F-Reg', desc: 'F-regression (univariate)' },
                    { key: 'MI',    desc: 'Mutual Information' },
                    { key: 'RF',    desc: `Random Forest (depth≤${result.n_match_rows < 100 ? 3 : result.n_match_rows < 300 ? 5 : '∞'})` },
                    { key: 'RFE',   desc: 'Recursive Feature Elim.' },
                  ].map(m => (
                    <div key={m.key} className="flex items-center gap-2">
                      <div
                        className="w-4 h-4 rounded flex items-center justify-center flex-shrink-0"
                        style={{ background: 'rgba(165,0,68,0.15)', border: '1px solid rgba(165,0,68,0.4)' }}
                      >
                        <span style={{ color: '#A50044', fontSize: 9 }}>✓</span>
                      </div>
                      <div>
                        <p className="text-[10px] font-semibold" style={{ color: '#94A3B8' }}>{m.key}</p>
                        <p className="text-[9px]" style={{ color: '#475569' }}>{m.desc}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
