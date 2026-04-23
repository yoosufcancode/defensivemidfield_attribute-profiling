import { useState, useEffect } from 'react'
import { usePipeline } from '../context/PipelineContext'
import { usePoll } from '../hooks/usePoll'
import { API } from '../lib/api'
import { Select, SelectGroup, SelectLabel, SelectTrigger, SelectValue, SelectContent, SelectItem } from './ui/select'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { AlertCircle, ChevronRight, ChevronDown, ChevronUp, TrendingDown, ArrowRight } from 'lucide-react'

const BUCKET_COLORS = { DM: '#3B82F6', CM: '#A50044', AM: '#22C55E', Mixed: '#F59E0B', MF: '#64748B' }
const ROLE_SHORT = r => (r || 'Unknown').split('-')[0].trim()

function StatCard({ label, value, sub, highlight }) {
  return (
    <div
      className="rounded-lg border p-4 space-y-1"
      style={{
        borderColor: highlight ? 'rgba(165,0,68,0.5)' : '#1A1A1A',
        background: highlight ? 'rgba(165,0,68,0.12)' : '#111111',
      }}
    >
      <p className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: '#64748B' }}>{label}</p>
      <p className="text-[28px] font-bold leading-tight" style={{ color: highlight ? '#A50044' : '#FFFFFF' }}>{value ?? '—'}</p>
      {sub && <p className="text-[11px]" style={{ color: '#475569' }}>{sub}</p>}
    </div>
  )
}

function ScoreBar({ value }) {
  const pct = Math.round(Math.max(0, Math.min(100, value || 0)))
  const color = pct < 33 ? '#22C55E' : pct < 66 ? '#F59E0B' : '#EF4444'
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 rounded-full overflow-hidden" style={{ background: '#1A1A1A' }}>
        <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="text-xs font-mono w-8 text-right tabular-nums" style={{ color: '#64748B' }}>{pct}</span>
    </div>
  )
}

function BucketBadge({ bucket }) {
  const color = BUCKET_COLORS[bucket] || '#64748B'
  return (
    <span
      className="text-[10px] font-bold px-1.5 py-0.5 rounded"
      style={{ color, background: `${color}18`, border: `1px solid ${color}40` }}
    >
      {bucket}
    </span>
  )
}

function ReplacementRow({ candidate, rank }) {
  const isTop = rank === 1
  return (
    <div
      className="flex items-center gap-4 p-3 rounded-lg"
      style={{
        background: isTop ? 'rgba(165,0,68,0.08)' : 'rgba(255,255,255,0.02)',
        border: `1px solid ${isTop ? 'rgba(165,0,68,0.3)' : '#1A1A1A'}`,
      }}
    >
      <span className="text-xs font-mono w-5 text-center flex-shrink-0" style={{ color: isTop ? '#A50044' : '#2D3748' }}>
        {rank}
      </span>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold truncate" style={{ color: isTop ? '#FFFFFF' : '#94A3B8' }}>
            {candidate.player_name}
          </span>
          {isTop && (
            <span className="text-[9px] font-bold px-1.5 py-0.5 rounded uppercase tracking-wider flex-shrink-0"
              style={{ color: '#A50044', background: 'rgba(165,0,68,0.15)', border: '1px solid rgba(165,0,68,0.3)' }}>
              Top Pick
            </span>
          )}
        </div>
        <p className="text-xs mt-0.5 truncate" style={{ color: '#475569' }}>
          {candidate.team} · {candidate.league}
        </p>
      </div>
      <BucketBadge bucket={candidate.position_bucket} />
      <div className="text-right flex-shrink-0">
        <p className="text-xs font-mono" style={{ color: '#94A3B8' }}>
          {candidate.bypasses_per_half?.toFixed(2)} <span style={{ color: '#475569' }}>byp/h</span>
        </p>
        <p className="text-xs font-mono" style={{ color: '#22C55E' }}>
          +{(candidate.improvement * 100).toFixed(1)}%
        </p>
      </div>
      <div className="text-right flex-shrink-0 w-14">
        <p className="text-lg font-bold font-mono leading-none"
          style={{ color: candidate.bypass_score < 33 ? '#22C55E' : candidate.bypass_score < 66 ? '#F59E0B' : '#EF4444' }}>
          {candidate.bypass_score?.toFixed(1)}
        </p>
        <p className="text-[9px] uppercase tracking-widest" style={{ color: '#2D3748' }}>score</p>
      </div>
    </div>
  )
}

export default function Stage6Replacement() {
  const {
    onStage6Done,
    stage4League, stage4Team,
    stage4ScoutingGrads, stage4ScoutingFeatures,
    stage4ModelSelected, stage4SpearmanTest, stage4SpearmanTrain,
  } = usePipeline()
  const { polling, progress, message, error, poll, reset } = usePoll()

  const [team, setTeam] = useState('')
  const [topN, setTopN] = useState(5)
  const [minMatches, setMinMatches] = useState(10)
  const [bypassCeilingPct, setBypassCeilingPct] = useState('')
  const [teamsByLeague, setTeamsByLeague] = useState({})
  const [result, setResult] = useState(null)
  const [expandedPlayer, setExpandedPlayer] = useState(null)

  const league = Object.entries(teamsByLeague).find(([, ts]) => ts.includes(team))?.[0] || ''

  // Pre-fill team from Stage 4 once it's available
  useEffect(() => {
    if (stage4Team && !team) setTeam(stage4Team)
  }, [stage4Team]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    fetch(`${API}/stage1/teams`)
      .then(r => r.json())
      .then(d => setTeamsByLeague(d.teams_by_league || {}))
      .catch(() => {})
  }, [])

  const topReplacement = result?.recommendations?.[0]?.replacements?.[0]

  // True when Stage 4 model matches the currently selected team
  const usingStage4Model = !!(
    stage4ScoutingGrads &&
    stage4League === league &&
    stage4Team   === team
  )

  async function run() {
    if (!team)   return alert('Select a team.')
    reset()
    setResult(null)
    setExpandedPlayer(null)
    try {
      const body = { league, team, top_n: topN, min_matches: minMatches }
      if (bypassCeilingPct !== '') body.bypass_ceiling_percentile = Number(bypassCeilingPct)

      // Pass Stage 4 pre-computed gradients to skip model re-training
      if (usingStage4Model) {
        body.scouting_grads    = stage4ScoutingGrads
        body.scouting_features = stage4ScoutingFeatures
        body.model_selected    = stage4ModelSelected
        body.spearman_test     = stage4SpearmanTest
        body.spearman_train    = stage4SpearmanTrain
      }

      const r = await fetch(`${API}/stage6/find-replacements`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const { job_id } = await r.json()
      poll(`${API}/stage6/status/${job_id}`, res => {
        setResult(res)
        onStage6Done(res)
        // Auto-expand the first (weakest) player
        if (res.squad?.length) setExpandedPlayer(res.squad[0].player_name)
      })
    } catch (e) {
      alert(`Failed: ${e}`)
    }
  }

  // Position breakdown counts
  const bucketCounts = result?.squad
    ? ['DM', 'CM', 'AM', 'Mixed'].reduce((acc, b) => {
        acc[b] = result.squad.filter(p => p.position_bucket === b).length
        return acc
      }, {})
    : null

  const stats = [
    { label: 'Squad Size',       value: result?.squad?.length ?? '—',          sub: 'midfielders analysed' },
    { label: 'Scouting Dims',    value: result?.scouting_features?.length ?? '—', sub: 'evaluation features' },
    { label: 'Top Replacement',  value: topReplacement?.player_name?.split(' ').slice(-1)[0] ?? '—',
      sub: topReplacement ? `${topReplacement.team} · ${topReplacement.league}` : '', highlight: true },
    { label: 'Best Improvement', value: topReplacement ? `+${(topReplacement.improvement * 100).toFixed(1)}%` : '—', sub: 'bypass score delta' },
  ]

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-[22px] font-bold text-white">Squad & Scout</h1>
          <div className="flex items-center gap-1.5 mt-0.5">
            <span className="text-xs font-semibold" style={{ color: '#A50044' }}>Stage 6</span>
            <span className="text-xs" style={{ color: '#475569' }}>·</span>
            <span className="text-xs" style={{ color: '#64748B' }}>Tactical role clustering and cross-league replacement scouting</span>
          </div>
        </div>
        <button
          onClick={run}
          disabled={polling || !team}
          className="flex items-center gap-2 text-white disabled:opacity-50 transition-opacity hover:opacity-90"
          style={{ background: '#A50044', borderRadius: 8, padding: '8px 16px', fontSize: 13, fontWeight: 600, border: 'none', cursor: (polling || !team) ? 'not-allowed' : 'pointer' }}
        >
          {polling ? 'Running…' : 'Find Replacements'} <ChevronRight className="h-3.5 w-3.5" />
        </button>
      </div>

      <div style={{ height: 1, background: '#1A1A1A' }} />

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4">
        {stats.map(s => <StatCard key={s.label} {...s} />)}
      </div>

      {/* Stage 4 model badge */}
      {usingStage4Model && (
        <div
          className="rounded-lg p-3 flex items-center gap-2 text-xs font-semibold"
          style={{ background: 'rgba(34,197,94,0.08)', border: '1px solid rgba(34,197,94,0.25)', color: '#22C55E' }}
        >
          <div className="h-2 w-2 rounded-full flex-shrink-0" style={{ background: '#22C55E' }} />
          Using Stage 4 model for <strong className="ml-1">{stage4Team}</strong> — model re-training will be skipped
        </div>
      )}

      {/* Config */}
      <div className="rounded-lg border border-border bg-card p-5 flex items-end gap-6 flex-wrap">
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
        <div className="space-y-1.5 w-28">
          <Label className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: '#64748B' }}>Top N</Label>
          <Input type="number" value={topN} min={1} max={20} onChange={e => setTopN(+e.target.value)} className="bg-secondary/50 border-border h-9 text-sm" />
        </div>
        <div className="space-y-1.5 w-28">
          <Label className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: '#64748B' }}>Min Matches</Label>
          <Input type="number" value={minMatches} min={3} max={50} onChange={e => setMinMatches(+e.target.value)} className="bg-secondary/50 border-border h-9 text-sm" />
        </div>
        <div className="space-y-1.5 w-36">
          <Label className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: '#64748B' }}>Bypass Ceiling %</Label>
          <Input
            type="number"
            value={bypassCeilingPct}
            min={1} max={99}
            placeholder="auto"
            onChange={e => setBypassCeilingPct(e.target.value)}
            className="bg-secondary/50 border-border h-9 text-sm"
          />
          <p className="text-[10px]" style={{ color: '#475569' }}>blank = per-role median</p>
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
          {/* Squad overview — full width table with expandable replacement rows */}
          <div className="rounded-lg border border-border bg-card overflow-hidden">
            {/* Section header */}
            <div className="flex items-center justify-between px-5 py-4" style={{ borderBottom: '1px solid #1A1A1A' }}>
              <h2 className="text-sm font-semibold text-white">Current Midfield Squad</h2>
              <div className="flex items-center gap-4">
                {bucketCounts && Object.entries(bucketCounts).filter(([, v]) => v > 0).map(([b, v]) => (
                  <div key={b} className="flex items-center gap-1.5">
                    <div className="w-2 h-2 rounded-full" style={{ background: BUCKET_COLORS[b] }} />
                    <span className="text-xs font-mono" style={{ color: '#64748B' }}>{b} <strong style={{ color: '#FFFFFF' }}>{v}</strong></span>
                  </div>
                ))}
                <span className="text-xs" style={{ color: '#475569' }}>Click a player to see replacements</span>
              </div>
            </div>

            {/* Table */}
            <table className="w-full text-sm">
              <thead>
                <tr style={{ borderBottom: '1px solid #1A1A1A' }}>
                  {['#', 'Player', 'Role', 'Position', 'Bypasses/Half', 'Score', 'Score Bar', 'Status'].map(h => (
                    <th key={h} className="text-left py-2.5 px-4 text-[10px] font-semibold uppercase tracking-widest" style={{ color: '#64748B' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(result.squad || []).map((p, i) => {
                  const rec = result.recommendations?.find(r => r.target_player?.player_name === p.player_name)
                  const replacements = rec?.replacements || []
                  const isExpanded = expandedPlayer === p.player_name

                  return (
                    <>
                      <tr
                        key={p.player_name}
                        onClick={() => setExpandedPlayer(isExpanded ? null : p.player_name)}
                        style={{
                          borderBottom: isExpanded ? 'none' : '1px solid #1A1A1A',
                          background: isExpanded ? 'rgba(165,0,68,0.05)' : p.is_weak ? 'rgba(239,68,68,0.04)' : 'transparent',
                          cursor: 'pointer',
                        }}
                        onMouseEnter={e => { if (!isExpanded) e.currentTarget.style.background = '#161616' }}
                        onMouseLeave={e => { if (!isExpanded) e.currentTarget.style.background = p.is_weak ? 'rgba(239,68,68,0.04)' : 'transparent' }}
                      >
                        <td className="py-3 px-4 font-mono text-xs" style={{ color: '#2D3748', width: 40 }}>{i + 1}</td>
                        <td className="py-3 px-4">
                          <div className="flex items-center gap-2">
                            <span className="font-medium text-white">{p.player_name}</span>
                            {p.is_weak && (
                              <div className="flex items-center gap-1">
                                <TrendingDown className="h-3 w-3" style={{ color: '#EF4444' }} />
                                <span className="text-[10px]" style={{ color: '#EF4444' }}>Weak</span>
                              </div>
                            )}
                          </div>
                          {p.is_weak && p.weakness_reason && (
                            <p className="text-[10px] mt-0.5 truncate max-w-xs" style={{ color: '#475569' }}>{p.weakness_reason}</p>
                          )}
                        </td>
                        <td className="py-3 px-4">
                          <span className="text-xs font-mono" style={{ color: '#64748B' }}>{ROLE_SHORT(p.tactical_role)}</span>
                        </td>
                        <td className="py-3 px-4">
                          <BucketBadge bucket={p.position_bucket} />
                        </td>
                        <td className="py-3 px-4 font-mono text-xs" style={{ color: '#94A3B8' }}>
                          {p.bypasses_per_half?.toFixed(3)}
                        </td>
                        <td className="py-3 px-4">
                          <span className="text-base font-bold font-mono"
                            style={{ color: p.bypass_score < 33 ? '#22C55E' : p.bypass_score < 66 ? '#F59E0B' : '#EF4444' }}>
                            {p.bypass_score?.toFixed(1)}
                          </span>
                        </td>
                        <td className="py-3 px-4 w-36">
                          <ScoreBar value={p.bypass_score} />
                        </td>
                        <td className="py-3 px-4">
                          <div className="flex items-center gap-2">
                            {replacements.length > 0 ? (
                              <span className="text-xs px-2 py-1 rounded" style={{ background: '#1A1A1A', color: '#64748B' }}>
                                {replacements.length} found
                              </span>
                            ) : (
                              <span className="text-xs" style={{ color: '#2D3748' }}>—</span>
                            )}
                            {isExpanded
                              ? <ChevronUp className="h-3.5 w-3.5 flex-shrink-0" style={{ color: '#A50044' }} />
                              : <ChevronDown className="h-3.5 w-3.5 flex-shrink-0" style={{ color: '#2D3748' }} />
                            }
                          </div>
                        </td>
                      </tr>

                      {/* Expanded replacement rows */}
                      {isExpanded && replacements.length > 0 && (
                        <tr key={`${p.player_name}-recs`}>
                          <td colSpan={8} className="px-4 pb-4 pt-2" style={{ borderBottom: '1px solid #1A1A1A', background: 'rgba(165,0,68,0.03)' }}>
                            <div className="flex items-center gap-2 mb-3">
                              <ArrowRight className="h-3.5 w-3.5" style={{ color: '#A50044' }} />
                              <span className="text-xs font-semibold" style={{ color: '#A50044' }}>
                                Replacements for {p.player_name}
                              </span>
                              <span className="text-xs" style={{ color: '#475569' }}>
                                · {rec?.match_filter}
                              </span>
                            </div>
                            <div className="grid grid-cols-1 gap-2">
                              {replacements.slice(0, topN).map((c, ci) => (
                                <ReplacementRow key={c.player_name} candidate={c} rank={ci + 1} />
                              ))}
                            </div>
                          </td>
                        </tr>
                      )}

                      {isExpanded && replacements.length === 0 && (
                        <tr key={`${p.player_name}-empty`}>
                          <td colSpan={8} className="px-4 py-3" style={{ borderBottom: '1px solid #1A1A1A', background: 'rgba(165,0,68,0.03)' }}>
                            <p className="text-xs" style={{ color: '#475569' }}>No replacement candidates found for this player's profile.</p>
                          </td>
                        </tr>
                      )}
                    </>
                  )
                })}
              </tbody>
            </table>
          </div>

          {/* Scouting features used */}
          {result.scouting_features?.length > 0 && (
            <div className="rounded-lg border border-border bg-card p-5 space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-semibold text-white">Scouting Dimensions</h2>
                <div className="flex items-center gap-3 text-xs" style={{ color: '#64748B' }}>
                  <span>Model: <strong style={{ color: '#FFFFFF' }}>{result.model_selected}</strong></span>
                  <span>Spearman ρ test: <strong style={{ color: '#A50044' }}>{result.spearman_test?.toFixed(3)}</strong></span>
                  <span>train: <strong style={{ color: '#94A3B8' }}>{result.spearman_train?.toFixed(3)}</strong></span>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2">
                {result.scouting_features.map(sf => (
                  <div
                    key={sf.feature}
                    className="flex items-center justify-between p-3 rounded-lg"
                    style={{ background: '#111111', border: '1px solid #1A1A1A' }}
                  >
                    <div className="min-w-0 flex-1">
                      <p className="text-xs font-mono text-white truncate">{sf.feature}</p>
                      <div className="flex items-center gap-2 mt-0.5">
                        <span
                          className="text-[10px] font-semibold"
                          style={{ color: sf.direction === 'lower_is_better' ? '#22C55E' : '#A50044' }}
                        >
                          {sf.direction === 'lower_is_better' ? '↓ Lower is better' : '↑ Higher is better'}
                        </span>
                        <span className="text-[10px]" style={{ color: '#2D3748' }}>
                          {sf.confidence_tier}
                        </span>
                      </div>
                    </div>
                    <div className="text-right flex-shrink-0 ml-3">
                      <p className="text-xs font-mono" style={{ color: '#475569' }}>
                        p = {sf.p_value < 0.001 ? '<0.001' : sf.p_value?.toFixed(3)}
                      </p>
                      <p className="text-xs font-mono" style={{ color: '#64748B' }}>
                        ∂y/∂x {sf.gradient >= 0 ? '+' : ''}{sf.gradient?.toFixed(4)}
                      </p>
                    </div>
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
