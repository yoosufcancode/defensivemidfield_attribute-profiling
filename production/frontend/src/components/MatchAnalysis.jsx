import { useState, useEffect, useMemo } from 'react'
import { API } from '../lib/api'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from './ui/select'
import { Input } from './ui/input'
import { Search, ChevronRight, Trophy } from 'lucide-react'

const ALL_LEAGUES = ['Spain', 'England', 'France', 'Germany', 'Italy']

const STATS = [
  { key: 'bypasses_per_halftime',    label: 'Bypasses',    fmt: v => v?.toFixed(1) },
  { key: 'passes_attempted',         label: 'Passes',      fmt: v => v?.toFixed(0) },
  { key: 'pass_completion_rate',     label: 'Pass%',       fmt: v => v != null ? `${(v * 100).toFixed(0)}%` : '—' },
  { key: 'progressive_passes',       label: 'Prog Pass',   fmt: v => v?.toFixed(1) },
  { key: 'key_passes',               label: 'Key Pass',    fmt: v => v?.toFixed(1) },
  { key: 'shot_creating_actions',    label: 'SCA',         fmt: v => v?.toFixed(1) },
  { key: 'ball_recoveries',          label: 'Recoveries',  fmt: v => v?.toFixed(1) },
  { key: 'tackles_won',              label: 'Tackles',     fmt: v => v?.toFixed(1) },
  { key: 'sliding_tackles',          label: 'Sliding',     fmt: v => v?.toFixed(1) },
  { key: 'zone14_touches',           label: 'Zone 14',     fmt: v => v?.toFixed(1) },
  { key: 'possession_time_seconds',  label: 'Poss (s)',    fmt: v => v?.toFixed(0) },
  { key: 'average_position_x',       label: 'Avg X',       fmt: v => v?.toFixed(1) },
]

function fmt(stat, value) {
  if (value == null || Number.isNaN(value)) return '—'
  return stat.fmt(value) ?? '—'
}

function BypassBar({ value }) {
  const pct = Math.min(100, Math.max(0, (value / 15) * 100))
  return (
    <div className="flex items-center gap-1.5">
      <div className="w-16 h-1 rounded-full overflow-hidden" style={{ background: '#1A1A1A' }}>
        <div className="h-full rounded-full" style={{ width: `${pct}%`, background: '#A50044' }} />
      </div>
      <span className="text-xs font-mono" style={{ color: '#94A3B8' }}>{value?.toFixed(1) ?? '—'}</span>
    </div>
  )
}

const BUCKET_COLORS = {
  DM:    { color: '#60A5FA', bg: '#60A5FA18', border: '#60A5FA40' },
  CM:    { color: '#A78BFA', bg: '#A78BFA18', border: '#A78BFA40' },
  AM:    { color: '#F59E0B', bg: '#F59E0B18', border: '#F59E0B40' },
  Mixed: { color: '#94A3B8', bg: '#94A3B818', border: '#94A3B840' },
  MF:    { color: '#64748B', bg: '#64748B18', border: '#64748B40' },
}

function PositionBadge({ bucket }) {
  const style = BUCKET_COLORS[bucket] || BUCKET_COLORS.MF
  return (
    <span
      className="text-[9px] font-bold px-1 py-0.5 rounded"
      style={{ color: style.color, background: style.bg, border: `1px solid ${style.border}` }}
    >
      {bucket}
    </span>
  )
}

function PlayerRow({ player, cols }) {
  return (
    <tr
      style={{ borderBottom: '1px solid #1A1A1A' }}
      onMouseEnter={e => e.currentTarget.style.background = '#161616'}
      onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
    >
      <td className="py-2 px-3 whitespace-nowrap">
        <div className="flex items-center gap-1.5">
          <span className="text-xs font-medium text-white">{player.player_name}</span>
          {player.position_bucket && <PositionBadge bucket={player.position_bucket} />}
        </div>
      </td>
      {cols.map(s => (
        <td key={s.key} className="py-2 px-3">
          {s.key === 'bypasses_per_halftime'
            ? <BypassBar value={player[s.key]} />
            : <span className="text-xs font-mono" style={{ color: '#64748B' }}>{fmt(s, player[s.key])}</span>
          }
        </td>
      ))}
    </tr>
  )
}

function TeamTable({ teamName, players, cols, score, isWinner }) {
  return (
    <div
      className="rounded-lg overflow-hidden"
      style={{ border: `1px solid ${isWinner ? 'rgba(34,197,94,0.4)' : '#1A1A1A'}` }}
    >
      <div className="px-4 py-3 flex items-center justify-between" style={{ borderBottom: '1px solid #1A1A1A', background: isWinner ? 'rgba(34,197,94,0.06)' : 'transparent' }}>
        <div className="flex items-center gap-2">
          {isWinner && <Trophy className="h-3.5 w-3.5" style={{ color: '#22C55E' }} />}
          <span className="text-sm font-semibold" style={{ color: isWinner ? '#22C55E' : '#FFFFFF' }}>{teamName}</span>
          {isWinner && <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded" style={{ color: '#22C55E', background: 'rgba(34,197,94,0.15)', border: '1px solid rgba(34,197,94,0.3)' }}>Winner</span>}
        </div>
        <div className="flex items-center gap-3">
          {score != null && <span className="text-lg font-bold font-mono" style={{ color: isWinner ? '#22C55E' : '#64748B' }}>{score}</span>}
          <span className="text-xs" style={{ color: '#64748B' }}>{players.length} players</span>
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr style={{ borderBottom: '1px solid #1A1A1A' }}>
              <th className="text-left py-2 px-3 text-[10px] font-semibold uppercase tracking-widest" style={{ color: '#64748B' }}>Player</th>
              {cols.map(s => (
                <th key={s.key} className="text-left py-2 px-3 text-[10px] font-semibold uppercase tracking-widest whitespace-nowrap" style={{ color: '#64748B' }}>
                  {s.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {players.map(p => <PlayerRow key={p.player_name} player={p} cols={cols} />)}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default function MatchAnalysis() {
  const [league, setLeague] = useState('Spain')
  const [matches, setMatches] = useState([])
  const [loadingMatches, setLoadingMatches] = useState(false)
  const [teamFilter, setTeamFilter] = useState('all')
  const [search, setSearch] = useState('')
  const [selectedMatch, setSelectedMatch] = useState(null)
  const [detail, setDetail] = useState(null)
  const [loadingDetail, setLoadingDetail] = useState(false)

  useEffect(() => {
    setMatches([])
    setSelectedMatch(null)
    setDetail(null)
    setTeamFilter('all')
    setSearch('')
    setLoadingMatches(true)
    fetch(`${API}/stage1/matches?league=${league}`)
      .then(r => r.json())
      .then(data => { setMatches(Array.isArray(data) ? data : []); setLoadingMatches(false) })
      .catch(() => setLoadingMatches(false))
  }, [league])

  useEffect(() => {
    if (!selectedMatch) { setDetail(null); return }
    setDetail(null)
    setLoadingDetail(true)
    fetch(`${API}/stage1/match-detail?league=${league}&match_id=${selectedMatch}`)
      .then(r => r.json())
      .then(data => { setDetail(data); setLoadingDetail(false) })
      .catch(() => setLoadingDetail(false))
  }, [selectedMatch, league])

  const allTeams = useMemo(() => {
    const teams = new Set()
    matches.forEach(m => m.teams.forEach(t => teams.add(t)))
    return ['all', ...Array.from(teams).sort()]
  }, [matches])

  const filtered = useMemo(() => {
    const q = search.toLowerCase()
    return matches.filter(m => {
      const matchTeam = teamFilter === 'all' || m.teams.includes(teamFilter)
      const matchSearch = !q || m.teams.some(t => t.toLowerCase().includes(q)) || m.match_id.includes(q)
      return matchTeam && matchSearch
    })
  }, [matches, teamFilter, search])

  const detailTeams = detail ? Object.entries(detail.teams) : []

  return (
    <div className="space-y-5">
      {/* Header */}
      <div>
        <h1 className="text-[22px] font-bold text-white">Match Analysis</h1>
        <div className="flex items-center gap-1.5 mt-0.5">
          <span className="text-xs font-semibold" style={{ color: '#A50044' }}>Analysis</span>
          <span className="text-xs" style={{ color: '#475569' }}>·</span>
          <span className="text-xs" style={{ color: '#64748B' }}>Per-match midfielder breakdown</span>
        </div>
      </div>

      <div style={{ height: 1, background: '#1A1A1A' }} />

      {/* Summary cards */}
      <div className="grid grid-cols-3 gap-4">
        {[
          { label: 'Matches',        value: matches.length || '—',  sub: `in ${league}` },
          { label: 'Filtered',       value: filtered.length || '—', sub: 'matching filters' },
          { label: 'Selected Match', value: selectedMatch ? `#${selectedMatch}` : '—', sub: detailTeams.map(([t]) => t).join(' vs ') || 'none selected' },
        ].map(c => (
          <div key={c.label} className="rounded-lg border border-border bg-card p-4 space-y-1">
            <p className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: '#64748B' }}>{c.label}</p>
            <p className="text-[22px] font-bold leading-tight text-white truncate">{c.value}</p>
            <p className="text-[11px] truncate" style={{ color: '#475569' }}>{c.sub}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-5 gap-4">
        {/* Match list */}
        <div className="col-span-2 rounded-lg border border-border bg-card p-4 space-y-3 flex flex-col" style={{ maxHeight: 620 }}>
          {/* League + filters */}
          <div className="space-y-2 flex-shrink-0">
            <Select value={league} onValueChange={setLeague}>
              <SelectTrigger className="bg-secondary/50 border-border h-9 text-sm w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {ALL_LEAGUES.map(lg => <SelectItem key={lg} value={lg}>{lg}</SelectItem>)}
              </SelectContent>
            </Select>
            <div className="flex gap-2">
              <div className="relative flex-1">
                <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3 w-3" style={{ color: '#475569' }} />
                <Input
                  value={search}
                  onChange={e => setSearch(e.target.value)}
                  placeholder="Search team…"
                  className="pl-7 bg-secondary/30 border-border h-8 text-xs"
                />
              </div>
              <Select value={teamFilter} onValueChange={setTeamFilter}>
                <SelectTrigger className="w-36 bg-secondary/30 border-border h-8 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {allTeams.map(t => <SelectItem key={t} value={t}>{t === 'all' ? 'All teams' : t}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* Match list */}
          <div className="overflow-y-auto flex-1 space-y-1 pr-1">
            {loadingMatches ? (
              <p className="text-xs text-center pt-8" style={{ color: '#64748B' }}>Loading matches…</p>
            ) : filtered.length === 0 ? (
              <p className="text-xs text-center pt-8" style={{ color: '#64748B' }}>No matches found.</p>
            ) : filtered.map(m => {
              const active = selectedMatch === m.match_id
              const scoreStr = m.scores && m.teams.length === 2
                ? `${m.scores[m.teams[0]] ?? '?'} - ${m.scores[m.teams[1]] ?? '?'}`
                : null
              return (
                <button
                  key={m.match_id}
                  onClick={() => setSelectedMatch(active ? null : m.match_id)}
                  className="w-full text-left rounded-lg px-3 py-2.5 transition-colors flex items-center justify-between gap-2"
                  style={{
                    background: active ? 'rgba(165,0,68,0.15)' : 'rgba(255,255,255,0.02)',
                    border: `1px solid ${active ? 'rgba(165,0,68,0.4)' : '#1A1A1A'}`,
                  }}
                  onMouseEnter={e => { if (!active) e.currentTarget.style.background = '#161616' }}
                  onMouseLeave={e => { if (!active) e.currentTarget.style.background = 'rgba(255,255,255,0.02)' }}
                >
                  <div className="min-w-0 flex-1">
                    <p className="text-xs font-semibold text-white truncate">
                      {m.teams[0]} <span style={{ color: '#475569' }}>vs</span> {m.teams[1]}
                    </p>
                    <div className="flex items-center gap-2 mt-0.5">
                      {m.date && <span className="text-[10px]" style={{ color: '#475569' }}>{m.date}</span>}
                      {scoreStr && <span className="text-[10px] font-mono font-semibold" style={{ color: '#94A3B8' }}>{scoreStr}</span>}
                      {m.winner && <span className="text-[10px]" style={{ color: '#22C55E' }}>↑ {m.winner}</span>}
                    </div>
                  </div>
                  <ChevronRight className="h-3.5 w-3.5 flex-shrink-0" style={{ color: active ? '#A50044' : '#2D3748' }} />
                </button>
              )
            })}
          </div>
        </div>

        {/* Match detail */}
        <div className="col-span-3 space-y-4">
          {!selectedMatch ? (
            <div className="rounded-lg border border-border bg-card flex items-center justify-center" style={{ height: 300 }}>
              <p className="text-sm" style={{ color: '#64748B' }}>Select a match to view player stats</p>
            </div>
          ) : loadingDetail ? (
            <div className="rounded-lg border border-border bg-card flex items-center justify-center" style={{ height: 300 }}>
              <p className="text-sm" style={{ color: '#64748B' }}>Loading…</p>
            </div>
          ) : detail ? (
            <>
              {/* Match header */}
              <div className="rounded-lg border border-border bg-card px-4 py-3 flex items-center justify-between">
                <div>
                  <p className="text-sm font-semibold text-white">
                    {detail.result?.label || detailTeams.map(([t]) => t).join(' vs ')}
                  </p>
                  <p className="text-xs font-mono mt-0.5" style={{ color: '#64748B' }}>
                    {detail.result?.date} · {detail.league} · stats averaged across both halves
                  </p>
                </div>
                <div className="flex items-center gap-3">
                  {detail.result?.winner && (
                    <div className="flex items-center gap-1.5">
                      <Trophy className="h-3.5 w-3.5" style={{ color: '#22C55E' }} />
                      <span className="text-xs font-semibold" style={{ color: '#22C55E' }}>{detail.result.winner}</span>
                    </div>
                  )}
                  <span className="text-xs font-semibold px-2 py-0.5 rounded" style={{ color: '#A50044', background: 'rgba(165,0,68,0.12)', border: '1px solid rgba(165,0,68,0.3)' }}>
                    {detailTeams.reduce((a, [, ps]) => a + ps.length, 0)} players
                  </span>
                </div>
              </div>

              {/* Team tables */}
              {detailTeams.map(([teamName, players]) => (
                <TeamTable
                  key={teamName}
                  teamName={teamName}
                  players={players}
                  cols={STATS}
                  score={detail.result?.scores?.[teamName]}
                  isWinner={detail.result?.winner === teamName}
                />
              ))}
            </>
          ) : null}
        </div>
      </div>
    </div>
  )
}
