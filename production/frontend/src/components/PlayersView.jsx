import { useState, useEffect, useMemo } from 'react'
import { API } from '../lib/api'
import { Input } from './ui/input'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from './ui/select'
import { Search } from 'lucide-react'

const PAGE_SIZE = 20
const ALL_LEAGUES = ['Spain', 'England', 'France', 'Germany', 'Italy']

const LEAGUE_COLORS = {
  Spain:   '#A50044',
  England: '#CF1B2B',
  France:  '#003189',
  Germany: '#000000',
  Italy:   '#009246',
}

export default function PlayersView() {
  const [players, setPlayers] = useState([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [leagueFilter, setLeagueFilter] = useState('all')
  const [teamFilter, setTeamFilter] = useState('all')
  const [page, setPage] = useState(1)

  useEffect(() => {
    fetch(`${API}/stage6/players`)
      .then(r => r.json())
      .then(data => { setPlayers(data); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  const teamsForLeague = useMemo(() => {
    const source = leagueFilter === 'all' ? players : players.filter(p => p.league === leagueFilter)
    return ['all', ...Array.from(new Set(source.map(p => p.team))).sort()]
  }, [players, leagueFilter])

  const filtered = useMemo(() => {
    const q = search.toLowerCase()
    return players.filter(p => {
      const matchLeague = leagueFilter === 'all' || p.league === leagueFilter
      const matchTeam   = teamFilter === 'all'   || p.team === teamFilter
      const matchSearch = !q || p.player_name.toLowerCase().includes(q) || p.team.toLowerCase().includes(q)
      return matchLeague && matchTeam && matchSearch
    })
  }, [players, search, leagueFilter, teamFilter])

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE))
  const currentPage = Math.min(page, totalPages)
  const pageRows = filtered.slice((currentPage - 1) * PAGE_SIZE, currentPage * PAGE_SIZE)

  function handleSearch(val) { setSearch(val); setPage(1) }

  function handleLeague(val) {
    setLeagueFilter(val)
    setTeamFilter('all')
    setPage(1)
  }

  function handleTeam(val) { setTeamFilter(val); setPage(1) }

  return (
    <div className="space-y-5">
      {/* Header */}
      <div>
        <h1 className="text-[22px] font-bold text-white">Players</h1>
        <div className="flex items-center gap-1.5 mt-0.5">
          <span className="text-xs font-semibold" style={{ color: '#A50044' }}>Database</span>
          <span className="text-xs" style={{ color: '#475569' }}>·</span>
          <span className="text-xs" style={{ color: '#64748B' }}>All midfielders across leagues</span>
        </div>
      </div>

      <div style={{ height: 1, background: '#1A1A1A' }} />

      {/* Summary cards */}
      <div className="grid grid-cols-3 gap-4">
        {[
          { label: 'Total Players', value: players.length.toLocaleString(), sub: 'unique midfielders' },
          { label: 'Filtered',      value: filtered.length.toLocaleString(), sub: 'matching filters' },
          { label: 'Leagues',       value: ALL_LEAGUES.length, sub: 'competitions covered' },
        ].map(c => (
          <div key={c.label} className="rounded-lg border border-border bg-card p-4 space-y-1">
            <p className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: '#64748B' }}>{c.label}</p>
            <p className="text-[28px] font-bold leading-tight text-white">{c.value}</p>
            <p className="text-[11px]" style={{ color: '#475569' }}>{c.sub}</p>
          </div>
        ))}
      </div>

      {/* Filters */}
      <div className="flex items-center gap-3">
        <div className="relative flex-1 max-w-xs">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5" style={{ color: '#475569' }} />
          <Input
            value={search}
            onChange={e => handleSearch(e.target.value)}
            placeholder="Search player or team…"
            className="pl-8 bg-card border-border h-9 text-sm"
          />
        </div>
        <Select value={leagueFilter} onValueChange={handleLeague}>
          <SelectTrigger className="w-36 bg-card border-border h-9 text-sm">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All leagues</SelectItem>
            {ALL_LEAGUES.map(lg => <SelectItem key={lg} value={lg}>{lg}</SelectItem>)}
          </SelectContent>
        </Select>
        <Select value={teamFilter} onValueChange={handleTeam} disabled={loading}>
          <SelectTrigger className="w-48 bg-card border-border h-9 text-sm">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {teamsForLeague.map(t => (
              <SelectItem key={t} value={t}>{t === 'all' ? 'All teams' : t}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Table */}
      <div className="rounded-lg border border-border bg-card overflow-hidden">
        {loading ? (
          <div className="flex items-center justify-center h-48">
            <p className="text-sm" style={{ color: '#64748B' }}>Loading players…</p>
          </div>
        ) : (
          <>
            <table className="w-full text-sm">
              <thead>
                <tr style={{ borderBottom: '1px solid #1A1A1A' }}>
                  {['#', 'Player', 'Team', 'League'].map(h => (
                    <th
                      key={h}
                      className="text-left py-3 px-4 text-[10px] font-semibold uppercase tracking-widest"
                      style={{ color: '#64748B' }}
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {pageRows.length === 0 ? (
                  <tr>
                    <td colSpan={4} className="py-12 text-center text-sm" style={{ color: '#64748B' }}>
                      No players match your filters.
                    </td>
                  </tr>
                ) : pageRows.map((p, i) => (
                  <tr
                    key={`${p.player_name}-${p.team}`}
                    style={{ borderBottom: '1px solid #1A1A1A' }}
                    onMouseEnter={e => e.currentTarget.style.background = '#161616'}
                    onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                  >
                    <td className="py-2.5 px-4 font-mono text-xs" style={{ color: '#2D3748', width: 48 }}>
                      {(currentPage - 1) * PAGE_SIZE + i + 1}
                    </td>
                    <td className="py-2.5 px-4 font-medium text-white">{p.player_name}</td>
                    <td className="py-2.5 px-4 text-xs" style={{ color: '#94A3B8' }}>{p.team}</td>
                    <td className="py-2.5 px-4">
                      <span
                        className="text-xs font-semibold px-2 py-0.5 rounded"
                        style={{
                          color: LEAGUE_COLORS[p.league] || '#64748B',
                          background: `${LEAGUE_COLORS[p.league] || '#64748B'}18`,
                          border: `1px solid ${LEAGUE_COLORS[p.league] || '#64748B'}40`,
                        }}
                      >
                        {p.league}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center justify-between px-4 py-3" style={{ borderTop: '1px solid #1A1A1A' }}>
                <span className="text-xs" style={{ color: '#64748B' }}>
                  {(currentPage - 1) * PAGE_SIZE + 1}–{Math.min(currentPage * PAGE_SIZE, filtered.length)} of {filtered.length}
                </span>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setPage(p => Math.max(1, p - 1))}
                    disabled={currentPage === 1}
                    className="text-xs px-3 py-1.5 rounded disabled:opacity-30 transition-opacity"
                    style={{ border: '1px solid #1A1A1A', color: '#94A3B8', background: 'transparent', cursor: currentPage === 1 ? 'not-allowed' : 'pointer' }}
                  >
                    Prev
                  </button>
                  <span className="text-xs font-mono" style={{ color: '#64748B' }}>{currentPage} / {totalPages}</span>
                  <button
                    onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                    disabled={currentPage === totalPages}
                    className="text-xs px-3 py-1.5 rounded disabled:opacity-30 transition-opacity"
                    style={{ border: '1px solid #1A1A1A', color: '#94A3B8', background: 'transparent', cursor: currentPage === totalPages ? 'not-allowed' : 'pointer' }}
                  >
                    Next
                  </button>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
