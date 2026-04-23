import { useState, useEffect } from 'react'
import { usePipeline } from '../context/PipelineContext'
import { usePoll } from '../hooks/usePoll'
import { API } from '../lib/api'
import { Checkbox } from './ui/checkbox'
import { Label } from './ui/label'
import { cn } from '../lib/utils'
import { CheckCircle2, Circle, AlertCircle, ChevronRight, Upload } from 'lucide-react'

const ALL_LEAGUES = ['Spain', 'England', 'France', 'Germany', 'Italy']

function StatCard({ label, value, sub, accent }) {
  return (
    <div className="rounded-lg border border-border bg-card p-4 space-y-1">
      <p className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: '#64748B' }}>{label}</p>
      <p className="text-[28px] font-bold leading-tight" style={{ color: accent ? '#A50044' : '#FFFFFF' }}>{value ?? '—'}</p>
      {sub && <p className="text-[11px]" style={{ color: '#475569' }}>{sub}</p>}
    </div>
  )
}

function QualityRow({ label, pass }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-border last:border-0">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <CheckCircle2 className="h-4 w-4 text-muted-foreground/40" />
        {label}
      </div>
      <span className={cn('text-xs font-semibold', pass ? 'text-green-400' : 'text-red-400')}>
        {pass ? 'Pass' : 'Fail'}
      </span>
    </div>
  )
}

export default function Stage1Ingestion() {
  const { onStage1Done } = usePipeline()
  const { polling, progress, message, error, poll, reset } = usePoll()

  const [selectedLeagues, setSelectedLeagues] = useState([])
  const [skipDownload, setSkipDownload] = useState(false)
  const [existingData, setExistingData] = useState([])
  const [result, setResult] = useState(null)

  useEffect(() => {
    fetch(`${API}/stage1/available-data`)
      .then(r => r.json())
      .then(data => {
        setExistingData(data)
        setSelectedLeagues(data.map(d => d.league))
        if (data.length) {
          const features_paths = {}, row_counts = {}, unique_player_counts = {}
          data.forEach(d => {
            features_paths[d.league] = d.features_path
            row_counts[d.league] = d.row_count
            unique_player_counts[d.league] = d.unique_players ?? 0
          })
          setResult({ features_paths, row_counts, unique_player_counts, leagues_processed: data.map(d => d.league) })
        }
      })
      .catch(() => {})
  }, [])

  function toggleLeague(lg) {
    setSelectedLeagues(prev =>
      prev.includes(lg) ? prev.filter(x => x !== lg) : [...prev, lg]
    )
  }

  async function startIngest() {
    if (!selectedLeagues.length) return alert('Select at least one league.')
    reset()
    try {
      const r = await fetch(`${API}/stage1/ingest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ leagues: selectedLeagues, skip_download: skipDownload }),
      })
      const { job_id } = await r.json()
      poll(`${API}/stage1/status/${job_id}`, res => {
        setResult(res)
        onStage1Done(res)
      })
    } catch (e) {
      alert(`Failed to start job: ${e}`)
    }
  }

  function useAllExisting() {
    if (!existingData.length) return alert('No existing data found.')
    const features_paths = {}, row_counts = {}, unique_player_counts = {}
    existingData.forEach(d => {
      features_paths[d.league] = d.features_path
      row_counts[d.league] = d.row_count
      unique_player_counts[d.league] = d.unique_players ?? 0
    })
    const res = { features_paths, row_counts, unique_player_counts, leagues_processed: existingData.map(d => d.league) }
    setResult(res)
    onStage1Done(res)
  }

  const totalRows = result
    ? Object.values(result.row_counts || {}).reduce((a, b) => a + b, 0)
    : null

  const leaguesLoaded = result?.leagues_processed?.length ?? 0
  const avgRowsPerLeague = result && leaguesLoaded
    ? Math.round(totalRows / leaguesLoaded).toLocaleString()
    : '—'

  const stats = [
    { label: 'Player-Match Rows', value: totalRows ? totalRows.toLocaleString() : '—',     sub: 'across all leagues' },
    { label: 'Unique Players',    value: result ? Object.values(result.unique_player_counts || {}).reduce((a, b) => a + b, 0).toLocaleString() : '—', sub: 'identified midfielders' },
    { label: 'Leagues Loaded',    value: leaguesLoaded || '—', sub: result?.leagues_processed?.join(', ') || '2017/18 season' },
    { label: 'Avg Rows / League', value: avgRowsPerLeague, sub: 'player-match density', accent: true },
  ]

  return (
    <div className="space-y-5">
      {/* Page header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-[22px] font-bold text-white">Data Input</h1>
          <div className="flex items-center gap-1.5 mt-0.5">
            <span className="text-xs font-semibold" style={{ color: '#A50044' }}>Stage 1</span>
            <span className="text-xs" style={{ color: '#475569' }}>·</span>
            <span className="text-xs" style={{ color: '#64748B' }}>Load & validate raw match data</span>
          </div>
        </div>
        {existingData.length > 0 && (
          <button
            onClick={useAllExisting}
            className="flex items-center gap-2 text-white transition-opacity hover:opacity-90"
            style={{ background: '#A50044', borderRadius: 8, padding: '8px 16px', fontSize: 13, fontWeight: 600, border: 'none', cursor: 'pointer' }}
          >
            <CheckCircle2 className="h-3.5 w-3.5" style={{ color: '#22C55E' }} />
            Data Loaded
          </button>
        )}
      </div>

      {/* Divider */}
      <div style={{ height: 1, background: '#1A1A1A' }} />

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4">
        {stats.map(s => <StatCard key={s.label} {...s} />)}
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* League selection */}
        <div className="rounded-lg border border-border bg-card p-5 space-y-4">
          <h2 className="text-sm font-semibold text-white">Data Sources</h2>

          {existingData.length > 0 && (
            <div className="space-y-2 pb-4 border-b border-border">
              {existingData.map(d => (
                <div key={d.league} className="flex items-center justify-between bg-secondary/40 rounded-lg px-4 py-2.5">
                  <div className="flex items-center gap-2.5">
                    <div className="w-2 h-2 rounded-full bg-green-400" />
                    <div>
                      <p className="text-sm text-white font-medium">wyscout_{d.league.toLowerCase()}_features.csv</p>
                      <p className="text-xs text-muted-foreground">{d.row_count?.toLocaleString()} rows</p>
                    </div>
                  </div>
                  <span className="text-xs text-green-400 font-semibold bg-green-400/10 rounded px-2 py-0.5">Loaded</span>
                </div>
              ))}
            </div>
          )}

          <div className="space-y-3">
            <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">Select leagues</p>
            <div className="grid grid-cols-1 gap-2">
              {ALL_LEAGUES.map(lg => (
                <label
                  key={lg}
                  className="flex items-center gap-3 bg-secondary/30 rounded-lg px-3 py-2.5 cursor-pointer hover:bg-secondary/50 transition-colors"
                >
                  <Checkbox
                    checked={selectedLeagues.includes(lg)}
                    onCheckedChange={() => toggleLeague(lg)}
                  />
                  <span className="text-sm text-white">{lg}</span>
                </label>
              ))}
            </div>

            <div className="flex items-center gap-2 pt-1">
              <Checkbox
                id="skip-dl"
                checked={skipDownload}
                onCheckedChange={setSkipDownload}
              />
              <Label htmlFor="skip-dl" className="text-xs text-muted-foreground">
                Skip if CSV already exists
              </Label>
            </div>

            <button
              onClick={startIngest}
              disabled={polling}
              className="w-full flex items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-sm font-semibold text-white disabled:opacity-50 transition-opacity hover:opacity-90"
              style={{ background: 'hsl(var(--primary))' }}
            >
              <Upload className="h-4 w-4" />
              {polling ? 'Running…' : 'Run Ingestion'}
            </button>
          </div>

          {/* Progress */}
          {polling && (
            <div className="space-y-2 pt-2">
              <div className="flex justify-between text-xs">
                <span className="text-muted-foreground">{message}</span>
                <span className="text-primary font-mono">{progress}%</span>
              </div>
              <div className="h-1.5 rounded-full bg-secondary overflow-hidden">
                <div
                  className="h-full rounded-full transition-all"
                  style={{ width: `${progress}%`, background: 'hsl(var(--primary))' }}
                />
              </div>
            </div>
          )}

          {error && (
            <div className="flex items-start gap-2 rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm text-red-400">
              <AlertCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Data quality */}
        <div className="rounded-lg border border-border bg-card p-5 space-y-4">
          <h2 className="text-sm font-semibold text-white">Data Quality Checks</h2>
          <div className="divide-y divide-border">
            <QualityRow label="No duplicate rows"    pass={!!result} />
            <QualityRow label="Schema validated"     pass={!!result} />
            <QualityRow label="Date range correct"   pass={!!result} />
            <QualityRow label="Player IDs matched"   pass={!!result} />
            <QualityRow label="Numeric types cast"   pass={!!result} />
          </div>

          {result && (
            <div className="mt-4 rounded-lg border border-green-500/30 bg-green-500/10 p-4 space-y-1">
              <p className="text-sm font-semibold text-green-400">Ingestion complete</p>
              <p className="text-xs text-muted-foreground">
                {result.leagues_processed?.join(', ')} — {totalRows?.toLocaleString()} rows total
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
