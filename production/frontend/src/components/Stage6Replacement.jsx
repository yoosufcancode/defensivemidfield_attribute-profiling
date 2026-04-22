import { useState, useEffect } from 'react'
import { usePipeline } from '../context/PipelineContext'
import { usePoll } from '../hooks/usePoll'
import { API } from '../lib/api'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from './ui/card'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Progress } from './ui/progress'
import { Alert, AlertDescription } from './ui/alert'
import { Badge } from './ui/badge'
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from './ui/table'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from './ui/select'
import { cn } from '../lib/utils'

const ALL_LEAGUES = ['Spain', 'England', 'France', 'Germany', 'Italy']

const ROLE_VARIANTS = {
  'Anchor 6': 'blue',
  'Ball-winning 8': 'yellow',
  'Hybrid 6/8': 'green',
}

function RoleBadge({ role }) {
  return <Badge variant={ROLE_VARIANTS[role] || 'blue'}>{role || 'Unknown'}</Badge>
}

function ScoreBar({ value }) {
  const pct = Math.round(Math.max(0, Math.min(100, value || 0)))
  const color = pct < 33 ? 'bg-green-500' : pct < 66 ? 'bg-yellow-500' : 'bg-red-500'
  const textColor = pct < 33 ? 'text-green-400' : pct < 66 ? 'text-yellow-400' : 'text-red-400'
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-2 bg-secondary rounded-full overflow-hidden">
        <div className={cn('h-full rounded-full transition-all', color)} style={{ width: `${pct}%` }} />
      </div>
      <span className={cn('text-xs font-mono w-8 text-right', textColor)}>{pct}</span>
    </div>
  )
}

export default function Stage6Replacement() {
  const { onStage6Done } = usePipeline()
  const { polling, progress, message, error, poll, reset } = usePoll()

  const [league, setLeague] = useState('')
  const [team, setTeam] = useState('')
  const [topN, setTopN] = useState(5)
  const [minMatches, setMinMatches] = useState(10)
  const [teamsByLeague, setTeamsByLeague] = useState({})
  const [result, setResult] = useState(null)

  useEffect(() => {
    fetch(`${API}/stage1/teams`)
      .then(r => r.json())
      .then(d => setTeamsByLeague(d.teams_by_league || {}))
      .catch(() => {})
  }, [])

  const teams = teamsByLeague[league] || []

  async function run() {
    if (!league) return alert('Select a league.')
    if (!team) return alert('Select a team.')
    reset()
    setResult(null)
    try {
      const r = await fetch(`${API}/stage6/find-replacements`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ league, team, top_n: topN, min_matches: minMatches }),
      })
      const { job_id } = await r.json()
      poll(`${API}/stage6/status/${job_id}`, (res) => {
        setResult(res)
        onStage6Done(res)
      })
    } catch (e) {
      alert(`Failed: ${e}`)
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Stage 6 — Replacement Finder</CardTitle>
          <CardDescription>
            Team-specific model selection + tactical role clustering + cross-league scouting.
            Select a league and team — the pipeline analyzes the full squad automatically.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="space-y-1.5">
              <Label>League</Label>
              <Select value={league} onValueChange={v => { setLeague(v); setTeam('') }}>
                <SelectTrigger><SelectValue placeholder="— select league —" /></SelectTrigger>
                <SelectContent>
                  {ALL_LEAGUES.map(lg => <SelectItem key={lg} value={lg}>{lg}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1.5">
              <Label>Team</Label>
              <Select value={team} onValueChange={setTeam} disabled={!league || !teams.length}>
                <SelectTrigger><SelectValue placeholder={league ? '— select team —' : '— select league first —'} /></SelectTrigger>
                <SelectContent>
                  {teams.map(t => <SelectItem key={t} value={t}>{t}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1.5">
              <Label>Top N replacements per player</Label>
              <Input type="number" value={topN} min={1} max={20} onChange={e => setTopN(+e.target.value)} />
            </div>
            <div className="space-y-1.5">
              <Label>Min half-match rows per player <span className="text-muted-foreground">(10 ≈ 5 full matches)</span></Label>
              <Input type="number" value={minMatches} min={3} max={50} onChange={e => setMinMatches(+e.target.value)} />
            </div>
          </div>
          <Button onClick={run} disabled={polling}>
            {polling ? 'Running…' : 'Find Replacements'}
          </Button>
        </CardContent>
      </Card>

      {polling && (
        <Card>
          <CardContent className="pt-6 space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">{message}</span>
              <span className="text-primary font-mono">{progress}%</span>
            </div>
            <Progress value={progress} />
          </CardContent>
        </Card>
      )}

      {error && (
        <Alert variant="destructive">
          <AlertDescription>Error: {error}</AlertDescription>
        </Alert>
      )}

      {result && (
        <>
          <Card>
            <CardContent className="pt-6 flex flex-wrap items-center gap-4">
              <div>
                <span className="text-muted-foreground text-sm">Team:</span>
                <span className="font-semibold ml-1">{result.team}</span>
                <span className="text-muted-foreground text-sm ml-1">({result.league})</span>
              </div>
              <div>
                <span className="text-muted-foreground text-sm">Model:</span>
                <Badge variant="green" className="ml-1">{result.model_selected}</Badge>
              </div>
              <div>
                <span className="text-muted-foreground text-sm">Spearman ρ test:</span>
                <span className={cn('font-mono font-semibold ml-1', result.spearman_test >= 0.3 ? 'text-green-400' : 'text-yellow-400')}>
                  {result.spearman_test?.toFixed(4)}
                </span>
              </div>
              <div>
                <span className="text-muted-foreground text-sm">train:</span>
                <span className="font-mono ml-1">{result.spearman_train?.toFixed(4)}</span>
              </div>
            </CardContent>
          </Card>

          {(result.scouting_features || []).length > 0 && (
            <Card>
              <CardHeader><CardTitle className="text-sm">Scouting Features ({result.scouting_features.length})</CardTitle></CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Feature</TableHead>
                      <TableHead className="text-center">Gradient</TableHead>
                      <TableHead className="text-center">Direction</TableHead>
                      <TableHead className="text-center">p-value</TableHead>
                      <TableHead className="text-center">Sign stable</TableHead>
                      <TableHead>Confidence</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {result.scouting_features.map((f, i) => (
                      <TableRow key={f.feature} className={i % 2 === 0 ? 'bg-secondary/20' : ''}>
                        <TableCell className="font-mono text-xs">{f.feature}</TableCell>
                        <TableCell className={cn('text-center font-mono text-xs', f.gradient > 0 ? 'text-red-400' : 'text-green-400')}>{f.gradient?.toFixed(4)}</TableCell>
                        <TableCell className="text-center text-xs text-muted-foreground">{f.direction}</TableCell>
                        <TableCell className={cn('text-center font-mono text-xs', f.p_value < 0.05 ? 'text-green-400' : f.p_value < 0.15 ? 'text-yellow-400' : 'text-red-400')}>{f.p_value?.toFixed(4)}</TableCell>
                        <TableCell className="text-center">{f.sign_stable ? <span className="text-green-400">✓</span> : <span className="text-yellow-400">~</span>}</TableCell>
                        <TableCell className="text-xs text-muted-foreground">{f.confidence_tier}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          )}

          {(result.squad || []).length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">{result.team} — Squad Bypass Scores</CardTitle>
                <p className="text-xs text-muted-foreground">Score = league percentile rank. Lower = better bypass prevention.</p>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Player</TableHead>
                      <TableHead>Role</TableHead>
                      <TableHead className="text-center">Pos</TableHead>
                      <TableHead className="text-center">AvgX</TableHead>
                      <TableHead className="min-w-[140px]">Bypass Score</TableHead>
                      <TableHead className="text-center">Halves</TableHead>
                      <TableHead className="text-center">Bypasses/half</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {result.squad.map((p, i) => (
                      <TableRow key={p.player_name} className={i % 2 === 0 ? 'bg-secondary/20' : ''}>
                        <TableCell className="font-semibold">{p.player_name}</TableCell>
                        <TableCell><RoleBadge role={p.tactical_role} /></TableCell>
                        <TableCell className="text-center text-muted-foreground">{p.position_bucket}</TableCell>
                        <TableCell className="text-center font-mono text-xs text-muted-foreground">{p.average_position_x?.toFixed(1) ?? '—'}</TableCell>
                        <TableCell><ScoreBar value={p.bypass_score} /></TableCell>
                        <TableCell className="text-center text-muted-foreground">{p.halves_played}</TableCell>
                        <TableCell className={cn('text-center font-mono text-xs', p.bypasses_per_half > 6 ? 'text-red-400' : '')}>{p.bypasses_per_half?.toFixed(2)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          )}

          {(result.recommendations || []).map(rec => {
            const p = rec.target_player
            const replacements = rec.replacements || []
            return (
              <Card key={p.player_name}>
                <CardContent className="pt-6">
                  <div className="flex flex-wrap items-center gap-3 mb-4">
                    <span className="text-base font-semibold">{p.player_name}</span>
                    <RoleBadge role={p.tactical_role} />
                    <Badge variant="blue">{p.position_bucket}</Badge>
                    <span className="text-muted-foreground text-sm">AvgX {p.average_position_x?.toFixed(1) ?? '—'}</span>
                    <div className="ml-auto flex gap-4 text-right">
                      <div>
                        <div className={cn('text-xl font-bold', p.bypass_score > 66 ? 'text-red-400' : p.bypass_score > 33 ? 'text-yellow-400' : 'text-green-400')}>
                          {p.bypass_score?.toFixed(1)}
                        </div>
                        <div className="text-xs text-muted-foreground">bypass score</div>
                      </div>
                      <div>
                        <div className="text-xl font-bold text-orange-400">{p.bypasses_per_half?.toFixed(2)}</div>
                        <div className="text-xs text-muted-foreground">bypasses/half</div>
                      </div>
                    </div>
                  </div>

                  <p className="text-xs text-muted-foreground mb-3">
                    Match filter: <span className="text-foreground/70">{rec.match_filter}</span>
                  </p>

                  {replacements.length === 0 ? (
                    <p className="text-muted-foreground text-sm">No candidates found with lower bypass score in this role + position.</p>
                  ) : (
                    <div className="space-y-3">
                      {replacements.map(c => {
                        const fc = c.feature_comparison || {}
                        const fcKeys = Object.keys(fc)
                        return (
                          <div key={c.player_name} className="bg-secondary/30 rounded-lg p-4">
                            <div className="flex flex-wrap items-start justify-between gap-2 mb-2">
                              <div>
                                <div className="flex items-center gap-2 flex-wrap">
                                  <span className="text-muted-foreground text-xs font-mono">#{c.rank}</span>
                                  <span className="font-semibold">{c.player_name}</span>
                                  <Badge variant="blue" className="text-xs">{c.team}</Badge>
                                  <span className="text-xs text-muted-foreground">{c.league}</span>
                                  <RoleBadge role={c.tactical_role} />
                                  <Badge variant="blue">{c.position_bucket}</Badge>
                                </div>
                                <div className="text-xs text-muted-foreground mt-1">AvgX {c.average_position_x?.toFixed(1) ?? '—'}</div>
                              </div>
                              <div className="flex gap-4 text-right">
                                <div>
                                  <div className="text-lg font-bold text-green-400">{c.bypass_score?.toFixed(1)}</div>
                                  <div className="text-xs text-muted-foreground">score</div>
                                </div>
                                <div>
                                  <div className="text-lg font-bold text-primary">+{c.improvement?.toFixed(3)}</div>
                                  <div className="text-xs text-muted-foreground">improvement</div>
                                </div>
                                <div>
                                  <div className="text-lg font-bold text-orange-400">{c.bypasses_per_half?.toFixed(2)}</div>
                                  <div className="text-xs text-muted-foreground">bypasses/half</div>
                                </div>
                              </div>
                            </div>

                            {fcKeys.length > 0 && (
                              <details className="mt-2">
                                <summary className="text-xs text-muted-foreground cursor-pointer hover:text-foreground">
                                  Feature comparison vs {p.player_name}
                                </summary>
                                <Table className="mt-2 text-xs">
                                  <TableHeader>
                                    <TableRow>
                                      <TableHead className="text-xs">Feature</TableHead>
                                      <TableHead className="text-right text-xs">Candidate</TableHead>
                                      <TableHead className="text-right text-xs">Target</TableHead>
                                      <TableHead className="text-right text-xs">Δ</TableHead>
                                      <TableHead className="text-xs pl-2">Direction</TableHead>
                                    </TableRow>
                                  </TableHeader>
                                  <TableBody>
                                    {fcKeys.map(f => {
                                      const v = fc[f]
                                      return (
                                        <TableRow key={f}>
                                          <TableCell className="font-mono text-xs py-0.5">{f}</TableCell>
                                          <TableCell className="text-right font-mono text-xs">{v.candidate?.toFixed(3)}</TableCell>
                                          <TableCell className="text-right font-mono text-xs text-muted-foreground">{v.target?.toFixed(3)}</TableCell>
                                          <TableCell className={cn('text-right font-mono text-xs', (v.delta ?? 0) > 0 ? 'text-green-400' : 'text-red-400')}>
                                            {(v.delta ?? 0) > 0 ? '+' : ''}{v.delta?.toFixed(3)}
                                          </TableCell>
                                          <TableCell className="pl-2 text-xs text-muted-foreground">{v.direction || ''}</TableCell>
                                        </TableRow>
                                      )
                                    })}
                                  </TableBody>
                                </Table>
                              </details>
                            )}
                          </div>
                        )
                      })}
                    </div>
                  )}
                </CardContent>
              </Card>
            )
          })}
        </>
      )}
    </div>
  )
}
