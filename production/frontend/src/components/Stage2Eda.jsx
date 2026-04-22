import { usePipeline } from '../context/PipelineContext'
import { usePoll } from '../hooks/usePoll'
import { API } from '../lib/api'
import { useState } from 'react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from './ui/card'
import { Button } from './ui/button'
import { Progress } from './ui/progress'
import { Alert, AlertDescription } from './ui/alert'
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from './ui/table'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts'

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
      poll(`${API}/stage2/status/${job_id}`, (res) => {
        setResult(res)
        onStage2Done(res)
      })
    } catch (e) {
      alert(`Failed: ${e}`)
    }
  }

  const distData = result?.bypass_distribution?.counts
    ? result.bypass_distribution.bin_edges.slice(0, -1).map((edge, i) => ({
        label: edge.toFixed(2),
        count: result.bypass_distribution.counts[i],
      }))
    : []

  const corrMatrix = result?.correlation_matrix || {}
  const corrCols = Object.keys(corrMatrix).slice(0, 15)

  const desc = result?.descriptive_stats || {}
  const statKeys = Object.keys(desc)
  const statCols = statKeys.length ? Object.keys(desc[statKeys[0]] || {}).slice(0, 12) : []

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Stage 2 — Exploratory Data Analysis</CardTitle>
          <CardDescription>
            Compute descriptive statistics, correlation matrix, and bypass distribution.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Button onClick={run} disabled={polling}>
            {polling ? 'Running…' : 'Run EDA'}
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
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            {[
              { value: result.row_count, label: 'Player-match rows' },
              { value: result.column_count, label: 'Columns' },
              { value: result.bypass_distribution?.mean?.toFixed(2), label: 'Mean bypasses/half' },
              { value: Object.keys(result.missing_values || {}).length, label: 'Cols w/ missing vals' },
            ].map(({ value, label }) => (
              <Card key={label}>
                <CardContent className="pt-6 text-center">
                  <div className="text-2xl font-bold text-primary">{value ?? '—'}</div>
                  <div className="text-xs text-muted-foreground mt-1">{label}</div>
                </CardContent>
              </Card>
            ))}
          </div>

          {distData.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Bypass Distribution (bypasses per halftime)</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={distData} margin={{ top: 4, right: 4, left: 0, bottom: 4 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="label" tick={{ fill: '#94a3b8', fontSize: 11 }} tickLine={false} />
                    <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} tickLine={false} axisLine={false} />
                    <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 6 }} />
                    <Bar dataKey="count" fill="#3b82f6" opacity={0.8} radius={[2, 2, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}

          {corrCols.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Correlation Matrix (top 15 features)</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto text-xs">
                  <div style={{ display: 'grid', gridTemplateColumns: `120px ${corrCols.map(() => '40px').join(' ')}`, gap: '1px' }}>
                    <div />
                    {corrCols.map(c => (
                      <div key={c} style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)', color: '#94a3b8', height: 90, textAlign: 'left', padding: '4px 2px', overflow: 'hidden', textOverflow: 'ellipsis' }}>{c}</div>
                    ))}
                    {corrCols.map(row => (
                      <>
                        <div key={`row-${row}`} style={{ color: '#cbd5e1', padding: 4, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: 120 }}>{row}</div>
                        {corrCols.map(col => {
                          const v = (corrMatrix[row] || {})[col] ?? 0
                          const abs = Math.abs(v)
                          const bg = v > 0 ? `rgba(59,130,246,${abs.toFixed(2)})` : `rgba(239,68,68,${abs.toFixed(2)})`
                          return (
                            <div key={`${row}-${col}`} title={`${row} × ${col}: ${v}`} style={{ background: bg, width: 40, height: 40, display: 'flex', alignItems: 'center', justifyContent: 'center', color: abs > 0.5 ? '#fff' : '#94a3b8', fontSize: 10 }}>
                              {v.toFixed(1)}
                            </div>
                          )
                        })}
                      </>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {statKeys.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Descriptive Statistics</CardTitle>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Stat</TableHead>
                      {statCols.map(c => <TableHead key={c} className="font-mono text-xs">{c}</TableHead>)}
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {statKeys.map((stat, i) => (
                      <TableRow key={stat} className={i % 2 === 0 ? 'bg-secondary/20' : ''}>
                        <TableCell className="font-medium">{stat}</TableCell>
                        {statCols.map(c => (
                          <TableCell key={c} className="font-mono text-xs text-muted-foreground">
                            {(desc[stat][c] ?? '—').toString().slice(0, 8)}
                          </TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  )
}
