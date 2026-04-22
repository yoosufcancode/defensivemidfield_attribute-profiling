import { useState } from 'react'
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
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'

function SpearmanBadge({ value }) {
  const v = typeof value === 'number' ? value : parseFloat(value)
  const variant = v >= 0.5 ? 'green' : v >= 0.3 ? 'yellow' : 'red'
  return <Badge variant={variant}>{v.toFixed(4)}</Badge>
}

export default function Stage4Models() {
  const { onStage4Done, stage1Result, stage3Result } = usePipeline()
  const { polling, progress, message, error, poll, reset } = usePoll()

  const [target, setTarget] = useState('bypasses_per_halftime')
  const [testSize, setTestSize] = useState(0.15)
  const [seed, setSeed] = useState(42)
  const [result, setResult] = useState(null)

  const featuresPath = Object.values(stage1Result?.features_paths || {})[0] || null
  const selectedFeatures = stage3Result?.selected_features || []

  async function run() {
    if (!featuresPath) return alert('Complete Stage 1 first.')
    if (!selectedFeatures.length) return alert('Complete Stage 3 first.')
    reset()
    try {
      const r = await fetch(`${API}/stage4/build`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          features_path: featuresPath,
          selected_features: selectedFeatures,
          target_col: target,
          test_size: testSize,
          random_state: seed,
        }),
      })
      const { job_id } = await r.json()
      poll(`${API}/stage4/status/${job_id}`, (res) => {
        setResult(res)
        onStage4Done(res)
      })
    } catch (e) {
      alert(`Failed: ${e}`)
    }
  }

  const chartData = result?.models?.map(m => ({
    name: m.name,
    'LOOCV ρ': m.loocv.spearman,
    'Test ρ': m.test.spearman,
  })) ?? []

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Stage 4 — Model Building</CardTitle>
          <CardDescription>
            Train MLR, Ridge (CV), and Lasso (CV). Best model selected by{' '}
            <strong className="text-foreground">LOOCV Spearman ρ</strong>.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="space-y-1.5">
              <Label htmlFor="s4-target">Target column</Label>
              <Input id="s4-target" value={target} onChange={e => setTarget(e.target.value)} />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="s4-testsize">Test size</Label>
              <Input id="s4-testsize" type="number" value={testSize} step={0.05} min={0.05} max={0.4} onChange={e => setTestSize(+e.target.value)} />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="s4-seed">Random state</Label>
              <Input id="s4-seed" type="number" value={seed} onChange={e => setSeed(+e.target.value)} />
            </div>
          </div>
          <Button onClick={run} disabled={polling}>
            {polling ? 'Running…' : 'Build Models'}
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
            <CardContent className="pt-6 flex flex-wrap items-center gap-3">
              <span className="text-muted-foreground text-sm">Best model (by LOOCV Spearman ρ):</span>
              <Badge variant="green">{result.best_model}</Badge>
              <span className="text-muted-foreground text-sm">Features used:</span>
              <strong>{result.feature_count}</strong>
            </CardContent>
          </Card>

          <Card>
            <CardHeader><CardTitle className="text-sm">Model Metrics</CardTitle></CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead rowSpan={2}>Model</TableHead>
                    <TableHead className="text-center" colSpan={4}>LOOCV</TableHead>
                    <TableHead className="text-center" colSpan={4}>Test Set</TableHead>
                  </TableRow>
                  <TableRow>
                    {['Spearman ρ','R²','RMSE','MAE','Spearman ρ','R²','RMSE','MAE'].map(h => (
                      <TableHead key={h} className="text-center text-xs">{h}</TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {result.models?.map((m, i) => (
                    <TableRow key={m.name} className={i % 2 === 0 ? 'bg-secondary/20' : ''}>
                      <TableCell className={`font-semibold ${m.name === result.best_model ? 'text-green-400' : ''}`}>{m.name}</TableCell>
                      <TableCell className="text-center"><SpearmanBadge value={m.loocv.spearman} /></TableCell>
                      <TableCell className="text-center font-mono text-xs text-muted-foreground">{m.loocv.r2}</TableCell>
                      <TableCell className="text-center font-mono text-xs text-muted-foreground">{m.loocv.rmse}</TableCell>
                      <TableCell className="text-center font-mono text-xs text-muted-foreground">{m.loocv.mae}</TableCell>
                      <TableCell className="text-center"><SpearmanBadge value={m.test.spearman} /></TableCell>
                      <TableCell className="text-center font-mono text-xs text-muted-foreground">{m.test.r2}</TableCell>
                      <TableCell className="text-center font-mono text-xs text-muted-foreground">{m.test.rmse}</TableCell>
                      <TableCell className="text-center font-mono text-xs text-muted-foreground">{m.test.mae}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          <Card>
            <CardHeader><CardTitle className="text-sm">Spearman ρ — LOOCV vs Test Set</CardTitle></CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={chartData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }} tickLine={false} />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} tickLine={false} axisLine={false} domain={[-0.1, 1]} />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 6 }} />
                  <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 12 }} />
                  <Bar dataKey="LOOCV ρ" fill="#3b82f6" opacity={0.85} radius={[2, 2, 0, 0]} />
                  <Bar dataKey="Test ρ" fill="#10b981" opacity={0.85} radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  )
}
