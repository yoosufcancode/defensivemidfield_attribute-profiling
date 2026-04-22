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
import { X } from 'lucide-react'

function FeatureTable({ title, items, scoreKey }) {
  if (!items?.length) return null
  return (
    <Card>
      <CardHeader><CardTitle className="text-sm">{title}</CardTitle></CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-8">#</TableHead>
              <TableHead>Feature</TableHead>
              <TableHead className="text-right">Score</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {items.map((it, i) => (
              <TableRow key={it.feature} className={i % 2 === 0 ? 'bg-secondary/20' : ''}>
                <TableCell className="text-muted-foreground">{it.rank ?? i + 1}</TableCell>
                <TableCell className="font-mono text-xs">{it.feature}</TableCell>
                <TableCell className="text-right font-mono text-xs text-primary">
                  {(it[scoreKey] ?? it.score ?? 0).toFixed(4)}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}

export default function Stage3Features() {
  const { onStage3Done, stage1Result } = usePipeline()
  const { polling, progress, message, error, poll, reset } = usePoll()

  const [target, setTarget] = useState('bypasses_per_halftime')
  const [nTop, setNTop] = useState(10)
  const [result, setResult] = useState(null)

  const featuresPath = Object.values(stage1Result?.features_paths || {})[0] || null

  async function run() {
    if (!featuresPath) return alert('Complete Stage 1 first.')
    reset()
    try {
      const r = await fetch(`${API}/stage3/select`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features_path: featuresPath, target_col: target, n_top: nTop }),
      })
      const { job_id } = await r.json()
      poll(`${API}/stage3/status/${job_id}`, (res) => setResult(res))
    } catch (e) {
      alert(`Failed: ${e}`)
    }
  }

  function removeFeature(f) {
    setResult(prev => ({ ...prev, selected_features: prev.selected_features.filter(x => x !== f) }))
  }

  function confirm() {
    if (!result) return
    onStage3Done(result)
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Stage 3 — Feature Selection</CardTitle>
          <CardDescription>
            F-regression, Mutual Information, Random Forest, and RFECV combined into a consensus ranking.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="space-y-1.5">
              <Label htmlFor="s3-target">Target column</Label>
              <Input id="s3-target" value={target} onChange={e => setTarget(e.target.value)} />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="s3-top">Top N features</Label>
              <Input id="s3-top" type="number" value={nTop} min={3} max={30} onChange={e => setNTop(+e.target.value)} />
            </div>
          </div>
          <Button onClick={run} disabled={polling}>
            {polling ? 'Running…' : 'Run Feature Selection'}
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
            <CardHeader><CardTitle className="text-sm">Consensus Selected Features</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div className="flex flex-wrap gap-2">
                {(result.selected_features || []).map(f => (
                  <Badge key={f} variant="blue" className="flex items-center gap-1 pr-1">
                    {f}
                    <button onClick={() => removeFeature(f)} className="ml-1 opacity-70 hover:opacity-100">
                      <X className="h-3 w-3" />
                    </button>
                  </Badge>
                ))}
              </div>
              <Button onClick={confirm}>Use these features →</Button>
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <FeatureTable title="Univariate F-regression" items={result.univariate} scoreKey="score" />
            <FeatureTable title="Mutual Information" items={result.mutual_info} scoreKey="score" />
            <FeatureTable title="Random Forest Importance" items={result.random_forest} scoreKey="score" />
            <FeatureTable title="RFECV Selected" items={result.rfe} scoreKey="score" />
          </div>

          <FeatureTable title="Consensus Ranking (avg rank across methods)" items={result.consensus} scoreKey="avg_rank" />
        </>
      )}
    </div>
  )
}
