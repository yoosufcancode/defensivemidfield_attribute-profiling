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
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'

export default function Stage5BestModel() {
  const { onStage5Done, stage1Result, stage3Result, stage4Result } = usePipeline()
  const { polling, progress, message, error, poll, reset } = usePoll()

  const best = stage4Result?.best_model
  const models = stage4Result?.models || []
  const match = models.find(m => m.name === best) || models[0]

  const [modelPath, setModelPath] = useState('')
  const [scalerPath, setScalerPath] = useState('')
  const [featuresPath, setFeaturesPath] = useState('')
  const [targetCol, setTargetCol] = useState('bypasses_per_halftime')
  const [result, setResult] = useState(null)

  useEffect(() => {
    if (match?.model_path) setModelPath(match.model_path)
    if (stage4Result?.scaler_path) setScalerPath(stage4Result.scaler_path)
    const fp = Object.values(stage1Result?.features_paths || {})[0] || ''
    setFeaturesPath(fp)
  }, [match, stage4Result, stage1Result])

  const selectedFeatures = stage3Result?.selected_features || []

  async function run() {
    if (!modelPath || !scalerPath || !featuresPath) return alert('Model path, scaler path, and features path are required.')
    if (!selectedFeatures.length) return alert('Complete Stage 3 first.')
    reset()
    try {
      const r = await fetch(`${API}/stage5/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_path: modelPath, scaler_path: scalerPath, features_path: featuresPath, selected_features: selectedFeatures, target_col: targetCol }),
      })
      const { job_id } = await r.json()
      poll(`${API}/stage5/status/${job_id}`, (res) => {
        setResult(res)
        onStage5Done(res)
      })
    } catch (e) {
      alert(`Failed: ${e}`)
    }
  }

  const coefs = result?.coefficients || []
  const grad = result?.gradient_sensitivity || []

  const coefChartData = coefs.map(c => ({ feature: c.feature, value: c.coefficient }))

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Stage 5 — Best Model Analysis</CardTitle>
          <CardDescription>Extract coefficients and gradient sensitivity from the best linear model.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {[
              { label: 'Model path (auto-filled)', value: modelPath, set: setModelPath, placeholder: 'models/lasso_model.pkl' },
              { label: 'Scaler path (auto-filled)', value: scalerPath, set: setScalerPath, placeholder: 'models/scaler.pkl' },
              { label: 'Features path (auto-filled)', value: featuresPath, set: setFeaturesPath, placeholder: 'data/processed/…_features.csv' },
              { label: 'Target column', value: targetCol, set: setTargetCol, placeholder: 'bypasses_per_halftime' },
            ].map(({ label, value, set, placeholder }) => (
              <div key={label} className="space-y-1.5">
                <Label>{label}</Label>
                <Input value={value} onChange={e => set(e.target.value)} placeholder={placeholder} />
              </div>
            ))}
          </div>
          <Button onClick={run} disabled={polling}>
            {polling ? 'Running…' : 'Analyze Model'}
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
            <CardContent className="pt-6 flex items-center gap-2">
              <span className="text-muted-foreground text-sm">Model:</span>
              <Badge variant="blue">{result.model_name || '—'}</Badge>
            </CardContent>
          </Card>

          <Card>
            <CardHeader><CardTitle className="text-sm">Feature Coefficients (by magnitude)</CardTitle></CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={Math.max(220, coefs.length * 32)}>
                <BarChart data={coefChartData} layout="vertical" margin={{ top: 4, right: 16, left: 8, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
                  <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 11 }} tickLine={false} axisLine={false} />
                  <YAxis type="category" dataKey="feature" tick={{ fill: '#cbd5e1', fontSize: 11 }} tickLine={false} axisLine={false} width={140} />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 6 }} />
                  <Bar dataKey="value" radius={[0, 2, 2, 0]}>
                    {coefChartData.map((entry, i) => (
                      <Cell key={i} fill={entry.value >= 0 ? '#10b981' : '#ef4444'} opacity={0.85} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <Card>
            <CardHeader><CardTitle className="text-sm">Coefficients</CardTitle></CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Feature</TableHead>
                    <TableHead className="text-right">Coefficient</TableHead>
                    <TableHead className="text-right">Importance %</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {coefs.map((c, i) => (
                    <TableRow key={c.feature} className={i % 2 === 0 ? 'bg-secondary/20' : ''}>
                      <TableCell className="font-mono text-xs">{c.feature}</TableCell>
                      <TableCell className={`text-right font-mono text-xs ${c.coefficient >= 0 ? 'text-green-400' : 'text-red-400'}`}>{c.coefficient}</TableCell>
                      <TableCell className="text-right font-mono text-xs text-muted-foreground">{c.relative_importance}%</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          <Card>
            <CardHeader><CardTitle className="text-sm">Gradient Sensitivity (coef × raw std)</CardTitle></CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Feature</TableHead>
                    <TableHead className="text-right">Sensitivity</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {grad.map((g, i) => (
                    <TableRow key={g.feature} className={i % 2 === 0 ? 'bg-secondary/20' : ''}>
                      <TableCell className="font-mono text-xs">{g.feature}</TableCell>
                      <TableCell className={`text-right font-mono text-xs ${g.sensitivity >= 0 ? 'text-green-400' : 'text-red-400'}`}>{g.sensitivity}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  )
}
