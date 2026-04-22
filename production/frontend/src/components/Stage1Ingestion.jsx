import { useState, useEffect } from 'react'
import { usePipeline } from '../context/PipelineContext'
import { usePoll } from '../hooks/usePoll'
import { API } from '../lib/api'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from './ui/card'
import { Button } from './ui/button'
import { Checkbox } from './ui/checkbox'
import { Label } from './ui/label'
import { Progress } from './ui/progress'
import { Alert, AlertDescription } from './ui/alert'
import { Badge } from './ui/badge'

const ALL_LEAGUES = ['Spain', 'England', 'France', 'Germany', 'Italy']

export default function Stage1Ingestion() {
  const { onStage1Done } = usePipeline()
  const { polling, progress, message, error, poll, reset } = usePoll()

  const [selectedLeagues, setSelectedLeagues] = useState([])
  const [skipDownload, setSkipDownload] = useState(false)
  const [existingData, setExistingData] = useState([])

  useEffect(() => {
    fetch(`${API}/stage1/available-data`)
      .then(r => r.json())
      .then(data => {
        setExistingData(data)
        setSelectedLeagues(data.map(d => d.league))
      })
      .catch(() => {})
  }, [])

  function toggleLeague(lg) {
    setSelectedLeagues(prev =>
      prev.includes(lg) ? prev.filter(x => x !== lg) : [...prev, lg]
    )
  }

  function useExisting(d) {
    onStage1Done({
      features_paths: { [d.league]: d.features_path },
      row_counts: { [d.league]: d.row_count },
      leagues_processed: [d.league],
    })
  }

  async function useAllExisting() {
    try {
      const r = await fetch(`${API}/stage1/available-data`)
      const data = await r.json()
      if (!data.length) return alert('No existing data found.')
      const features_paths = {}
      const row_counts = {}
      data.forEach(d => {
        features_paths[d.league] = d.features_path
        row_counts[d.league] = d.row_count
      })
      onStage1Done({ features_paths, row_counts, leagues_processed: data.map(d => d.league) })
    } catch (e) {
      alert(`Failed: ${e}`)
    }
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
      poll(`${API}/stage1/status/${job_id}`, onStage1Done)
    } catch (e) {
      alert(`Failed to start job: ${e}`)
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Stage 1 — Data Ingestion</CardTitle>
          <CardDescription>
            Compute midfielder features from the Wyscout Open Dataset (2017/18) for one or more leagues.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {existingData.length > 0 && (
            <div className="space-y-2">
              <p className="text-sm font-medium text-foreground">Existing processed data</p>
              {existingData.map(d => (
                <div key={d.league} className="flex items-center justify-between bg-secondary/50 rounded-lg px-4 py-2">
                  <div>
                    <span className="text-sm font-semibold">{d.league}</span>
                    <span className="text-xs text-muted-foreground ml-2">{d.row_count?.toLocaleString()} rows</span>
                  </div>
                  <Button variant="secondary" size="sm" onClick={() => useExisting(d)}>Use →</Button>
                </div>
              ))}
              <Button variant="secondary" size="sm" onClick={useAllExisting}>Use all existing data →</Button>
            </div>
          )}

          <div className="border-t border-border pt-4 space-y-4">
            <p className="text-sm font-medium">Select leagues to process</p>
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
              {ALL_LEAGUES.map(lg => (
                <label key={lg} className="flex items-center gap-2 bg-secondary/40 rounded-lg px-3 py-2 cursor-pointer hover:bg-secondary/60 transition-colors">
                  <Checkbox
                    checked={selectedLeagues.includes(lg)}
                    onCheckedChange={() => toggleLeague(lg)}
                  />
                  <span className="text-sm">{lg}</span>
                </label>
              ))}
            </div>

            <div className="flex items-center gap-2">
              <Checkbox
                id="skip-dl"
                checked={skipDownload}
                onCheckedChange={setSkipDownload}
              />
              <Label htmlFor="skip-dl">Skip feature engineering if CSV already exists</Label>
            </div>

            <Button onClick={startIngest} disabled={polling}>
              {polling ? 'Running…' : 'Run Ingestion'}
            </Button>
          </div>
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
    </div>
  )
}
