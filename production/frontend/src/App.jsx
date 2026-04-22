import { PipelineProvider, usePipeline } from './context/PipelineContext'
import Stage1Ingestion from './components/Stage1Ingestion'
import Stage2Eda from './components/Stage2Eda'
import Stage3Features from './components/Stage3Features'
import Stage4Models from './components/Stage4Models'
import Stage5BestModel from './components/Stage5BestModel'
import Stage6Replacement from './components/Stage6Replacement'
import { cn } from './lib/utils'

const STAGES = [
  { n: 1, label: '1 · Ingest',       Component: Stage1Ingestion },
  { n: 2, label: '2 · EDA',          Component: Stage2Eda },
  { n: 3, label: '3 · Features',     Component: Stage3Features },
  { n: 4, label: '4 · Models',       Component: Stage4Models },
  { n: 5, label: '5 · Best Model',   Component: Stage5BestModel },
  { n: 6, label: '6 · Replacements', Component: Stage6Replacement },
]

function PipelineApp() {
  const { currentStage, goToStage, unlockedStages, completedStages, health } = usePipeline()
  const { Component: ActiveStage } = STAGES.find(s => s.n === currentStage)

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="border-b border-border px-6 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-foreground">DM-Bypass Pipeline</h1>
          <p className="text-sm text-muted-foreground">Midfielder bypass analytics — 6-stage pipeline</p>
        </div>
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <span className={cn('w-2 h-2 rounded-full', {
            'bg-green-400': health === 'ok',
            'bg-red-500': health === 'error',
            'bg-slate-500': health === 'connecting',
          })} />
          <span>
            {health === 'ok' ? 'API connected' : health === 'error' ? 'API unreachable' : 'Connecting…'}
          </span>
        </div>
      </header>

      <nav className="px-6 py-3 flex gap-2 border-b border-border overflow-x-auto">
        {STAGES.map(({ n, label }) => {
          const unlocked = unlockedStages.has(n)
          const done = completedStages.has(n)
          const active = currentStage === n
          return (
            <button
              key={n}
              onClick={() => goToStage(n)}
              disabled={!unlocked}
              className={cn(
                'px-3 py-1.5 rounded-md text-sm font-medium transition-colors whitespace-nowrap',
                active && 'bg-primary text-primary-foreground',
                done && !active && 'bg-green-900 text-green-200',
                !unlocked && 'text-muted-foreground/40 cursor-not-allowed',
                unlocked && !active && !done && 'text-muted-foreground hover:bg-accent hover:text-accent-foreground',
              )}
            >
              {label}
            </button>
          )
        })}
      </nav>

      <main className="p-6 max-w-6xl mx-auto">
        <ActiveStage />
      </main>
    </div>
  )
}

export default function App() {
  return (
    <PipelineProvider>
      <PipelineApp />
    </PipelineProvider>
  )
}
