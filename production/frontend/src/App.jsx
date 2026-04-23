import { useState } from 'react'
import { PipelineProvider, usePipeline } from './context/PipelineContext'
import Stage1Ingestion from './components/Stage1Ingestion'
import Stage2Eda from './components/Stage2Eda'
import Stage3Features from './components/Stage3Features'
import Stage4Models from './components/Stage4Models'
import Stage5BestModel from './components/Stage5BestModel'
import Stage6Replacement from './components/Stage6Replacement'
import PlayersView from './components/PlayersView'
import MatchAnalysis from './components/MatchAnalysis'
import { cn } from './lib/utils'
import { LayoutDashboard, Users, ChartColumn, SlidersHorizontal, Cpu, Target } from 'lucide-react'

const NAV = [
  { id: 'dashboard', label: 'Dashboard',      icon: LayoutDashboard, stage: 1 },
  { id: 'players',   label: 'Players',         icon: Users           },
  { id: 'match',     label: 'Match Analysis',  icon: ChartColumn     },
  { id: 'features',  label: 'Features',        icon: SlidersHorizontal, stages: [2, 3] },
  { id: 'model',     label: 'Model Output',    icon: Cpu,             stages: [4, 5] },
  { id: 'squad',     label: 'Squad & Scout',   icon: Target,          stages: [6] },
]

function Placeholder({ title }) {
  return (
    <div className="flex flex-col items-center justify-center h-64 gap-3">
      <p className="text-sm font-semibold text-white">{title}</p>
      <p className="text-xs" style={{ color: '#64748B' }}>This section is coming soon.</p>
    </div>
  )
}

function FeaturesView({ onFeaturesConfirmed }) {
  const { completedStages, unlockedStages } = usePipeline()
  if (!unlockedStages.has(2) && !unlockedStages.has(3)) {
    return <Placeholder title="Features" />
  }
  if (!completedStages.has(2)) return <Stage2Eda />
  return <Stage3Features onConfirmed={onFeaturesConfirmed} />
}

function ModelView() {
  const { unlockedStages } = usePipeline()
  if (!unlockedStages.has(4)) return <Placeholder title="Model Output" />
  return (
    <div className="space-y-0">
      <Stage4Models />
      <Stage5BestModel />
    </div>
  )
}

function SquadView() {
  const { unlockedStages } = usePipeline()
  if (!unlockedStages.has(6)) return <Placeholder title="Squad & Scout" />
  return <Stage6Replacement />
}

const STAGE_META = {
  1: { nav: 'Dashboard',     label: 'Data Input',                 desc: 'Load & validate raw match data' },
  2: { nav: 'Features',      label: 'Exploratory Data Analysis',  desc: 'Feature distributions and correlations' },
  3: { nav: 'Features',      label: 'Feature Selection',          desc: 'Lasso + VIF consensus ranking' },
  4: { nav: 'Model Output',  label: 'Per-Team Model Training',     desc: 'Train MLR, Ridge, Lasso on team data — extract scouting gradients' },
  5: { nav: 'Model Output',  label: 'Best Model Analysis',        desc: 'Feature coefficients and gradient sensitivity from the per-team model' },
  6: { nav: 'Squad & Scout', label: 'Midfield Players & Replacements', desc: 'Cross-league scouting' },
}

function PipelineApp() {
  const { health, unlockedStages, completedStages } = usePipeline()
  const [view, setView] = useState('dashboard')
  const [mountedViews, setMountedViews] = useState(new Set(['dashboard']))

  function handleSetView(id) {
    setMountedViews(prev => new Set([...prev, id]))
    setView(id)
  }

  function handleFeaturesConfirmed() {
    handleSetView('model')
  }

  const CONTENT = {
    dashboard: <Stage1Ingestion />,
    players:   <PlayersView />,
    match:     <MatchAnalysis />,
    features:  <FeaturesView onFeaturesConfirmed={handleFeaturesConfirmed} />,
    model:     <ModelView />,
    squad:     <SquadView />,
  }

  function isNavLocked(item) {
    if (item.placeholder) return false
    if (item.stage)   return !unlockedStages.has(item.stage)
    if (item.stages)  return !item.stages.some(s => unlockedStages.has(s))
    return false
  }

  function isNavDone(item) {
    if (item.stage)   return completedStages.has(item.stage)
    if (item.stages)  return item.stages.every(s => completedStages.has(s))
    return false
  }

  const healthColor = health === 'ok' ? '#22C55E' : health === 'error' ? '#EF4444' : '#64748B'

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: '#0D0D0D' }}>

      {/* ── Sidebar ── */}
      <aside
        className="flex-shrink-0 flex flex-col"
        style={{ width: 220, background: '#111111', paddingTop: 32, paddingBottom: 32 }}
      >
        {/* Logo */}
        <div className="flex items-center gap-2.5" style={{ paddingLeft: 24, paddingRight: 24 }}>
          <div
            className="flex items-center justify-center flex-shrink-0"
            style={{ width: 32, height: 32, borderRadius: 6, background: '#A50044', fontSize: 14, fontWeight: 700, color: '#FFFFFF' }}
          >
            ◈
          </div>
          <span style={{ color: '#FFFFFF', fontSize: 12, fontWeight: 700, letterSpacing: 2 }}>
            DM BYPASS
          </span>
        </div>

        {/* Divider */}
        <div style={{ height: 1, background: '#1A1A1A', margin: '24px 0' }} />

        {/* Nav */}
        <nav className="flex flex-col" style={{ gap: 4, paddingLeft: 24, paddingRight: 24 }}>
          {NAV.map(item => {
            const active = view === item.id
            const locked = isNavLocked(item)
            const Icon = item.icon
            return (
              <button
                key={item.id}
                onClick={() => !locked && handleSetView(item.id)}
                className="flex items-center w-full text-left transition-colors"
                style={{
                  height: 40,
                  borderRadius: 8,
                  gap: 10,
                  padding: '0 12px',
                  background: active ? '#A50044' : 'transparent',
                  color: active ? '#FFFFFF' : locked ? '#2D3748' : '#64748B',
                  cursor: locked ? 'not-allowed' : 'pointer',
                  fontSize: 13,
                  fontWeight: active ? 600 : 500,
                  border: 'none',
                  outline: 'none',
                }}
                onMouseEnter={e => { if (!active && !locked) e.currentTarget.style.background = '#1A1A1A' }}
                onMouseLeave={e => { if (!active) e.currentTarget.style.background = 'transparent' }}
              >
                <Icon
                  style={{
                    width: 16, height: 16, flexShrink: 0,
                    color: active ? '#FFFFFF' : locked ? '#2D3748' : '#475569',
                  }}
                />
                <span style={{ flex: 1 }}>{item.label}</span>
                {isNavDone(item) && !active && (
                  <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#22C55E', flexShrink: 0 }} />
                )}
              </button>
            )
          })}
        </nav>

        {/* Spacer */}
        <div style={{ flex: 1 }} />

        {/* User avatar */}
        <div className="flex items-center" style={{ gap: 10, paddingLeft: 24, paddingRight: 24 }}>
          <div
            className="flex items-center justify-center flex-shrink-0"
            style={{ width: 32, height: 32, borderRadius: 16, background: '#A50044', color: '#FFFFFF', fontSize: 13, fontWeight: 700 }}
          >
            A
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <span style={{ color: '#FFFFFF', fontSize: 12, fontWeight: 600 }}>Analyst</span>
            <span style={{ color: '#64748B', fontSize: 11 }}>
              {health === 'ok' ? '● API connected' : health === 'error' ? '● API unreachable' : '● Connecting…'}
            </span>
          </div>
        </div>
      </aside>

      {/* ── Main Content ── */}
      <main
        className="flex-1 overflow-y-auto"
        style={{ background: '#0D0D0D', padding: 32, display: 'flex', flexDirection: 'column', gap: 20 }}
      >
        {Object.entries(CONTENT).map(([id, content]) =>
          mountedViews.has(id) ? (
            <div key={id} style={{ display: view === id ? 'contents' : 'none' }}>
              {content}
            </div>
          ) : null
        )}
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
