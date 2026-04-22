import { createContext, useContext, useState, useEffect, useCallback } from 'react'
import { API } from '../lib/api'

const PipelineContext = createContext(null)

export function PipelineProvider({ children }) {
  const [currentStage, setCurrentStage] = useState(1)
  const [unlockedStages, setUnlockedStages] = useState(new Set([1]))
  const [completedStages, setCompletedStages] = useState(new Set())
  const [health, setHealth] = useState('connecting')

  const [stage1Result, setStage1Result] = useState(null)
  const [stage3Result, setStage3Result] = useState(null)
  const [stage4Result, setStage4Result] = useState(null)
  const [stage5Result, setStage5Result] = useState(null)

  const unlock = useCallback((n) => {
    setUnlockedStages(prev => new Set([...prev, n]))
  }, [])

  const complete = useCallback((n) => {
    setCompletedStages(prev => new Set([...prev, n]))
  }, [])

  const goToStage = useCallback((n) => {
    setUnlockedStages(prev => {
      if (!prev.has(n)) return prev
      setCurrentStage(n)
      return prev
    })
  }, [])

  const onStage1Done = useCallback((result) => {
    setStage1Result(result)
    complete(1)
    unlock(2)
    unlock(6)
  }, [complete, unlock])

  const onStage2Done = useCallback(() => {
    complete(2)
    unlock(3)
  }, [complete, unlock])

  const onStage3Done = useCallback((result) => {
    setStage3Result(result)
    complete(3)
    unlock(4)
  }, [complete, unlock])

  const onStage4Done = useCallback((result) => {
    setStage4Result(result)
    complete(4)
    unlock(5)
  }, [complete, unlock])

  const onStage5Done = useCallback((result) => {
    setStage5Result(result)
    complete(5)
    unlock(6)
  }, [complete, unlock])

  const onStage6Done = useCallback(() => {
    complete(6)
  }, [complete])

  useEffect(() => {
    fetch(`${API}/health`)
      .then(r => r.json())
      .then(d => setHealth(d.status === 'ok' ? 'ok' : 'error'))
      .catch(() => setHealth('error'))
  }, [])

  useEffect(() => {
    fetch(`${API}/pipeline/state`)
      .then(r => r.json())
      .then(data => {
        const stages = data.stages || {}

        if (stages['1']) {
          const r1 = {
            features_paths: data.features_paths || {},
            row_counts: data.row_counts || {},
            leagues_processed: data.leagues_processed || [],
          }
          setStage1Result(r1)
          complete(1)
          unlock(2)
          unlock(6)
        }
        if (stages['2']) { complete(2); unlock(3) }
        if (stages['3']) {
          setStage3Result({ selected_features: data.selected_features || [] })
          complete(3)
          unlock(4)
        }
        if (stages['4']) {
          const modelPaths = data.model_paths || {}
          const nameMap = { mlr: 'MLR', ridge: 'Ridge', lasso: 'Lasso' }
          const modelEntries = Object.entries(modelPaths).map(([stem, path]) => {
            const key = stem.replace(/_model$/, '').toLowerCase()
            return { name: nameMap[key] || stem, model_path: path }
          })
          setStage4Result({
            models: modelEntries,
            best_model: modelEntries[0]?.name || '',
            scaler_path: data.scaler_path || '',
            feature_count: 0,
          })
          complete(4)
          unlock(5)
        }
        if (stages['5']) { complete(5); unlock(6) }
      })
      .catch(() => {})
  }, [complete, unlock])

  return (
    <PipelineContext.Provider value={{
      currentStage, goToStage,
      unlockedStages, completedStages,
      health,
      stage1Result, stage3Result, stage4Result, stage5Result,
      onStage1Done, onStage2Done, onStage3Done,
      onStage4Done, onStage5Done, onStage6Done,
    }}>
      {children}
    </PipelineContext.Provider>
  )
}

export function usePipeline() {
  return useContext(PipelineContext)
}
