import { useState, useCallback, useRef } from 'react'
import { startPolling } from '../lib/api'

export function usePoll() {
  const [polling, setPolling] = useState(false)
  const [progress, setProgress] = useState(0)
  const [message, setMessage] = useState('')
  const [error, setError] = useState(null)
  const stopRef = useRef(null)

  const poll = useCallback((statusUrl, onDone) => {
    setPolling(true)
    setProgress(0)
    setMessage('Starting…')
    setError(null)

    stopRef.current = startPolling(
      statusUrl,
      (data) => {
        setProgress(data.progress || 0)
        setMessage(data.message || '…')
      },
      (result) => {
        setPolling(false)
        setProgress(100)
        setMessage('Complete')
        onDone(result)
      },
      (err) => {
        setPolling(false)
        setError(err)
      }
    )
  }, [])

  const reset = useCallback(() => {
    stopRef.current?.()
    setPolling(false)
    setProgress(0)
    setMessage('')
    setError(null)
  }, [])

  return { polling, progress, message, error, setError, poll, reset }
}
