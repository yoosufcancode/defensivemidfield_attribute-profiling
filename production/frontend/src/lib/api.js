export const API = '/api/v1'

export function startPolling(statusUrl, onProgress, onDone, onError) {
  const interval = setInterval(async () => {
    try {
      const r = await fetch(statusUrl)
      const data = await r.json()
      onProgress(data)
      if (data.status === 'completed') {
        clearInterval(interval)
        onDone(data.result)
      } else if (data.status === 'failed') {
        clearInterval(interval)
        onError(data.error || data.message)
      }
    } catch (e) {
      clearInterval(interval)
      onError(String(e))
    }
  }, 1000)
  return () => clearInterval(interval)
}
