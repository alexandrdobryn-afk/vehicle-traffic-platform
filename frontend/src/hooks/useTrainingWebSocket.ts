'use client'
import { useEffect, useRef, useState, useCallback } from 'react'
import { TrainingProgress } from '@/types/training'
import { TRAINING_WS_URL } from '@/lib/trainingApi'

export function useTrainingWebSocket(jobId: number | null) {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectRef = useRef<ReturnType<typeof setTimeout>>()
  const shouldReconnectRef = useRef(true)
  const terminalRef = useRef(false)
  const [connected, setConnected] = useState(false)
  const [progress, setProgress] = useState<TrainingProgress | null>(null)

  const connect = useCallback(() => {
    if (!jobId) return
    const token = localStorage.getItem('vtp_token')
    if (!token) return

    const url = `${TRAINING_WS_URL}/ws/training/${jobId}?token=${token}`
    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => setConnected(true)

    ws.onmessage = (e) => {
      if (e.data === 'pong') return
      try {
        const data = JSON.parse(e.data)
        if (data.type === 'heartbeat') return
        terminalRef.current = ['completed', 'failed', 'cancelled'].includes(data.status)
        setProgress(data as TrainingProgress)
      } catch {}
    }

    ws.onclose = () => {
      setConnected(false)
      wsRef.current = null
      // Don't reconnect if job is terminal
      if (shouldReconnectRef.current && !terminalRef.current) {
        reconnectRef.current = setTimeout(connect, 4000)
      }
    }

    ws.onerror = () => ws.close()
  }, [jobId])

  useEffect(() => {
    shouldReconnectRef.current = true
    terminalRef.current = false
    connect()
    return () => {
      shouldReconnectRef.current = false
      clearTimeout(reconnectRef.current)
      wsRef.current?.close()
    }
  }, [connect])

  return { connected, progress }
}
