'use client'
import { useEffect, useRef, useState, useCallback } from 'react'
import { WSFrame } from '@/types'

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000'

interface UseWebSocketOptions {
  cameraId: number
  onFrame?: (frame: WSFrame) => void
  onAlert?: (alert: WSFrame) => void
  enabled?: boolean
}

export function useWebSocket({ cameraId, onFrame, onAlert, enabled = true }: UseWebSocketOptions) {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectRef = useRef<ReturnType<typeof setTimeout>>()
  const shouldReconnectRef = useRef(true)
  const [connected, setConnected] = useState(false)
  const [latestFrame, setLatestFrame] = useState<WSFrame | null>(null)

  const connect = useCallback(() => {
    if (!enabled) return
    const token = localStorage.getItem('vtp_token')
    if (!token) return

    const url = `${WS_URL}/ws/live/${cameraId}?token=${token}`
    const ws = new WebSocket(url)
    wsRef.current = ws
    let ping: ReturnType<typeof setInterval> | undefined

    ws.onopen = () => {
      setConnected(true)
      ping = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) ws.send('ping')
      }, 15000)
    }

    ws.onmessage = (e) => {
      try {
        const data: WSFrame = JSON.parse(e.data)
        if (data.type === 'alert') {
          onAlert?.(data)
        } else {
          setLatestFrame(data)
          onFrame?.(data)
        }
      } catch {}
    }

    ws.onclose = () => {
      if (ping) clearInterval(ping)
      setConnected(false)
      wsRef.current = null
      if (shouldReconnectRef.current) {
        reconnectRef.current = setTimeout(connect, 3000)
      }
    }

    ws.onerror = () => ws.close()
  }, [cameraId, enabled, onFrame, onAlert])

  useEffect(() => {
    shouldReconnectRef.current = true
    connect()
    return () => {
      shouldReconnectRef.current = false
      clearTimeout(reconnectRef.current)
      wsRef.current?.close()
    }
  }, [connect])

  return { connected, latestFrame }
}
