'use client'

import { useEffect, useState } from 'react'

import api from '@/lib/api'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export function useStreamUrl(
  sourceId: number | null,
  enabled = true,
  resource: 'stream' | 'snapshot' = 'stream',
) {
  const [url, setUrl] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined

    const issueTicket = async () => {
      if (!sourceId || !enabled) {
        setUrl(null)
        return
      }
      try {
        const response = await api.post(`/stream/${sourceId}/ticket`)
        if (!cancelled) {
          const suffix = resource === 'snapshot' ? '/snapshot' : ''
          setUrl(`${API_URL}/api/v1/stream/${sourceId}${suffix}?ticket=${encodeURIComponent(response.data.ticket)}`)
          timer = setTimeout(issueTicket, Math.max(30, response.data.expires_in - 30) * 1000)
        }
      } catch {
        if (!cancelled) {
          setUrl(null)
          timer = setTimeout(issueTicket, 3000)
        }
      }
    }

    issueTicket()
    return () => {
      cancelled = true
      if (timer) clearTimeout(timer)
    }
  }, [sourceId, enabled, resource])

  return url
}
