'use client'

import { useEffect, useState } from 'react'
import trainingApi from '@/lib/trainingApi'

export function useTrainingImageUrl(endpoint?: string) {
  const [url, setUrl] = useState<string | null>(null)
  const [error, setError] = useState(false)

  useEffect(() => {
    if (!endpoint) {
      setUrl(null)
      setError(false)
      return
    }

    const controller = new AbortController()
    let objectUrl: string | null = null
    setUrl(null)
    setError(false)

    trainingApi.get(endpoint, {
      responseType: 'blob',
      signal: controller.signal,
    }).then(response => {
      objectUrl = URL.createObjectURL(response.data)
      setUrl(objectUrl)
    }).catch(err => {
      if (err?.code !== 'ERR_CANCELED') setError(true)
    })

    return () => {
      controller.abort()
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [endpoint])

  return { url, error, loading: Boolean(endpoint) && !url && !error }
}

interface AuthenticatedTrainingImageProps {
  endpoint: string
  alt: string
  className?: string
}

export default function AuthenticatedTrainingImage({
  endpoint,
  alt,
  className = '',
}: AuthenticatedTrainingImageProps) {
  const { url, error } = useTrainingImageUrl(endpoint)

  if (error) {
    return (
      <div
        role="img"
        aria-label={alt}
        className={`${className} bg-red-500/10 border border-red-500/30`}
      />
    )
  }

  if (!url) {
    return <div aria-hidden="true" className={`${className} bg-muted animate-pulse`} />
  }

  return <img src={url} alt={alt} loading="lazy" className={className} />
}
