import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatConfidence(v: number | null | undefined) {
  const numeric = Number(v)
  return Number.isFinite(numeric) ? `${Math.round(numeric * 100)}%` : '-'
}

export function formatDuration(seconds: number, locale?: string) {
  const secondLabel = locale === 'uk-UA' ? 'с' : 's'
  const minuteLabel = locale === 'uk-UA' ? 'хв' : 'm'
  if (seconds < 60) return `${Math.round(seconds)}${secondLabel}`
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds % 60)
  return `${m}${minuteLabel} ${s}${secondLabel}`
}

export function formatDateTime(iso: string | null | undefined, locale?: string) {
  if (!iso) return '-'
  const date = new Date(iso)
  return Number.isNaN(date.getTime()) ? '-' : date.toLocaleString(locale)
}

export function formatTime(iso: string) {
  return new Date(iso).toLocaleTimeString()
}
