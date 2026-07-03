import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatConfidence(v: number) {
  return `${Math.round(v * 100)}%`
}

export function formatDuration(seconds: number, locale?: string) {
  const secondLabel = locale === 'uk-UA' ? 'с' : 's'
  const minuteLabel = locale === 'uk-UA' ? 'хв' : 'm'
  if (seconds < 60) return `${Math.round(seconds)}${secondLabel}`
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds % 60)
  return `${m}${minuteLabel} ${s}${secondLabel}`
}

export function formatDateTime(iso: string, locale?: string) {
  return new Date(iso).toLocaleString(locale)
}

export function formatTime(iso: string) {
  return new Date(iso).toLocaleTimeString()
}
