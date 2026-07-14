export function browserEndpoint(defaultUrl: string, port: string, ws = false): string {
  if (typeof window === 'undefined') return defaultUrl
  try {
    const configured = new URL(defaultUrl)
    const browserHost = window.location.hostname
    const configuredIsLocalhost = ['localhost', '127.0.0.1', '::1'].includes(configured.hostname)
    const browserIsLocalhost = ['localhost', '127.0.0.1', '::1'].includes(browserHost)
    if (configuredIsLocalhost && !browserIsLocalhost) {
      const protocol = ws ? (window.location.protocol === 'https:' ? 'wss:' : 'ws:') : window.location.protocol
      return `${protocol}//${browserHost}:${port}`
    }
  } catch {
    // Fall back to the configured URL below.
  }
  return defaultUrl
}
