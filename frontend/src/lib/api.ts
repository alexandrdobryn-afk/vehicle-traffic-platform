import axios from 'axios'
import { browserEndpoint } from './runtimeUrl'

const CONFIGURED_API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8100'
const CONFIGURED_WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8100'

export const API_URL = browserEndpoint(CONFIGURED_API_URL, '8100')
export const WS_URL = browserEndpoint(CONFIGURED_WS_URL, '8100', true)

export function storageUrl(path: string | null | undefined): string | null {
  if (!path) return null
  const normalized = path.replace(/\\/g, '/').replace(/^.*\/storage\//, '/storage/')
  return normalized.startsWith('/storage/') ? `${API_URL}${normalized}` : null
}

export const api = axios.create({
  baseURL: `${API_URL}/api/v1`,
  headers: { 'Content-Type': 'application/json' },
  timeout: 20000,
})

export function apiErrorMessage(error: any, fallback: string) {
  const detail = error?.response?.data?.detail
  if (error?.response?.status === 401) return 'Session expired. Log in again.'
  if (error?.response?.status === 403) return 'Access denied for this API request. Log in again or use an account with the required role.'
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) return detail.map((item) => item?.msg || JSON.stringify(item)).join('; ')
  if (error?.code === 'ECONNABORTED') return 'API did not respond in time. Check that backend, database, cache and Docker/WSL are running.'
  if (error?.message === 'Network Error') return 'Backend API is unreachable. Check Docker/WSL and the backend port.'
  return error?.message || fallback
}

// Attach JWT token to every request
api.interceptors.request.use((config) => {
  if (typeof window !== 'undefined') {
    const token = localStorage.getItem('bevp_token')
    if (token) config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Redirect to login on 401
api.interceptors.response.use(
  (r) => r,
  (err) => {
    if (err.response?.status === 401 && typeof window !== 'undefined') {
      localStorage.removeItem('bevp_token')
      window.location.href = '/login'
    }
    return Promise.reject(err)
  }
)

export default api
