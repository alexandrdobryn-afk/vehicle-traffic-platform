import axios from 'axios'

const TRAINING_URL = process.env.NEXT_PUBLIC_TRAINING_API_URL || 'http://localhost:8001'

export const trainingApi = axios.create({
  baseURL: `${TRAINING_URL}/api/v1/training`,
  headers: { 'Content-Type': 'application/json' },
})

trainingApi.interceptors.request.use((config) => {
  if (typeof window !== 'undefined') {
    const token = localStorage.getItem('vtp_token')
    if (token) config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

trainingApi.interceptors.response.use(
  (r) => r,
  (err) => {
    if (err.response?.status === 401 && typeof window !== 'undefined') {
      localStorage.removeItem('vtp_token')
      window.location.href = '/login'
    }
    return Promise.reject(err)
  }
)

export const TRAINING_WS_URL =
  process.env.NEXT_PUBLIC_TRAINING_WS_URL || 'ws://localhost:8001'

export default trainingApi
