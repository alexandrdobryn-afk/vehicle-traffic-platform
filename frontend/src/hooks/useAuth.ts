'use client'
import { create } from 'zustand'
import api from '@/lib/api'

interface AuthState {
  token: string | null
  role: string | null
  email: string | null
  isAuthenticated: boolean
  login: (email: string, password: string) => Promise<void>
  logout: () => void
  init: () => void
}

export const useAuth = create<AuthState>((set) => ({
  token: null,
  role: null,
  email: null,
  isAuthenticated: false,

  init: () => {
    if (typeof window === 'undefined') return
    const token = localStorage.getItem('vtp_token')
    const role = localStorage.getItem('vtp_role')
    const email = localStorage.getItem('vtp_email')
    if (token) set({ token, role, email, isAuthenticated: true })
  },

  login: async (email, password) => {
    const res = await api.post('/auth/login', { email, password })
    const { access_token, role } = res.data
    localStorage.setItem('vtp_token', access_token)
    localStorage.setItem('vtp_role', role)
    localStorage.setItem('vtp_email', email)
    set({ token: access_token, role, email, isAuthenticated: true })
  },

  logout: () => {
    localStorage.removeItem('vtp_token')
    localStorage.removeItem('vtp_role')
    localStorage.removeItem('vtp_email')
    set({ token: null, role: null, email: null, isAuthenticated: false })
    window.location.href = '/login'
  },
}))
