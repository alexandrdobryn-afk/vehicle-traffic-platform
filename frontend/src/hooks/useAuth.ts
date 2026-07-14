'use client'
import { create } from 'zustand'
import api from '@/lib/api'

interface AuthState {
  token: string | null
  role: string | null
  email: string | null
  isAuthenticated: boolean
  initialized: boolean
  login: (email: string, password: string) => Promise<void>
  logout: () => void
  init: () => void
}

export const useAuth = create<AuthState>((set) => ({
  token: null,
  role: null,
  email: null,
  isAuthenticated: false,
  initialized: false,

  init: () => {
    if (typeof window === 'undefined') return
    const token = localStorage.getItem('bevp_token')
    const role = localStorage.getItem('bevp_role')
    const email = localStorage.getItem('bevp_email')
    set({
      token,
      role,
      email,
      isAuthenticated: Boolean(token),
      initialized: true,
    })
  },

  login: async (email, password) => {
    const res = await api.post('/auth/login', { email, password })
    const { access_token, role } = res.data
    localStorage.setItem('bevp_token', access_token)
    localStorage.setItem('bevp_role', role)
    localStorage.setItem('bevp_email', email)
    set({ token: access_token, role, email, isAuthenticated: true, initialized: true })
  },

  logout: () => {
    localStorage.removeItem('bevp_token')
    localStorage.removeItem('bevp_role')
    localStorage.removeItem('bevp_email')
    set({ token: null, role: null, email: null, isAuthenticated: false, initialized: true })
    window.location.href = '/login'
  },
}))
