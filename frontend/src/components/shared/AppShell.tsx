'use client'
import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import Sidebar from '@/components/shared/Sidebar'
import { useAuth } from '@/hooks/useAuth'

export default function AppShell({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, init } = useAuth()
  const router = useRouter()

  useEffect(() => {
    init()
  }, [init])

  useEffect(() => {
    if (!isAuthenticated) router.push('/login')
  }, [isAuthenticated, router])

  if (!isAuthenticated) return null

  return (
    <div className="flex min-h-screen bg-background">
      <Sidebar />
      <main className="ml-60 flex-1 min-h-screen overflow-auto">
        {children}
      </main>
    </div>
  )
}
