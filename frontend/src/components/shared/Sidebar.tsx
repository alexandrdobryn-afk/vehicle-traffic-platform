'use client'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import {
  LayoutDashboard, Camera, PlayCircle, List, Car,
  BarChart2, AlertTriangle, Settings, Activity, LogOut,
  Shield, Zap, Database, Archive, ChevronDown, ChevronRight, FileVideo
} from 'lucide-react'
import { useAuth } from '@/hooks/useAuth'
import { cn } from '@/lib/utils'
import { useState } from 'react'
import { useTranslation } from '@/lib/i18n'

const NAV_INFERENCE = [
  { href: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { href: '/cameras', label: 'Cameras', icon: Camera },
  { href: '/videos', label: 'Videos', icon: FileVideo },
  { href: '/live', label: 'Live View', icon: PlayCircle },
  { href: '/events', label: 'Events', icon: List },
  { href: '/vehicles', label: 'Vehicles', icon: Car },
  { href: '/analytics', label: 'Analytics', icon: BarChart2 },
  { href: '/watchlist', label: 'Watchlist', icon: AlertTriangle },
  { href: '/settings', label: 'Settings', icon: Settings },
  { href: '/health', label: 'System Health', icon: Activity },
]

const NAV_TRAINING = [
  { href: '/training', label: 'Training Hub', icon: Zap },
  { href: '/training/datasets', label: 'Datasets', icon: Database },
  { href: '/training/jobs', label: 'Jobs', icon: PlayCircle },
  { href: '/training/registry', label: 'Model Registry', icon: Archive },
]

function NavItem({ href, label, icon: Icon }: { href: string; label: string; icon: any }) {
  const pathname = usePathname()
  const { t } = useTranslation()
  const active = pathname === href || (href !== '/training' && pathname.startsWith(href))
  return (
    <Link
      href={href}
      className={cn(
        'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors',
        active
          ? 'bg-primary/10 text-primary'
          : 'text-muted-foreground hover:text-foreground hover:bg-accent'
      )}
    >
      <Icon className="w-4 h-4 shrink-0" />
      {t(label)}
    </Link>
  )
}

export default function Sidebar() {
  const { logout, email, role } = useAuth()
  const { t } = useTranslation()
  const pathname = usePathname()
  const [trainingOpen, setTrainingOpen] = useState(pathname.startsWith('/training'))

  return (
    <aside className="fixed left-0 top-0 h-full w-60 bg-card border-r border-border flex flex-col z-40">
      {/* Logo */}
      <div className="p-5 border-b border-border">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-primary/10 rounded-lg flex items-center justify-center">
            <Shield className="w-4 h-4 text-primary" />
          </div>
          <div>
            <p className="text-sm font-bold text-foreground leading-tight">VTP</p>
            <p className="text-xs text-muted-foreground leading-tight">{t('Traffic Platform')}</p>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 p-3 space-y-0.5 overflow-y-auto">
        {/* Inference section */}
        <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide px-3 py-2 mt-1">
          {t('Monitoring')}
        </p>
        {NAV_INFERENCE.map((item) => (
          <NavItem key={item.href} {...item} />
        ))}

        {/* Training section */}
        <button
          onClick={() => setTrainingOpen(o => !o)}
          className="w-full flex items-center justify-between px-3 py-2 mt-3 text-xs font-semibold text-muted-foreground uppercase tracking-wide hover:text-foreground"
        >
          <span>{t('AI Training')}</span>
          {trainingOpen
            ? <ChevronDown className="w-3.5 h-3.5" />
            : <ChevronRight className="w-3.5 h-3.5" />}
        </button>
        {trainingOpen && NAV_TRAINING.map((item) => (
          <NavItem key={item.href} {...item} />
        ))}
      </nav>

      {/* User */}
      <div className="p-3 border-t border-border">
        <div className="flex items-center justify-between px-3 py-2">
          <div className="min-w-0">
            <p className="text-xs font-medium text-foreground truncate">{email}</p>
            <p className="text-xs text-muted-foreground capitalize">
              {role ? t(role.charAt(0).toUpperCase() + role.slice(1)) : ''}
            </p>
          </div>
          <button
            onClick={logout}
            className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
            title={t('Sign out')}
            aria-label={t('Sign out')}
          >
            <LogOut className="w-4 h-4" />
          </button>
        </div>
      </div>
    </aside>
  )
}
