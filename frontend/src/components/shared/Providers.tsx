'use client'

import { LanguageProvider } from '@/lib/i18n'
import LanguageSwitcher from './LanguageSwitcher'

export default function Providers({ children }: { children: React.ReactNode }) {
  return (
    <LanguageProvider>
      <LanguageSwitcher />
      {children}
    </LanguageProvider>
  )
}
