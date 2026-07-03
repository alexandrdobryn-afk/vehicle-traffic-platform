'use client'

import { useTranslation } from '@/lib/i18n'

export default function LanguageSwitcher() {
  const { language, setLanguage, t } = useTranslation()

  return (
    <div className="fixed right-6 top-5 z-[60] flex items-center rounded-lg border border-border bg-card/95 p-1 shadow-lg backdrop-blur" aria-label={t('Language')}>
      <button
        type="button"
        onClick={() => setLanguage('uk')}
        className={`rounded-md px-3 py-1.5 text-xs font-semibold transition-colors ${language === 'uk' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground'}`}
        aria-pressed={language === 'uk'}
        title={t('Ukrainian')}
      >
        UA
      </button>
      <button
        type="button"
        onClick={() => setLanguage('en')}
        className={`rounded-md px-3 py-1.5 text-xs font-semibold transition-colors ${language === 'en' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground'}`}
        aria-pressed={language === 'en'}
        title={t('English')}
      >
        EN
      </button>
    </div>
  )
}
