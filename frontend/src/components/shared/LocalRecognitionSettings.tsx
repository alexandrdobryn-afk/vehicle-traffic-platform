'use client'

import { useCallback, useEffect, useId, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { HelpCircle } from 'lucide-react'
import { useTranslation } from '@/lib/i18n'

export const LOCAL_RECOGNITION_DEFAULTS = {
  vehicle_confidence_threshold: 0.45,
  plate_confidence_threshold: 0.40,
  ocr_threshold: 0.60,
  plate_regex_profile: 'AUTO',
  input_resolution: '1280x720',
  ocr_voting_window: 10,
  track_missing_grace_frames: 15,
  minimum_plate_width: 60,
  minimum_plate_height: 20,
  frame_skip: 0,
}

const SETTING_HELP: Record<string, {
  description: string
  recommendation: string
  load: string
}> = {
  vehicle_confidence_threshold: {
    description: 'Vehicle confidence help',
    recommendation: 'Vehicle confidence recommendation',
    load: 'Vehicle confidence load',
  },
  plate_confidence_threshold: {
    description: 'Plate confidence help',
    recommendation: 'Plate confidence recommendation',
    load: 'Plate confidence load',
  },
  ocr_threshold: {
    description: 'OCR confidence help',
    recommendation: 'OCR confidence recommendation',
    load: 'OCR confidence load',
  },
  plate_regex_profile: {
    description: 'Plate profile help',
    recommendation: 'Plate profile recommendation',
    load: 'Plate profile load',
  },
  input_resolution: {
    description: 'Input resolution help',
    recommendation: 'Input resolution recommendation',
    load: 'Input resolution load',
  },
  ocr_voting_window: {
    description: 'Voting window help',
    recommendation: 'Voting window recommendation',
    load: 'Voting window load',
  },
  track_missing_grace_frames: {
    description: 'Track grace help',
    recommendation: 'Track grace recommendation',
    load: 'Track grace load',
  },
  minimum_plate_width: {
    description: 'Minimum plate width help',
    recommendation: 'Minimum plate width recommendation',
    load: 'Minimum plate size load',
  },
  minimum_plate_height: {
    description: 'Minimum plate height help',
    recommendation: 'Minimum plate height recommendation',
    load: 'Minimum plate size load',
  },
  frame_skip: {
    description: 'Frame skip override help',
    recommendation: 'Frame skip override recommendation',
    load: 'Frame skip override load',
  },
}

const ENGLISH_SETTING_HELP: typeof SETTING_HELP = {
  vehicle_confidence_threshold: {
    description: 'Controls which vehicle detections enter tracking. A higher value reduces false detections but may miss distant, dark, or partly hidden vehicles. A lower value finds more objects but creates more false tracks.',
    recommendation: '0.40–0.55; 0.45 is the balanced value.',
    load: 'Indirect. Lower values can increase tracking, plate detection, and OCR work; higher values reduce it.',
  },
  plate_confidence_threshold: {
    description: 'Controls which plate candidates go to quality checks and OCR. A higher value rejects more blurred or distant plates; a lower value sends more weak and false crops to OCR.',
    recommendation: '0.35–0.50; 0.40 is the balanced value.',
    load: 'Moderate and indirect. A lower threshold runs OCR for more candidates.',
  },
  ocr_threshold: {
    description: 'Minimum confidence required to accept an OCR result. A higher value produces fewer wrong plates but more unread results. A lower value accepts more results with a greater error risk.',
    recommendation: '0.55–0.70; 0.60 is the balanced value.',
    load: 'Almost none: this changes acceptance of an OCR result that has already been produced.',
  },
  plate_regex_profile: {
    description: 'Validates and normalizes recognized text using a country format. A wrong profile can reject valid plates.',
    recommendation: 'AUTO for mixed traffic; UA for cameras that mostly observe Ukrainian plates.',
    load: 'Negligible; this is a fast text check after OCR.',
  },
  input_resolution: {
    description: 'Frames are resized to this resolution before detection. Higher resolution preserves more detail in small and distant plates; lower resolution improves FPS but loses detail.',
    recommendation: '1280×720 for balance; 1920×1080 for distant or small plates; 640×360 only when throughput is constrained.',
    load: 'High and direct. 1080p has 2.25× as many pixels as 720p; 360p has 4× fewer.',
  },
  ocr_voting_window: {
    description: 'Number of OCR observations from one track used to choose the final plate. A larger window is more resistant to a bad frame but takes longer to stabilize.',
    recommendation: '8–12; 10 is the balanced value.',
    load: 'Low. A larger window stores more candidates but does not itself run extra OCR.',
  },
  track_missing_grace_frames: {
    description: 'Number of processed frames for which a track survives after the object disappears. A higher value handles occlusion better but keeps stale tracks longer. A lower value can split one vehicle into several tracks.',
    recommendation: '10–20 processed frames; 15 is the default.',
    load: 'Low. Higher values retain inactive track state for longer.',
  },
  minimum_plate_width: {
    description: 'Plate crops narrower than this are rejected before OCR. A higher minimum avoids weak readings but misses distant plates.',
    recommendation: '50–80 pixels; 60 is balanced for 720p.',
    load: 'Moderate and indirect. Lower limits send more small crops to OCR.',
  },
  minimum_plate_height: {
    description: 'Plate crops shorter than this are rejected before OCR. A higher minimum avoids weak readings but misses distant plates.',
    recommendation: '16–24 pixels; 20 is balanced for 720p.',
    load: 'Moderate and indirect. Lower limits send more small crops to OCR.',
  },
  frame_skip: {
    description: '0 uses the selected mode default; 1 processes every frame; 2 every second frame; 10 every tenth frame. Higher values improve FPS but can miss short appearances and reduce temporal voting data.',
    recommendation: '0 for normal operation; 1 only for maximum quality when enough compute is available.',
    load: 'Very high and direct. 1 is maximum load; increasing the number reduces processed frames almost proportionally.',
  },
}

function SettingLabel({
  fieldId,
  label,
  settingKey,
}: {
  fieldId: string
  label: string
  settingKey: string
}) {
  const { t, language } = useTranslation()
  const buttonRef = useRef<HTMLButtonElement>(null)
  const [isOpen, setIsOpen] = useState(false)
  const [position, setPosition] = useState({ top: 16, left: 16, width: 320 })
  const help = SETTING_HELP[settingKey]
  const copy = language === 'uk' ? {
    description: t(help.description),
    recommendation: t(help.recommendation),
    load: t(help.load),
  } : ENGLISH_SETTING_HELP[settingKey]

  const openTooltip = useCallback(() => {
    const button = buttonRef.current
    if (!button || typeof window === 'undefined') return

    const rect = button.getBoundingClientRect()
    const edge = 16
    const gap = 12
    const width = Math.min(320, window.innerWidth - edge * 2)
    let left = rect.right + gap

    if (left + width > window.innerWidth - edge) left = rect.left - width - gap
    if (left < edge) left = Math.max(edge, Math.min(window.innerWidth - width - edge, rect.left))

    const estimatedHeight = Math.min(380, window.innerHeight * 0.7)
    const top = Math.max(edge, Math.min(rect.top - 40, window.innerHeight - estimatedHeight - edge))
    setPosition({ top, left, width })
    setIsOpen(true)
  }, [])

  useEffect(() => {
    if (!isOpen) return
    const close = () => setIsOpen(false)
    window.addEventListener('resize', close)
    window.addEventListener('scroll', close, true)
    return () => {
      window.removeEventListener('resize', close)
      window.removeEventListener('scroll', close, true)
    }
  }, [isOpen])

  const tooltip = isOpen && typeof document !== 'undefined' ? createPortal(
    <div
      id={`${fieldId}-help`}
      role="tooltip"
      style={position}
      className="pointer-events-none fixed z-[200] max-h-[70vh] overflow-y-auto rounded-xl border border-primary/30 bg-slate-950 p-4 text-left text-xs leading-5 text-slate-200 opacity-100 shadow-[0_18px_60px_rgba(0,0,0,0.75)] ring-1 ring-white/5"
    >
      <div className="flex items-center gap-2 border-b border-white/10 pb-2">
        <HelpCircle className="h-4 w-4 shrink-0 text-primary" aria-hidden="true" />
        <span className="font-semibold text-white">{t(label)}</span>
      </div>
      <p className="mt-3">{copy.description}</p>
      <p className="mt-3"><strong className="text-white">{t('Recommended')}:</strong> {copy.recommendation}</p>
      <p className="mt-2"><strong className="text-white">{t('System load')}:</strong> {copy.load}</p>
    </div>,
    document.body,
  ) : null

  return (
    <div className="flex items-start gap-1.5">
      <label id={`${fieldId}-label`} htmlFor={fieldId} className="leading-4 text-muted-foreground">
        {t(label)}
      </label>
      <span className="relative inline-flex shrink-0">
        <button
          ref={buttonRef}
          type="button"
          className="mt-0.5 rounded-full text-muted-foreground/70 outline-none transition-colors hover:text-primary focus-visible:text-primary focus-visible:ring-2 focus-visible:ring-primary/50"
          aria-label={`${t('About this setting')}: ${t(label)}`}
          aria-describedby={`${fieldId}-help`}
          aria-expanded={isOpen}
          onMouseEnter={openTooltip}
          onMouseLeave={() => setIsOpen(false)}
          onFocus={openTooltip}
          onBlur={() => setIsOpen(false)}
          onClick={() => isOpen ? setIsOpen(false) : openTooltip()}
        >
          <HelpCircle className="h-3.5 w-3.5" aria-hidden="true" />
        </button>
        {tooltip}
      </span>
    </div>
  )
}

export default function LocalRecognitionSettings({
  value,
  onChange,
}: {
  value: Record<string, any>
  onChange: (value: Record<string, any>) => void
}) {
  const { t } = useTranslation()
  const idPrefix = useId()
  const set = (key: string, next: string | number) => onChange({ ...value, [key]: next })
  const input = 'w-full px-3 py-2 bg-background border border-input rounded-lg text-sm text-foreground'

  return (
    <details className="rounded-xl border border-border bg-muted/20 p-4" open>
      <summary className="cursor-pointer text-sm font-medium text-foreground">{t('Recognition settings for this source')}</summary>
      <p className="mt-2 text-xs leading-5 text-muted-foreground">
        {t('These values apply only to this camera or video and are not overridden by Global Settings.')}
      </p>
      <div className="mt-4 grid grid-cols-2 gap-3">
        {[
          ['vehicle_confidence_threshold', 'Vehicle Confidence Threshold', 0.1, 0.99, 0.05],
          ['plate_confidence_threshold', 'Plate Confidence Threshold', 0.1, 0.99, 0.05],
          ['ocr_threshold', 'OCR Confidence Threshold', 0.1, 0.99, 0.05],
        ].map(([key, label, min, max, step]) => {
          const fieldId = `${idPrefix}-${String(key)}`
          return (
          <div key={String(key)} className="text-xs text-muted-foreground">
            <SettingLabel fieldId={fieldId} label={String(label)} settingKey={String(key)} />
            <input id={fieldId} aria-labelledby={`${fieldId}-label`} className={`${input} mt-1`} type="number" min={Number(min)} max={Number(max)} step={Number(step)} value={value[String(key)]}
              onChange={(e) => set(String(key), Number(e.target.value))} />
          </div>
          )
        })}
        <div className="text-xs text-muted-foreground">
          <SettingLabel fieldId={`${idPrefix}-plate_regex_profile`} label="Regional Plate Profile" settingKey="plate_regex_profile" />
          <select id={`${idPrefix}-plate_regex_profile`} aria-labelledby={`${idPrefix}-plate_regex_profile-label`} className={`${input} mt-1`} value={value.plate_regex_profile} onChange={(e) => set('plate_regex_profile', e.target.value)}>
            {['AUTO', 'UA', 'UK', 'IN', 'EU', 'US'].map((profile) => <option key={profile} value={profile}>{profile}</option>)}
          </select>
        </div>
        <div className="text-xs text-muted-foreground">
          <SettingLabel fieldId={`${idPrefix}-input_resolution`} label="Input Resolution" settingKey="input_resolution" />
          <select id={`${idPrefix}-input_resolution`} aria-labelledby={`${idPrefix}-input_resolution-label`} className={`${input} mt-1`} value={value.input_resolution} onChange={(e) => set('input_resolution', e.target.value)}>
            <option value="640x360">640x360</option><option value="1280x720">1280x720</option><option value="1920x1080">1920x1080</option>
          </select>
        </div>
        {[
          ['ocr_voting_window', 'Voting Window', 3, 30],
          ['track_missing_grace_frames', 'Track Missing Grace', 1, 120],
          ['minimum_plate_width', 'Min Plate Width (px)', 20, 300],
          ['minimum_plate_height', 'Min Plate Height (px)', 10, 150],
          ['frame_skip', 'Frame Skip Override (0 = mode default)', 0, 10],
        ].map(([key, label, min, max]) => {
          const fieldId = `${idPrefix}-${String(key)}`
          return (
          <div key={String(key)} className="text-xs text-muted-foreground">
            <SettingLabel fieldId={fieldId} label={String(label)} settingKey={String(key)} />
            <input id={fieldId} aria-labelledby={`${fieldId}-label`} className={`${input} mt-1`} type="number" min={Number(min)} max={Number(max)} value={value[String(key)]}
              onChange={(e) => set(String(key), Number(e.target.value))} />
          </div>
          )
        })}
      </div>
    </details>
  )
}
