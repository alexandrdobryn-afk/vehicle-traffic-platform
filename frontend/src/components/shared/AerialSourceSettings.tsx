'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { HelpCircle } from 'lucide-react'

import { useTranslation } from '@/lib/i18n'

export const AERIAL_SOURCE_DEFAULTS = {
  object_confidence_threshold: 0.55,
  frame_skip: 0,
  track_missing_grace_frames: 6,
  min_track_frames_for_event: 4,
  min_track_duration_seconds: 0,
  min_box_width: 14,
  min_box_height: 14,
  min_box_area_ratio: 0.00008,
  max_box_area_ratio: 0.25,
  max_box_aspect_ratio: 6.0,
  target_classes: [] as string[],
  aerial: {
    enabled: true,
    tile_size: 1024,
    tile_overlap: 0.2,
    nms_iou: 0.35,
    enhance: false,
  },
  object_memory: {
    enabled: true,
    max_gap_frames: 90,
    merge_threshold: 0.48,
    duplicate_iou: 0.45,
    duplicate_contained: 0.78,
    appearance_threshold: 0.68,
  },
  kalman_prediction: { enabled: true, smoothing: true, max_prediction_frames: 2 },
  classification: { enabled: false, confidence_threshold: 0.35, interval_frames: 15 },
  segmentation: { enabled: false, confidence_threshold: 0.35, interval_frames: 5 },
  ocr: { enabled: false, engine: 'auto', confidence_threshold: 0.35, interval_frames: 30 },
  reid: { enabled: false, model: 'hsv_histogram_v1', similarity_threshold: 0.72, store_vector: false },
  geo: { enabled: false, telemetry_source: 'metadata', coordinate_output: false },
  super_resolution: { enabled: false, engine: 'auto', min_object_size_px: 32, max_crops_per_frame: 8 },
}

type Config = Record<string, any>

function HelpTooltip({ text }: { text: string }) {
  const { t } = useTranslation()
  const anchorRef = useRef<HTMLButtonElement | null>(null)
  const [open, setOpen] = useState(false)
  const [position, setPosition] = useState<{ left: number; top: number } | null>(null)
  const translated = t(text)

  const updatePosition = useCallback(() => {
    const rect = anchorRef.current?.getBoundingClientRect()
    if (!rect) return
    const width = Math.min(320, window.innerWidth - 24)
    const left = Math.min(Math.max(12, rect.left + rect.width / 2 - width / 2), window.innerWidth - width - 12)
    const top = Math.min(rect.bottom + 8, window.innerHeight - 120)
    setPosition({ left, top })
  }, [])

  const show = () => {
    updatePosition()
    setOpen(true)
  }

  useEffect(() => {
    if (!open) return
    updatePosition()
    window.addEventListener('resize', updatePosition)
    window.addEventListener('scroll', updatePosition, true)
    return () => {
      window.removeEventListener('resize', updatePosition)
      window.removeEventListener('scroll', updatePosition, true)
    }
  }, [open, updatePosition])

  return (
    <>
      <button
        ref={anchorRef}
        type="button"
        className="inline-flex shrink-0 cursor-help items-center rounded-full text-muted-foreground/80 transition-colors hover:text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/60"
        onClick={(event) => {
          event.preventDefault()
          event.stopPropagation()
        }}
        onMouseDown={(event) => {
          event.preventDefault()
          event.stopPropagation()
        }}
        onMouseEnter={show}
        onMouseLeave={() => setOpen(false)}
        onFocus={show}
        onBlur={() => setOpen(false)}
        aria-label={translated}
      >
        <HelpCircle className="h-3.5 w-3.5" />
      </button>
      {open && position && createPortal(
        <div
          className="pointer-events-none fixed z-[9999] max-w-[min(20rem,calc(100vw-1.5rem))] rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-left text-xs font-normal leading-5 text-slate-50 shadow-2xl shadow-black/60"
          style={{ left: position.left, top: position.top }}
        >
          {translated}
        </div>,
        document.body,
      )}
    </>
  )
}

function FieldLabel({ label, help }: { label: string; help: string }) {
  const { t } = useTranslation()
  return (
    <span className="mb-1 flex items-center gap-1.5 text-xs text-muted-foreground">
      {t(label)}
      <HelpTooltip text={help} />
    </span>
  )
}

function NumberField({
  label,
  help,
  value,
  onChange,
  min,
  max,
  step,
}: {
  label: string
  help: string
  value: number
  onChange: (value: number) => void
  min?: number
  max?: number
  step?: number
}) {
  return (
    <label className="block">
      <FieldLabel label={label} help={help} />
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="w-full rounded-lg border border-input bg-background px-3 py-2 text-sm"
      />
    </label>
  )
}

function ToggleRow({
  label,
  help,
  checked,
  onChange,
}: {
  label: string
  help: string
  checked: boolean
  onChange: (checked: boolean) => void
}) {
  const { t } = useTranslation()
  return (
    <label className="flex items-center justify-between gap-3 rounded-lg border border-border/60 bg-background/40 px-3 py-2 text-sm">
      <span className="flex items-center gap-1.5">
        {t(label)}
        <HelpTooltip text={help} />
      </span>
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
    </label>
  )
}

export default function AerialSourceSettings({ value, onChange }: { value: Config; onChange: (next: Config) => void }) {
  const { t } = useTranslation()
  const update = (patch: Config) => onChange({ ...value, ...patch })
  const updateModule = (name: string, patch: Config) => update({ [name]: { ...(value[name] || {}), ...patch } })
  const aerial = value.aerial || AERIAL_SOURCE_DEFAULTS.aerial

  return (
    <div className="grid gap-4 xl:grid-cols-2 xl:items-start">
      <section className="space-y-3 rounded-xl border border-cyan-500/25 bg-cyan-500/5 p-4">
        <div>
          <p className="text-sm font-medium text-foreground">{t('Detection and tiling')}</p>
          <p className="mt-1 text-xs leading-5 text-muted-foreground">
            {t('These settings control what the detector sees before tracking starts. Empty classes means the model may return any class it knows.')}
          </p>
        </div>

        <ToggleRow
          label="Enable tiled inference"
          help="Tiling keeps small objects visible by processing high-resolution frame fragments instead of shrinking the whole image."
          checked={Boolean(aerial.enabled)}
          onChange={(checked) => updateModule('aerial', { enabled: checked })}
        />

        <div className="grid grid-cols-2 gap-3">
          <NumberField
            label="Detection confidence"
            help="Minimum detector confidence. Higher values reduce false positives but can miss weak or distant objects."
            min={0.05}
            max={0.99}
            step={0.05}
            value={value.object_confidence_threshold ?? 0.55}
            onChange={(next) => update({ object_confidence_threshold: next })}
          />
          <NumberField
            label="Frame skip"
            help="0 uses the selected mode default. Higher values process fewer frames and run faster, but short events can be missed."
            min={0}
            max={10}
            value={value.frame_skip ?? 0}
            onChange={(next) => update({ frame_skip: next })}
          />
        </div>

        {Boolean(aerial.enabled) && (
          <div className="grid grid-cols-2 gap-3 rounded-lg border border-cyan-500/20 bg-background/35 p-3">
            <NumberField
              label="Tile size"
              help="Pixel size of each inference tile. Larger tiles see more context; smaller tiles can help tiny objects."
              min={320}
              max={2048}
              step={64}
              value={aerial.tile_size ?? 1024}
              onChange={(next) => updateModule('aerial', { tile_size: next })}
            />
            <NumberField
              label="Tile overlap"
              help="Overlap between neighboring tiles. More overlap reduces edge misses but costs more compute."
              min={0}
              max={0.5}
              step={0.05}
              value={aerial.tile_overlap ?? 0.2}
              onChange={(next) => updateModule('aerial', { tile_overlap: next })}
            />
            <div className="col-span-2">
              <ToggleRow
                label="Image enhancement"
                help="Optional contrast/sharpness enhancement before detection. Use it only when the source is blurry or low contrast."
                checked={Boolean(aerial.enhance)}
                onChange={(checked) => updateModule('aerial', { enhance: checked })}
              />
            </div>
          </div>
        )}

        <label className="block">
          <FieldLabel
            label="Target object classes"
            help="Comma-separated class names. Leave empty for universal detection; fill it only when this source must ignore other classes."
          />
          <input
            value={(value.target_classes || []).join(', ')}
            onChange={(event) => update({ target_classes: event.target.value.split(',').map((item) => item.trim()).filter(Boolean) })}
            placeholder={t('person, car, building, road (empty = all model classes)')}
            className="w-full rounded-lg border border-input bg-background px-3 py-2 text-sm"
          />
        </label>

        <div className="grid grid-cols-2 gap-3">
          <NumberField
            label="Confirm track after N frames"
            help="A new object becomes a real event only after it appears for this many processed frames."
            min={1}
            max={120}
            value={value.min_track_frames_for_event ?? 4}
            onChange={(next) => update({ min_track_frames_for_event: next })}
          />
          <NumberField
            label="Minimum object size, px"
            help="Small boxes below this width and height are discarded as fragments/noise."
            min={1}
            max={4096}
            value={value.min_box_width ?? 14}
            onChange={(next) => update({ min_box_width: next, min_box_height: next })}
          />
        </div>

      </section>

      <section className="space-y-4 rounded-xl border border-violet-500/25 bg-violet-500/5 p-4">
        <div>
          <p className="text-sm font-medium text-foreground">{t('Tracking and optional modules')}</p>
          <p className="mt-1 text-xs leading-5 text-muted-foreground">
            {t('Tracking is always part of the pipeline. Optional modules run only when enabled and when their model artifact is installed.')}
          </p>
        </div>

        <div className="grid gap-3 lg:grid-cols-2 xl:grid-cols-1 2xl:grid-cols-2">
          <ModuleRow
            name="object_memory"
            label="Object Memory"
            help="Builds a canonical object card above the tracker, merging short ID resets and suppressing duplicate report rows."
            config={value.object_memory}
            update={updateModule}
          />
          <ModuleRow
            name="kalman_prediction"
            label="Kalman prediction"
            help="Keeps object IDs stable across very short detector gaps. It should stay enabled for video and camera sources."
            config={value.kalman_prediction}
            update={updateModule}
          />
          <ModuleRow
            name="classification"
            label="Object classification"
            help="Runs YOLO11n-cls ImageNet starter or the project-trained ONNX classifier on object crops."
            config={value.classification}
            update={updateModule}
            interval
          />
          <ModuleRow
            name="segmentation"
            label="Instance segmentation"
            help="Runs YOLO11n-seg COCO starter or the project-trained segmenter to produce masks and polygons."
            config={value.segmentation}
            update={updateModule}
            interval
          />
          <ModuleRow
            name="ocr"
            label="OCR text recognition"
            help="Reads visible text in detected crops. Keep it off unless text is part of the task."
            config={value.ocr}
            update={updateModule}
            interval
          />
          <ModuleRow
            name="reid"
            label="Embedding / ReID"
            help="Creates an object embedding vector for Object Memory matching. Current implementation uses HSV histogram embeddings; OSNet/FastReID is the production upgrade path."
            config={value.reid}
            update={updateModule}
          />
          <ModuleRow
            name="geo"
            label="Geo telemetry"
            help="Attaches image-space or telemetry-backed coordinates to objects when source metadata is available."
            config={value.geo}
            update={updateModule}
          />
          <ModuleRow
            name="super_resolution"
            label="Super resolution"
            help="Uses OpenCV Lanczos crop upscaling now; Real-ESRGAN/SwinIR can be added later for neural SR."
            config={value.super_resolution}
            update={updateModule}
          />
        </div>
      </section>
    </div>
  )
}

function ModuleRow({
  name,
  label,
  help,
  config = {},
  update,
  interval = false,
}: {
  name: string
  label: string
  help: string
  config?: Config
  update: (name: string, patch: Config) => void
  interval?: boolean
}) {
  const { t } = useTranslation()
  return (
    <div className="rounded-lg border border-border/70 bg-background/50 p-3">
      <label className="flex items-center justify-between gap-3 text-sm font-medium">
        <span className="flex items-center gap-1.5">
          {t(label)}
          <HelpTooltip text={help} />
        </span>
        <input type="checkbox" checked={Boolean(config.enabled)} onChange={(event) => update(name, { enabled: event.target.checked })} />
      </label>
      <p className="mt-1 text-xs leading-5 text-muted-foreground">{t(help)}</p>
      {config.enabled && (
        <div className="mt-3 grid grid-cols-2 gap-3">
          {name !== 'kalman_prediction' && name !== 'object_memory' && name !== 'reid' && (
            <NumberField
              label="Confidence threshold"
              help="Minimum confidence accepted from this optional module."
              min={0}
              max={1}
              step={0.05}
              value={config.confidence_threshold ?? 0.35}
              onChange={(next) => update(name, { confidence_threshold: next })}
            />
          )}
          {interval && (
            <NumberField
              label="Run every N frames"
              help="How often this optional module runs on tracked objects."
              min={1}
              max={300}
              value={config.interval_frames ?? 5}
              onChange={(next) => update(name, { interval_frames: next })}
            />
          )}
          {name === 'kalman_prediction' && (
            <NumberField
              label="Maximum prediction frames"
              help="How many frames a track may be predicted without a fresh detector box. Keep this low to avoid duplicate phantom tracks."
              min={0}
              max={10}
              value={config.max_prediction_frames ?? 2}
              onChange={(next) => update(name, { max_prediction_frames: next })}
            />
          )}
          {name === 'object_memory' && (
            <NumberField
              label="Memory gap frames"
              help="How long an object card remains eligible for merging after the tracker loses it."
              min={1}
              max={600}
              value={config.max_gap_frames ?? 90}
              onChange={(next) => update(name, { max_gap_frames: next })}
            />
          )}
          {name === 'object_memory' && (
            <NumberField
              label="Merge threshold"
              help="Minimum association score for treating a new tracker ID as the same physical object."
              min={0}
              max={1}
              step={0.01}
              value={config.merge_threshold ?? 0.48}
              onChange={(next) => update(name, { merge_threshold: next })}
            />
          )}
          {name === 'object_memory' && (
            <NumberField
              label="Duplicate IoU"
              help="Same-frame boxes above this overlap are treated as duplicate views of one object."
              min={0}
              max={1}
              step={0.01}
              value={config.duplicate_iou ?? 0.45}
              onChange={(next) => update(name, { duplicate_iou: next })}
            />
          )}
          {name === 'object_memory' && (
            <NumberField
              label="Appearance threshold"
              help="Minimum crop-signature similarity used when geometry alone is not enough."
              min={0}
              max={1}
              step={0.01}
              value={config.appearance_threshold ?? 0.68}
              onChange={(next) => update(name, { appearance_threshold: next })}
            />
          )}
          {name === 'reid' && (
            <NumberField
              label="Similarity threshold"
              help="Higher values require more similar embeddings before tracks are reconnected."
              min={0}
              max={1}
              step={0.01}
              value={config.similarity_threshold ?? 0.72}
              onChange={(next) => update(name, { similarity_threshold: next })}
            />
          )}
          {name === 'reid' && (
            <label className="col-span-2 flex items-center justify-between gap-3 text-xs text-muted-foreground">
              <span>{t('Store embedding vector in object card')}</span>
              <input type="checkbox" checked={Boolean(config.store_vector)} onChange={(event) => update(name, { store_vector: event.target.checked })} />
            </label>
          )}
          {name === 'geo' && (
            <label className="col-span-2 flex items-center justify-between gap-3 text-xs text-muted-foreground">
              <span>{t('Output object coordinates')}</span>
              <input type="checkbox" checked={Boolean(config.coordinate_output)} onChange={(event) => update(name, { coordinate_output: event.target.checked })} />
            </label>
          )}
          {name === 'super_resolution' && (
            <NumberField
              label="Minimum crop size"
              help="Only crops smaller than this are upscaled before optional analysis."
              min={4}
              max={512}
              value={config.min_object_size_px ?? 32}
              onChange={(next) => update(name, { min_object_size_px: next })}
            />
          )}
          {name === 'super_resolution' && (
            <NumberField
              label="Max crops per frame"
              help="Limits extra work from super-resolution on crowded frames."
              min={1}
              max={100}
              value={config.max_crops_per_frame ?? 8}
              onChange={(next) => update(name, { max_crops_per_frame: next })}
            />
          )}
        </div>
      )}
    </div>
  )
}
