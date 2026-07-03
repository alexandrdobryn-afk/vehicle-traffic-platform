'use client'
import { useEffect, useRef, useState, useCallback } from 'react'
import AppShell from '@/components/shared/AppShell'
import trainingApi from '@/lib/trainingApi'
import { TDataset, TDatasetImage, TAnnotation } from '@/types/training'
import {
  ChevronLeft, ChevronRight, Save, Trash2,
  ZoomIn, ZoomOut, Copy, Wand2, CheckCircle
} from 'lucide-react'
import toast from 'react-hot-toast'
import { useParams } from 'next/navigation'
import { useTranslation } from '@/lib/i18n'

interface BBox { x: number; y: number; w: number; h: number }
interface DrawingBox { startX: number; startY: number; endX: number; endY: number }

const CLASS_COLORS = [
  '#ef4444','#3b82f6','#22c55e','#f59e0b','#8b5cf6',
  '#ec4899','#06b6d4','#f97316','#14b8a6','#a855f7',
]

export default function AnnotatePage() {
  const { t } = useTranslation()
  const params = useParams<{ datasetId: string }>()
  const datasetId = parseInt(params.datasetId)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef = useRef<HTMLImageElement | null>(null)

  const [dataset, setDataset] = useState<TDataset | null>(null)
  const [images, setImages] = useState<TDatasetImage[]>([])
  const [currentIdx, setCurrentIdx] = useState(0)
  const [annotations, setAnnotations] = useState<TAnnotation[]>([])
  const [selectedClass, setSelectedClass] = useState(0)
  const [isDrawing, setIsDrawing] = useState(false)
  const [drawBox, setDrawBox] = useState<DrawingBox | null>(null)
  const [zoom, setZoom] = useState(1)
  const [imgLoaded, setImgLoaded] = useState(false)
  const [saving, setSaving] = useState(false)
  const [autoAnnotating, setAutoAnnotating] = useState(false)

  const TRAINING_URL = process.env.NEXT_PUBLIC_TRAINING_API_URL || 'http://localhost:8001'

  useEffect(() => {
    trainingApi.get(`/datasets/${datasetId}`).then(r => setDataset(r.data))
    trainingApi.get(`/datasets/${datasetId}/images?limit=500`).then(r => setImages(r.data))
  }, [datasetId])

  const currentImage = images[currentIdx]

  useEffect(() => {
    if (!currentImage) return
    loadAnnotations(currentImage.id)
    setImgLoaded(false)
    const img = new Image()
    img.src = `${TRAINING_URL}/api/v1/training/datasets/${datasetId}/images/${currentImage.id}/file`
    img.onload = () => { imgRef.current = img; setImgLoaded(true) }
  }, [currentImage?.id])

  useEffect(() => {
    if (imgLoaded) drawCanvas()
  }, [imgLoaded, annotations, drawBox, zoom])

  const loadAnnotations = async (imageId: number) => {
    const r = await trainingApi.get(`/annotations/image/${imageId}`)
    setAnnotations(r.data)
  }

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current
    const img = imgRef.current
    if (!canvas || !img) return
    const ctx = canvas.getContext('2d')!
    const W = img.width * zoom
    const H = img.height * zoom
    canvas.width = W
    canvas.height = H
    ctx.clearRect(0, 0, W, H)
    ctx.drawImage(img, 0, 0, W, H)

    // Draw saved annotations
    annotations.forEach((ann, idx) => {
      if (ann.annotation_type !== 'bbox') return
      const x = (ann.x_center! - ann.bbox_width! / 2) * W
      const y = (ann.y_center! - ann.bbox_height! / 2) * H
      const w = ann.bbox_width! * W
      const h = ann.bbox_height! * H
      const color = CLASS_COLORS[ann.class_id || 0] || CLASS_COLORS[0]

      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.strokeRect(x, y, w, h)
      ctx.fillStyle = color + '30'
      ctx.fillRect(x, y, w, h)

      // Label
      const label = t(ann.class_name || 'obj')
      ctx.fillStyle = color
      ctx.fillRect(x, y - 18, label.length * 7 + 8, 18)
      ctx.fillStyle = '#fff'
      ctx.font = '12px monospace'
      ctx.fillText(label || `class_${ann.class_id}`, x + 4, y - 4)

      if (ann.is_auto) {
        ctx.fillStyle = '#fbbf24'
        ctx.font = '10px sans-serif'
        ctx.fillText(t('auto'), x + 4, y + 14)
      }
    })

    // Draw current drawing box
    if (drawBox) {
      const x = Math.min(drawBox.startX, drawBox.endX)
      const y = Math.min(drawBox.startY, drawBox.endY)
      const w = Math.abs(drawBox.endX - drawBox.startX)
      const h = Math.abs(drawBox.endY - drawBox.startY)
      const color = CLASS_COLORS[selectedClass] || CLASS_COLORS[0]
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.setLineDash([4, 4])
      ctx.strokeRect(x, y, w, h)
      ctx.setLineDash([])
    }
  }, [annotations, drawBox, zoom, selectedClass, t])

  const getCanvasCoords = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current!
    const rect = canvas.getBoundingClientRect()
    return {
      x: (e.clientX - rect.left) * (canvas.width / rect.width),
      y: (e.clientY - rect.top) * (canvas.height / rect.height),
    }
  }

  const onMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const { x, y } = getCanvasCoords(e)
    setIsDrawing(true)
    setDrawBox({ startX: x, startY: y, endX: x, endY: y })
  }

  const onMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return
    const { x, y } = getCanvasCoords(e)
    setDrawBox(b => b ? { ...b, endX: x, endY: y } : null)
  }

  const onMouseUp = async () => {
    if (!isDrawing || !drawBox || !imgRef.current || !currentImage) return
    setIsDrawing(false)
    const W = imgRef.current.width * zoom
    const H = imgRef.current.height * zoom
    const x1 = Math.min(drawBox.startX, drawBox.endX) / W
    const y1 = Math.min(drawBox.startY, drawBox.endY) / H
    const x2 = Math.max(drawBox.startX, drawBox.endX) / W
    const y2 = Math.max(drawBox.startY, drawBox.endY) / H
    const w = x2 - x1, h = y2 - y1

    if (w < 0.01 || h < 0.01) { setDrawBox(null); return }

    const cls = dataset?.classes[selectedClass] || 'unknown'
    setSaving(true)
    try {
      await trainingApi.post(`/annotations/image/${currentImage.id}`, {
        annotation_type: 'bbox',
        class_name: cls,
        class_id: selectedClass,
        x_center: x1 + w / 2,
        y_center: y1 + h / 2,
        bbox_width: w,
        bbox_height: h,
      })
      await loadAnnotations(currentImage.id)
    } catch { toast.error(t('Failed to save annotation')) }
    finally { setSaving(false); setDrawBox(null) }
  }

  const deleteAnnotation = async (id: number) => {
    await trainingApi.delete(`/annotations/${id}`)
    setAnnotations(prev => prev.filter(a => a.id !== id))
  }

  const clearAll = async () => {
    if (!currentImage) return
    await trainingApi.delete(`/annotations/image/${currentImage.id}/all`)
    setAnnotations([])
  }

  const copyToNext = async () => {
    if (!currentImage || !images[currentIdx + 1]) return
    await trainingApi.post(`/annotations/image/${currentImage.id}/copy-to/${images[currentIdx + 1].id}`)
    toast.success(t('Annotations copied to next image'))
  }

  const handleAutoAnnotate = async () => {
    setAutoAnnotating(true)
    try {
      await trainingApi.post(`/datasets/${datasetId}/auto-annotate`, {
        image_ids: [currentImage?.id],
        confidence_threshold: 0.5,
        overwrite_existing: false,
      })
      toast.success(t('Auto-annotation queued'))
      setTimeout(() => loadAnnotations(currentImage!.id), 3000)
    } catch { toast.error(t('Auto-annotate failed')) }
    finally { setAutoAnnotating(false) }
  }

  const navigate = (dir: -1 | 1) => {
    const next = currentIdx + dir
    if (next >= 0 && next < images.length) setCurrentIdx(next)
  }

  return (
    <AppShell>
      <div className="mt-16 flex h-[calc(100vh-4rem)]">
        {/* Left panel — class selector + annotation list */}
        <div className="w-56 bg-card border-r border-border flex flex-col shrink-0">
          <div className="p-3 border-b border-border">
            <p className="text-xs font-semibold text-muted-foreground uppercase mb-2">{t('Classes')}</p>
            {dataset?.classes.map((cls, idx) => (
              <button key={cls} onClick={() => setSelectedClass(idx)}
                className={`w-full flex items-center gap-2 px-2 py-1.5 rounded-lg text-sm mb-0.5 transition-colors ${
                  selectedClass === idx ? 'bg-primary/10 text-primary' : 'text-muted-foreground hover:text-foreground hover:bg-accent'
                }`}>
                <span className="w-3 h-3 rounded-sm shrink-0" style={{ background: CLASS_COLORS[idx] }} />
                {t(cls)}
              </button>
            ))}
          </div>

          <div className="flex-1 overflow-y-auto p-3">
            <p className="text-xs font-semibold text-muted-foreground uppercase mb-2">
              {t('Annotations')} ({annotations.length})
            </p>
            {annotations.map((ann) => (
              <div key={ann.id} className="flex items-center justify-between py-1.5 px-2 rounded hover:bg-accent/50 group">
                <div className="flex items-center gap-1.5 min-w-0">
                  <span className="w-2.5 h-2.5 rounded-sm shrink-0"
                    style={{ background: CLASS_COLORS[ann.class_id || 0] }} />
                  <span className="text-xs text-foreground truncate">{ann.class_name ? t(ann.class_name) : ''}</span>
                  {ann.is_auto && <span className="text-xs text-amber-400">{t('auto')}</span>}
                </div>
                <button onClick={() => deleteAnnotation(ann.id)} title={t('Delete')} aria-label={t('Delete')}
                  className="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-red-400">
                  <Trash2 className="w-3 h-3" />
                </button>
              </div>
            ))}
          </div>

          <div className="p-3 border-t border-border space-y-1.5">
            <button onClick={handleAutoAnnotate} disabled={autoAnnotating}
              className="w-full flex items-center justify-center gap-1.5 py-1.5 bg-violet-500/10 text-violet-400 rounded-lg text-xs hover:bg-violet-500/20 disabled:opacity-50">
              <Wand2 className="w-3.5 h-3.5" />{autoAnnotating ? t('Queuing...') : t('Auto-Annotate')}
            </button>
            <button onClick={copyToNext}
              className="w-full flex items-center justify-center gap-1.5 py-1.5 bg-muted text-muted-foreground rounded-lg text-xs hover:text-foreground">
              <Copy className="w-3.5 h-3.5" /> {t('Copy to Next')}
            </button>
            <button onClick={clearAll}
              className="w-full flex items-center justify-center gap-1.5 py-1.5 bg-red-500/10 text-red-400 rounded-lg text-xs hover:bg-red-500/20">
              <Trash2 className="w-3.5 h-3.5" /> {t('Clear All')}
            </button>
          </div>
        </div>

        {/* Main canvas area */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* Toolbar */}
          <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-card">
            <div className="flex items-center gap-3">
              <button onClick={() => navigate(-1)} disabled={currentIdx === 0} title={t('Previous image')} aria-label={t('Previous image')}
                className="p-1.5 rounded text-muted-foreground hover:text-foreground disabled:opacity-30">
                <ChevronLeft className="w-5 h-5" />
              </button>
              <span className="text-sm text-foreground font-mono">
                {currentIdx + 1} / {images.length}
              </span>
              <button onClick={() => navigate(1)} disabled={currentIdx >= images.length - 1} title={t('Next image')} aria-label={t('Next image')}
                className="p-1.5 rounded text-muted-foreground hover:text-foreground disabled:opacity-30">
                <ChevronRight className="w-5 h-5" />
              </button>
              <span className="text-xs text-muted-foreground truncate max-w-[200px]">
                {currentImage?.filename}
              </span>
              {currentImage?.is_annotated && (
                <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0" />
              )}
            </div>
            <div className="flex items-center gap-2">
              <button onClick={() => setZoom(z => Math.max(0.25, z - 0.25))} title={t('Zoom out')} aria-label={t('Zoom out')}
                className="p-1.5 rounded text-muted-foreground hover:text-foreground">
                <ZoomOut className="w-4 h-4" />
              </button>
              <span className="text-xs text-muted-foreground w-12 text-center">{Math.round(zoom * 100)}%</span>
              <button onClick={() => setZoom(z => Math.min(4, z + 0.25))} title={t('Zoom in')} aria-label={t('Zoom in')}
                className="p-1.5 rounded text-muted-foreground hover:text-foreground">
                <ZoomIn className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Canvas */}
          <div className="flex-1 overflow-auto bg-[#1a1a2e] flex items-start justify-start p-4">
            {currentImage ? (
              <canvas
                ref={canvasRef}
                className="cursor-crosshair"
                style={{ userSelect: 'none' }}
                onMouseDown={onMouseDown}
                onMouseMove={onMouseMove}
                onMouseUp={onMouseUp}
                onMouseLeave={() => { if (isDrawing) { setIsDrawing(false); setDrawBox(null) } }}
              />
            ) : (
              <div className="flex-1 flex items-center justify-center text-muted-foreground">
                {t('No images in dataset')}
              </div>
            )}
          </div>
        </div>

        {/* Right panel — image list */}
        <div className="w-44 bg-card border-l border-border overflow-y-auto shrink-0">
          <div className="p-2 border-b border-border">
            <p className="text-xs font-semibold text-muted-foreground uppercase">{t('Images')}</p>
          </div>
          {images.map((img, idx) => (
            <button key={img.id} onClick={() => setCurrentIdx(idx)}
              className={`w-full flex items-center gap-2 px-2 py-2 text-left border-b border-border/50 transition-colors ${
                idx === currentIdx ? 'bg-primary/10' : 'hover:bg-accent/50'
              }`}>
              <span className={`w-2 h-2 rounded-full shrink-0 ${img.is_annotated ? 'bg-emerald-400' : 'bg-muted-foreground/30'}`} />
              <span className="text-xs text-muted-foreground truncate">{img.filename}</span>
            </button>
          ))}
        </div>
      </div>
    </AppShell>
  )
}
