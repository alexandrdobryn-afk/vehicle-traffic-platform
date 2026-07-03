'use client'
import { useEffect, useRef, useState, useCallback } from 'react'
import AppShell from '@/components/shared/AppShell'
import trainingApi from '@/lib/trainingApi'
import { TDataset, TDatasetImage, TAnnotation } from '@/types/training'
import {
  ChevronLeft, ChevronRight, Trash2,
  ZoomIn, ZoomOut, Copy, Wand2, CheckCircle, MousePointer2, Square
} from 'lucide-react'
import toast from 'react-hot-toast'
import { useParams } from 'next/navigation'
import { useTranslation } from '@/lib/i18n'
import AuthenticatedTrainingImage, { useTrainingImageUrl } from '@/components/training/AuthenticatedTrainingImage'

interface DrawingBox { startX: number; startY: number; endX: number; endY: number }
interface NormalizedBox { xCenter: number; yCenter: number; width: number; height: number }
type ResizeHandle = 'nw' | 'ne' | 'se' | 'sw'
interface EditInteraction {
  mode: 'move' | 'resize'
  annotationId: number
  handle?: ResizeHandle
  startX: number
  startY: number
  original: NormalizedBox
}

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
  const [activeTool, setActiveTool] = useState<'select' | 'draw'>('select')
  const [selectedAnnotationId, setSelectedAnnotationId] = useState<number | null>(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [drawBox, setDrawBox] = useState<DrawingBox | null>(null)
  const [editInteraction, setEditInteraction] = useState<EditInteraction | null>(null)
  const [draftEdit, setDraftEdit] = useState<(NormalizedBox & { annotationId: number }) | null>(null)
  const draftEditRef = useRef<(NormalizedBox & { annotationId: number }) | null>(null)
  const [zoom, setZoom] = useState(1)
  const [imgLoaded, setImgLoaded] = useState(false)
  const [imageDecodeError, setImageDecodeError] = useState(false)
  const [autoAnnotating, setAutoAnnotating] = useState(false)

  useEffect(() => {
    trainingApi.get(`/datasets/${datasetId}`).then(r => setDataset(r.data))
    trainingApi.get(`/datasets/${datasetId}/images?limit=500`).then(r => setImages(r.data))
  }, [datasetId])

  const currentImage = images[currentIdx]
  const selectedAnnotation = annotations.find(ann => ann.id === selectedAnnotationId) || null
  const imageEndpoint = currentImage
    ? `/datasets/${datasetId}/images/${currentImage.id}/file`
    : undefined
  const { url: imageUrl, error: imageRequestError } = useTrainingImageUrl(imageEndpoint)

  useEffect(() => {
    if (!currentImage) return
    loadAnnotations(currentImage.id)
    setImgLoaded(false)
    setImageDecodeError(false)
    setSelectedAnnotationId(null)
    setEditInteraction(null)
    setDraftEdit(null)
    draftEditRef.current = null
  }, [currentImage?.id])

  useEffect(() => {
    if (!imageUrl) return
    const img = new Image()
    img.onload = () => { imgRef.current = img; setImgLoaded(true) }
    img.onerror = () => setImageDecodeError(true)
    img.src = imageUrl
  }, [imageUrl])

  useEffect(() => {
    if (imgLoaded) drawCanvas()
  }, [imgLoaded, annotations, drawBox, zoom])

  const loadAnnotations = async (imageId: number) => {
    const r = await trainingApi.get(`/annotations/image/${imageId}`)
    setAnnotations(r.data)
  }

  const annotationBox = (ann: TAnnotation): NormalizedBox | null => {
    if (
      ann.x_center == null || ann.y_center == null ||
      ann.bbox_width == null || ann.bbox_height == null
    ) return null
    return {
      xCenter: ann.x_center,
      yCenter: ann.y_center,
      width: ann.bbox_width,
      height: ann.bbox_height,
    }
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
    annotations.forEach((ann) => {
      if (ann.annotation_type !== 'bbox') return
      const stored = annotationBox(ann)
      if (!stored) return
      const box = draftEdit?.annotationId === ann.id ? draftEdit : stored
      const x = (box.xCenter - box.width / 2) * W
      const y = (box.yCenter - box.height / 2) * H
      const w = box.width * W
      const h = box.height * H
      const color = CLASS_COLORS[ann.class_id || 0] || CLASS_COLORS[0]
      const selected = ann.id === selectedAnnotationId

      ctx.strokeStyle = color
      ctx.lineWidth = selected ? 4 : 2
      ctx.strokeRect(x, y, w, h)
      ctx.fillStyle = color + '30'
      ctx.fillRect(x, y, w, h)

      if (selected) {
        const handleSize = 10
        ctx.fillStyle = '#fff'
        ctx.strokeStyle = '#111827'
        ;[
          [x, y], [x + w, y], [x + w, y + h], [x, y + h],
        ].forEach(([handleX, handleY]) => {
          ctx.fillRect(handleX - handleSize / 2, handleY - handleSize / 2, handleSize, handleSize)
          ctx.strokeRect(handleX - handleSize / 2, handleY - handleSize / 2, handleSize, handleSize)
        })
      }

      // Label
      const label = t(ann.class_name || 'obj')
      const labelY = Math.max(0, y - 18)
      ctx.fillStyle = color
      ctx.fillRect(x, labelY, label.length * 7 + 8, 18)
      ctx.fillStyle = '#fff'
      ctx.font = '12px monospace'
      ctx.fillText(label || `class_${ann.class_id}`, x + 4, labelY + 13)

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
  }, [annotations, drawBox, draftEdit, zoom, selectedClass, selectedAnnotationId, t])

  const getCanvasCoords = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current!
    const rect = canvas.getBoundingClientRect()
    return {
      x: (e.clientX - rect.left) * (canvas.width / rect.width),
      y: (e.clientY - rect.top) * (canvas.height / rect.height),
    }
  }

  const updateAnnotation = async (
    ann: TAnnotation,
    changes: Partial<NormalizedBox> & { classId?: number },
  ) => {
    const box = annotationBox(ann)
    if (!box) return
    const classId = changes.classId ?? ann.class_id ?? 0
    const response = await trainingApi.put(`/annotations/${ann.id}`, {
      annotation_type: ann.annotation_type,
      class_name: dataset?.classes[classId] || ann.class_name || 'unknown',
      class_id: classId,
      x_center: changes.xCenter ?? box.xCenter,
      y_center: changes.yCenter ?? box.yCenter,
      bbox_width: changes.width ?? box.width,
      bbox_height: changes.height ?? box.height,
      confidence: ann.confidence,
      is_auto: false,
      is_verified: true,
    })
    setAnnotations(prev => prev.map(item => item.id === ann.id ? response.data : item))
  }

  const getResizeHandle = (ann: TAnnotation, x: number, y: number): ResizeHandle | null => {
    const canvas = canvasRef.current
    const box = annotationBox(ann)
    if (!canvas || !box) return null
    const left = (box.xCenter - box.width / 2) * canvas.width
    const top = (box.yCenter - box.height / 2) * canvas.height
    const right = (box.xCenter + box.width / 2) * canvas.width
    const bottom = (box.yCenter + box.height / 2) * canvas.height
    const handles: Array<[ResizeHandle, number, number]> = [
      ['nw', left, top], ['ne', right, top],
      ['se', right, bottom], ['sw', left, bottom],
    ]
    return handles.find(([, hx, hy]) => Math.abs(x - hx) <= 12 && Math.abs(y - hy) <= 12)?.[0] || null
  }

  const findAnnotationAt = (x: number, y: number) => {
    const canvas = canvasRef.current
    if (!canvas) return null
    return [...annotations].reverse().find(ann => {
      const box = annotationBox(ann)
      if (!box) return false
      const left = (box.xCenter - box.width / 2) * canvas.width
      const top = (box.yCenter - box.height / 2) * canvas.height
      const right = (box.xCenter + box.width / 2) * canvas.width
      const bottom = (box.yCenter + box.height / 2) * canvas.height
      return x >= left && x <= right && y >= top && y <= bottom
    }) || null
  }

  const setDraftBox = (value: (NormalizedBox & { annotationId: number }) | null) => {
    draftEditRef.current = value
    setDraftEdit(value)
  }

  const onMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const { x, y } = getCanvasCoords(e)
    if (activeTool === 'draw') {
      if (!dataset?.classes.length) {
        toast.error(t('Configure at least one dataset class before annotating'))
        return
      }
      setSelectedAnnotationId(null)
      setIsDrawing(true)
      setDrawBox({ startX: x, startY: y, endX: x, endY: y })
      return
    }

    if (selectedAnnotation) {
      const handle = getResizeHandle(selectedAnnotation, x, y)
      const original = annotationBox(selectedAnnotation)
      if (handle && original) {
        setEditInteraction({
          mode: 'resize', annotationId: selectedAnnotation.id,
          handle, startX: x, startY: y, original,
        })
        setDraftBox({ annotationId: selectedAnnotation.id, ...original })
        return
      }
    }

    const hit = findAnnotationAt(x, y)
    const hitBox = hit ? annotationBox(hit) : null
    if (hit && hitBox) {
      setSelectedAnnotationId(hit.id)
      setSelectedClass(hit.class_id ?? 0)
      setEditInteraction({
        mode: 'move', annotationId: hit.id,
        startX: x, startY: y, original: hitBox,
      })
      setDraftBox({ annotationId: hit.id, ...hitBox })
      return
    }

    setSelectedAnnotationId(null)
  }

  const onMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const { x, y } = getCanvasCoords(e)
    if (editInteraction && canvasRef.current) {
      const W = canvasRef.current.width
      const H = canvasRef.current.height
      const dx = (x - editInteraction.startX) / W
      const dy = (y - editInteraction.startY) / H
      const original = editInteraction.original
      let next: NormalizedBox

      if (editInteraction.mode === 'move') {
        next = {
          ...original,
          xCenter: Math.min(1 - original.width / 2, Math.max(original.width / 2, original.xCenter + dx)),
          yCenter: Math.min(1 - original.height / 2, Math.max(original.height / 2, original.yCenter + dy)),
        }
      } else {
        let left = original.xCenter - original.width / 2
        let top = original.yCenter - original.height / 2
        let right = original.xCenter + original.width / 2
        let bottom = original.yCenter + original.height / 2
        if (editInteraction.handle?.includes('w')) left = Math.min(right - 0.01, Math.max(0, left + dx))
        if (editInteraction.handle?.includes('e')) right = Math.max(left + 0.01, Math.min(1, right + dx))
        if (editInteraction.handle?.includes('n')) top = Math.min(bottom - 0.01, Math.max(0, top + dy))
        if (editInteraction.handle?.includes('s')) bottom = Math.max(top + 0.01, Math.min(1, bottom + dy))
        next = {
          xCenter: (left + right) / 2,
          yCenter: (top + bottom) / 2,
          width: right - left,
          height: bottom - top,
        }
      }
      setDraftBox({ annotationId: editInteraction.annotationId, ...next })
      return
    }

    if (isDrawing) setDrawBox(b => b ? { ...b, endX: x, endY: y } : null)
  }

  const onMouseUp = async () => {
    if (editInteraction) {
      const ann = annotations.find(item => item.id === editInteraction.annotationId)
      const edited = draftEditRef.current
      setEditInteraction(null)
      setDraftBox(null)
      if (ann && edited) {
        try {
          await updateAnnotation(ann, edited)
          toast.success(t('Annotation updated'))
        } catch {
          toast.error(t('Failed to update annotation'))
        }
      }
      return
    }

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
    try {
      const response = await trainingApi.post(`/annotations/image/${currentImage.id}`, {
        annotation_type: 'bbox',
        class_name: cls,
        class_id: selectedClass,
        x_center: x1 + w / 2,
        y_center: y1 + h / 2,
        bbox_width: w,
        bbox_height: h,
        is_auto: false,
        is_verified: true,
      })
      setAnnotations(prev => [...prev, response.data])
      setSelectedAnnotationId(response.data.id)
      setActiveTool('select')
      setImages(prev => prev.map(image => image.id === currentImage.id ? { ...image, is_annotated: true } : image))
    } catch { toast.error(t('Failed to save annotation')) }
    finally { setDrawBox(null) }
  }

  const deleteAnnotation = async (id: number) => {
    await trainingApi.delete(`/annotations/${id}`)
    const remaining = annotations.filter(a => a.id !== id)
    setAnnotations(remaining)
    if (currentImage && remaining.length === 0) {
      setImages(prev => prev.map(image => image.id === currentImage.id ? { ...image, is_annotated: false } : image))
    }
    if (selectedAnnotationId === id) setSelectedAnnotationId(null)
  }

  const selectAnnotation = (ann: TAnnotation) => {
    setActiveTool('select')
    setSelectedAnnotationId(ann.id)
    setSelectedClass(ann.class_id ?? 0)
  }

  const changeSelectedClass = async (classId: number) => {
    if (!selectedAnnotation) return
    setSelectedClass(classId)
    try {
      await updateAnnotation(selectedAnnotation, { classId })
      toast.success(t('Annotation updated'))
    } catch {
      toast.error(t('Failed to update annotation'))
    }
  }

  const clearAll = async () => {
    if (!currentImage) return
    await trainingApi.delete(`/annotations/image/${currentImage.id}/all`)
    setAnnotations([])
    setSelectedAnnotationId(null)
    setImages(prev => prev.map(image => image.id === currentImage.id ? { ...image, is_annotated: false } : image))
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
        <div className="w-64 bg-card border-r border-border flex flex-col shrink-0">
          <div className="p-3 border-b border-border">
            <p className="text-xs font-semibold text-muted-foreground uppercase mb-2">{t('Classes')}</p>
            <div className="grid grid-cols-2 gap-1 mb-3">
              <button onClick={() => setActiveTool('select')}
                className={`flex items-center justify-center gap-1 py-1.5 rounded text-xs ${
                  activeTool === 'select' ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground hover:text-foreground'
                }`}>
                <MousePointer2 className="w-3.5 h-3.5" /> {t('Select / Edit')}
              </button>
              <button onClick={() => setActiveTool('draw')}
                className={`flex items-center justify-center gap-1 py-1.5 rounded text-xs ${
                  activeTool === 'draw' ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground hover:text-foreground'
                }`}>
                <Square className="w-3.5 h-3.5" /> {t('Draw box')}
              </button>
            </div>
            {dataset?.classes.map((cls, idx) => (
              <button key={cls} onClick={() => { setSelectedClass(idx); setActiveTool('draw') }}
                className={`w-full flex items-center gap-2 px-2 py-1.5 rounded-lg text-sm mb-0.5 transition-colors ${
                  selectedClass === idx ? 'bg-primary/10 text-primary' : 'text-muted-foreground hover:text-foreground hover:bg-accent'
                }`}>
                <span className="w-3 h-3 rounded-sm shrink-0" style={{ background: CLASS_COLORS[idx] }} />
                {t(cls)}
              </button>
            ))}
            {dataset?.classes.length === 0 && (
              <p className="text-xs text-amber-400">{t('No classes configured for this dataset')}</p>
            )}
            <p className="mt-2 text-[11px] leading-4 text-muted-foreground">
              {t('Select a class, then draw a new box on the image.')}
            </p>
          </div>

          <div className="flex-1 overflow-y-auto p-3">
            <p className="text-xs font-semibold text-muted-foreground uppercase mb-2">
              {t('Annotations')} ({annotations.length})
            </p>
            {annotations.map((ann) => (
              <div key={ann.id} className={`flex items-center rounded group border mb-1 ${
                ann.id === selectedAnnotationId ? 'border-primary bg-primary/10' : 'border-transparent hover:bg-accent/50'
              }`}>
                <button onClick={() => selectAnnotation(ann)}
                  className="flex-1 flex items-center gap-1.5 min-w-0 py-2 px-2 text-left">
                  <span className="w-2.5 h-2.5 rounded-sm shrink-0"
                    style={{ background: CLASS_COLORS[ann.class_id || 0] }} />
                  <span className="text-xs text-foreground truncate">{ann.class_name ? t(ann.class_name) : ''}</span>
                  <span className={`ml-auto text-[9px] uppercase ${ann.is_auto ? 'text-amber-400' : 'text-emerald-400'}`}>
                    {ann.is_auto ? t('Auto-generated') : t('Manual')}
                  </span>
                </button>
                <button onClick={() => deleteAnnotation(ann.id)} title={t('Delete')} aria-label={t('Delete')}
                  className="mr-2 opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-red-400">
                  <Trash2 className="w-3 h-3" />
                </button>
              </div>
            ))}

            {selectedAnnotation && (
              <div className="mt-3 p-3 rounded-lg border border-primary/30 bg-primary/5 space-y-3">
                <div className="flex items-center gap-2 text-xs font-medium text-foreground">
                  <MousePointer2 className="w-3.5 h-3.5 text-primary" />
                  {t('Selected annotation')}
                </div>
                <div>
                  <label className="block text-[10px] uppercase text-muted-foreground mb-1">{t('Class')}</label>
                  <select
                    value={selectedAnnotation.class_id ?? 0}
                    onChange={event => changeSelectedClass(Number(event.target.value))}
                    className="w-full px-2 py-1.5 bg-background border border-input rounded text-xs text-foreground"
                  >
                    {dataset?.classes.map((cls, idx) => (
                      <option key={cls} value={idx}>{t(cls)}</option>
                    ))}
                  </select>
                </div>
                <div className="flex items-center justify-between text-[11px]">
                  <span className="text-muted-foreground">{t('Source')}</span>
                  <span className={selectedAnnotation.is_auto ? 'text-amber-400' : 'text-emerald-400'}>
                    {selectedAnnotation.is_auto ? t('Auto-generated') : t('Manual')}
                  </span>
                </div>
                {selectedAnnotation.confidence != null && (
                  <div className="flex items-center justify-between text-[11px]">
                    <span className="text-muted-foreground">{t('Confidence')}</span>
                    <span className="text-foreground">{Math.round(selectedAnnotation.confidence * 100)}%</span>
                  </div>
                )}
                <p className="text-[10px] leading-4 text-muted-foreground">
                  {t('Drag the box to move it. Drag a corner handle to resize it. Manual edits are marked as reviewed.')}
                </p>
              </div>
            )}
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
          <div className="relative flex-1 overflow-auto bg-[#1a1a2e] flex items-start justify-start p-4">
            {currentImage ? (
              <>
                {!imgLoaded && !imageRequestError && !imageDecodeError && (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="w-7 h-7 rounded-full border-2 border-primary border-t-transparent animate-spin" />
                  </div>
                )}
                {(imageRequestError || imageDecodeError) && (
                  <div className="absolute inset-0 flex items-center justify-center text-sm text-red-400">
                    {t('Failed to load image')}
                  </div>
                )}
                <canvas
                  ref={canvasRef}
                  className={`${editInteraction ? 'cursor-grabbing' : activeTool === 'draw' ? 'cursor-crosshair' : 'cursor-default'} ${imgLoaded ? '' : 'invisible'}`}
                  style={{ userSelect: 'none' }}
                  onMouseDown={onMouseDown}
                  onMouseMove={onMouseMove}
                  onMouseUp={onMouseUp}
                  onMouseLeave={() => {
                    if (isDrawing) { setIsDrawing(false); setDrawBox(null) }
                    if (editInteraction) { setEditInteraction(null); setDraftBox(null) }
                  }}
                />
              </>
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
              className={`w-full flex items-center gap-2 p-2 text-left border-b border-border/50 transition-colors ${
                idx === currentIdx ? 'bg-primary/10' : 'hover:bg-accent/50'
              }`}>
              <div className="relative w-12 h-9 shrink-0">
                <AuthenticatedTrainingImage
                  endpoint={`/datasets/${datasetId}/images/${img.id}/file`}
                  alt={img.filename}
                  className="w-12 h-9 object-cover rounded border border-border"
                />
                <span className={`absolute top-0.5 right-0.5 w-2 h-2 rounded-full border border-card ${img.is_annotated ? 'bg-emerald-400' : 'bg-muted-foreground/50'}`} />
              </div>
              <span className="text-[11px] text-muted-foreground truncate">{img.filename}</span>
            </button>
          ))}
        </div>
      </div>
    </AppShell>
  )
}
