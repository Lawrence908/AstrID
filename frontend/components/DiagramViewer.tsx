'use client'

import { useState, useEffect, useRef } from 'react'
import { Loader2, AlertCircle, ZoomIn, ZoomOut, RotateCcw } from 'lucide-react'

interface DiagramViewerProps {
  file: string
}

export default function DiagramViewer({ file }: DiagramViewerProps) {
  const [svgContent, setSvgContent] = useState<string>('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [scale, setScale] = useState<number>(1)
  const containerRef = useRef<HTMLDivElement | null>(null)
  const innerRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const fetchSvg = async () => {
      try {
        setLoading(true)
        setError(null)

        // For development, we'll serve files from the public directory
        // In production, you might want to serve these from your API
        const response = await fetch(file)

        if (!response.ok) {
          throw new Error(`Failed to load diagram: ${response.statusText}`)
        }

        const content = await response.text()
        setSvgContent(content)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load diagram')
      } finally {
        setLoading(false)
      }
    }

    fetchSvg()
  }, [file])

  const zoomIn = () => setScale((s) => Math.min(10, parseFloat((s + 1).toFixed(2))))
  const zoomOut = () => setScale((s) => Math.max(0.25, parseFloat((s - 1).toFixed(2))))
  const reset = () => setScale(1)

  // Mouse drag to pan
  const isPanningRef = useRef(false)
  const startRef = useRef<{x:number;y:number;left:number;top:number}>({x:0,y:0,left:0,top:0})

  const onMouseDown = (e: React.MouseEvent) => {
    if (!containerRef.current) return
    // only left button
    if (e.button !== 0) return
    isPanningRef.current = true
    startRef.current = {
      x: e.clientX,
      y: e.clientY,
      left: containerRef.current.scrollLeft,
      top: containerRef.current.scrollTop
    }
    // prevent text selection while dragging
    e.preventDefault()
  }

  const onMouseMove = (e: React.MouseEvent) => {
    if (!isPanningRef.current || !containerRef.current) return
    const dx = e.clientX - startRef.current.x
    const dy = e.clientY - startRef.current.y
    containerRef.current.scrollLeft = startRef.current.left - dx
    containerRef.current.scrollTop = startRef.current.top - dy
  }

  const endPan = () => { isPanningRef.current = false }

  // Touch: pinch-zoom and one-finger pan
  const lastTouchDistance = useRef<number | null>(null)
  const getDistance = (a: React.Touch, b: React.Touch) => {
    const dx = a.clientX - b.clientX
    const dy = a.clientY - b.clientY
    return Math.hypot(dx, dy)
  }

  const onTouchStart = (e: React.TouchEvent) => {
    if (!containerRef.current) return
    if (e.touches.length === 1) {
      isPanningRef.current = true
      startRef.current = {
        x: e.touches[0].clientX,
        y: e.touches[0].clientY,
        left: containerRef.current.scrollLeft,
        top: containerRef.current.scrollTop
      }
    } else if (e.touches.length === 2) {
      lastTouchDistance.current = getDistance(e.touches[0], e.touches[1])
    }
  }

  const onTouchMove = (e: React.TouchEvent) => {
    if (!containerRef.current) return
    if (e.touches.length === 1 && isPanningRef.current) {
      const t = e.touches[0]
      const dx = t.clientX - startRef.current.x
      const dy = t.clientY - startRef.current.y
      containerRef.current.scrollLeft = startRef.current.left - dx
      containerRef.current.scrollTop = startRef.current.top - dy
    } else if (e.touches.length === 2) {
      const dist = getDistance(e.touches[0], e.touches[1])
      if (lastTouchDistance.current) {
        const delta = dist - lastTouchDistance.current
        const step = delta / 150
        setScale((s) => Math.min(10, Math.max(0.25, parseFloat((s + step).toFixed(2)))))
      }
      lastTouchDistance.current = dist
      e.preventDefault()
    }
  }

  const onTouchEnd = () => {
    isPanningRef.current = false
    lastTouchDistance.current = null
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="w-8 h-8 animate-spin text-astrid-blue" />
        <span className="ml-2 text-gray-400">Loading diagram...</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-96">
        <AlertCircle className="w-8 h-8 text-red-500" />
        <div className="ml-4">
          <p className="text-red-500 font-medium">Error loading diagram</p>
          <p className="text-gray-400 text-sm">{error}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="diagram-container">
      <div className="flex items-center justify-end mb-2 gap-2">
        <button onClick={zoomOut} className="px-2 py-1 text-sm bg-gray-800 hover:bg-gray-700 rounded border border-gray-700 text-gray-200 inline-flex items-center">
          <ZoomOut className="w-4 h-4" />
        </button>
        <span className="text-xs text-gray-400 w-14 text-center">{Math.round(scale * 100)}%</span>
        <button onClick={zoomIn} className="px-2 py-1 text-sm bg-gray-800 hover:bg-gray-700 rounded border border-gray-700 text-gray-200 inline-flex items-center">
          <ZoomIn className="w-4 h-4" />
        </button>
        <button onClick={reset} className="px-2 py-1 text-xs bg-gray-800 hover:bg-gray-700 rounded border border-gray-700 text-gray-300 inline-flex items-center">
          <RotateCcw className="w-4 h-4 mr-1" /> Reset
        </button>
      </div>
      {(() => {
        // Make the SVG responsive and preserve aspect ratio.
        // 1) Ensure preserveAspectRatio is set to meet (no skewing)
        // 2) Remove fixed width/height that force scaling
        // 3) Let it render at its intrinsic aspect ratio and allow scrolling for very wide diagrams
        const enhanced = svgContent
          .replace(/<svg(\s[^>]*)?>/, (match) => {
            // Remove width/height attributes if present
            const withoutSize = match
              .replace(/\swidth="[^"]*"/g, '')
              .replace(/\sheight="[^"]*"/g, '')
              .replace(/\spreserveAspectRatio="[^"]*"/g, '')
            // Add our preferred attributes/styles
            const insert = ' preserveAspectRatio="xMidYMid meet" style="height:auto; max-width:none; display:block;"'
            return withoutSize.replace('<svg', `<svg${insert}`)
          })
        return (
          <div
            ref={containerRef}
            className="w-full overflow-auto border border-gray-800 rounded select-none"
            style={{ cursor: isPanningRef.current ? 'grabbing' : 'grab' }}
            onMouseDown={onMouseDown}
            onMouseMove={onMouseMove}
            onMouseUp={endPan}
            onMouseLeave={endPan}
            onTouchStart={onTouchStart}
            onTouchMove={onTouchMove}
            onTouchEnd={onTouchEnd}
          >
            <div
              ref={innerRef}
              style={{ transform: `scale(${scale})`, transformOrigin: 'top left' }}
              // We intentionally allow horizontal scroll so wide diagrams keep their scale
              dangerouslySetInnerHTML={{ __html: enhanced }}
            />
          </div>
        )
      })()}
    </div>
  )
}
