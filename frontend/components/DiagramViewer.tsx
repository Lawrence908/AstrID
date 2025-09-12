'use client'

import { useState, useEffect } from 'react'
import { Loader2, AlertCircle } from 'lucide-react'

interface DiagramViewerProps {
  file: string
}

export default function DiagramViewer({ file }: DiagramViewerProps) {
  const [svgContent, setSvgContent] = useState<string>('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

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
      <div
        className="w-full overflow-auto"
        dangerouslySetInnerHTML={{ __html: svgContent }}
      />
    </div>
  )
}
