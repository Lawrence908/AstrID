'use client'

import { useState, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Loader2, AlertCircle, Code } from 'lucide-react'

interface DocumentationViewerProps {
  file: string
}

export default function DocumentationViewer({ file }: DocumentationViewerProps) {
  const [content, setContent] = useState<string>('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchContent = async () => {
      try {
        setLoading(true)
        setError(null)

        const response = await fetch(file)

        if (!response.ok) {
          throw new Error(`Failed to load content: ${response.statusText}`)
        }

        const text = await response.text()
        setContent(text)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load content')
      } finally {
        setLoading(false)
      }
    }

    fetchContent()
  }, [file])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="w-8 h-8 animate-spin text-astrid-blue" />
        <span className="ml-2 text-gray-400">Loading content...</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-96">
        <AlertCircle className="w-8 h-8 text-red-500" />
        <div className="ml-4">
          <p className="text-red-500 font-medium">Error loading content</p>
          <p className="text-gray-400 text-sm">{error}</p>
        </div>
      </div>
    )
  }

  const isCodeFile = file.endsWith('.py') || file.endsWith('.js') || file.endsWith('.ts') || file.endsWith('.tsx')

  if (isCodeFile) {
    return (
      <div className="space-y-4">
        <div className="flex items-center space-x-2 text-sm text-gray-400">
          <Code className="w-4 h-4" />
          <span>{file.split('/').pop()}</span>
        </div>
        <pre className="bg-gray-900 p-6 rounded-lg overflow-x-auto">
          <code className="text-gray-300 text-sm leading-relaxed">
            {content}
          </code>
        </pre>
      </div>
    )
  }

  return (
    <div className="markdown-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ children }) => (
            <h1 className="text-3xl font-bold text-astrid-blue mb-6 border-b border-gray-700 pb-2">
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-2xl font-semibold text-astrid-blue mb-4 mt-8">
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-xl font-medium text-astrid-blue mb-3 mt-6">
              {children}
            </h3>
          ),
          p: ({ children }) => (
            <p className="text-gray-300 mb-4 leading-relaxed">
              {children}
            </p>
          ),
          code: ({ children, className }) => {
            const isInline = !className
            return isInline ? (
              <code className="bg-gray-800 text-astrid-blue px-2 py-1 rounded text-sm">
                {children}
              </code>
            ) : (
              <code className={className}>{children}</code>
            )
          },
          pre: ({ children }) => (
            <pre className="bg-gray-900 p-4 rounded-lg overflow-x-auto mb-4">
              {children}
            </pre>
          ),
          ul: ({ children }) => (
            <ul className="text-gray-300 mb-4 list-disc list-inside space-y-2">
              {children}
            </ul>
          ),
          ol: ({ children }) => (
            <ol className="text-gray-300 mb-4 list-decimal list-inside space-y-2">
              {children}
            </ol>
          ),
          li: ({ children }) => (
            <li className="text-gray-300">{children}</li>
          ),
          blockquote: ({ children }) => (
            <blockquote className="border-l-4 border-astrid-blue pl-4 italic text-gray-400 mb-4">
              {children}
            </blockquote>
          ),
          table: ({ children }) => (
            <div className="overflow-x-auto mb-4">
              <table className="min-w-full border border-gray-700 rounded-lg">
                {children}
              </table>
            </div>
          ),
          th: ({ children }) => (
            <th className="px-4 py-2 bg-gray-800 text-astrid-blue font-semibold border-b border-gray-700">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="px-4 py-2 text-gray-300 border-b border-gray-700">
              {children}
            </td>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}
