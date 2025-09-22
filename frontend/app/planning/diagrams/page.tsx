'use client'

import { useState } from 'react'
import Link from 'next/link'
import { ArrowLeft, Image, Download, Eye, Calendar, FileText } from 'lucide-react'
import DiagramViewer from '@/components/DiagramViewer'

const diagrams = [
  {
    id: 'data-flow-pipeline',
    name: 'Data Flow Pipeline',
    file: '/docs/diagrams/data-flow-pipeline.svg',
    description: 'Complete data processing pipeline from observation to detection',
    category: 'Architecture',
    lastUpdated: '2025-09-21'
  },
  {
    id: 'database-schema',
    name: 'Database Schema',
    file: '/docs/diagrams/database-schema.svg',
    description: 'Database structure and relationships',
    category: 'Data',
    lastUpdated: '2025-09-21'
  },
  {
    id: 'domain-interactions',
    name: 'Domain Interactions',
    file: '/docs/diagrams/domain-interactions.svg',
    description: 'Service interactions and domain boundaries',
    category: 'Architecture',
    lastUpdated: '2025-09-21'
  },
  {
    id: 'external-integrations',
    name: 'External Integrations',
    file: '/docs/diagrams/external-integrations.svg',
    description: 'External API integrations and data sources',
    category: 'Integration',
    lastUpdated: '2025-09-21'
  },
  {
    id: 'system-architecture',
    name: 'System Architecture',
    file: '/docs/diagrams/system-architecture.svg',
    description: 'High-level system architecture overview',
    category: 'Architecture',
    lastUpdated: '2025-09-21'
  },
  {
    id: 'workflow-orchestration',
    name: 'Workflow Orchestration',
    file: '/docs/diagrams/workflow-orchestration.svg',
    description: 'Task orchestration and workflow management',
    category: 'Workflow',
    lastUpdated: '2025-09-21'
  },
  {
    id: 'sequence-observation-processing',
    name: 'Sequence Diagram - Observation Processing',
    file: '/docs/diagrams/sequence-observation-processing.svg',
    description: 'Sequence of operations for observation processing',
    category: 'Sequence',
    lastUpdated: '2025-09-21'
  },
  {
    id: 'linear-tickets-mapping',
    name: 'Linear Tickets Mapping',
    file: '/docs/diagrams/linear-tickets-mapping.svg',
    description: 'Project tickets and development workflow mapping',
    category: 'Project',
    lastUpdated: '2025-09-21'
  }
]

const categories = ['All', 'Architecture', 'Data', 'Integration', 'Workflow', 'Sequence', 'Project']

export default function DiagramsPage() {
  const [selectedDiagram, setSelectedDiagram] = useState(diagrams[0])
  const [selectedCategory, setSelectedCategory] = useState('All')
  const [isFullscreen, setIsFullscreen] = useState(false)

  const filteredDiagrams = selectedCategory === 'All'
    ? diagrams
    : diagrams.filter(diagram => diagram.category === selectedCategory)

  const handleDownload = () => {
    const link = document.createElement('a')
    link.href = selectedDiagram.file
    link.download = `${selectedDiagram.name}.svg`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const handleFullscreen = () => {
    setIsFullscreen(true)
  }

  const handleCloseFullscreen = () => {
    setIsFullscreen(false)
  }

  return (
    <div className="min-h-[calc(100vh-4rem)] bg-astrid-dark">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center space-x-4 mb-4">
            <Link
              href="/planning"
              className="inline-flex items-center px-3 py-2 text-sm font-medium text-gray-400 hover:text-white transition-colors"
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Planning
            </Link>
          </div>
          <div className="flex items-center space-x-6 mb-4">
            <div className="flex-shrink-0">
              <div className="w-16 h-16 bg-astrid-blue/20 rounded-lg flex items-center justify-center">
                <Image className="w-8 h-8 text-astrid-blue" />
              </div>
            </div>
            <div>
              <h1 className="text-4xl font-bold text-white mb-2">System Diagrams</h1>
              <p className="text-xl text-gray-400">
                Architecture and system diagrams for AstrID
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <span className="text-sm text-gray-300">8 diagrams available</span>
            </div>
            <div className="flex items-center space-x-2">
              <Calendar className="w-4 h-4 text-gray-400" />
              <span className="text-sm text-gray-300">Last updated: Sep 21, 2025</span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Sidebar - Diagram List */}
          <div className="lg:col-span-1">
            <div className="bg-gray-800 rounded-lg border border-gray-700">
              <div className="p-6 border-b border-gray-700">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold text-white">Diagrams</h2>
                </div>

                {/* Category Filter */}
                <div className="flex flex-wrap gap-2 mb-4">
                  {categories.map((category) => (
                    <button
                      key={category}
                      onClick={() => setSelectedCategory(category)}
                      className={`px-3 py-1 text-xs font-medium rounded-full transition-colors ${
                        selectedCategory === category
                          ? 'bg-astrid-blue text-white'
                          : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                      }`}
                    >
                      {category}
                    </button>
                  ))}
                </div>
              </div>

              <div className="p-4 space-y-2 max-h-96 overflow-y-auto">
                {filteredDiagrams.map((diagram) => (
                  <button
                    key={diagram.id}
                    onClick={() => setSelectedDiagram(diagram)}
                    className={`w-full text-left p-3 rounded-lg transition-colors ${
                      selectedDiagram.id === diagram.id
                        ? 'bg-astrid-blue/20 border border-astrid-blue/50'
                        : 'hover:bg-gray-700'
                    }`}
                  >
                    <div className="flex items-start space-x-3">
                      <div className="flex-shrink-0 mt-1">
                        <Image className="w-4 h-4 text-astrid-blue" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <h3 className="text-sm font-medium text-white truncate">
                          {diagram.name}
                        </h3>
                        <p className="text-xs text-gray-400 mt-1 line-clamp-2">
                          {diagram.description}
                        </p>
                        <div className="flex items-center space-x-2 mt-2">
                          <span className="text-xs text-gray-500">{diagram.category}</span>
                        </div>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Main Content - Diagram Viewer */}
          <div className="lg:col-span-2">
            <div className="bg-gray-800 rounded-lg border border-gray-700">
              <div className="p-6 border-b border-gray-700">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-xl font-bold text-white mb-2">
                      {selectedDiagram.name}
                    </h2>
                    <p className="text-gray-400">{selectedDiagram.description}</p>
                  </div>
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={handleDownload}
                      className="inline-flex items-center px-3 py-2 text-sm font-medium text-gray-300 hover:text-white transition-colors"
                    >
                      <Download className="w-4 h-4 mr-2" />
                      Download
                    </button>
                    <button
                      onClick={handleFullscreen}
                      className="inline-flex items-center px-3 py-2 text-sm font-medium text-gray-300 hover:text-white transition-colors"
                    >
                      <Eye className="w-4 h-4 mr-2" />
                      Fullscreen
                    </button>
                  </div>
                </div>
                <div className="flex items-center space-x-4 mt-4 text-sm text-gray-500">
                  <span>Category: {selectedDiagram.category}</span>
                  <span>•</span>
                  <span>Updated: {selectedDiagram.lastUpdated}</span>
                </div>
              </div>

              <div className="p-6">
                <div className="bg-gray-900 rounded-lg p-4 min-h-96">
                  <DiagramViewer file={selectedDiagram.file} />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Fullscreen Modal */}
      {isFullscreen && (
        <div className="fixed inset-0 bg-black bg-opacity-90 z-50 flex items-center justify-center p-4">
          <div className="relative w-full h-full flex flex-col">
            <div className="flex items-center justify-between p-4 bg-gray-800 border-b border-gray-700">
              <h2 className="text-xl font-bold text-white">{selectedDiagram.name}</h2>
              <button
                onClick={handleCloseFullscreen}
                className="inline-flex items-center px-4 py-2 text-sm font-medium text-gray-300 hover:text-white transition-colors"
              >
                <Eye className="w-4 h-4 mr-2" />
                Exit Fullscreen
              </button>
            </div>
            <div className="flex-1 p-4 overflow-auto">
              <div className="bg-gray-900 rounded-lg p-4 h-full">
                <DiagramViewer file={selectedDiagram.file} />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
