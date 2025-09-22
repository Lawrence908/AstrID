'use client'

import { useState } from 'react'
import Link from 'next/link'
import { ArrowLeft, FileText, Download, Eye, Calendar, Search, BookOpen, Code, Settings, Database } from 'lucide-react'
import DocumentationViewer from '@/components/DocumentationViewer'

const documents = [
  {
    id: 'architecture',
    name: 'Architecture Overview',
    file: '/docs/architecture.md',
    description: 'High-level system architecture and design principles',
    category: 'Architecture',
    lastUpdated: '2025-09-21',
    icon: <Settings className="w-4 h-4" />
  },
  {
    id: 'database-schema-design',
    name: 'Database Schema Design',
    file: '/docs/database-schema-design.md',
    description: 'Detailed database design and schema documentation',
    category: 'Database',
    lastUpdated: '2025-09-21',
    icon: <Database className="w-4 h-4" />
  },
  {
    id: 'design-overview',
    name: 'Design Overview',
    file: '/docs/design-overview.md',
    description: 'UI/UX design principles and component guidelines',
    category: 'Design',
    lastUpdated: '2025-09-21',
    icon: <BookOpen className="w-4 h-4" />
  },
  {
    id: 'development',
    name: 'Development Guide',
    file: '/docs/development.md',
    description: 'Developer setup, coding standards, and contribution guidelines',
    category: 'Development',
    lastUpdated: '2025-09-21',
    icon: <Code className="w-4 h-4" />
  },
  {
    id: 'linear-tickets',
    name: 'Linear Tickets',
    file: '/docs/linear-tickets.md',
    description: 'Project management and ticket tracking documentation',
    category: 'Project',
    lastUpdated: '2025-09-21',
    icon: <FileText className="w-4 h-4" />
  },
  {
    id: 'logging-guide',
    name: 'Logging Guide',
    file: '/docs/logging-guide.md',
    description: 'Logging standards and debugging practices',
    category: 'Development',
    lastUpdated: '2025-09-21',
    icon: <Code className="w-4 h-4" />
  },
  {
    id: 'migration-strategy',
    name: 'Migration Strategy',
    file: '/docs/migration-strategy.md',
    description: 'Data migration and system upgrade procedures',
    category: 'Operations',
    lastUpdated: '2025-09-21',
    icon: <Settings className="w-4 h-4" />
  },
  {
    id: 'tech-stack',
    name: 'Tech Stack',
    file: '/docs/tech-stack.md',
    description: 'Technology stack overview and dependency management',
    category: 'Architecture',
    lastUpdated: '2025-09-21',
    icon: <Settings className="w-4 h-4" />
  }
]

const categories = ['All', 'Architecture', 'Database', 'Design', 'Development', 'Project', 'Operations']

export default function DocumentationPage() {
  const [selectedDocument, setSelectedDocument] = useState(documents[0])
  const [selectedCategory, setSelectedCategory] = useState('All')
  const [searchQuery, setSearchQuery] = useState('')
  const [isFullscreen, setIsFullscreen] = useState(false)

  const filteredDocuments = documents.filter(doc => {
    const matchesCategory = selectedCategory === 'All' || doc.category === selectedCategory
    const matchesSearch = doc.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         doc.description.toLowerCase().includes(searchQuery.toLowerCase())
    return matchesCategory && matchesSearch
  })

  const handleDownload = () => {
    const link = document.createElement('a')
    link.href = selectedDocument.file
    link.download = `${selectedDocument.name}.md`
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
                <FileText className="w-8 h-8 text-astrid-blue" />
              </div>
            </div>
            <div>
              <h1 className="text-4xl font-bold text-white mb-2">Documentation</h1>
              <p className="text-xl text-gray-400">
                Technical documentation and guides for AstrID
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <span className="text-sm text-gray-300">8 documents available</span>
            </div>
            <div className="flex items-center space-x-2">
              <Calendar className="w-4 h-4 text-gray-400" />
              <span className="text-sm text-gray-300">Last updated: Sep 21, 2025</span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Sidebar - Document List */}
          <div className="lg:col-span-1">
            <div className="bg-gray-800 rounded-lg border border-gray-700">
              <div className="p-6 border-b border-gray-700">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold text-white">Documents</h2>
                </div>

                {/* Search */}
                <div className="relative mb-4">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search documents..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full pl-10 pr-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-astrid-blue focus:border-transparent"
                  />
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
                {filteredDocuments.map((document) => (
                  <button
                    key={document.id}
                    onClick={() => setSelectedDocument(document)}
                    className={`w-full text-left p-3 rounded-lg transition-colors ${
                      selectedDocument.id === document.id
                        ? 'bg-astrid-blue/20 border border-astrid-blue/50'
                        : 'hover:bg-gray-700'
                    }`}
                  >
                    <div className="flex items-start space-x-3">
                      <div className="flex-shrink-0 mt-1 text-astrid-blue">
                        {document.icon}
                      </div>
                      <div className="flex-1 min-w-0">
                        <h3 className="text-sm font-medium text-white truncate">
                          {document.name}
                        </h3>
                        <p className="text-xs text-gray-400 mt-1 line-clamp-2">
                          {document.description}
                        </p>
                        <div className="flex items-center space-x-2 mt-2">
                          <span className="text-xs text-gray-500">{document.category}</span>
                        </div>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Main Content - Document Viewer */}
          <div className="lg:col-span-2">
            <div className="bg-gray-800 rounded-lg border border-gray-700">
              <div className="p-6 border-b border-gray-700">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-xl font-bold text-white mb-2">
                      {selectedDocument.name}
                    </h2>
                    <p className="text-gray-400">{selectedDocument.description}</p>
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
                  <span>Category: {selectedDocument.category}</span>
                  <span>•</span>
                  <span>Updated: {selectedDocument.lastUpdated}</span>
                </div>
              </div>

              <div className="p-6">
                <div className="bg-gray-900 rounded-lg p-4 min-h-96">
                  <DocumentationViewer file={selectedDocument.file} />
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
              <h2 className="text-xl font-bold text-white">{selectedDocument.name}</h2>
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
                <DocumentationViewer file={selectedDocument.file} />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
