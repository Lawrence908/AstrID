'use client'

import { useState } from 'react'
import { FileText, Image, Database, GitBranch, Workflow, Users, Settings, BookOpen } from 'lucide-react'
import DiagramViewer from '@/components/DiagramViewer'
import DocumentationViewer from '@/components/DocumentationViewer'

const sections = [
  {
    id: 'diagrams',
    title: 'System Diagrams',
    icon: <Image className="w-5 h-5" />,
    items: [
      { name: 'Data Flow Pipeline', file: '/docs/diagrams/data-flow-pipeline.svg', type: 'svg' },
      { name: 'Database Schema', file: '/docs/diagrams/database-schema.svg', type: 'svg' },
      { name: 'Domain Interactions', file: '/docs/diagrams/domain-interactions.svg', type: 'svg' },
      { name: 'External Integrations', file: '/docs/diagrams/external-integrations.svg', type: 'svg' },
      { name: 'System Architecture', file: '/docs/diagrams/system-architecture.svg', type: 'svg' },
      { name: 'Workflow Orchestration', file: '/docs/diagrams/workflow-orchestration.svg', type: 'svg' },
      { name: 'Sequence Diagram', file: '/docs/diagrams/sequence-observation-processing.svg', type: 'svg' },
      { name: 'Linear Tickets Mapping', file: '/docs/diagrams/linear-tickets-mapping.svg', type: 'svg' },
    ]
  },
  {
    id: 'documentation',
    title: 'Documentation',
    icon: <FileText className="w-5 h-5" />,
    items: [
      { name: 'Architecture Overview', file: '/docs/architecture.md', type: 'markdown' },
      { name: 'Database Schema Design', file: '/docs/database-schema-design.md', type: 'markdown' },
      { name: 'Design Overview', file: '/docs/design-overview.md', type: 'markdown' },
      { name: 'Development Guide', file: '/docs/development.md', type: 'markdown' },
      { name: 'Linear Tickets', file: '/docs/linear-tickets.md', type: 'markdown' },
      { name: 'Logging Guide', file: '/docs/logging-guide.md', type: 'markdown' },
      { name: 'Migration Strategy', file: '/docs/migration-strategy.md', type: 'markdown' },
      { name: 'Tech Stack', file: '/docs/tech-stack.md', type: 'markdown' },
    ]
  },
  {
    id: 'models',
    title: 'Data Models',
    icon: <Database className="w-5 h-5" />,
    items: [
      { name: 'Consolidated Models', file: '/docs/consolidated-models.py', type: 'code' },
    ]
  }
]

export default function Home() {
  const [activeSection, setActiveSection] = useState('diagrams')
  const [activeItem, setActiveItem] = useState(sections[0].items[0])

  const handleItemClick = (item: any) => {
    setActiveItem(item)
  }

  return (
    <div className="min-h-screen bg-astrid-dark">
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <h1 className="text-2xl font-bold text-astrid-blue">AstrID</h1>
              </div>
              <div className="ml-4">
                <p className="text-sm text-gray-400">Planning Dashboard</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-400">localhost:3000</span>
            </div>
          </div>
        </div>
      </header>

      <div className="flex h-[calc(100vh-4rem)]">
        {/* Sidebar */}
        <div className="w-80 bg-gray-900 border-r border-gray-700 overflow-y-auto">
          <div className="p-4">
            <nav className="space-y-2">
              {sections.map((section) => (
                <div key={section.id}>
                  <button
                    onClick={() => setActiveSection(section.id)}
                    className={`w-full flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                      activeSection === section.id
                        ? 'bg-astrid-blue text-white'
                        : 'text-gray-300 hover:bg-gray-800 hover:text-white'
                    }`}
                  >
                    {section.icon}
                    <span className="ml-3">{section.title}</span>
                  </button>

                  {activeSection === section.id && (
                    <div className="mt-2 ml-6 space-y-1">
                      {section.items.map((item, index) => (
                        <button
                          key={index}
                          onClick={() => handleItemClick(item)}
                          className={`w-full text-left px-3 py-2 text-sm rounded-md transition-colors ${
                            activeItem === item
                              ? 'bg-gray-700 text-astrid-blue'
                              : 'text-gray-400 hover:bg-gray-800 hover:text-white'
                          }`}
                        >
                          {item.name}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </nav>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 overflow-y-auto">
          <div className="p-6">
            <div className="mb-6">
              <h2 className="text-3xl font-bold text-white mb-2">{activeItem.name}</h2>
              <p className="text-gray-400">
                {activeItem.type === 'svg' && 'System diagram visualization'}
                {activeItem.type === 'markdown' && 'Documentation and guides'}
                {activeItem.type === 'code' && 'Data models and schemas'}
              </p>
            </div>

            <div className="bg-gray-800 rounded-lg p-6">
              {activeItem.type === 'svg' && (
                <DiagramViewer file={activeItem.file} />
              )}
              {activeItem.type === 'markdown' && (
                <DocumentationViewer file={activeItem.file} />
              )}
              {activeItem.type === 'code' && (
                <DocumentationViewer file={activeItem.file} />
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
