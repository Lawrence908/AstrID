'use client'

import { useState } from 'react'
import {
  Workflow,
  Play,
  Pause,
  Square,
  RefreshCw,
  AlertTriangle,
  CheckCircle,
  Clock,
  BarChart3,
  Settings,
  Eye,
  Download,
  Activity,
  Zap
} from 'lucide-react'

// MOCK DATA for demonstration - All data below is simulated
const mockWorkflows = [
  {
    id: 'wf-001',
    name: 'Observation Processing Pipeline',
    status: 'running',
    progress: 75,
    startTime: '2025-01-15T10:00:00Z',
    duration: '2h 15m',
    tasks: {
      total: 8,
      completed: 6,
      failed: 0,
      pending: 2
    },
    resources: {
      cpu: 45,
      memory: 67,
      storage: 23
    }
  },
  {
    id: 'wf-002',
    name: 'Detection Analysis Workflow',
    status: 'completed',
    progress: 100,
    startTime: '2025-01-15T08:30:00Z',
    duration: '1h 45m',
    tasks: {
      total: 5,
      completed: 5,
      failed: 0,
      pending: 0
    },
    resources: {
      cpu: 0,
      memory: 0,
      storage: 0
    }
  },
  {
    id: 'wf-003',
    name: 'Data Export Pipeline',
    status: 'failed',
    progress: 30,
    startTime: '2025-01-15T12:00:00Z',
    duration: '45m',
    tasks: {
      total: 4,
      completed: 1,
      failed: 1,
      pending: 2
    },
    resources: {
      cpu: 0,
      memory: 0,
      storage: 0
    },
    error: 'Storage quota exceeded'
  },
  {
    id: 'wf-004',
    name: 'Model Training Job',
    status: 'pending',
    progress: 0,
    startTime: null,
    duration: null,
    tasks: {
      total: 12,
      completed: 0,
      failed: 0,
      pending: 12
    },
    resources: {
      cpu: 0,
      memory: 0,
      storage: 0
    }
  }
]

const statusConfig = {
  running: { color: 'text-blue-500', bg: 'bg-blue-900', icon: Play, label: 'Running' },
  completed: { color: 'text-green-500', bg: 'bg-green-900', icon: CheckCircle, label: 'Completed' },
  failed: { color: 'text-red-500', bg: 'bg-red-900', icon: AlertTriangle, label: 'Failed' },
  pending: { color: 'text-yellow-500', bg: 'bg-yellow-900', icon: Clock, label: 'Pending' },
  paused: { color: 'text-gray-500', bg: 'bg-gray-900', icon: Pause, label: 'Paused' }
}

export default function WorkflowsPage() {
  const [selectedWorkflow, setSelectedWorkflow] = useState<string | null>(null)
  const [filters, setFilters] = useState({
    status: '',
    type: '',
    dateRange: ''
  })

  const handleFilterChange = (key: string, value: string) => {
    setFilters(prev => ({ ...prev, [key]: value }))
  }

  const handleWorkflowAction = (workflowId: string, action: string) => {
    console.log(`Action ${action} on workflow ${workflowId}`)
    // In a real app, this would trigger the actual workflow action
  }

  return (
    <div className="min-h-[calc(100vh-4rem)] bg-astrid-dark">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-white mb-2">Workflows</h1>
              <p className="text-gray-400">Monitor processing pipelines and orchestration</p>
            </div>
            <div className="flex space-x-3">
              <button className="flex items-center px-4 py-2 bg-astrid-blue text-white rounded-lg hover:bg-blue-600 transition-colors">
                <Workflow className="w-4 h-4 mr-2" />
                New Workflow
              </button>
              <button className="flex items-center px-4 py-2 bg-gray-800 text-gray-200 rounded-lg hover:bg-gray-700 transition-colors border border-gray-700">
                <RefreshCw className="w-4 h-4 mr-2" />
                Refresh
              </button>
            </div>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Total Workflows</p>
                <p className="text-2xl font-bold text-white">24</p>
                <p className="text-xs text-yellow-500 mt-1 font-medium">(MOCK data)</p>
              </div>
              <Workflow className="w-8 h-8 text-astrid-blue" />
            </div>
          </div>
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Running</p>
                <p className="text-2xl font-bold text-blue-500">3</p>
                <p className="text-xs text-yellow-500 mt-1 font-medium">(MOCK data)</p>
              </div>
              <Play className="w-8 h-8 text-blue-500" />
            </div>
          </div>
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Completed</p>
                <p className="text-2xl font-bold text-green-500">18</p>
                <p className="text-xs text-yellow-500 mt-1 font-medium">(MOCK data)</p>
              </div>
              <CheckCircle className="w-8 h-8 text-green-500" />
            </div>
          </div>
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Failed</p>
                <p className="text-2xl font-bold text-red-500">3</p>
                <p className="text-xs text-yellow-500 mt-1 font-medium">(MOCK data)</p>
              </div>
              <AlertTriangle className="w-8 h-8 text-red-500" />
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Filters Sidebar */}
          <div className="lg:col-span-1">
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h3 className="text-lg font-semibold text-white mb-4">Filters</h3>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Status</label>
                  <select
                    value={filters.status}
                    onChange={(e) => handleFilterChange('status', e.target.value)}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-astrid-blue focus:border-transparent"
                  >
                    <option value="">All Status</option>
                    <option value="running">Running</option>
                    <option value="completed">Completed</option>
                    <option value="failed">Failed</option>
                    <option value="pending">Pending</option>
                    <option value="paused">Paused</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Type</label>
                  <select
                    value={filters.type}
                    onChange={(e) => handleFilterChange('type', e.target.value)}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-astrid-blue focus:border-transparent"
                  >
                    <option value="">All Types</option>
                    <option value="processing">Processing</option>
                    <option value="analysis">Analysis</option>
                    <option value="export">Export</option>
                    <option value="training">Training</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Date Range</label>
                  <input
                    type="date"
                    value={filters.dateRange}
                    onChange={(e) => handleFilterChange('dateRange', e.target.value)}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-astrid-blue focus:border-transparent"
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Workflows List */}
          <div className="lg:col-span-3">
            <div className="bg-gray-800 rounded-lg border border-gray-700">
              <div className="px-6 py-4 border-b border-gray-700">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-white">Workflows</h3>
                  <div className="flex items-center space-x-2">
                    <button className="p-2 text-gray-400 hover:text-white transition-colors">
                      <BarChart3 className="w-4 h-4" />
                    </button>
                    <button className="p-2 text-gray-400 hover:text-white transition-colors">
                      <Settings className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>

              <div className="divide-y divide-gray-700">
                {mockWorkflows.map((workflow) => {
                  const statusInfo = statusConfig[workflow.status as keyof typeof statusConfig]
                  const StatusIcon = statusInfo.icon

                  return (
                    <div key={workflow.id} className="p-6 hover:bg-gray-700 transition-colors">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center space-x-3 mb-2">
                            <h4 className="text-lg font-semibold text-white">{workflow.name}</h4>
                            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${statusInfo.bg} ${statusInfo.color}`}>
                              <StatusIcon className="w-3 h-3 mr-1" />
                              {statusInfo.label}
                            </span>
                          </div>

                          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                            <div>
                              <p className="text-sm text-gray-400">Progress</p>
                              <div className="flex items-center space-x-2">
                                <div className="flex-1 bg-gray-700 rounded-full h-2">
                                  <div
                                    className="bg-astrid-blue h-2 rounded-full transition-all duration-300"
                                    style={{ width: `${workflow.progress}%` }}
                                  ></div>
                                </div>
                                <span className="text-sm text-white">{workflow.progress}%</span>
                              </div>
                            </div>

                            <div>
                              <p className="text-sm text-gray-400">Tasks</p>
                              <p className="text-sm text-white">
                                {workflow.tasks.completed}/{workflow.tasks.total} completed
                                {workflow.tasks.failed > 0 && (
                                  <span className="text-red-500 ml-1">({workflow.tasks.failed} failed)</span>
                                )}
                              </p>
                            </div>

                            <div>
                              <p className="text-sm text-gray-400">Duration</p>
                              <p className="text-sm text-white">{workflow.duration || 'Not started'}</p>
                            </div>
                          </div>

                          {workflow.status === 'running' && (
                            <div className="grid grid-cols-3 gap-4 mb-4">
                              <div>
                                <p className="text-sm text-gray-400">CPU Usage</p>
                                <p className="text-sm text-white">{workflow.resources.cpu}%</p>
                              </div>
                              <div>
                                <p className="text-sm text-gray-400">Memory</p>
                                <p className="text-sm text-white">{workflow.resources.memory}%</p>
                              </div>
                              <div>
                                <p className="text-sm text-gray-400">Storage</p>
                                <p className="text-sm text-white">{workflow.resources.storage}%</p>
                              </div>
                            </div>
                          )}

                          {workflow.error && (
                            <div className="bg-red-900 bg-opacity-20 border border-red-800 rounded-lg p-3 mb-4">
                              <div className="flex items-center space-x-2">
                                <AlertTriangle className="w-4 h-4 text-red-500" />
                                <span className="text-sm text-red-300">{workflow.error}</span>
                              </div>
                            </div>
                          )}

                          <div className="flex items-center space-x-4 text-sm text-gray-400">
                            <div className="flex items-center space-x-1">
                              <Clock className="w-3 h-3" />
                              <span>Started: {workflow.startTime ? new Date(workflow.startTime).toLocaleString() : 'Not started'}</span>
                            </div>
                            <div className="flex items-center space-x-1">
                              <Activity className="w-3 h-3" />
                              <span>ID: {workflow.id}</span>
                            </div>
                          </div>
                        </div>

                        <div className="flex items-center space-x-2 ml-6">
                          {workflow.status === 'running' && (
                            <>
                              <button
                                onClick={() => handleWorkflowAction(workflow.id, 'pause')}
                                className="p-2 text-yellow-500 hover:text-yellow-400 transition-colors"
                                title="Pause"
                              >
                                <Pause className="w-4 h-4" />
                              </button>
                              <button
                                onClick={() => handleWorkflowAction(workflow.id, 'stop')}
                                className="p-2 text-red-500 hover:text-red-400 transition-colors"
                                title="Stop"
                              >
                                <Square className="w-4 h-4" />
                              </button>
                            </>
                          )}
                          {workflow.status === 'paused' && (
                            <button
                              onClick={() => handleWorkflowAction(workflow.id, 'resume')}
                              className="p-2 text-green-500 hover:text-green-400 transition-colors"
                              title="Resume"
                            >
                              <Play className="w-4 h-4" />
                            </button>
                          )}
                          {workflow.status === 'pending' && (
                            <button
                              onClick={() => handleWorkflowAction(workflow.id, 'start')}
                              className="p-2 text-blue-500 hover:text-blue-400 transition-colors"
                              title="Start"
                            >
                              <Play className="w-4 h-4" />
                            </button>
                          )}
                          <button
                            onClick={() => setSelectedWorkflow(workflow.id)}
                            className="p-2 text-astrid-blue hover:text-blue-400 transition-colors"
                            title="View Details"
                          >
                            <Eye className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => handleWorkflowAction(workflow.id, 'download')}
                            className="p-2 text-gray-400 hover:text-white transition-colors"
                            title="Download Logs"
                          >
                            <Download className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          </div>
        </div>

        {/* Workflow Detail Modal */}
        {selectedWorkflow && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-gray-800 rounded-lg p-6 max-w-4xl w-full mx-4 border border-gray-700">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-white">Workflow Details</h3>
                <button
                  onClick={() => setSelectedWorkflow(null)}
                  className="text-gray-400 hover:text-white transition-colors"
                >
                  ×
                </button>
              </div>
              <div className="space-y-4">
                <p className="text-gray-300">Detailed workflow information will be displayed here.</p>
                <p className="text-sm text-gray-500">This is a placeholder for the full workflow detail view with task breakdown, logs, and metrics.</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
