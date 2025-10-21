'use client'

import { useState, useEffect } from 'react'
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
import { dashboardApi, WorkflowStats } from '@/lib/api/dashboard'

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
  const [workflows, setWorkflows] = useState<any[]>([])
  const [stats, setStats] = useState<WorkflowStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const handleFilterChange = (key: string, value: string) => {
    setFilters(prev => ({ ...prev, [key]: value }))
  }

  const handleWorkflowAction = (workflowId: string, action: string) => {
    console.log(`Action ${action} on workflow ${workflowId}`)
    // In a real app, this would trigger the actual workflow action
  }

  const fetchData = async () => {
    try {
      setLoading(true)
      setError(null)
      
      const [statsData, workflowsData] = await Promise.all([
        dashboardApi.getWorkflowStats(),
        dashboardApi.getWorkflows(filters)
      ])
      
      setStats(statsData)
      setWorkflows(workflowsData)
    } catch (err) {
      console.error('Failed to fetch workflows data:', err)
      setError('Failed to load workflows data')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [filters])

  const handleRefresh = () => {
    fetchData()
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
              <button 
                onClick={handleRefresh}
                disabled={loading}
                className="flex items-center px-4 py-2 bg-gray-800 text-gray-200 rounded-lg hover:bg-gray-700 transition-colors border border-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
                {loading ? 'Loading...' : 'Refresh'}
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
                <p className="text-2xl font-bold text-white">
                  {loading ? '—' : stats?.total_workflows?.toLocaleString() || '0'}
                </p>
                {!loading && stats && (
                  <p className="text-xs text-green-500 mt-1 font-medium">Live data</p>
                )}
              </div>
              <Workflow className="w-8 h-8 text-astrid-blue" />
            </div>
          </div>
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Running</p>
                <p className="text-2xl font-bold text-blue-500">
                  {loading ? '—' : stats?.running?.toLocaleString() || '0'}
                </p>
                {!loading && stats && (
                  <p className="text-xs text-green-500 mt-1 font-medium">Live data</p>
                )}
              </div>
              <Play className="w-8 h-8 text-blue-500" />
            </div>
          </div>
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Completed</p>
                <p className="text-2xl font-bold text-green-500">
                  {loading ? '—' : stats?.completed?.toLocaleString() || '0'}
                </p>
                {!loading && stats && (
                  <p className="text-xs text-green-500 mt-1 font-medium">Live data</p>
                )}
              </div>
              <CheckCircle className="w-8 h-8 text-green-500" />
            </div>
          </div>
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Failed</p>
                <p className="text-2xl font-bold text-red-500">
                  {loading ? '—' : stats?.failed?.toLocaleString() || '0'}
                </p>
                {!loading && stats && (
                  <p className="text-xs text-green-500 mt-1 font-medium">Live data</p>
                )}
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
                {loading ? (
                  <div className="p-8 text-center text-gray-400">
                    <div className="flex items-center justify-center space-x-2">
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      <span>Loading workflows...</span>
                    </div>
                  </div>
                ) : error ? (
                  <div className="p-8 text-center text-red-400">
                    <div className="flex items-center justify-center space-x-2">
                      <AlertTriangle className="w-4 h-4" />
                      <span>{error}</span>
                    </div>
                  </div>
                ) : workflows.length === 0 ? (
                  <div className="p-8 text-center text-gray-400">
                    No workflows found
                  </div>
                ) : (
                  workflows.map((workflow) => {
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
                                    style={{ width: `${workflow.progress || 0}%` }}
                                  ></div>
                                </div>
                                <span className="text-sm text-white">{workflow.progress || 0}%</span>
                              </div>
                            </div>

                            <div>
                              <p className="text-sm text-gray-400">Tasks</p>
                              <p className="text-sm text-white">
                                {workflow.tasks?.completed || 0}/{workflow.tasks?.total || 0} completed
                                {(workflow.tasks?.failed || 0) > 0 && (
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
                                <p className="text-sm text-white">{workflow.resources?.cpu || 0}%</p>
                              </div>
                              <div>
                                <p className="text-sm text-gray-400">Memory</p>
                                <p className="text-sm text-white">{workflow.resources?.memory || 0}%</p>
                              </div>
                              <div>
                                <p className="text-sm text-gray-400">Storage</p>
                                <p className="text-sm text-white">{workflow.resources?.storage || 0}%</p>
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
                  })
                )}
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
