'use client'

import { useAuth } from '@/lib/auth/AuthProvider'
import Link from 'next/link'
import { useEffect, useState } from 'react'
import {
  Eye,
  Search,
  Users,
  Settings,
  BarChart3,
  FileText,
  Image,
  Database,
  GitBranch,
  Workflow,
  BookOpen,
  Star,
  Activity,
  AlertTriangle,
  Server,
  Gauge,
  Cpu,
  Zap,
  TrendingUp
} from 'lucide-react'
import { dashboardApi } from '@/lib/api/dashboard'

const mainSections = [
  {
    id: 'observations',
    title: 'Observations',
    description: 'Manage astronomical observations and survey data',
    icon: <Eye className="w-8 h-8" />,
    href: '/dashboard/observations',
    status: 'in-development',
    features: ['Observation Overview', 'Survey Integration', 'FITS Processing', 'Quality Assessment']
  },
  {
    id: 'detections',
    title: 'Detections',
    description: 'View and analyze anomaly detections',
    icon: <Search className="w-8 h-8" />,
    href: '/dashboard/detections',
    status: 'planned',
    features: ['Detection Visualization', 'Image Analysis', 'Anomaly Scoring', 'Human Validation']
  },
  {
    id: 'workflows',
    title: 'Workflows',
    description: 'Monitor processing pipelines and orchestration',
    icon: <Workflow className="w-8 h-8" />,
    href: '/dashboard/workflows',
    status: 'planned',
    features: ['Pipeline Monitoring', 'Task Management', 'Error Handling', 'Performance Metrics']
  },
  {
    id: 'analytics',
    title: 'Analytics',
    description: 'System metrics and performance analytics',
    icon: <BarChart3 className="w-8 h-8" />,
    href: '/dashboard/analytics',
    status: 'planned',
    features: ['System Metrics', 'Performance Charts', 'Usage Statistics', 'Trend Analysis']
  },
  {
    id: 'users',
    title: 'User Management',
    description: 'Manage users, roles, and permissions',
    icon: <Users className="w-8 h-8" />,
    href: '/dashboard/users',
    status: 'planned',
    features: ['User Administration', 'Role Management', 'Permission Control', 'Activity Tracking']
  },
  {
    id: 'settings',
    title: 'Settings',
    description: 'System configuration and preferences',
    icon: <Settings className="w-8 h-8" />,
    href: '/dashboard/settings',
    status: 'planned',
    features: ['System Configuration', 'API Settings', 'Notification Preferences', 'Data Management']
  }
]

const planningSections = [
  {
    id: 'diagrams',
    title: 'System Diagrams',
    icon: <Image className="w-5 h-5" />,
    href: '/planning/diagrams',
    description: 'Architecture and system diagrams'
  },
  {
    id: 'documentation',
    title: 'Documentation',
    icon: <FileText className="w-5 h-5" />,
    href: '/planning/documentation',
    description: 'Technical documentation and guides'
  },
  {
    id: 'tickets',
    title: 'Linear Tickets',
    icon: <GitBranch className="w-5 h-5" />,
    href: '/planning/tickets',
    description: 'Project tickets and progress tracking'
  }
]

export default function Home() {
  const { session } = useAuth()
  const [workersHealth, setWorkersHealth] = useState<any | null>(null)
  const [workersMetrics, setWorkersMetrics] = useState<any | null>(null)
  const [queueStatus, setQueueStatus] = useState<any[] | null>(null)
  const [workersLoading, setWorkersLoading] = useState<boolean>(true)
  const [workersError, setWorkersError] = useState<string | null>(null)
  const [modelMetrics, setModelMetrics] = useState<any | null>(null)
  const [modelLoading, setModelLoading] = useState<boolean>(true)
  const [modelError, setModelError] = useState<string | null>(null)

  useEffect(() => {
    const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000'
    let isMounted = true

    const withTimeout = async (url: string, ms: number) => {
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), ms)
      try {
        const res = await fetch(url, { signal: controller.signal })
        return res.ok ? await res.json().catch(() => null) : null
      } catch {
        return null
      } finally {
        clearTimeout(timeoutId)
      }
    }

    const fetchWorkers = async () => {
      try {
        setWorkersError(null)
        setWorkersLoading(true)

        const results = await Promise.allSettled([
          withTimeout(`${baseUrl}/workers/health`, 3000),
          withTimeout(`${baseUrl}/workers/metrics?time_window_hours=24`, 3000),
          withTimeout(`${baseUrl}/workers/queues`, 3000)
        ])

        if (!isMounted) return

        const health = results[0].status === 'fulfilled' ? results[0].value : null
        const metrics = results[1].status === 'fulfilled' ? results[1].value : null
        const queues = results[2].status === 'fulfilled' ? results[2].value : null

        setWorkersHealth(health)
        setWorkersMetrics(metrics)
        setQueueStatus(Array.isArray(queues) ? queues : queues?.queues || null)
        setWorkersLoading(false)
      } catch (e: any) {
        if (!isMounted) return
        setWorkersError('Unable to reach workers API')
        setWorkersLoading(false)
      }
    }

    fetchWorkers()
    const id = setInterval(fetchWorkers, 10000)
    return () => {
      isMounted = false
      clearInterval(id)
    }
  }, [])

  // Fetch model metrics
  useEffect(() => {
    const fetchModelMetrics = async () => {
      try {
        setModelLoading(true)
        setModelError(null)
        const metrics = await dashboardApi.getLatestModelMetrics()
        setModelMetrics(metrics)
      } catch (err) {
        console.error('Failed to fetch model metrics:', err)
        setModelError('Failed to load model metrics')
      } finally {
        setModelLoading(false)
      }
    }

    fetchModelMetrics()
  }, [])

  if (!session) {
    return (
      <div className="min-h-[calc(100vh-4rem)] flex items-center justify-center px-4">
        <div className="text-center">
          <div className="mb-8">
            <div className="relative inline-block">
              <img
                src="/images/jwst.png"
                alt="JWST Space Telescope"
                className="w-64 h-64 mx-auto rounded-full object-cover shadow-2xl transform rotate-3 transition-all duration-500 ease-out hover:scale-110 hover:rotate-0 hover:shadow-3xl"
              />
              <div className="absolute inset-0 rounded-full bg-gradient-to-br from-astrid-blue/20 via-transparent to-purple-500/20 animate-pulse"></div>
              <div className="absolute -inset-4 rounded-full bg-gradient-to-r from-astrid-blue/30 via-transparent to-astrid-blue/30 blur-xl opacity-50"></div>
            </div>
          </div>
          <h1 className="text-4xl font-bold text-white mb-4">Welcome to AstrID</h1>
          <p className="text-xl text-gray-400 mb-8">Astronomical Identification System</p>
          <Link
            href="/login"
            className="inline-flex items-center px-6 py-3 bg-astrid-blue text-white font-semibold rounded-lg hover:bg-blue-600 transition-colors"
          >
            Sign In to Continue
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-[calc(100vh-4rem)] bg-astrid-dark">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center space-x-6 mb-4">
            <div className="flex-shrink-0">
              <img
                src="/images/jwst.png"
                alt="JWST Space Telescope"
                className="w-16 h-16 rounded-lg object-cover border-2 border-astrid-blue"
              />
            </div>
            <div>
              <h1 className="text-4xl font-bold text-white mb-2">AstrID Dashboard</h1>
              <p className="text-xl text-gray-400">
                Astronomical Identification: Temporal Dataset Preparation and Anomaly Detection
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <span className="text-sm text-gray-300">System Online</span>
            </div>
            <div className="flex items-center space-x-2">
              <Activity className="w-4 h-4 text-gray-400" />
              <span className="text-sm text-gray-300">14/32 tickets completed (43.8%)</span>
            </div>
          </div>
        </div>

        {/* Main Sections Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
          {mainSections.map((section) => (
            <Link
              key={section.id}
              href={section.href}
              className="group bg-gray-800 rounded-lg p-6 hover:bg-gray-700 transition-colors border border-gray-700 hover:border-gray-600"
            >
              <div className="flex items-start space-x-4">
                <div className="flex-shrink-0 text-astrid-blue group-hover:text-blue-400 transition-colors">
                  {section.icon}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-lg font-semibold text-white group-hover:text-blue-100 transition-colors">
                      {section.title}
                    </h3>
                    <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                      section.status === 'in-development'
                        ? 'bg-yellow-900 text-yellow-300'
                        : 'bg-gray-700 text-gray-300'
                    }`}>
                      {section.status === 'in-development' ? 'In Development' : 'Planned'}
                    </span>
                  </div>
                  <p className="text-gray-400 text-sm mb-3">{section.description}</p>
                  <div className="space-y-1">
                    {section.features.map((feature, index) => (
                      <div key={index} className="flex items-center space-x-2">
                        <Star className="w-3 h-3 text-gray-500" />
                        <span className="text-xs text-gray-500">{feature}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </Link>
          ))}
        </div>

        {/* Planning Dashboard Section */}
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-2xl font-bold text-white mb-2">Planning Dashboard</h2>
              <p className="text-gray-400">Development documentation, diagrams, and project tracking</p>
            </div>
            <Link
              href="/planning"
              className="inline-flex items-center px-4 py-2 bg-astrid-blue text-white font-medium rounded-lg hover:bg-blue-600 transition-colors"
            >
              <BookOpen className="w-4 h-4 mr-2" />
              Open Planning Dashboard
            </Link>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {planningSections.map((section) => (
              <Link
                key={section.id}
                href={section.href}
                className="group bg-gray-700 rounded-lg p-4 hover:bg-gray-600 transition-colors"
              >
                <div className="flex items-center space-x-3">
                  <div className="text-astrid-blue group-hover:text-blue-400 transition-colors">
                    {section.icon}
                  </div>
                  <div>
                    <h3 className="font-medium text-white group-hover:text-blue-100 transition-colors">
                      {section.title}
                    </h3>
                    <p className="text-sm text-gray-400">{section.description}</p>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </div>

        {/* Quick Stats */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <div className="flex items-center space-x-2">
              <Eye className="w-5 h-5 text-astrid-blue" />
              <span className="text-sm font-medium text-gray-300">Observations</span>
            </div>
            <p className="text-2xl font-bold text-white mt-2">0</p>
            <p className="text-xs text-gray-500">Ready for processing</p>
            <p className="text-xs text-green-500 mt-1 font-medium">Live data</p>
          </div>
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <div className="flex items-center space-x-2">
              <Search className="w-5 h-5 text-green-500" />
              <span className="text-sm font-medium text-gray-300">Detections</span>
            </div>
            <p className="text-2xl font-bold text-white mt-2">0</p>
            <p className="text-xs text-gray-500">Anomalies found</p>
            <p className="text-xs text-green-500 mt-1 font-medium">Live data</p>
          </div>
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <div className="flex items-center space-x-2">
              <Activity className="w-5 h-5 text-yellow-500" />
              <span className="text-sm font-medium text-gray-300">Processing</span>
            </div>
            <p className="text-2xl font-bold text-white mt-2">0</p>
            <p className="text-xs text-gray-500">Active workflows</p>
            <p className="text-xs text-green-500 mt-1 font-medium">Live data</p>
          </div>
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <div className="flex items-center space-x-2">
              <AlertTriangle className="w-5 h-5 text-red-500" />
              <span className="text-sm font-medium text-gray-300">Alerts</span>
            </div>
            <p className="text-2xl font-bold text-white mt-2">0</p>
            <p className="text-xs text-gray-500">System alerts</p>
            <p className="text-xs text-green-500 mt-1 font-medium">Live data</p>
          </div>
        </div>

        {/* Model Performance Metrics */}
        {modelMetrics && (
          <div className="mt-8">
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <TrendingUp className="w-5 h-5 text-astrid-blue" />
                  <h2 className="text-xl font-bold text-white">Latest Model Performance</h2>
                  <span className="px-2 py-1 text-xs font-medium rounded-full bg-green-900 text-green-300">
                    Live Training Data
                  </span>
                </div>
                <div className="flex items-center space-x-2">
                  <Zap className="w-4 h-4 text-yellow-500" />
                  <span className="text-sm text-gray-300">
                    Last training: {new Date(modelMetrics.last_training).toLocaleDateString()}
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-700">
                  <div className="flex items-center space-x-2">
                    <BarChart3 className="w-4 h-4 text-green-400" />
                    <span className="text-sm text-gray-300">Accuracy</span>
                  </div>
                  <p className="text-2xl font-bold text-white mt-2">
                    {(modelMetrics.final_accuracy * 100).toFixed(1)}%
                  </p>
                  <p className="text-xs text-gray-500">Final test accuracy</p>
                </div>

                <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-700">
                  <div className="flex items-center space-x-2">
                    <Star className="w-4 h-4 text-blue-400" />
                    <span className="text-sm text-gray-300">F1 Score</span>
                  </div>
                  <p className="text-2xl font-bold text-white mt-2">
                    {(modelMetrics.final_f1_score * 100).toFixed(1)}%
                  </p>
                  <p className="text-xs text-gray-500">Macro F1 score</p>
                </div>

                <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-700">
                  <div className="flex items-center space-x-2">
                    <Gauge className="w-4 h-4 text-purple-400" />
                    <span className="text-sm text-gray-300">Best Val Loss</span>
                  </div>
                  <p className="text-2xl font-bold text-white mt-2">
                    {modelMetrics.best_val_loss.toFixed(4)}
                  </p>
                  <p className="text-xs text-gray-500">Lowest validation loss</p>
                </div>

                <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-700">
                  <div className="flex items-center space-x-2">
                    <Zap className="w-4 h-4 text-yellow-400" />
                    <span className="text-sm text-gray-300">Energy Used</span>
                  </div>
                  <p className="text-2xl font-bold text-white mt-2">
                    {modelMetrics.training_energy_wh.toFixed(1)} Wh
                  </p>
                  <p className="text-xs text-gray-500">
                    CO₂: {modelMetrics.training_carbon_footprint_kg.toFixed(6)} kg
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Workers Status Card */}
        <div className="mt-8">
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-3">
                <Server className="w-5 h-5 text-astrid-blue" />
                <h2 className="text-xl font-bold text-white">Workers</h2>
                <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                  workersHealth?.status === 'healthy'
                    ? 'bg-green-900 text-green-300'
                    : workersHealth?.status === 'degraded'
                    ? 'bg-yellow-900 text-yellow-300'
                    : 'bg-gray-700 text-gray-300'
                }`}>
                  {workersLoading ? 'Checking…' : workersHealth?.status ?? 'Unknown'}
                </span>
              </div>
              <div className="flex items-center space-x-2">
                <Link
                  href="/dashboard/workflows"
                  className="text-sm px-3 py-1.5 rounded-md bg-gray-700 hover:bg-gray-600 text-gray-200"
                >
                  View Workflows
                </Link>
              </div>
            </div>

            {workersError && (
              <div className="flex items-center space-x-2 text-red-400 text-sm mb-4">
                <AlertTriangle className="w-4 h-4" />
                <span>{workersError}</span>
              </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-700">
                <div className="flex items-center space-x-2">
                  <Activity className="w-4 h-4 text-green-400" />
                  <span className="text-sm text-gray-300">Active Workers</span>
                </div>
                <p className="text-2xl font-bold text-white mt-2">
                  {workersLoading ? '—' : workersHealth?.active_workers ?? workersHealth?.total_workers ?? 0}
                </p>
                <p className="text-xs text-gray-500">Healthy: {workersLoading ? '—' : workersHealth?.healthy_workers ?? 0}</p>
              </div>

              <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-700">
                <div className="flex items-center space-x-2">
                  <Gauge className="w-4 h-4 text-astrid-blue" />
                  <span className="text-sm text-gray-300">Throughput (24h)</span>
                </div>
                <p className="text-2xl font-bold text-white mt-2">
                  {workersLoading ? '—' : workersMetrics?.total_tasks_processed ?? 0}
                </p>
                <p className="text-xs text-gray-500">Failed: {workersLoading ? '—' : workersMetrics?.total_tasks_failed ?? 0}</p>
              </div>

              <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-700">
                <div className="flex items-center space-x-2">
                  <Cpu className="w-4 h-4 text-yellow-400" />
                  <span className="text-sm text-gray-300">Avg Proc Time</span>
                </div>
                <p className="text-2xl font-bold text-white mt-2">
                  {workersLoading ? '—' : (workersMetrics?.average_processing_time?.toFixed?.(2) ?? '0.00')}s
                </p>
                <p className="text-xs text-gray-500">Failure Rate: {workersLoading ? '—' : ((workersMetrics?.failure_rate ?? 0) * 100).toFixed(2)}%</p>
              </div>

              <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-700">
                <div className="flex items-center space-x-2">
                  <Database className="w-4 h-4 text-purple-400" />
                  <span className="text-sm text-gray-300">Queues</span>
                </div>
                <p className="text-2xl font-bold text-white mt-2">
                  {workersLoading ? '—' : (queueStatus?.length ?? 0)}
                </p>
                <p className="text-xs text-gray-500">Showing enabled queues</p>
              </div>
            </div>

            {/* Queue list preview */}
            {!workersLoading && queueStatus && queueStatus.length > 0 && (
              <div className="mt-6">
                <div className="overflow-x-auto">
                  <table className="min-w-full text-sm">
                    <thead>
                      <tr className="text-left text-gray-400">
                        <th className="py-2 pr-4 font-medium">Queue</th>
                        <th className="py-2 pr-4 font-medium">Type</th>
                        <th className="py-2 pr-4 font-medium">Priority</th>
                        <th className="py-2 pr-4 font-medium">Concurrency</th>
                        <th className="py-2 pr-4 font-medium">Timeout</th>
                      </tr>
                    </thead>
                    <tbody>
                      {queueStatus.slice(0, 6).map((q: any, idx: number) => (
                        <tr key={idx} className="border-t border-gray-700 text-gray-300">
                          <td className="py-2 pr-4">{q.queue_name ?? q.name ?? '—'}</td>
                          <td className="py-2 pr-4">{q.worker_type?.value ?? q.worker_type ?? '—'}</td>
                          <td className="py-2 pr-4">{q.priority ?? '—'}</td>
                          <td className="py-2 pr-4">{q.concurrency ?? '—'}</td>
                          <td className="py-2 pr-4">{q.timeout ? `${q.timeout}s` : '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {queueStatus.length > 6 && (
                  <p className="text-xs text-gray-500 mt-2">+{queueStatus.length - 6} more queues…</p>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
