'use client'

import { useAuth } from '@/lib/auth/AuthProvider'
import Link from 'next/link'
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
  AlertTriangle
} from 'lucide-react'

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
            <p className="text-xs text-yellow-500 mt-1 font-medium">(MOCK data)</p>
          </div>
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <div className="flex items-center space-x-2">
              <Search className="w-5 h-5 text-green-500" />
              <span className="text-sm font-medium text-gray-300">Detections</span>
            </div>
            <p className="text-2xl font-bold text-white mt-2">0</p>
            <p className="text-xs text-gray-500">Anomalies found</p>
            <p className="text-xs text-yellow-500 mt-1 font-medium">(MOCK data)</p>
          </div>
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <div className="flex items-center space-x-2">
              <Activity className="w-5 h-5 text-yellow-500" />
              <span className="text-sm font-medium text-gray-300">Processing</span>
            </div>
            <p className="text-2xl font-bold text-white mt-2">0</p>
            <p className="text-xs text-gray-500">Active workflows</p>
            <p className="text-xs text-yellow-500 mt-1 font-medium">(MOCK data)</p>
          </div>
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <div className="flex items-center space-x-2">
              <AlertTriangle className="w-5 h-5 text-red-500" />
              <span className="text-sm font-medium text-gray-300">Alerts</span>
            </div>
            <p className="text-2xl font-bold text-white mt-2">0</p>
            <p className="text-xs text-gray-500">System alerts</p>
            <p className="text-xs text-yellow-500 mt-1 font-medium">(MOCK data)</p>
          </div>
        </div>
      </div>
    </div>
  )
}
