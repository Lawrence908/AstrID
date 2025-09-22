'use client'

import Link from 'next/link'
import { ArrowLeft, Image, FileText, GitBranch, BookOpen, Calendar, Activity, CheckCircle, Clock, AlertCircle } from 'lucide-react'

const planningSections = [
  {
    id: 'diagrams',
    title: 'System Diagrams',
    description: 'Architecture and system diagrams',
    icon: <Image className="w-8 h-8" />,
    href: '/planning/diagrams',
    status: 'completed',
    count: 8,
    lastUpdated: 'Sep 21, 2025',
    features: ['Data Flow Pipeline', 'Database Schema', 'System Architecture', 'Workflow Orchestration']
  },
  {
    id: 'documentation',
    title: 'Documentation',
    description: 'Technical documentation and guides',
    icon: <FileText className="w-8 h-8" />,
    href: '/planning/documentation',
    status: 'completed',
    count: 8,
    lastUpdated: 'Sep 21, 2025',
    features: ['Architecture Overview', 'Development Guide', 'API Documentation', 'Migration Strategy']
  },
  {
    id: 'tickets',
    title: 'Linear Tickets',
    description: 'Project tickets and progress tracking',
    icon: <GitBranch className="w-8 h-8" />,
    href: '/planning/tickets',
    status: 'in-progress',
    count: 8,
    lastUpdated: 'Sep 21, 2025',
    features: ['Progress Tracking', 'Task Management', 'Team Collaboration', 'Project Milestones']
  }
]

const recentActivity = [
  {
    id: 1,
    type: 'diagram',
    title: 'Updated System Architecture diagram',
    description: 'Added new microservices and updated data flow',
    timestamp: '2 hours ago',
    icon: <Image className="w-4 h-4" />
  },
  {
    id: 2,
    type: 'documentation',
    title: 'Updated Development Guide',
    description: 'Added new setup instructions for Docker environment',
    timestamp: '4 hours ago',
    icon: <FileText className="w-4 h-4" />
  },
  {
    id: 3,
    type: 'ticket',
    title: 'Completed AST-001: Data ingestion pipeline',
    description: 'Successfully implemented FITS file processing',
    timestamp: '1 day ago',
    icon: <CheckCircle className="w-4 h-4" />
  },
  {
    id: 4,
    type: 'ticket',
    title: 'In Progress: AST-002: Anomaly detection algorithm',
    description: '75% complete - implementing ML model training',
    timestamp: '2 days ago',
    icon: <Clock className="w-4 h-4" />
  }
]

export default function PlanningPage() {
  return (
    <div className="min-h-[calc(100vh-4rem)] bg-astrid-dark">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center space-x-4 mb-4">
            <Link
              href="/"
              className="inline-flex items-center px-3 py-2 text-sm font-medium text-gray-400 hover:text-white transition-colors"
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Dashboard
            </Link>
          </div>
          <div className="flex items-center space-x-6 mb-4">
            <div className="flex-shrink-0">
              <div className="w-16 h-16 bg-astrid-blue/20 rounded-lg flex items-center justify-center">
                <BookOpen className="w-8 h-8 text-astrid-blue" />
              </div>
            </div>
            <div>
              <h1 className="text-4xl font-bold text-white mb-2">Planning Dashboard</h1>
              <p className="text-xl text-gray-400">
                Development documentation, diagrams, and project tracking
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <span className="text-sm text-gray-300">All systems operational</span>
            </div>
            <div className="flex items-center space-x-2">
              <Activity className="w-4 h-4 text-gray-400" />
              <span className="text-sm text-gray-300">Last updated: Sep 21, 2025</span>
            </div>
          </div>
        </div>

        {/* Main Sections Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          {planningSections.map((section) => (
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
                      section.status === 'completed'
                        ? 'bg-green-900 text-green-300'
                        : section.status === 'in-progress'
                        ? 'bg-yellow-900 text-yellow-300'
                        : 'bg-gray-700 text-gray-300'
                    }`}>
                      {section.status === 'completed' ? 'Ready' : section.status === 'in-progress' ? 'Active' : 'Planned'}
                    </span>
                  </div>
                  <p className="text-gray-400 text-sm mb-3">{section.description}</p>
                  <div className="space-y-1 mb-4">
                    {section.features.map((feature, index) => (
                      <div key={index} className="flex items-center space-x-2">
                        <CheckCircle className="w-3 h-3 text-gray-500" />
                        <span className="text-xs text-gray-500">{feature}</span>
                      </div>
                    ))}
                  </div>
                  <div className="flex items-center justify-between text-xs text-gray-500">
                    <span>{section.count} items</span>
                    <span>Updated {section.lastUpdated}</span>
                  </div>
                </div>
              </div>
            </Link>
          ))}
        </div>

        {/* Recent Activity */}
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold text-white">Recent Activity</h2>
            <Link
              href="/planning/activity"
              className="text-sm text-astrid-blue hover:text-blue-400 transition-colors"
            >
              View all activity
            </Link>
          </div>

          <div className="space-y-4">
            {recentActivity.map((activity) => (
              <div key={activity.id} className="flex items-start space-x-4 p-4 bg-gray-700/50 rounded-lg hover:bg-gray-700 transition-colors">
                <div className="flex-shrink-0 mt-1 text-astrid-blue">
                  {activity.icon}
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="text-sm font-medium text-white">{activity.title}</h3>
                  <p className="text-sm text-gray-400 mt-1">{activity.description}</p>
                  <p className="text-xs text-gray-500 mt-2">{activity.timestamp}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Quick Stats */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <div className="flex items-center space-x-2">
              <Image className="w-5 h-5 text-astrid-blue" />
              <span className="text-sm font-medium text-gray-300">Diagrams</span>
            </div>
            <p className="text-2xl font-bold text-white mt-2">8</p>
            <p className="text-xs text-gray-500">System diagrams available</p>
            <p className="text-xs text-green-500 mt-1 font-medium">Up to date</p>
          </div>
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <div className="flex items-center space-x-2">
              <FileText className="w-5 h-5 text-green-500" />
              <span className="text-sm font-medium text-gray-300">Documents</span>
            </div>
            <p className="text-2xl font-bold text-white mt-2">8</p>
            <p className="text-xs text-gray-500">Technical docs</p>
            <p className="text-xs text-green-500 mt-1 font-medium">Current</p>
          </div>
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <div className="flex items-center space-x-2">
              <GitBranch className="w-5 h-5 text-yellow-500" />
              <span className="text-sm font-medium text-gray-300">Tickets</span>
            </div>
            <p className="text-2xl font-bold text-white mt-2">8</p>
            <p className="text-xs text-gray-500">3 completed (37.5%)</p>
            <p className="text-xs text-yellow-500 mt-1 font-medium">In progress</p>
          </div>
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <div className="flex items-center space-x-2">
              <Activity className="w-5 h-5 text-purple-500" />
              <span className="text-sm font-medium text-gray-300">Activity</span>
            </div>
            <p className="text-2xl font-bold text-white mt-2">12</p>
            <p className="text-xs text-gray-500">Updates this week</p>
            <p className="text-xs text-green-500 mt-1 font-medium">Active</p>
          </div>
        </div>
      </div>
    </div>
  )
}
