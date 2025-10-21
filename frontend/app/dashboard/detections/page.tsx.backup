'use client'

import { useState } from 'react'
import {
  Search,
  Filter,
  Eye,
  Download,
  Star,
  AlertTriangle,
  CheckCircle,
  Clock,
  BarChart3,
  Image as ImageIcon,
  MapPin,
  Calendar,
  Zap,
  Target
} from 'lucide-react'

// MOCK DATA for demonstration - All data below is simulated
const mockDetections = [
  {
    id: 'det-001',
    observationId: 'obs-001',
    ra: 123.456,
    dec: 45.678,
    confidence: 0.95,
    magnitude: 18.5,
    status: 'confirmed',
    timestamp: '2025-01-15T10:35:00Z',
    imageUrl: '/api/placeholder/400/300',
    annotations: [
      { x: 150, y: 200, type: 'source', label: 'Anomaly' },
      { x: 300, y: 150, type: 'reference', label: 'Reference' }
    ]
  },
  {
    id: 'det-002',
    observationId: 'obs-002',
    ra: 234.567,
    dec: 56.789,
    confidence: 0.78,
    magnitude: 19.2,
    status: 'pending',
    timestamp: '2025-01-15T11:20:00Z',
    imageUrl: '/api/placeholder/400/300',
    annotations: [
      { x: 200, y: 250, type: 'source', label: 'Candidate' }
    ]
  },
  {
    id: 'det-003',
    observationId: 'obs-003',
    ra: 345.678,
    dec: 67.890,
    confidence: 0.45,
    magnitude: 20.1,
    status: 'rejected',
    timestamp: '2025-01-15T12:05:00Z',
    imageUrl: '/api/placeholder/400/300',
    annotations: []
  }
]

const statusConfig = {
  confirmed: { color: 'text-green-500', bg: 'bg-green-900', icon: CheckCircle, label: 'Confirmed' },
  pending: { color: 'text-yellow-500', bg: 'bg-yellow-900', icon: Clock, label: 'Pending Review' },
  rejected: { color: 'text-red-500', bg: 'bg-red-900', icon: AlertTriangle, label: 'Rejected' },
  false_positive: { color: 'text-gray-500', bg: 'bg-gray-900', icon: AlertTriangle, label: 'False Positive' }
}

const confidenceConfig = {
  high: { color: 'text-green-500', threshold: 0.8 },
  medium: { color: 'text-yellow-500', threshold: 0.6 },
  low: { color: 'text-red-500', threshold: 0.0 }
}

export default function DetectionsPage() {
  const [selectedDetection, setSelectedDetection] = useState<string | null>(null)
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid')
  const [filters, setFilters] = useState({
    status: '',
    confidence: '',
    magnitude: '',
    dateRange: ''
  })

  const handleFilterChange = (key: string, value: string) => {
    setFilters(prev => ({ ...prev, [key]: value }))
  }

  const getConfidenceLevel = (confidence: number) => {
    if (confidence >= confidenceConfig.high.threshold) return 'high'
    if (confidence >= confidenceConfig.medium.threshold) return 'medium'
    return 'low'
  }

  return (
    <div className="min-h-[calc(100vh-4rem)] bg-astrid-dark">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-white mb-2">Detections</h1>
              <p className="text-gray-400">View and analyze anomaly detections</p>
            </div>
            <div className="flex space-x-3">
              <div className="flex bg-gray-800 rounded-lg p-1 border border-gray-700">
                <button
                  onClick={() => setViewMode('grid')}
                  className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                    viewMode === 'grid'
                      ? 'bg-astrid-blue text-white'
                      : 'text-gray-400 hover:text-white'
                  }`}
                >
                  Grid
                </button>
                <button
                  onClick={() => setViewMode('list')}
                  className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                    viewMode === 'list'
                      ? 'bg-astrid-blue text-white'
                      : 'text-gray-400 hover:text-white'
                  }`}
                >
                  List
                </button>
              </div>
              <button className="flex items-center px-4 py-2 bg-astrid-blue text-white rounded-lg hover:bg-blue-600 transition-colors">
                <Download className="w-4 h-4 mr-2" />
                Export
              </button>
            </div>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Total Detections</p>
                <p className="text-2xl font-bold text-white">342</p>
                <p className="text-xs text-yellow-500 mt-1 font-medium">(MOCK data)</p>
              </div>
              <Search className="w-8 h-8 text-astrid-blue" />
            </div>
          </div>
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Confirmed</p>
                <p className="text-2xl font-bold text-green-500">156</p>
                <p className="text-xs text-yellow-500 mt-1 font-medium">(MOCK data)</p>
              </div>
              <CheckCircle className="w-8 h-8 text-green-500" />
            </div>
          </div>
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Pending</p>
                <p className="text-2xl font-bold text-yellow-500">89</p>
                <p className="text-xs text-yellow-500 mt-1 font-medium">(MOCK data)</p>
              </div>
              <Clock className="w-8 h-8 text-yellow-500" />
            </div>
          </div>
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">High Confidence</p>
                <p className="text-2xl font-bold text-blue-500">203</p>
                <p className="text-xs text-yellow-500 mt-1 font-medium">(MOCK data)</p>
              </div>
              <Star className="w-8 h-8 text-blue-500" />
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
                    <option value="confirmed">Confirmed</option>
                    <option value="pending">Pending</option>
                    <option value="rejected">Rejected</option>
                    <option value="false_positive">False Positive</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Confidence</label>
                  <select
                    value={filters.confidence}
                    onChange={(e) => handleFilterChange('confidence', e.target.value)}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-astrid-blue focus:border-transparent"
                  >
                    <option value="">All Confidence</option>
                    <option value="high">High (≥80%)</option>
                    <option value="medium">Medium (60-79%)</option>
                    <option value="low">Low (<60%)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Magnitude Range</label>
                  <select
                    value={filters.magnitude}
                    onChange={(e) => handleFilterChange('magnitude', e.target.value)}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-astrid-blue focus:border-transparent"
                  >
                    <option value="">All Magnitudes</option>
                    <option value="bright">Bright (<18)</option>
                    <option value="medium">Medium (18-20)</option>
                    <option value="faint">Faint (>20)</option>
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

          {/* Detections Content */}
          <div className="lg:col-span-3">
            {viewMode === 'grid' ? (
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
                {mockDetections.map((detection) => {
                  const statusInfo = statusConfig[detection.status as keyof typeof statusConfig]
                  const StatusIcon = statusInfo.icon
                  const confidenceLevel = getConfidenceLevel(detection.confidence)
                  const confidenceColor = confidenceConfig[confidenceLevel as keyof typeof confidenceConfig].color

                  return (
                    <div key={detection.id} className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden hover:border-gray-600 transition-colors">
                      {/* Image */}
                      <div className="relative h-48 bg-gray-700">
                        <div className="absolute inset-0 flex items-center justify-center">
                          <ImageIcon className="w-12 h-12 text-gray-500" />
                        </div>
                        <div className="absolute top-2 right-2">
                          <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${statusInfo.bg} ${statusInfo.color}`}>
                            <StatusIcon className="w-3 h-3 mr-1" />
                            {statusInfo.label}
                          </span>
                        </div>
                        <div className="absolute bottom-2 left-2">
                          <span className={`px-2 py-1 bg-black bg-opacity-50 rounded text-xs font-medium ${confidenceColor}`}>
                            {(detection.confidence * 100).toFixed(0)}% confidence
                          </span>
                        </div>
                      </div>

                      {/* Content */}
                      <div className="p-4">
                        <div className="flex items-center justify-between mb-2">
                          <h3 className="font-semibold text-white">{detection.id}</h3>
                          <div className="flex items-center space-x-1 text-yellow-500">
                            <Star className="w-4 h-4" />
                            <span className="text-sm">{detection.magnitude}</span>
                          </div>
                        </div>

                        <div className="space-y-2 text-sm text-gray-300">
                          <div className="flex items-center space-x-2">
                            <MapPin className="w-3 h-3" />
                            <span>{detection.ra.toFixed(3)}°, {detection.dec.toFixed(3)}°</span>
                          </div>
                          <div className="flex items-center space-x-2">
                            <Calendar className="w-3 h-3" />
                            <span>{new Date(detection.timestamp).toLocaleDateString()}</span>
                          </div>
                          <div className="flex items-center space-x-2">
                            <Target className="w-3 h-3" />
                            <span>{detection.annotations.length} annotations</span>
                          </div>
                        </div>

                        <div className="flex items-center justify-between mt-4">
                          <button
                            onClick={() => setSelectedDetection(detection.id)}
                            className="flex items-center px-3 py-1.5 bg-astrid-blue text-white rounded text-sm hover:bg-blue-600 transition-colors"
                          >
                            <Eye className="w-3 h-3 mr-1" />
                            View
                          </button>
                          <button className="text-gray-400 hover:text-white transition-colors">
                            <Download className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            ) : (
              <div className="bg-gray-800 rounded-lg border border-gray-700">
                <div className="px-6 py-4 border-b border-gray-700">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold text-white">Detections</h3>
                    <div className="flex items-center space-x-2">
                      <div className="relative">
                        <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                        <input
                          type="text"
                          placeholder="Search detections..."
                          className="pl-10 pr-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:ring-2 focus:ring-astrid-blue focus:border-transparent"
                        />
                      </div>
                      <button className="p-2 text-gray-400 hover:text-white transition-colors">
                        <BarChart3 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead className="bg-gray-700">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">ID</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Coordinates</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Magnitude</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Confidence</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Status</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Actions</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-700">
                      {mockDetections.map((detection) => {
                        const statusInfo = statusConfig[detection.status as keyof typeof statusConfig]
                        const StatusIcon = statusInfo.icon
                        const confidenceLevel = getConfidenceLevel(detection.confidence)
                        const confidenceColor = confidenceConfig[confidenceLevel as keyof typeof confidenceConfig].color

                        return (
                          <tr key={detection.id} className="hover:bg-gray-700 transition-colors">
                            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-white">
                              {detection.id}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                              <div className="flex items-center space-x-1">
                                <MapPin className="w-3 h-3" />
                                <span>{detection.ra.toFixed(3)}°, {detection.dec.toFixed(3)}°</span>
                              </div>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                              <div className="flex items-center space-x-1">
                                <Star className="w-3 h-3 text-yellow-500" />
                                <span>{detection.magnitude}</span>
                              </div>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                              <span className={`font-medium ${confidenceColor}`}>
                                {(detection.confidence * 100).toFixed(0)}%
                              </span>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${statusInfo.bg} ${statusInfo.color}`}>
                                <StatusIcon className="w-3 h-3 mr-1" />
                                {statusInfo.label}
                              </span>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                              <div className="flex items-center space-x-2">
                                <button
                                  onClick={() => setSelectedDetection(detection.id)}
                                  className="text-astrid-blue hover:text-blue-400 transition-colors"
                                >
                                  <Eye className="w-4 h-4" />
                                </button>
                                <button className="text-gray-400 hover:text-white transition-colors">
                                  <Download className="w-4 h-4" />
                                </button>
                              </div>
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Detection Detail Modal */}
        {selectedDetection && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-gray-800 rounded-lg p-6 max-w-4xl w-full mx-4 border border-gray-700">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-white">Detection Details</h3>
                <button
                  onClick={() => setSelectedDetection(null)}
                  className="text-gray-400 hover:text-white transition-colors"
                >
                  ×
                </button>
              </div>
              <div className="space-y-4">
                <p className="text-gray-300">Detailed detection information will be displayed here.</p>
                <p className="text-sm text-gray-500">This is a placeholder for the full detection detail view with image viewer and annotations.</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
