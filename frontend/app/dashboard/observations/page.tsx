'use client'

import { useState } from 'react'
import {
  Eye,
  Filter,
  Search,
  Download,
  Upload,
  RefreshCw,
  Eye as EyeIcon,
  Calendar,
  MapPin,
  Star,
  AlertCircle,
  CheckCircle,
  Clock,
  BarChart3
} from 'lucide-react'

// MOCK DATA for demonstration - All data below is simulated
const mockObservations = [
  {
    id: 'obs-001',
    survey: 'ZTF',
    ra: 123.456,
    dec: 45.678,
    filter: 'r',
    exposureTime: 30,
    status: 'processed',
    quality: 0.95,
    timestamp: '2025-01-15T10:30:00Z',
    processingTime: 45
  },
  {
    id: 'obs-002',
    survey: 'ATLAS',
    ra: 234.567,
    dec: 56.789,
    filter: 'g',
    exposureTime: 60,
    status: 'processing',
    quality: null,
    timestamp: '2025-01-15T11:15:00Z',
    processingTime: null
  },
  {
    id: 'obs-003',
    survey: 'ZTF',
    ra: 345.678,
    dec: 67.890,
    filter: 'i',
    exposureTime: 45,
    status: 'failed',
    quality: null,
    timestamp: '2025-01-15T12:00:00Z',
    processingTime: null
  }
]

const statusConfig = {
  processed: { color: 'text-green-500', bg: 'bg-green-900', icon: CheckCircle, label: 'Processed' },
  processing: { color: 'text-yellow-500', bg: 'bg-yellow-900', icon: Clock, label: 'Processing' },
  failed: { color: 'text-red-500', bg: 'bg-red-900', icon: AlertCircle, label: 'Failed' },
  pending: { color: 'text-gray-500', bg: 'bg-gray-900', icon: Clock, label: 'Pending' }
}

export default function ObservationsPage() {
  const [selectedObservation, setSelectedObservation] = useState<string | null>(null)
  const [filters, setFilters] = useState({
    survey: '',
    status: '',
    filter: '',
    dateRange: ''
  })

  const handleFilterChange = (key: string, value: string) => {
    setFilters(prev => ({ ...prev, [key]: value }))
  }

  return (
    <div className="min-h-[calc(100vh-4rem)] bg-astrid-dark">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-white mb-2">Observations</h1>
              <p className="text-gray-400">Manage astronomical observations and survey data</p>
            </div>
            <div className="flex space-x-3">
              <button className="flex items-center px-4 py-2 bg-astrid-blue text-white rounded-lg hover:bg-blue-600 transition-colors">
                <Upload className="w-4 h-4 mr-2" />
                Upload FITS
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
                <p className="text-sm font-medium text-gray-400">Total Observations</p>
                <p className="text-2xl font-bold text-white">1,247</p>
                <p className="text-xs text-yellow-500 mt-1 font-medium">(MOCK data)</p>
              </div>
              <Eye className="w-8 h-8 text-astrid-blue" />
            </div>
          </div>
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Processed</p>
                <p className="text-2xl font-bold text-green-500">1,156</p>
                <p className="text-xs text-yellow-500 mt-1 font-medium">(MOCK data)</p>
              </div>
              <CheckCircle className="w-8 h-8 text-green-500" />
            </div>
          </div>
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Processing</p>
                <p className="text-2xl font-bold text-yellow-500">23</p>
                <p className="text-xs text-yellow-500 mt-1 font-medium">(MOCK data)</p>
              </div>
              <Clock className="w-8 h-8 text-yellow-500" />
            </div>
          </div>
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Failed</p>
                <p className="text-2xl font-bold text-red-500">68</p>
                <p className="text-xs text-yellow-500 mt-1 font-medium">(MOCK data)</p>
              </div>
              <AlertCircle className="w-8 h-8 text-red-500" />
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Filters Sidebar */}
          <div className="lg:col-span-1">
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h3 className="text-lg font-semibold text-white mb-4">Filters</h3>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Survey</label>
                  <select
                    value={filters.survey}
                    onChange={(e) => handleFilterChange('survey', e.target.value)}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-astrid-blue focus:border-transparent"
                  >
                    <option value="">All Surveys</option>
                    <option value="ztf">ZTF</option>
                    <option value="atlas">ATLAS</option>
                    <option value="lsst">LSST</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Status</label>
                  <select
                    value={filters.status}
                    onChange={(e) => handleFilterChange('status', e.target.value)}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-astrid-blue focus:border-transparent"
                  >
                    <option value="">All Status</option>
                    <option value="processed">Processed</option>
                    <option value="processing">Processing</option>
                    <option value="failed">Failed</option>
                    <option value="pending">Pending</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Filter</label>
                  <select
                    value={filters.filter}
                    onChange={(e) => handleFilterChange('filter', e.target.value)}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-astrid-blue focus:border-transparent"
                  >
                    <option value="">All Filters</option>
                    <option value="g">g-band</option>
                    <option value="r">r-band</option>
                    <option value="i">i-band</option>
                    <option value="z">z-band</option>
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

          {/* Observations Table */}
          <div className="lg:col-span-2">
            <div className="bg-gray-800 rounded-lg border border-gray-700">
              <div className="px-6 py-4 border-b border-gray-700">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-white">Observations</h3>
                  <div className="flex items-center space-x-2">
                    <div className="relative">
                      <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                      <input
                        type="text"
                        placeholder="Search observations..."
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
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Survey</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Coordinates</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Filter</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Status</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Quality</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-700">
                    {mockObservations.map((obs) => {
                      const statusInfo = statusConfig[obs.status as keyof typeof statusConfig]
                      const StatusIcon = statusInfo.icon

                      return (
                        <tr key={obs.id} className="hover:bg-gray-700 transition-colors">
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-white">
                            {obs.id}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                            {obs.survey}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                            <div className="flex items-center space-x-1">
                              <MapPin className="w-3 h-3" />
                              <span>{obs.ra.toFixed(3)}°, {obs.dec.toFixed(3)}°</span>
                            </div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                            <span className="px-2 py-1 bg-gray-700 rounded text-xs">{obs.filter}</span>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${statusInfo.bg} ${statusInfo.color}`}>
                              <StatusIcon className="w-3 h-3 mr-1" />
                              {statusInfo.label}
                            </span>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                            {obs.quality ? (
                              <div className="flex items-center space-x-1">
                                <Star className="w-3 h-3 text-yellow-500" />
                                <span>{(obs.quality * 100).toFixed(1)}%</span>
                              </div>
                            ) : (
                              <span className="text-gray-500">-</span>
                            )}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                            <div className="flex items-center space-x-2">
                              <button
                                onClick={() => setSelectedObservation(obs.id)}
                                className="text-astrid-blue hover:text-blue-400 transition-colors"
                              >
                                <EyeIcon className="w-4 h-4" />
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
          </div>
        </div>

        {/* Observation Detail Modal */}
        {selectedObservation && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-gray-800 rounded-lg p-6 max-w-2xl w-full mx-4 border border-gray-700">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-white">Observation Details</h3>
                <button
                  onClick={() => setSelectedObservation(null)}
                  className="text-gray-400 hover:text-white transition-colors"
                >
                  ×
                </button>
              </div>
              <div className="space-y-4">
                <p className="text-gray-300">Detailed observation information will be displayed here.</p>
                <p className="text-sm text-gray-500">This is a placeholder for the full observation detail view.</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
