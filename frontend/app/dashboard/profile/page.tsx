'use client'

import { useState } from 'react'
import {
  User,
  Mail,
  Shield,
  Calendar,
  Star,
  Settings,
  Bell,
  Download,
  Upload,
  Edit,
  Save,
  X,
  Heart,
  Eye,
  BarChart3,
  Activity
} from 'lucide-react'
import { useAuth } from '@/lib/auth/AuthProvider'

// Mock data for demonstration
const mockFavorites = [
  {
    id: 'fav-001',
    type: 'detection',
    title: 'High-confidence anomaly in M31',
    description: 'Unusual brightness variation detected in Andromeda Galaxy',
    confidence: 0.95,
    timestamp: '2025-01-15T10:30:00Z',
    tags: ['anomaly', 'm31', 'high-confidence']
  },
  {
    id: 'fav-002',
    type: 'observation',
    title: 'ZTF observation obs-001',
    description: 'Clean observation with excellent quality metrics',
    quality: 0.98,
    timestamp: '2025-01-14T15:45:00Z',
    tags: ['ztf', 'high-quality', 'reference']
  },
  {
    id: 'fav-003',
    type: 'detection',
    title: 'Potential supernova candidate',
    description: 'New source detected in NGC 4565 field',
    confidence: 0.87,
    timestamp: '2025-01-13T08:20:00Z',
    tags: ['supernova', 'ngc4565', 'candidate']
  }
]

const mockActivity = [
  {
    id: 'act-001',
    action: 'viewed_detection',
    target: 'det-001',
    timestamp: '2025-01-15T14:30:00Z',
    description: 'Viewed detection det-001'
  },
  {
    id: 'act-002',
    action: 'favorited_detection',
    target: 'det-003',
    timestamp: '2025-01-15T12:15:00Z',
    description: 'Added detection det-003 to favorites'
  },
  {
    id: 'act-003',
    action: 'exported_data',
    target: 'observations_batch_001',
    timestamp: '2025-01-15T09:45:00Z',
    description: 'Exported observation batch 001'
  }
]

export default function ProfilePage() {
  const { profile, session } = useAuth()
  const [isEditing, setIsEditing] = useState(false)
  const [userSettings, setUserSettings] = useState({
    email: profile?.email || '',
    role: 'Researcher',
    notifications: {
      email: true,
      push: false,
      weekly: true
    },
    preferences: {
      theme: 'dark',
      timezone: 'UTC',
      dateFormat: 'ISO'
    }
  })

  const handleSave = () => {
    // In a real app, this would save to the backend
    setIsEditing(false)
  }

  const handleCancel = () => {
    setIsEditing(false)
  }

  const getRoleColor = (role: string) => {
    switch (role.toLowerCase()) {
      case 'admin': return 'text-red-500 bg-red-900'
      case 'researcher': return 'text-blue-500 bg-blue-900'
      case 'curator': return 'text-green-500 bg-green-900'
      case 'viewer': return 'text-gray-500 bg-gray-900'
      default: return 'text-gray-500 bg-gray-900'
    }
  }

  return (
    <div className="min-h-[calc(100vh-4rem)] bg-astrid-dark">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Profile</h1>
          <p className="text-gray-400">Manage your account settings and view your activity</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Profile Card */}
          <div className="lg:col-span-1">
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <div className="text-center mb-6">
                <div className="w-24 h-24 bg-astrid-blue rounded-full flex items-center justify-center mx-auto mb-4">
                  <User className="w-12 h-12 text-white" />
                </div>
                <h2 className="text-xl font-semibold text-white mb-1">
                  {profile?.email?.split('@')[0] || 'User'}
                </h2>
                <p className="text-gray-400">{profile?.email}</p>
                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium mt-2 ${getRoleColor(userSettings.role)}`}>
                  <Shield className="w-3 h-3 mr-1" />
                  {userSettings.role}
                </span>
              </div>

              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-400">Member since</span>
                  <span className="text-sm text-white">Jan 2025</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-400">Last active</span>
                  <span className="text-sm text-white">2 hours ago</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-400">Favorites</span>
                  <span className="text-sm text-white">{mockFavorites.length}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-400">Activity</span>
                  <span className="text-sm text-white">{mockActivity.length} items</span>
                </div>
              </div>

              <button
                onClick={() => setIsEditing(!isEditing)}
                className="w-full mt-6 flex items-center justify-center px-4 py-2 bg-astrid-blue text-white rounded-lg hover:bg-blue-600 transition-colors"
              >
                <Edit className="w-4 h-4 mr-2" />
                {isEditing ? 'Cancel Edit' : 'Edit Profile'}
              </button>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-2 space-y-8">
            {/* Settings */}
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold text-white">Settings</h3>
                {isEditing && (
                  <div className="flex space-x-2">
                    <button
                      onClick={handleSave}
                      className="flex items-center px-3 py-1.5 bg-green-600 text-white rounded text-sm hover:bg-green-700 transition-colors"
                    >
                      <Save className="w-3 h-3 mr-1" />
                      Save
                    </button>
                    <button
                      onClick={handleCancel}
                      className="flex items-center px-3 py-1.5 bg-gray-600 text-white rounded text-sm hover:bg-gray-700 transition-colors"
                    >
                      <X className="w-3 h-3 mr-1" />
                      Cancel
                    </button>
                  </div>
                )}
              </div>

              <div className="space-y-6">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Email</label>
                  <input
                    type="email"
                    value={userSettings.email}
                    onChange={(e) => setUserSettings(prev => ({ ...prev, email: e.target.value }))}
                    disabled={!isEditing}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white disabled:bg-gray-800 disabled:text-gray-400 focus:ring-2 focus:ring-astrid-blue focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Role</label>
                  <select
                    value={userSettings.role}
                    onChange={(e) => setUserSettings(prev => ({ ...prev, role: e.target.value }))}
                    disabled={!isEditing}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white disabled:bg-gray-800 disabled:text-gray-400 focus:ring-2 focus:ring-astrid-blue focus:border-transparent"
                  >
                    <option value="Researcher">Researcher</option>
                    <option value="Curator">Curator</option>
                    <option value="Admin">Admin</option>
                    <option value="Viewer">Viewer</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-3">Notifications</label>
                  <div className="space-y-3">
                    <label className="flex items-center">
                      <input
                        type="checkbox"
                        checked={userSettings.notifications.email}
                        onChange={(e) => setUserSettings(prev => ({
                          ...prev,
                          notifications: { ...prev.notifications, email: e.target.checked }
                        }))}
                        disabled={!isEditing}
                        className="w-4 h-4 text-astrid-blue bg-gray-700 border-gray-600 rounded focus:ring-astrid-blue disabled:opacity-50"
                      />
                      <span className="ml-3 text-sm text-gray-300">Email notifications</span>
                    </label>
                    <label className="flex items-center">
                      <input
                        type="checkbox"
                        checked={userSettings.notifications.push}
                        onChange={(e) => setUserSettings(prev => ({
                          ...prev,
                          notifications: { ...prev.notifications, push: e.target.checked }
                        }))}
                        disabled={!isEditing}
                        className="w-4 h-4 text-astrid-blue bg-gray-700 border-gray-600 rounded focus:ring-astrid-blue disabled:opacity-50"
                      />
                      <span className="ml-3 text-sm text-gray-300">Push notifications</span>
                    </label>
                    <label className="flex items-center">
                      <input
                        type="checkbox"
                        checked={userSettings.notifications.weekly}
                        onChange={(e) => setUserSettings(prev => ({
                          ...prev,
                          notifications: { ...prev.notifications, weekly: e.target.checked }
                        }))}
                        disabled={!isEditing}
                        className="w-4 h-4 text-astrid-blue bg-gray-700 border-gray-600 rounded focus:ring-astrid-blue disabled:opacity-50"
                      />
                      <span className="ml-3 text-sm text-gray-300">Weekly summary</span>
                    </label>
                  </div>
                </div>
              </div>
            </div>

            {/* Favorites */}
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold text-white">Favorites</h3>
                <span className="text-sm text-gray-400">{mockFavorites.length} items</span>
              </div>

              <div className="space-y-4">
                {mockFavorites.map((favorite) => (
                  <div key={favorite.id} className="bg-gray-700 rounded-lg p-4 hover:bg-gray-600 transition-colors">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-2">
                          <h4 className="font-medium text-white">{favorite.title}</h4>
                          <span className="px-2 py-1 bg-gray-600 rounded text-xs text-gray-300">
                            {favorite.type}
                          </span>
                        </div>
                        <p className="text-sm text-gray-400 mb-2">{favorite.description}</p>
                        <div className="flex items-center space-x-4 text-xs text-gray-500">
                          <div className="flex items-center space-x-1">
                            <Calendar className="w-3 h-3" />
                            <span>{new Date(favorite.timestamp).toLocaleDateString()}</span>
                          </div>
                          {favorite.confidence && (
                            <div className="flex items-center space-x-1">
                              <Star className="w-3 h-3" />
                              <span>{(favorite.confidence * 100).toFixed(0)}% confidence</span>
                            </div>
                          )}
                          {favorite.quality && (
                            <div className="flex items-center space-x-1">
                              <BarChart3 className="w-3 h-3" />
                              <span>{(favorite.quality * 100).toFixed(0)}% quality</span>
                            </div>
                          )}
                        </div>
                        <div className="flex flex-wrap gap-1 mt-2">
                          {favorite.tags.map((tag, index) => (
                            <span key={index} className="px-2 py-1 bg-astrid-blue bg-opacity-20 text-astrid-blue rounded text-xs">
                              {tag}
                            </span>
                          ))}
                        </div>
                      </div>
                      <div className="flex items-center space-x-2 ml-4">
                        <button className="text-gray-400 hover:text-white transition-colors">
                          <Eye className="w-4 h-4" />
                        </button>
                        <button className="text-gray-400 hover:text-red-500 transition-colors">
                          <Heart className="w-4 h-4 fill-current" />
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Recent Activity */}
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold text-white">Recent Activity</h3>
                <span className="text-sm text-gray-400">Last 7 days</span>
              </div>

              <div className="space-y-4">
                {mockActivity.map((activity) => (
                  <div key={activity.id} className="flex items-center space-x-4">
                    <div className="w-8 h-8 bg-gray-700 rounded-full flex items-center justify-center">
                      <Activity className="w-4 h-4 text-gray-400" />
                    </div>
                    <div className="flex-1">
                      <p className="text-sm text-white">{activity.description}</p>
                      <p className="text-xs text-gray-500">
                        {new Date(activity.timestamp).toLocaleString()}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
