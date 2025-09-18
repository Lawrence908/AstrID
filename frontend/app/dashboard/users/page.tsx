'use client'

import { useState } from 'react'
import {
  Users,
  UserPlus,
  Search,
  Filter,
  MoreVertical,
  Edit,
  Trash2,
  Shield,
  Mail,
  Calendar,
  Activity,
  CheckCircle,
  XCircle,
  Clock,
  Eye,
  Settings
} from 'lucide-react'

// User data - First entry is real user, rest are MOCK examples
const mockUsers = [
  {
    id: 'user-001',
    email: 'astronomical.identification@gmail.com',
    name: 'Chris Lawrence',
    role: 'Admin',
    status: 'active',
    lastActive: new Date().toISOString(),
    joinDate: '2025-01-01T00:00:00Z',
    permissions: ['read', 'write', 'admin'],
    activity: {
      observations: 0, // Real data - will be populated from actual system
      detections: 0,   // Real data - will be populated from actual system
      workflows: 0     // Real data - will be populated from actual system
    },
    isReal: true
  },
  // MOCK DATA - Example users for different roles
  {
    id: 'user-002',
    email: 'researcher@astrid.com', // MOCK
    name: 'Dr. Sarah Chen', // MOCK
    role: 'Researcher',
    status: 'active',
    lastActive: '2025-01-15T12:15:00Z',
    joinDate: '2025-01-05T00:00:00Z',
    permissions: ['read', 'write'],
    activity: {
      observations: 89, // MOCK
      detections: 45, // MOCK
      workflows: 12 // MOCK
    },
    isReal: false
  },
  {
    id: 'user-003',
    email: 'curator@astrid.com', // MOCK
    name: 'Mike Rodriguez', // MOCK
    role: 'Curator',
    status: 'active',
    lastActive: '2025-01-15T10:45:00Z',
    joinDate: '2025-01-08T00:00:00Z',
    permissions: ['read', 'write', 'curate'],
    activity: {
      observations: 234, // MOCK
      detections: 156, // MOCK
      workflows: 8 // MOCK
    },
    isReal: false
  },
  {
    id: 'user-004',
    email: 'viewer@astrid.com', // MOCK
    name: 'Alex Johnson', // MOCK
    role: 'Viewer',
    status: 'inactive',
    lastActive: '2025-01-10T16:20:00Z',
    joinDate: '2025-01-12T00:00:00Z',
    permissions: ['read'],
    activity: {
      observations: 12, // MOCK
      detections: 3, // MOCK
      workflows: 0 // MOCK
    },
    isReal: false
  }
]

const roleConfig = {
  admin: { color: 'text-red-500', bg: 'bg-red-900', icon: Shield },
  researcher: { color: 'text-blue-500', bg: 'bg-blue-900', icon: Users },
  curator: { color: 'text-green-500', bg: 'bg-green-900', icon: CheckCircle },
  viewer: { color: 'text-gray-500', bg: 'bg-gray-900', icon: Eye }
}

const statusConfig = {
  active: { color: 'text-green-500', bg: 'bg-green-900', icon: CheckCircle, label: 'Active' },
  inactive: { color: 'text-gray-500', bg: 'bg-gray-900', icon: Clock, label: 'Inactive' },
  suspended: { color: 'text-red-500', bg: 'bg-red-900', icon: XCircle, label: 'Suspended' }
}

export default function UsersPage() {
  const [selectedUser, setSelectedUser] = useState<string | null>(null)
  const [showUserForm, setShowUserForm] = useState(false)
  const [filters, setFilters] = useState({
    role: '',
    status: '',
    search: ''
  })

  const handleFilterChange = (key: string, value: string) => {
    setFilters(prev => ({ ...prev, [key]: value }))
  }

  const handleUserAction = (userId: string, action: string) => {
    console.log(`Action ${action} on user ${userId}`)
    // In a real app, this would trigger the actual user action
  }

  const filteredUsers = mockUsers.filter(user => {
    const matchesRole = !filters.role || user.role.toLowerCase() === filters.role
    const matchesStatus = !filters.status || user.status === filters.status
    const matchesSearch = !filters.search ||
      user.name.toLowerCase().includes(filters.search.toLowerCase()) ||
      user.email.toLowerCase().includes(filters.search.toLowerCase())

    return matchesRole && matchesStatus && matchesSearch
  })

  return (
    <div className="min-h-[calc(100vh-4rem)] bg-astrid-dark">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-white mb-2">User Management</h1>
              <p className="text-gray-400">Manage users, roles, and permissions</p>
            </div>
            <div className="flex space-x-3">
              <button
                onClick={() => setShowUserForm(true)}
                className="flex items-center px-4 py-2 bg-astrid-blue text-white rounded-lg hover:bg-blue-600 transition-colors"
              >
                <UserPlus className="w-4 h-4 mr-2" />
                Add User
              </button>
              <button className="flex items-center px-4 py-2 bg-gray-800 text-gray-200 rounded-lg hover:bg-gray-700 transition-colors border border-gray-700">
                <Settings className="w-4 h-4 mr-2" />
                Settings
              </button>
            </div>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Total Users</p>
                <p className="text-2xl font-bold text-white">{mockUsers.length}</p>
                <p className="text-xs text-yellow-500 mt-1 font-medium">(1 real + 3 MOCK)</p>
              </div>
              <Users className="w-8 h-8 text-astrid-blue" />
            </div>
          </div>
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Active</p>
                <p className="text-2xl font-bold text-green-500">
                  {mockUsers.filter(u => u.status === 'active').length}
                </p>
                <p className="text-xs text-yellow-500 mt-1 font-medium">(MOCK data)</p>
              </div>
              <CheckCircle className="w-8 h-8 text-green-500" />
            </div>
          </div>
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Admins</p>
                <p className="text-2xl font-bold text-red-500">
                  {mockUsers.filter(u => u.role.toLowerCase() === 'admin').length}
                </p>
                <p className="text-xs text-yellow-500 mt-1 font-medium">(You + MOCK)</p>
              </div>
              <Shield className="w-8 h-8 text-red-500" />
            </div>
          </div>
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Researchers</p>
                <p className="text-2xl font-bold text-blue-500">
                  {mockUsers.filter(u => u.role.toLowerCase() === 'researcher').length}
                </p>
                <p className="text-xs text-yellow-500 mt-1 font-medium">(MOCK data)</p>
              </div>
              <Users className="w-8 h-8 text-blue-500" />
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
                  <label className="block text-sm font-medium text-gray-300 mb-2">Search</label>
                  <div className="relative">
                    <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                    <input
                      type="text"
                      value={filters.search}
                      onChange={(e) => handleFilterChange('search', e.target.value)}
                      placeholder="Search users..."
                      className="w-full pl-10 pr-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:ring-2 focus:ring-astrid-blue focus:border-transparent"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Role</label>
                  <select
                    value={filters.role}
                    onChange={(e) => handleFilterChange('role', e.target.value)}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-astrid-blue focus:border-transparent"
                  >
                    <option value="">All Roles</option>
                    <option value="admin">Admin</option>
                    <option value="researcher">Researcher</option>
                    <option value="curator">Curator</option>
                    <option value="viewer">Viewer</option>
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
                    <option value="active">Active</option>
                    <option value="inactive">Inactive</option>
                    <option value="suspended">Suspended</option>
                  </select>
                </div>
              </div>
            </div>
          </div>

          {/* Users Table */}
          <div className="lg:col-span-3">
            <div className="bg-gray-800 rounded-lg border border-gray-700">
              <div className="px-6 py-4 border-b border-gray-700">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-white">Users ({filteredUsers.length})</h3>
                  <div className="flex items-center space-x-2">
                    <button className="p-2 text-gray-400 hover:text-white transition-colors">
                      <Filter className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-700">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">User</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Role</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Status</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Last Active</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Activity</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-700">
                    {filteredUsers.map((user) => {
                      const roleInfo = roleConfig[user.role.toLowerCase() as keyof typeof roleConfig]
                      const statusInfo = statusConfig[user.status as keyof typeof statusConfig]
                      const StatusIcon = statusInfo.icon
                      const RoleIcon = roleInfo.icon

                      return (
                        <tr key={user.id} className="hover:bg-gray-700 transition-colors">
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="flex items-center">
                              <div className="w-10 h-10 bg-astrid-blue rounded-full flex items-center justify-center">
                                <span className="text-white font-medium text-sm">
                                  {user.name.split(' ').map(n => n[0]).join('')}
                                </span>
                              </div>
                              <div className="ml-4">
                                <div className="flex items-center space-x-2">
                                  <div className="text-sm font-medium text-white">{user.name}</div>
                                  {!user.isReal && (
                                    <span className="px-2 py-0.5 bg-yellow-900 text-yellow-300 text-xs rounded-full font-medium">
                                      MOCK
                                    </span>
                                  )}
                                </div>
                                <div className="text-sm text-gray-400">{user.email}</div>
                              </div>
                            </div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${roleInfo.bg} ${roleInfo.color}`}>
                              <RoleIcon className="w-3 h-3 mr-1" />
                              {user.role}
                            </span>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${statusInfo.bg} ${statusInfo.color}`}>
                              <StatusIcon className="w-3 h-3 mr-1" />
                              {statusInfo.label}
                            </span>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                            <div className="flex items-center space-x-1">
                              <Calendar className="w-3 h-3" />
                              <span>{new Date(user.lastActive).toLocaleDateString()}</span>
                            </div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                            <div className="space-y-1">
                              <div className="flex items-center space-x-1">
                                <Activity className="w-3 h-3" />
                                <span>{user.activity.observations} obs</span>
                              </div>
                              <div className="flex items-center space-x-1">
                                <Activity className="w-3 h-3" />
                                <span>{user.activity.detections} det</span>
                              </div>
                            </div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                            <div className="flex items-center space-x-2">
                              <button
                                onClick={() => setSelectedUser(user.id)}
                                className="text-astrid-blue hover:text-blue-400 transition-colors"
                                title="View Details"
                              >
                                <Eye className="w-4 h-4" />
                              </button>
                              <button
                                onClick={() => handleUserAction(user.id, 'edit')}
                                className="text-gray-400 hover:text-white transition-colors"
                                title="Edit User"
                              >
                                <Edit className="w-4 h-4" />
                              </button>
                              <button
                                onClick={() => handleUserAction(user.id, 'delete')}
                                className="text-red-400 hover:text-red-300 transition-colors"
                                title="Delete User"
                              >
                                <Trash2 className="w-4 h-4" />
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

        {/* User Detail Modal */}
        {selectedUser && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-gray-800 rounded-lg p-6 max-w-2xl w-full mx-4 border border-gray-700">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-white">User Details</h3>
                <button
                  onClick={() => setSelectedUser(null)}
                  className="text-gray-400 hover:text-white transition-colors"
                >
                  ×
                </button>
              </div>
              <div className="space-y-4">
                <p className="text-gray-300">Detailed user information will be displayed here.</p>
                <p className="text-sm text-gray-500">This is a placeholder for the full user detail view with permissions, activity history, and settings.</p>
              </div>
            </div>
          </div>
        )}

        {/* Add User Modal */}
        {showUserForm && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-gray-800 rounded-lg p-6 max-w-md w-full mx-4 border border-gray-700">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-white">Add New User</h3>
                <button
                  onClick={() => setShowUserForm(false)}
                  className="text-gray-400 hover:text-white transition-colors"
                >
                  ×
                </button>
              </div>
              <div className="space-y-4">
                <p className="text-gray-300">User creation form will be displayed here.</p>
                <p className="text-sm text-gray-500">This is a placeholder for the user creation form with email, role, and permissions.</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
