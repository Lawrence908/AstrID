'use client'

import { useEffect, useState } from 'react'
import { Shield, KeyRound, Plus, Trash2, RefreshCcw, Info } from 'lucide-react'
import { useAuth } from '@/lib/auth/AuthProvider'

type APIKey = {
  id: string
  name: string
  description?: string
  key_prefix: string
  permissions: string[]
  scopes: string[]
  expires_at?: string
  last_used_at?: string
  usage_count: string
  is_active: boolean
  is_expired: boolean
  is_valid: boolean
  created_at: string
  updated_at: string
}

export default function AdminPage() {
  const { session, getToken } = useAuth()
  const [apiKeys, setApiKeys] = useState<APIKey[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [roleInfo, setRoleInfo] = useState<{ email: string; role?: string } | null>(null)
  const [creating, setCreating] = useState(false)
  const [newName, setNewName] = useState('prefect-workflows')
  const [newDesc, setNewDesc] = useState('For Prefect automated workflows')
  const [permissionSet, setPermissionSet] = useState('prefect_workflows')
  const [createdKey, setCreatedKey] = useState<string | null>(null)

  const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:9001'

  const authHeaders = async () => {
    const token = await getToken?.()
    return token
      ? { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' }
      : { 'Content-Type': 'application/json' }
  }

  const fetchKeys = async () => {
    setLoading(true)
    setError(null)
    try {
      const headers = await authHeaders()
      const res = await fetch(`${apiBase}/api-keys/?active_only=true&limit=100`, { headers })
      const json = await res.json().catch(() => ({}))
      if (!res.ok) throw new Error(json?.error?.message || 'Failed to load API keys')
      setApiKeys(Array.isArray(json?.data) ? json.data : [])
    } catch (e: any) {
      setError(e.message || 'Failed to load API keys')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    const load = async () => {
      if (!session) return
      // fetch role
      try {
        const headers = await authHeaders()
        const res = await fetch(`${apiBase}/me`, { headers })
        const json = await res.json().catch(() => ({}))
        if (res.ok && json?.data) {
          setRoleInfo({ email: json.data.email, role: json.data.role })
        } else {
          setRoleInfo(null)
        }
      } catch {
        setRoleInfo(null)
      }
      await fetchKeys()
    }
    load()
  }, [session])

  const createKey = async () => {
    setCreating(true)
    setError(null)
    setCreatedKey(null)
    try {
      const headers = await authHeaders()
      const res = await fetch(`${apiBase}/api-keys/`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          name: newName,
          description: newDesc,
          permission_set: permissionSet
        })
      })
      const json = await res.json().catch(() => ({}))
      if (!res.ok) throw new Error(json?.error?.message || 'Failed to create API key')
      const key = json?.data?.key as string | undefined
      if (key) setCreatedKey(key)
      await fetchKeys()
    } catch (e: any) {
      setError(e.message || 'Failed to create key')
    } finally {
      setCreating(false)
    }
  }

  const revokeKey = async (id: string) => {
    setError(null)
    try {
      const headers = await authHeaders()
      const res = await fetch(`${apiBase}/api-keys/${id}/revoke`, {
        method: 'POST',
        headers
      })
      if (!res.ok) {
        const json = await res.json().catch(() => ({}))
        throw new Error(json?.error?.message || 'Failed to revoke key')
      }
      await fetchKeys()
    } catch (e: any) {
      setError(e.message || 'Failed to revoke key')
    }
  }

  if (!session) {
    return (
      <div className="max-w-5xl mx-auto px-4 py-10 text-gray-300">Sign in to access Admin.</div>
    )
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <Shield className="w-6 h-6 text-astrid-blue" />
          <h1 className="text-2xl font-bold text-white">Admin</h1>
          {roleInfo && (
            <span className="ml-3 px-2 py-0.5 text-xs rounded-full border border-gray-700 text-gray-300 bg-gray-800">
              Role: {roleInfo.role ?? 'unknown'}
            </span>
          )}
          {roleInfo && (
            <span className={`px-2 py-0.5 text-xs rounded-full ${roleInfo.role?.toLowerCase?.() === 'admin' ? 'bg-green-900 text-green-300' : 'bg-yellow-900 text-yellow-300'} border border-gray-700`}>
              {roleInfo.role?.toLowerCase?.() === 'admin' ? 'Admin access' : 'Limited access'}
            </span>
          )}
        </div>
        <button
          onClick={fetchKeys}
          className="flex items-center px-3 py-1.5 rounded bg-gray-800 text-gray-200 text-sm border border-gray-700 hover:bg-gray-700"
        >
          <RefreshCcw className="w-4 h-4 mr-2" /> Refresh
        </button>
      </div>

      {error && (
        <div className="mb-4 text-sm text-red-400">{error}</div>
      )}

      {/* API Keys */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700 mb-8">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-2">
            <KeyRound className="w-5 h-5 text-astrid-blue" />
            <h2 className="text-xl font-semibold text-white">API Keys</h2>
          </div>
        </div>

        {/* Create */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3 mb-4">
          <label className="text-xs text-gray-400">Key name
          <input
            className="px-3 py-2 rounded bg-gray-900 border border-gray-700 text-gray-200 text-sm"
            placeholder="Name"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
          />
          </label>
          <label className="text-xs text-gray-400">Description
          <input
            className="px-3 py-2 rounded bg-gray-900 border border-gray-700 text-gray-200 text-sm"
            placeholder="Description"
            value={newDesc}
            onChange={(e) => setNewDesc(e.target.value)}
          />
          </label>
          <label className="text-xs text-gray-400">Permission set
          <select
            className="px-3 py-2 rounded bg-gray-900 border border-gray-700 text-gray-200 text-sm"
            value={permissionSet}
            onChange={(e) => setPermissionSet(e.target.value)}
          >
            <option value="training_pipeline">training_pipeline</option>
            <option value="prefect_workflows">prefect_workflows</option>
            <option value="read_only">read_only</option>
            <option value="full_access">full_access</option>
          </select>
          </label>
          <button
            onClick={createKey}
            disabled={creating}
            className="flex items-center justify-center px-3 py-2 rounded bg-astrid-blue text-white text-sm hover:bg-blue-600 disabled:opacity-60"
          >
            <Plus className="w-4 h-4 mr-2" /> {creating ? 'Creating…' : 'Create Key'}
          </button>
        </div>

        {createdKey && (
          <div className="mb-4 text-sm text-green-400 flex items-center">
            <Info className="w-4 h-4 mr-2" />
            <span>Save this API key now (shown only once):</span>
            <code className="ml-2 px-2 py-1 bg-gray-900 border border-gray-700 rounded text-gray-200">{createdKey}</code>
          </div>
        )}

        {/* List */}
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="text-left text-gray-400">
                <th className="py-2 pr-4 font-medium">Name</th>
                <th className="py-2 pr-4 font-medium">Prefix</th>
                <th className="py-2 pr-4 font-medium">Permissions</th>
                <th className="py-2 pr-4 font-medium">Usage</th>
                <th className="py-2 pr-4 font-medium">Status</th>
                <th className="py-2 pr-4 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td className="py-3 text-gray-400" colSpan={6}>Loading…</td></tr>
              ) : apiKeys.length === 0 ? (
                <tr><td className="py-3 text-gray-400" colSpan={6}>No API keys</td></tr>
              ) : (
                apiKeys.map((k) => (
                  <tr key={k.id} className="border-t border-gray-700 text-gray-300">
                    <td className="py-2 pr-4">{k.name}</td>
                    <td className="py-2 pr-4">{k.key_prefix}</td>
                    <td className="py-2 pr-4">{k.permissions.join(', ')}</td>
                    <td className="py-2 pr-4">{k.usage_count}</td>
                    <td className="py-2 pr-4">{k.is_active ? 'Active' : 'Revoked'}</td>
                    <td className="py-2 pr-4">
                      <button
                        onClick={() => revokeKey(k.id)}
                        className="inline-flex items-center px-2 py-1 rounded bg-red-900/40 text-red-300 border border-red-800 hover:bg-red-900/60"
                      >
                        <Trash2 className="w-4 h-4 mr-1" /> Revoke
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Surveys placeholder */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h2 className="text-xl font-semibold text-white mb-2">Surveys</h2>
        <p className="text-sm text-gray-400">Survey admin will appear here. For now, create via API/Swagger. This page will handle create/list/edit (RBAC: admin only).</p>
      </div>
    </div>
  )
}
