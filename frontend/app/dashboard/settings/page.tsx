'use client'

import { useState } from 'react'
import {
  Settings,
  Save,
  RefreshCw,
  Database,
  Server,
  Shield,
  Bell,
  Palette,
  Globe,
  Key,
  Upload,
  Download,
  AlertTriangle,
  CheckCircle,
  Info
} from 'lucide-react'

const settingsSections = [
  {
    id: 'general',
    title: 'General',
    icon: Settings,
    description: 'Basic system configuration'
  },
  {
    id: 'database',
    title: 'Database',
    icon: Database,
    description: 'Database connection and settings'
  },
  {
    id: 'api',
    title: 'API',
    icon: Server,
    description: 'API endpoints and authentication'
  },
  {
    id: 'security',
    title: 'Security',
    icon: Shield,
    description: 'Security and access control'
  },
  {
    id: 'notifications',
    title: 'Notifications',
    icon: Bell,
    description: 'Alert and notification settings'
  },
  {
    id: 'appearance',
    title: 'Appearance',
    icon: Palette,
    description: 'UI themes and display options'
  }
]

export default function SettingsPage() {
  const [activeSection, setActiveSection] = useState('general')
  const [hasChanges, setHasChanges] = useState(false)
  const [settings, setSettings] = useState({
    general: {
      siteName: 'AstrID',
      siteDescription: 'Astronomical Identification System',
      timezone: 'UTC',
      dateFormat: 'ISO',
      language: 'en'
    },
    database: {
      host: 'localhost',
      port: 5432,
      name: 'astrid_db',
      ssl: true,
      connectionPool: 10
    },
    api: {
      baseUrl: 'https://api.astrid.chrislawrence.ca',
      version: 'v1',
      rateLimit: 1000,
      timeout: 30,
      cors: true
    },
    security: {
      jwtSecret: '••••••••••••••••',
      sessionTimeout: 24,
      requireMFA: false,
      passwordPolicy: 'strong',
      maxLoginAttempts: 5
    },
    notifications: {
      email: {
        enabled: true,
        smtp: {
          host: 'smtp.gmail.com',
          port: 587,
          secure: false
        }
      },
      push: {
        enabled: false,
        vapidKey: '••••••••••••••••'
      },
      webhook: {
        enabled: false,
        url: '',
        secret: '••••••••••••••••'
      }
    },
    appearance: {
      theme: 'dark',
      primaryColor: '#2FA4E7',
      fontSize: 'medium',
      sidebarCollapsed: false,
      animations: true
    }
  })

  const handleSettingChange = (section: string, key: string, value: any) => {
    setSettings(prev => ({
      ...prev,
      [section]: {
        ...prev[section as keyof typeof prev],
        [key]: value
      }
    }))
    setHasChanges(true)
  }

  const handleSave = () => {
    // In a real app, this would save to the backend
    setHasChanges(false)
    console.log('Settings saved:', settings)
  }

  const handleReset = () => {
    // In a real app, this would reset to default values
    setHasChanges(false)
    console.log('Settings reset to defaults')
  }

  const renderGeneralSettings = () => (
    <div className="space-y-6">
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">Site Name</label>
        <input
          type="text"
          value={settings.general.siteName}
          onChange={(e) => handleSettingChange('general', 'siteName', e.target.value)}
          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-astrid-blue focus:border-transparent"
        />
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">Site Description</label>
        <textarea
          value={settings.general.siteDescription}
          onChange={(e) => handleSettingChange('general', 'siteDescription', e.target.value)}
          rows={3}
          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-astrid-blue focus:border-transparent"
        />
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Timezone</label>
          <select
            value={settings.general.timezone}
            onChange={(e) => handleSettingChange('general', 'timezone', e.target.value)}
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-astrid-blue focus:border-transparent"
          >
            <option value="UTC">UTC</option>
            <option value="America/New_York">Eastern Time</option>
            <option value="America/Chicago">Central Time</option>
            <option value="America/Denver">Mountain Time</option>
            <option value="America/Los_Angeles">Pacific Time</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Date Format</label>
          <select
            value={settings.general.dateFormat}
            onChange={(e) => handleSettingChange('general', 'dateFormat', e.target.value)}
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-astrid-blue focus:border-transparent"
          >
            <option value="ISO">ISO 8601 (2025-01-15)</option>
            <option value="US">US (01/15/2025)</option>
            <option value="EU">European (15/01/2025)</option>
          </select>
        </div>
      </div>
    </div>
  )

  const renderDatabaseSettings = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Host</label>
          <input
            type="text"
            value={settings.database.host}
            onChange={(e) => handleSettingChange('database', 'host', e.target.value)}
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-astrid-blue focus:border-transparent"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Port</label>
          <input
            type="number"
            value={settings.database.port}
            onChange={(e) => handleSettingChange('database', 'port', parseInt(e.target.value))}
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-astrid-blue focus:border-transparent"
          />
        </div>
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">Database Name</label>
        <input
          type="text"
          value={settings.database.name}
          onChange={(e) => handleSettingChange('database', 'name', e.target.value)}
          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-astrid-blue focus:border-transparent"
        />
      </div>
      <div className="flex items-center space-x-4">
        <label className="flex items-center">
          <input
            type="checkbox"
            checked={settings.database.ssl}
            onChange={(e) => handleSettingChange('database', 'ssl', e.target.checked)}
            className="w-4 h-4 text-astrid-blue bg-gray-700 border-gray-600 rounded focus:ring-astrid-blue"
          />
          <span className="ml-2 text-sm text-gray-300">Enable SSL</span>
        </label>
      </div>
    </div>
  )

  const renderAPISettings = () => (
    <div className="space-y-6">
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">Base URL</label>
        <input
          type="url"
          value={settings.api.baseUrl}
          onChange={(e) => handleSettingChange('api', 'baseUrl', e.target.value)}
          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-astrid-blue focus:border-transparent"
        />
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">API Version</label>
          <select
            value={settings.api.version}
            onChange={(e) => handleSettingChange('api', 'version', e.target.value)}
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-astrid-blue focus:border-transparent"
          >
            <option value="v1">v1</option>
            <option value="v2">v2</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Rate Limit (req/min)</label>
          <input
            type="number"
            value={settings.api.rateLimit}
            onChange={(e) => handleSettingChange('api', 'rateLimit', parseInt(e.target.value))}
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-astrid-blue focus:border-transparent"
          />
        </div>
      </div>
      <div className="flex items-center space-x-4">
        <label className="flex items-center">
          <input
            type="checkbox"
            checked={settings.api.cors}
            onChange={(e) => handleSettingChange('api', 'cors', e.target.checked)}
            className="w-4 h-4 text-astrid-blue bg-gray-700 border-gray-600 rounded focus:ring-astrid-blue"
          />
          <span className="ml-2 text-sm text-gray-300">Enable CORS</span>
        </label>
      </div>
    </div>
  )

  const renderSecuritySettings = () => (
    <div className="space-y-6">
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">JWT Secret</label>
        <div className="flex items-center space-x-2">
          <input
            type="password"
            value={settings.security.jwtSecret}
            onChange={(e) => handleSettingChange('security', 'jwtSecret', e.target.value)}
            className="flex-1 px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-astrid-blue focus:border-transparent"
          />
          <button className="p-2 text-gray-400 hover:text-white transition-colors">
            <Key className="w-4 h-4" />
          </button>
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Session Timeout (hours)</label>
          <input
            type="number"
            value={settings.security.sessionTimeout}
            onChange={(e) => handleSettingChange('security', 'sessionTimeout', parseInt(e.target.value))}
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-astrid-blue focus:border-transparent"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Max Login Attempts</label>
          <input
            type="number"
            value={settings.security.maxLoginAttempts}
            onChange={(e) => handleSettingChange('security', 'maxLoginAttempts', parseInt(e.target.value))}
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-astrid-blue focus:border-transparent"
          />
        </div>
      </div>
      <div className="space-y-3">
        <label className="flex items-center">
          <input
            type="checkbox"
            checked={settings.security.requireMFA}
            onChange={(e) => handleSettingChange('security', 'requireMFA', e.target.checked)}
            className="w-4 h-4 text-astrid-blue bg-gray-700 border-gray-600 rounded focus:ring-astrid-blue"
          />
          <span className="ml-2 text-sm text-gray-300">Require Multi-Factor Authentication</span>
        </label>
      </div>
    </div>
  )

  const renderNotificationSettings = () => (
    <div className="space-y-6">
      <div className="bg-gray-700 rounded-lg p-4">
        <h4 className="text-lg font-medium text-white mb-4">Email Notifications</h4>
        <div className="space-y-4">
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={settings.notifications.email.enabled}
              onChange={(e) => handleSettingChange('notifications', 'email', { ...settings.notifications.email, enabled: e.target.checked })}
              className="w-4 h-4 text-astrid-blue bg-gray-700 border-gray-600 rounded focus:ring-astrid-blue"
            />
            <span className="ml-2 text-sm text-gray-300">Enable email notifications</span>
          </label>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">SMTP Host</label>
              <input
                type="text"
                value={settings.notifications.email.smtp.host}
                onChange={(e) => handleSettingChange('notifications', 'email', {
                  ...settings.notifications.email,
                  smtp: { ...settings.notifications.email.smtp, host: e.target.value }
                })}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-astrid-blue focus:border-transparent"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">SMTP Port</label>
              <input
                type="number"
                value={settings.notifications.email.smtp.port}
                onChange={(e) => handleSettingChange('notifications', 'email', {
                  ...settings.notifications.email,
                  smtp: { ...settings.notifications.email.smtp, port: parseInt(e.target.value) }
                })}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-astrid-blue focus:border-transparent"
              />
            </div>
          </div>
        </div>
      </div>

      <div className="bg-gray-700 rounded-lg p-4">
        <h4 className="text-lg font-medium text-white mb-4">Push Notifications</h4>
        <label className="flex items-center">
          <input
            type="checkbox"
            checked={settings.notifications.push.enabled}
            onChange={(e) => handleSettingChange('notifications', 'push', { ...settings.notifications.push, enabled: e.target.checked })}
            className="w-4 h-4 text-astrid-blue bg-gray-700 border-gray-600 rounded focus:ring-astrid-blue"
          />
          <span className="ml-2 text-sm text-gray-300">Enable push notifications</span>
        </label>
      </div>
    </div>
  )

  const renderAppearanceSettings = () => (
    <div className="space-y-6">
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">Theme</label>
        <select
          value={settings.appearance.theme}
          onChange={(e) => handleSettingChange('appearance', 'theme', e.target.value)}
          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-astrid-blue focus:border-transparent"
        >
          <option value="dark">Dark</option>
          <option value="light">Light</option>
          <option value="auto">Auto</option>
        </select>
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">Primary Color</label>
        <div className="flex items-center space-x-2">
          <input
            type="color"
            value={settings.appearance.primaryColor}
            onChange={(e) => handleSettingChange('appearance', 'primaryColor', e.target.value)}
            className="w-12 h-10 bg-gray-700 border border-gray-600 rounded cursor-pointer"
          />
          <input
            type="text"
            value={settings.appearance.primaryColor}
            onChange={(e) => handleSettingChange('appearance', 'primaryColor', e.target.value)}
            className="flex-1 px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-astrid-blue focus:border-transparent"
          />
        </div>
      </div>
      <div className="space-y-3">
        <label className="flex items-center">
          <input
            type="checkbox"
            checked={settings.appearance.animations}
            onChange={(e) => handleSettingChange('appearance', 'animations', e.target.checked)}
            className="w-4 h-4 text-astrid-blue bg-gray-700 border-gray-600 rounded focus:ring-astrid-blue"
          />
          <span className="ml-2 text-sm text-gray-300">Enable animations</span>
        </label>
      </div>
    </div>
  )

  const renderSettingsContent = () => {
    switch (activeSection) {
      case 'general': return renderGeneralSettings()
      case 'database': return renderDatabaseSettings()
      case 'api': return renderAPISettings()
      case 'security': return renderSecuritySettings()
      case 'notifications': return renderNotificationSettings()
      case 'appearance': return renderAppearanceSettings()
      default: return renderGeneralSettings()
    }
  }

  return (
    <div className="min-h-[calc(100vh-4rem)] bg-astrid-dark">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-white mb-2">Settings</h1>
              <p className="text-gray-400">System configuration and preferences</p>
            </div>
            <div className="flex space-x-3">
              {hasChanges && (
                <button
                  onClick={handleReset}
                  className="flex items-center px-4 py-2 bg-gray-800 text-gray-200 rounded-lg hover:bg-gray-700 transition-colors border border-gray-700"
                >
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Reset
                </button>
              )}
              <button
                onClick={handleSave}
                disabled={!hasChanges}
                className={`flex items-center px-4 py-2 rounded-lg transition-colors ${
                  hasChanges
                    ? 'bg-astrid-blue text-white hover:bg-blue-600'
                    : 'bg-gray-800 text-gray-400 cursor-not-allowed'
                }`}
              >
                <Save className="w-4 h-4 mr-2" />
                Save Changes
              </button>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Settings Navigation */}
          <div className="lg:col-span-1">
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <nav className="space-y-2">
                {settingsSections.map((section) => {
                  const Icon = section.icon
                  return (
                    <button
                      key={section.id}
                      onClick={() => setActiveSection(section.id)}
                      className={`w-full flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                        activeSection === section.id
                          ? 'bg-astrid-blue text-white'
                          : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                      }`}
                    >
                      <Icon className="w-4 h-4 mr-3" />
                      {section.title}
                    </button>
                  )
                })}
              </nav>
            </div>
          </div>

          {/* Settings Content */}
          <div className="lg:col-span-3">
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <div className="mb-6">
                <h2 className="text-xl font-semibold text-white mb-2">
                  {settingsSections.find(s => s.id === activeSection)?.title}
                </h2>
                <p className="text-gray-400">
                  {settingsSections.find(s => s.id === activeSection)?.description}
                </p>
              </div>

              {renderSettingsContent()}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
