'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { User, BookOpen, Eye, Search, BarChart3, Users, Settings, Workflow } from 'lucide-react'
import { useAuth } from '@/lib/auth/AuthProvider'
import { useRouter } from 'next/navigation'

const navigation = [
  { name: 'Dashboard', href: '/', icon: BarChart3 },
  { name: 'Observations', href: '/dashboard/observations', icon: Eye },
  { name: 'Detections', href: '/dashboard/detections', icon: Search },
  { name: 'Workflows', href: '/dashboard/workflows', icon: Workflow },
  { name: 'Users', href: '/dashboard/users', icon: Users },
  { name: 'Settings', href: '/dashboard/settings', icon: Settings },
]

export default function Navbar() {
  const { session, profile, signOut } = useAuth()
  const router = useRouter()
  const pathname = usePathname()

  const handleLogout = async () => {
    await signOut()
    router.replace('/login')
  }

  return (
    <header className="bg-gray-900 border-b border-gray-700">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-8">
            <Link href="/" className="flex items-center space-x-2 text-2xl font-bold text-astrid-blue">
              <img
                src="/images/jwst.png"
                alt="JWST"
                className="w-8 h-8 rounded object-cover"
              />
              <span>AstrID</span>
            </Link>

            {/* Navigation Links */}
            {session && (
              <nav className="hidden md:flex space-x-1">
                {navigation.map((item) => {
                  const isActive = pathname === item.href
                  return (
                    <Link
                      key={item.name}
                      href={item.href}
                      className={`flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                        isActive
                          ? 'bg-astrid-blue text-white'
                          : 'text-gray-300 hover:bg-gray-800 hover:text-white'
                      }`}
                    >
                      <item.icon className="w-4 h-4 mr-2" />
                      {item.name}
                    </Link>
                  )
                })}
              </nav>
            )}
          </div>

          <div className="flex items-center space-x-4">
            {!session ? (
              <Link href="/login" className="px-3 py-1.5 rounded bg-astrid-blue text-white text-sm">Login</Link>
            ) : (
              <>
                {/* Planning Dashboard Button */}
                <Link
                  href="/planning"
                  className="flex items-center px-3 py-1.5 rounded bg-gray-800 text-gray-200 text-sm border border-gray-700 hover:bg-gray-700 transition-colors"
                >
                  <BookOpen className="w-4 h-4 mr-2" />
                  Planning
                </Link>

                {/* Profile Dropdown */}
                <div className="flex items-center space-x-2 text-sm text-gray-300">
                  <User className="w-4 h-4" />
                  <span>{profile?.email ?? 'User'}</span>
                </div>

                <button
                  onClick={handleLogout}
                  className="px-3 py-1.5 rounded bg-gray-800 text-gray-200 text-sm border border-gray-700 hover:bg-gray-700 transition-colors"
                >
                  Logout
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </header>
  )
}
