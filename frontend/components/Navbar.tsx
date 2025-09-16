'use client'

import Link from 'next/link'
import { User } from 'lucide-react'
import { useAuth } from '@/lib/auth/AuthProvider'
import { useRouter } from 'next/navigation'

export default function Navbar() {
  const { session, profile, signOut } = useAuth()
  const router = useRouter()

  const handleLogout = async () => {
    await signOut()
    router.replace('/login')
  }

  return (
    <header className="bg-gray-900 border-b border-gray-700">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <Link href="/" className="text-2xl font-bold text-astrid-blue">AstrID</Link>
            <div className="ml-4">
              <p className="text-sm text-gray-400">Planning Dashboard</p>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            {!session ? (
              <Link href="/login" className="px-3 py-1.5 rounded bg-astrid-blue text-white text-sm">Login</Link>
            ) : (
              <>
                <div className="flex items-center space-x-2 text-sm text-gray-300">
                  <User className="w-4 h-4" />
                  <span>{profile?.email ?? 'User'}</span>
                </div>
                <button onClick={handleLogout} className="px-3 py-1.5 rounded bg-gray-800 text-gray-200 text-sm border border-gray-700 hover:bg-gray-700">
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
