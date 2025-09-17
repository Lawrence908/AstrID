'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { Auth } from '@supabase/auth-ui-react'
import { ThemeSupa } from '@supabase/auth-ui-shared'
import { useAuth } from '@/lib/auth/AuthProvider'

export default function LoginPage() {
  const { supabase, session } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (session) router.replace('/')
  }, [session, router])

  return (
    <div className="min-h-[calc(100vh-4rem)] flex items-center justify-center px-4 bg-gradient-to-b from-gray-950 via-gray-900 to-gray-950">
      <div className="w-full max-w-md bg-gray-900/80 backdrop-blur border border-gray-800 rounded-xl p-6 shadow-2xl">
        <div className="mb-6 text-center">
          <div className="inline-flex items-center gap-2">
            <span className="text-2xl font-bold text-astrid-blue">AstrID</span>
            <span className="text-sm text-gray-400">Planning Dashboard</span>
          </div>
          <h1 className="mt-3 text-xl font-semibold text-white">Welcome back</h1>
          <p className="mt-1 text-sm text-gray-400">Sign in to continue</p>
        </div>
        {supabase ? (
          <Auth
            supabaseClient={supabase}
            appearance={{
              theme: ThemeSupa,
              variables: {
                default: {
                  colors: {
                    brand: '#2FA4E7',
                    brandAccent: '#1F6FA1',
                    inputBackground: '#111827',
                    inputText: 'white',
                    inputBorder: '#374151',
                    inputLabelText: '#D1D5DB',
                    inputPlaceholder: '#9CA3AF',
                  },
                },
              },
              style: {
                button: {
                  background: '#2FA4E7',
                  color: 'white',
                  borderRadius: '0.5rem',
                  fontWeight: 600,
                  padding: '0.625rem 0.75rem',
                },
                anchor: {
                  color: '#9CA3AF',
                },
                input: {
                  borderRadius: '0.5rem',
                },
                label: {
                  color: '#D1D5DB',
                },
                divider: {
                  background: '#1F2937',
                },
                message: {
                  color: '#D1D5DB',
                },
              },
            }}
            providers={['google']}
            onlyThirdPartyProviders={false}
            theme="dark"
            redirectTo={typeof window !== 'undefined' ? `${window.location.origin}` : undefined}
          />
        ) : (
          <div className="space-y-2 text-sm text-gray-300">
            <p className="font-medium text-astrid-blue">Authentication is not configured</p>
            <p>Set the following environment variables and restart the dev server:</p>
            <ul className="list-disc list-inside text-gray-400">
              <li><code>NEXT_PUBLIC_SUPABASE_URL</code></li>
              <li><code>NEXT_PUBLIC_SUPABASE_ANON_KEY</code></li>
            </ul>
          </div>
        )}
      </div>
    </div>
  )
}
