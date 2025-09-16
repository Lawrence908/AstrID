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
    <div className="min-h-[calc(100vh-4rem)] flex items-center justify-center px-4">
      <div className="w-full max-w-md bg-gray-900 border border-gray-800 rounded-lg p-6">
        {supabase && (
          <Auth
            supabaseClient={supabase}
            appearance={{ theme: ThemeSupa }}
            providers={['google']}
            onlyThirdPartyProviders={false}
            theme="dark"
            redirectTo={typeof window !== 'undefined' ? `${window.location.origin}` : undefined}
          />
        )}
      </div>
    </div>
  )
}
