'use client'

import { createContext, useContext, useEffect, useMemo, useState } from 'react'
import { createClient, Session, SupabaseClient } from '@supabase/supabase-js'

type AuthContextValue = {
  supabase: SupabaseClient | null
  session: Session | null
  profile: { email?: string } | null
  signOut: () => Promise<void>
  getToken?: () => Promise<string | null>
}

const AuthContext = createContext<AuthContextValue>({
  supabase: null,
  session: null,
  profile: null,
  signOut: async () => {},
  getToken: async () => null,
})

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const supabase = useMemo(() => {
    const url = process.env.NEXT_PUBLIC_SUPABASE_URL
    const key = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
    if (!url || !key) return null
    return createClient(url, key)
  }, [])

  const [session, setSession] = useState<Session | null>(null)

  useEffect(() => {
    if (!supabase) return

    let isMounted = true

    supabase.auth.getSession().then(({ data }) => {
      if (!isMounted) return
      setSession(data.session ?? null)
    })

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, sess) => {
      setSession(sess ?? null)
    })

    return () => {
      isMounted = false
      subscription.unsubscribe()
    }
  }, [supabase])

  const profile = useMemo(() => {
    if (!session) return null
    return { email: session.user?.email ?? undefined }
  }, [session])

  const signOut = async () => {
    if (!supabase) return
    await supabase.auth.signOut()
  }

  const getToken = async () => {
    // Prefer live session token
    if (session?.access_token) return session.access_token
    // Fallback to fetching from supabase
    if (!supabase) return null
    const { data } = await supabase.auth.getSession()
    return data.session?.access_token ?? null
  }

  return (
    <AuthContext.Provider value={{ supabase, session, profile, signOut, getToken }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => useContext(AuthContext)
