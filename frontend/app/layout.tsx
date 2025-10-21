import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { AuthProvider } from '../lib/auth/AuthProvider'
import Navbar from '../components/Navbar'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'AstrID - Planning Dashboard',
  description: 'Astronomical Identification: Temporal Dataset Preparation and Anomaly Detection',
  icons: {
    icon: [
      { url: '/favicon-16x16.png', sizes: '16x16', type: 'image/png' },
      { url: '/favicon-32x32.png', sizes: '32x32', type: 'image/png' },
      { url: '/favicon.ico', sizes: 'any' }
    ],
    apple: [
      { url: '/android-chrome-192x192.png', sizes: '192x192', type: 'image/png' }
    ],
    other: [
      { url: '/android-chrome-512x512.png', sizes: '512x512', type: 'image/png' }
    ]
  }
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <head>
        <link rel="manifest" href="/site.webmanifest" />
      </head>
      <body className={inter.className}>
        <div className="min-h-screen bg-astrid-dark text-white">
          <AuthProvider>
            <Navbar />
            {children}
          </AuthProvider>
        </div>
      </body>
    </html>
  )
}
