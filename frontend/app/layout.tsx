import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'AstrID - Planning Dashboard',
  description: 'Astronomical Identification: Temporal Dataset Preparation and Anomaly Detection',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="min-h-screen bg-astrid-dark text-white">
          {children}
        </div>
      </body>
    </html>
  )
}
