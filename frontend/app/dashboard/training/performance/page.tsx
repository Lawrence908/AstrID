'use client'

import { useState, useEffect, useMemo } from 'react'
import Link from 'next/link'
import {
  ArrowLeft,
  TrendingUp,
  AlertCircle,
  RefreshCw,
  Target,
  Award,
  Image as ImageIcon
} from 'lucide-react'

const BASE = '/training-data'

interface HistoryEntry {
  epoch: number
  train_loss: number
  val_precision: number
  val_recall: number
  val_f1: number
  val_aucpr: number
}

const CHART_HEIGHT = 240
const CHART_PAD = { top: 12, right: 12, bottom: 28, left: 44 }

function TrainingChart({ history }: { history: HistoryEntry[] }) {
  const [metric, setMetric] = useState<'train_loss' | 'val_f1' | 'val_precision' | 'val_recall' | 'val_aucpr'>('val_f1')
  const epochs = history.map((h) => h.epoch)
  const values = history.map((h) => h[metric])
  const isLoss = metric === 'train_loss'
  const minVal = Math.min(...values)
  const maxVal = Math.max(...values)
  const range = maxVal - minVal || 1
  const width = Math.max(400, epochs.length * 12)
  const innerWidth = width - CHART_PAD.left - CHART_PAD.right
  const innerHeight = CHART_HEIGHT - CHART_PAD.top - CHART_PAD.bottom

  const scaleX = (e: number) =>
    CHART_PAD.left + (innerWidth * (e - epochs[0])) / (epochs[epochs.length - 1] - epochs[0] || 1)
  const scaleY = (v: number) => {
    if (isLoss) {
      return CHART_PAD.top + innerHeight * (1 - (v - minVal) / range)
    }
    return CHART_PAD.top + innerHeight * (1 - (v - minVal) / range)
  }

  const pathD = values
    .map((v, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(epochs[i])} ${scaleY(v)}`)
    .join(' ')

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
      <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
        <h3 className="text-lg font-semibold text-white">Training curves</h3>
        <div className="flex gap-2">
          {(
            [
              ['val_f1', 'F1'],
              ['val_aucpr', 'AUCPR'],
              ['val_precision', 'Precision'],
              ['val_recall', 'Recall'],
              ['train_loss', 'Train loss']
            ] as const
          ).map(([key, label]) => (
            <button
              key={key}
              onClick={() => setMetric(key)}
              className={`px-3 py-1 text-xs font-medium rounded ${
                metric === key ? 'bg-astrid-blue text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
      <div className="overflow-x-auto">
        <svg width={width} height={CHART_HEIGHT} className="text-astrid-blue">
          <line
            x1={CHART_PAD.left}
            y1={CHART_PAD.top}
            x2={CHART_PAD.left}
            y2={CHART_HEIGHT - CHART_PAD.bottom}
            stroke="currentColor"
            strokeOpacity={0.3}
            strokeWidth={1}
          />
          <line
            x1={CHART_PAD.left}
            y1={CHART_HEIGHT - CHART_PAD.bottom}
            x2={width - CHART_PAD.right}
            y2={CHART_HEIGHT - CHART_PAD.bottom}
            stroke="currentColor"
            strokeOpacity={0.3}
            strokeWidth={1}
          />
          <path
            d={pathD}
            fill="none"
            stroke="currentColor"
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </div>
      <p className="text-xs text-gray-500 mt-2">
        {metric} (min: {minVal.toFixed(4)}, max: {maxVal.toFixed(4)})
      </p>
    </div>
  )
}

export default function PerformancePage() {
  const [history, setHistory] = useState<HistoryEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        setError(null)
        const res = await fetch(`${BASE}/model-history.json`)
        if (!res.ok) throw new Error('Failed to load model history')
        const data = await res.json()
        setHistory(Array.isArray(data) ? data : [])
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load model history')
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [])

  const best = useMemo(() => {
    if (history.length === 0) return null
    const bestF1 = history.reduce((a, b) => (b.val_f1 > a.val_f1 ? b : a), history[0])
    const bestAucpr = history.reduce((a, b) => (b.val_aucpr > a.val_aucpr ? b : a), history[0])
    const bestLoss = history.reduce((a, b) => (b.train_loss < a.train_loss ? b : a), history[0])
    return { bestF1, bestAucpr, bestLoss }
  }, [history])

  return (
    <div className="min-h-[calc(100vh-4rem)] bg-astrid-dark">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8 flex flex-wrap items-center gap-4">
          <Link
            href="/dashboard/training"
            className="inline-flex items-center text-gray-400 hover:text-white transition-colors"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Training Dataset
          </Link>
        </div>

        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Model Performance</h1>
          <p className="text-gray-400">Training curves and validation metrics for the real/bogus CNN</p>
        </div>

        {error && (
          <div className="mb-6 p-4 bg-red-900/30 border border-red-700 rounded-lg flex items-center gap-2 text-red-400">
            <AlertCircle className="w-5 h-5 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {loading ? (
          <div className="flex items-center justify-center py-20">
            <RefreshCw className="w-8 h-8 animate-spin text-astrid-blue" />
            <span className="ml-2 text-gray-400">Loading history...</span>
          </div>
        ) : (
          <>
            {best && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-400">Best validation F1</p>
                      <p className="text-2xl font-bold text-green-500">
                        {best.bestF1.val_f1.toFixed(4)}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">Epoch {best.bestF1.epoch}</p>
                    </div>
                    <Award className="w-8 h-8 text-green-500" />
                  </div>
                </div>
                <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-400">Best AUCPR</p>
                      <p className="text-2xl font-bold text-astrid-blue">
                        {best.bestAucpr.val_aucpr.toFixed(4)}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">Epoch {best.bestAucpr.epoch}</p>
                    </div>
                    <TrendingUp className="w-8 h-8 text-astrid-blue" />
                  </div>
                </div>
                <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-400">Lowest train loss</p>
                      <p className="text-2xl font-bold text-white">
                        {best.bestLoss.train_loss.toFixed(4)}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">Epoch {best.bestLoss.epoch}</p>
                    </div>
                    <Target className="w-8 h-8 text-gray-400" />
                  </div>
                </div>
              </div>
            )}

            {history.length > 0 && <TrainingChart history={history} />}

            <div className="mt-8 bg-gray-800 rounded-lg border border-gray-700 p-6">
              <h3 className="text-lg font-semibold text-white mb-2 flex items-center gap-2">
                <ImageIcon className="w-5 h-5" />
                Validation predictions
              </h3>
              <p className="text-gray-400 text-sm">
                After retraining on the full dataset, validation prediction images will appear here.
                The app will look for images in <code className="bg-gray-700 px-1 rounded">/training-data/validation/</code>.
                Symlink your validation output directory to <code className="bg-gray-700 px-1 rounded">frontend/public/training-data/validation</code> to enable this gallery.
              </p>
              <div className="mt-4 p-6 border border-dashed border-gray-600 rounded-lg text-center text-gray-500">
                Validation prediction gallery placeholder — add validation images post-training to inspect model performance on held-out samples.
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
