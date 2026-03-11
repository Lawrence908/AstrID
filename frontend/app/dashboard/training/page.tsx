'use client'

import { useState, useEffect, useMemo } from 'react'
import Link from 'next/link'
import {
  BarChart3,
  RefreshCw,
  Search,
  AlertCircle,
  CheckCircle,
  Image as ImageIcon,
  ChevronLeft,
  ChevronRight,
  X,
  TrendingUp,
  Target
} from 'lucide-react'

const PER_PAGE = 50
const BASE = '/training-data'

interface Summary {
  total_sne?: number
  processed?: number
  failed?: number
  real_samples?: number
  bogus_samples?: number
  total_samples?: number
  cutout_size?: number
  bogus_ratio?: number
  augmentation?: boolean
}

interface TripletMeta {
  sn_name: string
  mission: string
  filter: string
  center_x: number
  center_y: number
  ref_date?: string
  sci_date?: string
  overlap?: number
  sig_max?: number
}

type TabKind = 'original' | 'augmented' | 'predictions'

interface GalleryItem {
  id: string
  imagePath: string
  label: 'real' | 'bogus'
  snName?: string
  mission?: string
  filter?: string
  meta?: TripletMeta
  /** From predictions.json: model prediction and correctness */
  predProb?: number
  correct?: boolean
}

interface PredictionSample {
  path: string
  true_label: number
  pred_prob: number
  correct: boolean
}

interface PredictionsData {
  checkpoint?: string
  triplet_dir?: string
  val_only?: boolean
  n_rendered: number
  n_total: number
  samples: PredictionSample[]
}

function buildOriginalItems(
  realMeta: TripletMeta[],
  bogusMeta: TripletMeta[]
): GalleryItem[] {
  const items: GalleryItem[] = []
  realMeta.forEach((m, i) => {
    const filename = `${m.sn_name}_${m.mission}_${m.filter}_real_${String(i).padStart(3, '0')}.png`
    items.push({
      id: `real-${i}`,
      imagePath: `${BASE}/triplets/real/${filename}`,
      label: 'real',
      snName: m.sn_name,
      mission: m.mission,
      filter: m.filter,
      meta: m
    })
  })
  bogusMeta.forEach((m, i) => {
    const filename = `${m.sn_name}_${m.mission}_${m.filter}_bogus_${String(i).padStart(3, '0')}.png`
    items.push({
      id: `bogus-${i}`,
      imagePath: `${BASE}/triplets/bogus/${filename}`,
      label: 'bogus',
      snName: m.sn_name,
      mission: m.mission,
      filter: m.filter,
      meta: m
    })
  })
  return items
}

function buildAugmentedItems(realCount: number, bogusCount: number): GalleryItem[] {
  const items: GalleryItem[] = []
  const realViz = Math.min(500, realCount)
  const bogusViz = Math.min(500, bogusCount)
  for (let i = 0; i < realViz; i++) {
    items.push({
      id: `aug-real-${i}`,
      imagePath: `${BASE}/augmented/real/real_${String(i).padStart(4, '0')}.png`,
      label: 'real'
    })
  }
  for (let i = 0; i < bogusViz; i++) {
    items.push({
      id: `aug-bogus-${i}`,
      imagePath: `${BASE}/augmented/bogus/bogus_${String(i).padStart(4, '0')}.png`,
      label: 'bogus'
    })
  }
  return items
}

export default function TrainingPage() {
  const [tab, setTab] = useState<TabKind>('original')
  const [filterLabel, setFilterLabel] = useState<'all' | 'real' | 'bogus'>('all')
  const [missionFilter, setMissionFilter] = useState<string>('')
  const [searchSn, setSearchSn] = useState('')
  const [page, setPage] = useState(0)

  const [tripletsSummary, setTripletsSummary] = useState<Summary | null>(null)
  const [augmentedSummary, setAugmentedSummary] = useState<Summary | null>(null)
  const [realMeta, setRealMeta] = useState<TripletMeta[]>([])
  const [bogusMeta, setBogusMeta] = useState<TripletMeta[]>([])
  const [predictionsData, setPredictionsData] = useState<PredictionsData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const [selectedItem, setSelectedItem] = useState<GalleryItem | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        setError(null)
        const [tripletsRes, augRes, realRes, bogusRes, predRes] = await Promise.all([
          fetch(`${BASE}/triplets-summary.json`),
          fetch(`${BASE}/augmented-summary.json`),
          fetch(`${BASE}/real-metadata.json`),
          fetch(`${BASE}/bogus-metadata.json`),
          fetch(`${BASE}/validation/predictions.json`)
        ])
        if (tripletsRes.ok) setTripletsSummary(await tripletsRes.json())
        else setTripletsSummary(null)
        if (augRes.ok) setAugmentedSummary(await augRes.json())
        else setAugmentedSummary(null)
        if (realRes.ok) setRealMeta(await realRes.json())
        else setRealMeta([])
        if (bogusRes.ok) setBogusMeta(await bogusRes.json())
        else setBogusMeta([])
        if (predRes.ok) {
          const data = await predRes.json()
          if (data?.samples?.length) setPredictionsData(data)
          else setPredictionsData(null)
        } else setPredictionsData(null)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load training data')
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [])

  const allOriginal = useMemo(
    () => buildOriginalItems(realMeta, bogusMeta),
    [realMeta, bogusMeta]
  )
  const allAugmented = useMemo(() => {
    const r = augmentedSummary?.real_samples ?? 0
    const b = augmentedSummary?.bogus_samples ?? 0
    return buildAugmentedItems(r, b)
  }, [augmentedSummary])

  const allPredictions = useMemo((): GalleryItem[] => {
    if (!predictionsData?.samples?.length) return []
    const validationBase = `${BASE}/validation`
    return predictionsData.samples.map((s, i) => ({
      id: `pred-${i}`,
      imagePath: `${validationBase}/${s.path}`,
      label: s.true_label === 1 ? 'real' : 'bogus',
      predProb: s.pred_prob,
      correct: s.correct
    }))
  }, [predictionsData])

  const missions = useMemo(() => {
    const set = new Set<string>()
    allOriginal.forEach((item) => item.mission && set.add(item.mission))
    return Array.from(set).sort()
  }, [allOriginal])

  const filteredOriginal = useMemo(() => {
    return allOriginal.filter((item) => {
      if (filterLabel !== 'all' && item.label !== filterLabel) return false
      if (missionFilter && item.mission !== missionFilter) return false
      if (searchSn && (!item.snName || !item.snName.toLowerCase().includes(searchSn.toLowerCase()))) return false
      return true
    })
  }, [allOriginal, filterLabel, missionFilter, searchSn])

  const filteredAugmented = useMemo(() => {
    return allAugmented.filter((item) => {
      if (filterLabel !== 'all' && item.label !== filterLabel) return false
      return true
    })
  }, [allAugmented, filterLabel])

  const filteredPredictions = useMemo(() => {
    return allPredictions.filter((item) => {
      if (filterLabel !== 'all' && item.label !== filterLabel) return false
      return true
    })
  }, [allPredictions, filterLabel])

  const items = tab === 'original'
    ? filteredOriginal
    : tab === 'augmented'
      ? filteredAugmented
      : filteredPredictions
  const totalPages = Math.max(1, Math.ceil(items.length / PER_PAGE))
  const currentPage = Math.min(page, totalPages - 1)
  const paginatedItems = items.slice(currentPage * PER_PAGE, (currentPage + 1) * PER_PAGE)

  useEffect(() => {
    setPage(0)
  }, [tab, filterLabel, missionFilter, searchSn])

  const summary = tab === 'original'
    ? tripletsSummary
    : tab === 'augmented'
      ? augmentedSummary
      : null
  const totalSamples = tab === 'predictions'
    ? (predictionsData?.n_rendered ?? 0)
    : (summary?.total_samples ?? 0)
  const realSamples = tab === 'predictions'
    ? (predictionsData?.samples?.filter((s) => s.true_label === 1).length ?? 0)
    : (summary?.real_samples ?? 0)
  const bogusSamples = tab === 'predictions'
    ? (predictionsData?.samples?.filter((s) => s.true_label === 0).length ?? 0)
    : (summary?.bogus_samples ?? 0)
  const balanceRatio = bogusSamples && realSamples ? (bogusSamples / realSamples).toFixed(2) : '—'
  const predictionsCorrect = predictionsData?.samples?.filter((s) => s.correct).length ?? 0

  return (
    <div className="min-h-[calc(100vh-4rem)] bg-astrid-dark">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8 flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">Training Dataset</h1>
            <p className="text-gray-400">Browse triplet visualizations used for real/bogus CNN training</p>
          </div>
          <div className="flex items-center gap-3">
            <Link
              href="/dashboard/training/performance"
              className="inline-flex items-center px-4 py-2 bg-astrid-blue text-white rounded-lg hover:bg-blue-600 transition-colors"
            >
              <TrendingUp className="w-4 h-4 mr-2" />
              Model Performance
            </Link>
          </div>
        </div>

        {error && (
          <div className="mb-6 p-4 bg-red-900/30 border border-red-700 rounded-lg flex items-center gap-2 text-red-400">
            <AlertCircle className="w-5 h-5 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {!loading && !tripletsSummary && !augmentedSummary && !predictionsData && (
          <div className="mb-6 p-4 bg-gray-800 border border-gray-700 rounded-lg text-gray-400 text-sm">
            No training metadata or predictions loaded. For the triplet gallery, add summary and metadata JSON under{' '}
            <code className="bg-gray-700 px-1 rounded">public/training-data/</code>. For the Predictions tab, run{' '}
            <code className="bg-gray-700 px-1 rounded">visualize_real_bogus_predictions.py</code> and symlink its output to{' '}
            <code className="bg-gray-700 px-1 rounded">public/training-data/validation</code>.
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6 mb-8">
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Total samples</p>
                <p className="text-2xl font-bold text-white">{loading ? '—' : totalSamples.toLocaleString()}</p>
              </div>
              <ImageIcon className="w-8 h-8 text-astrid-blue" />
            </div>
          </div>
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Real</p>
                <p className="text-2xl font-bold text-green-500">{loading ? '—' : realSamples.toLocaleString()}</p>
              </div>
              <CheckCircle className="w-8 h-8 text-green-500" />
            </div>
          </div>
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Bogus</p>
                <p className="text-2xl font-bold text-amber-500">{loading ? '—' : bogusSamples.toLocaleString()}</p>
              </div>
              <AlertCircle className="w-8 h-8 text-amber-500" />
            </div>
          </div>
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Class balance (b:r)</p>
                <p className="text-2xl font-bold text-white">{loading ? '—' : balanceRatio}</p>
              </div>
            </div>
          </div>
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">
                  {tab === 'predictions' ? 'Correct (predicted)' : 'SNe processed / failed'}
                </p>
                <p className="text-2xl font-bold text-white">
                  {loading
                    ? '—'
                    : tab === 'predictions'
                      ? `${predictionsCorrect} / ${totalSamples}`
                      : `${tripletsSummary?.processed ?? 0} / ${tripletsSummary?.failed ?? 0}`}
                </p>
              </div>
              <Target className="w-8 h-8 text-gray-400" />
            </div>
          </div>
        </div>

        <div className="mb-6 flex flex-wrap items-center gap-4">
          <div className="flex rounded-lg border border-gray-700 overflow-hidden">
            <button
              onClick={() => setTab('original')}
              className={`px-4 py-2 text-sm font-medium transition-colors ${
                tab === 'original'
                  ? 'bg-astrid-blue text-white'
                  : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
              }`}
            >
              Original Triplets ({tripletsSummary?.total_samples ?? 0})
            </button>
            <button
              onClick={() => setTab('augmented')}
              className={`px-4 py-2 text-sm font-medium transition-colors ${
                tab === 'augmented'
                  ? 'bg-astrid-blue text-white'
                  : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
              }`}
            >
              Augmented ({augmentedSummary?.total_samples ?? 0})
            </button>
            <button
              onClick={() => setTab('predictions')}
              className={`px-4 py-2 text-sm font-medium transition-colors ${
                tab === 'predictions'
                  ? 'bg-astrid-blue text-white'
                  : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
              }`}
            >
              Predictions ({predictionsData?.n_rendered ?? 0})
            </button>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-400">Label:</span>
            {(['all', 'real', 'bogus'] as const).map((l) => (
              <button
                key={l}
                onClick={() => setFilterLabel(l)}
                className={`px-3 py-1 text-xs font-medium rounded-full ${
                  filterLabel === l ? 'bg-astrid-blue text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                {l === 'all' ? 'All' : l}
              </button>
            ))}
          </div>

          {tab === 'original' && missions.length > 0 && (
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-400">Mission:</span>
              <select
                value={missionFilter}
                onChange={(e) => setMissionFilter(e.target.value)}
                className="px-3 py-1.5 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm focus:ring-2 focus:ring-astrid-blue"
              >
                <option value="">All</option>
                {missions.map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </div>
          )}

          <div className="flex items-center gap-2 flex-1 min-w-[200px]">
            <Search className="w-4 h-4 text-gray-400 flex-shrink-0" />
            <input
              type="text"
              placeholder="Search by SN name..."
              value={searchSn}
              onChange={(e) => setSearchSn(e.target.value)}
              className="flex-1 px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 text-sm focus:ring-2 focus:ring-astrid-blue"
            />
          </div>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-20">
            <RefreshCw className="w-8 h-8 animate-spin text-astrid-blue" />
            <span className="ml-2 text-gray-400">Loading...</span>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4 mb-6">
              {paginatedItems.map((item) => (
                <button
                  key={item.id}
                  onClick={() => setSelectedItem(item)}
                  className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden hover:border-astrid-blue/50 transition-colors text-left"
                >
                  <div className="aspect-square bg-gray-900 relative">
                    <img
                      src={item.imagePath}
                      alt={item.snName || item.id}
                      className="w-full h-full object-contain"
                    />
                    <span
                      className={`absolute top-2 right-2 px-2 py-0.5 text-xs font-medium rounded ${
                        item.label === 'real' ? 'bg-green-900 text-green-300' : 'bg-amber-900 text-amber-300'
                      }`}
                    >
                      {item.label}
                    </span>
                    {item.predProb != null && (
                      <>
                        <span
                          className={`absolute bottom-2 left-2 px-2 py-0.5 text-xs font-medium rounded ${
                            item.correct ? 'bg-green-900/90 text-green-300' : 'bg-red-900/90 text-red-300'
                          }`}
                        >
                          {item.correct ? '✓' : '✗'} P(real)={(item.predProb * 100).toFixed(0)}%
                        </span>
                      </>
                    )}
                  </div>
                  <div className="p-2 text-xs text-gray-300 truncate">
                    {item.snName ?? item.id}
                    {item.mission && ` · ${item.mission} ${item.filter}`}
                    {item.predProb != null && ` · P(real)=${(item.predProb * 100).toFixed(0)}%`}
                  </div>
                </button>
              ))}
            </div>

            {totalPages > 1 && (
              <div className="flex items-center justify-center gap-4">
                <button
                  onClick={() => setPage((p) => Math.max(0, p - 1))}
                  disabled={currentPage === 0}
                  className="p-2 rounded-lg bg-gray-800 border border-gray-700 text-white disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-700"
                >
                  <ChevronLeft className="w-5 h-5" />
                </button>
                <span className="text-sm text-gray-400">
                  Page {currentPage + 1} of {totalPages} ({items.length} items)
                </span>
                <button
                  onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                  disabled={currentPage >= totalPages - 1}
                  className="p-2 rounded-lg bg-gray-800 border border-gray-700 text-white disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-700"
                >
                  <ChevronRight className="w-5 h-5" />
                </button>
              </div>
            )}
          </>
        )}
      </div>

      {selectedItem && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
          <div className="bg-gray-800 rounded-lg border border-gray-700 max-w-2xl w-full max-h-[90vh] overflow-auto">
            <div className="p-4 border-b border-gray-700 flex items-center justify-between">
              <h3 className="text-lg font-semibold text-white">
                {selectedItem.snName ?? selectedItem.id} · {selectedItem.label}
                {selectedItem.mission && ` · ${selectedItem.mission} ${selectedItem.filter}`}
              </h3>
              <button
                onClick={() => setSelectedItem(null)}
                className="p-2 text-gray-400 hover:text-white rounded"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="p-4">
              <img
                src={selectedItem.imagePath}
                alt={selectedItem.snName ?? selectedItem.id}
                className="w-full rounded-lg mb-4"
              />
              {selectedItem.predProb != null && (
                <div className="mb-4 flex items-center gap-2">
                  <span className={`px-2 py-1 text-sm font-medium rounded ${selectedItem.correct ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'}`}>
                    {selectedItem.correct ? '✓ Correct' : '✗ Wrong'}
                  </span>
                  <span className="text-gray-400">P(real) = {(selectedItem.predProb * 100).toFixed(1)}%</span>
                </div>
              )}
              {selectedItem.meta && (
                <dl className="grid grid-cols-2 gap-2 text-sm">
                  <dt className="text-gray-500">Center (x, y)</dt>
                  <dd className="text-gray-300">
                    {selectedItem.meta.center_x?.toFixed(1)}, {selectedItem.meta.center_y?.toFixed(1)}
                  </dd>
                  {selectedItem.meta.overlap != null && (
                    <>
                      <dt className="text-gray-500">Overlap %</dt>
                      <dd className="text-gray-300">{selectedItem.meta.overlap.toFixed(2)}</dd>
                    </>
                  )}
                  {selectedItem.meta.sig_max != null && (
                    <>
                      <dt className="text-gray-500">sig_max</dt>
                      <dd className="text-gray-300">{Number(selectedItem.meta.sig_max).toExponential(2)}</dd>
                    </>
                  )}
                </dl>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
