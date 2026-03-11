import { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { getTradeIdeas } from '../api'
import MetricCard from '../components/MetricCard'
import LoadingSkeleton from '../components/LoadingSkeleton'

const formatCompact = (v) => {
  if (!v) return '$0'
  if (v >= 1e9) return '$' + (v / 1e9).toFixed(1) + 'B'
  if (v >= 1e6) return '$' + (v / 1e6).toFixed(1) + 'M'
  if (v >= 1e3) return '$' + (v / 1e3).toFixed(0) + 'K'
  return '$' + v.toFixed(0)
}

function ScoreBar({ score, label }) {
  const getColor = (s) => {
    if (s >= 70) return 'bg-green-500'
    if (s >= 50) return 'bg-green-600'
    if (s >= 30) return 'bg-yellow-500'
    return 'bg-gray-600'
  }
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="text-gray-500 w-16 shrink-0">{label}</span>
      <div className="flex-1 bg-gray-800 rounded-full h-1.5">
        <div className={`h-1.5 rounded-full ${getColor(score)}`} style={{ width: `${Math.min(score, 100)}%` }} />
      </div>
      <span className="text-gray-400 w-8 text-right">{Math.round(score)}</span>
    </div>
  )
}

function SignalScoreRing({ score }) {
  const circumference = 2 * Math.PI * 28
  const offset = circumference - (score / 100) * circumference
  const color = score >= 70 ? '#22c55e' : score >= 50 ? '#84cc16' : score >= 30 ? '#eab308' : '#6b7280'

  return (
    <div className="relative w-20 h-20 shrink-0">
      <svg className="w-20 h-20 -rotate-90" viewBox="0 0 64 64">
        <circle cx="32" cy="32" r="28" fill="none" stroke="#1f2937" strokeWidth="4" />
        <circle
          cx="32" cy="32" r="28" fill="none" stroke={color} strokeWidth="4"
          strokeDasharray={circumference} strokeDashoffset={offset}
          strokeLinecap="round"
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-lg font-bold" style={{ color }}>{Math.round(score)}</span>
      </div>
    </div>
  )
}

export default function TradeIdeas() {
  const navigate = useNavigate()
  const [ideas, setIdeas] = useState([])
  const [loading, setLoading] = useState(true)
  const [minScore, setMinScore] = useState(20)
  const [sectorFilter, setSectorFilter] = useState('')
  const [lastUpdated, setLastUpdated] = useState(null)

  const fetchData = useCallback(async () => {
    setLoading(true)
    try {
      const params = { min_score: minScore, limit: 30 }
      if (sectorFilter) params.sector = sectorFilter
      const data = await getTradeIdeas(params)
      setIdeas(data)
      setLastUpdated(new Date())
    } catch (err) {
      console.error('Trade ideas fetch error:', err)
    } finally {
      setLoading(false)
    }
  }, [minScore, sectorFilter])

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 10 * 60 * 1000)
    return () => clearInterval(interval)
  }, [fetchData])

  const topScore = ideas.length > 0 ? ideas[0].signal_score : 0
  const avgScore = ideas.length > 0
    ? Math.round(ideas.reduce((s, i) => s + i.signal_score, 0) / ideas.length)
    : 0
  const highConviction = ideas.filter(i => i.signal_score >= 60).length
  const totalBuyValue = ideas.reduce((s, i) => s + (i.total_buy_value || 0), 0)

  if (loading) {
    return (
      <div className="space-y-6">
        <h2 className="text-xl font-bold">Trade Ideas</h2>
        <LoadingSkeleton type="cards" />
        <LoadingSkeleton rows={10} />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold">Trade Ideas</h2>
          <p className="text-xs text-gray-500 mt-1">
            Ranked by Insider Signal Score (0-100) — higher = stronger signal
          </p>
        </div>
        {lastUpdated && (
          <span className="text-xs text-gray-500">
            Updated {lastUpdated.toLocaleTimeString()}
          </span>
        )}
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          label="Top Signal"
          value={Math.round(topScore)}
          color={topScore >= 60 ? 'text-green-400' : 'text-yellow-400'}
          tooltip="Highest Insider Signal Score currently"
        />
        <MetricCard
          label="Avg Score"
          value={avgScore}
          color="text-gray-300"
          tooltip="Average score across all active ideas"
        />
        <MetricCard
          label="High Conviction"
          value={highConviction}
          color="text-green-400"
          subtext="score 60+"
          tooltip="Ideas with signal score >= 60"
        />
        <MetricCard
          label="Total Buy Volume"
          value={formatCompact(totalBuyValue)}
          color="text-green-400"
          tooltip="Combined insider buying across all ideas"
        />
      </div>

      {/* Filters */}
      <div className="flex gap-3 items-center flex-wrap">
        <label className="text-xs text-gray-500">Min Score:</label>
        <div className="flex gap-1">
          {[0, 20, 40, 60, 80].map(s => (
            <button
              key={s}
              onClick={() => setMinScore(s)}
              className={`px-2.5 py-1 text-xs rounded transition-colors ${
                minScore === s
                  ? 'bg-blue-600 text-white'
                  : 'bg-dark-card border border-dark-border text-gray-400 hover:text-gray-200'
              }`}
            >
              {s}+
            </button>
          ))}
        </div>
        <select
          value={sectorFilter}
          onChange={e => setSectorFilter(e.target.value)}
          className="bg-dark-card border border-dark-border rounded px-2 py-1 text-xs text-gray-300"
        >
          <option value="">All Sectors</option>
          {['Defense & Aerospace', 'AI & Semiconductors', 'Energy',
            'Healthcare & Pharma', 'Financials & Banking', 'Technology',
            'Real Estate', 'Consumer Discretionary', 'Industrials'].map(s =>
            <option key={s} value={s}>{s}</option>
          )}
        </select>
      </div>

      {/* Ideas List */}
      {ideas.length === 0 ? (
        <div className="text-center py-16 text-gray-500">
          <p className="text-lg">No trade ideas match your filters</p>
          <p className="text-sm mt-1">Try lowering the minimum score</p>
        </div>
      ) : (
        <div className="space-y-3">
          {ideas.map((idea, idx) => (
            <div
              key={idea.ticker}
              onClick={() => navigate(`/ticker/${idea.ticker}`)}
              className="bg-dark-card border border-dark-border rounded-lg p-4 hover:border-blue-500/50 cursor-pointer transition-colors"
            >
              <div className="flex items-start gap-4">
                {/* Score Ring */}
                <SignalScoreRing score={idea.signal_score} />

                {/* Main Content */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-3 mb-1">
                    <span className="text-xs text-gray-600 font-mono">#{idx + 1}</span>
                    <span className="text-lg font-bold font-mono text-white">{idea.ticker}</span>
                    {idea.company_name && (
                      <span className="text-sm text-gray-400 truncate">{idea.company_name}</span>
                    )}
                    {idea.sector && (
                      <span className="text-xs bg-dark-hover text-gray-500 px-2 py-0.5 rounded hidden md:inline">
                        {idea.sector}
                      </span>
                    )}
                    {idea.current_price && (
                      <span className="text-sm text-gray-300 ml-auto font-mono">
                        ${idea.current_price.toFixed(2)}
                      </span>
                    )}
                  </div>

                  {/* Thesis */}
                  <p className="text-sm text-gray-300 mb-3">{idea.thesis}</p>

                  {/* Score Components */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-x-6 gap-y-1">
                    <ScoreBar score={idea.components.cluster} label="Cluster" />
                    <ScoreBar score={idea.components.conviction} label="Convict." />
                    <ScoreBar score={idea.components.seniority} label="Senior." />
                    <ScoreBar score={idea.components.political_convergence} label="Politic." />
                    <ScoreBar score={idea.components.timeliness} label="Timely" />
                    <ScoreBar score={idea.components.price_context} label="Price" />
                    <ScoreBar score={idea.components.filer_accuracy} label="Accurcy" />
                    <div className="flex items-center gap-2 text-xs">
                      <span className="text-gray-500 w-16 shrink-0">Buy Vol</span>
                      <span className="text-green-400 font-mono">{formatCompact(idea.total_buy_value)}</span>
                      <span className="text-gray-600">/ {idea.unique_buyers} buyers</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
