import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RTooltip, ResponsiveContainer
} from 'recharts'
import { getTickerActivity } from '../api'
import TransactionTable from '../components/TransactionTable'
import MetricCard from '../components/MetricCard'
import LoadingSkeleton from '../components/LoadingSkeleton'
import InfoTooltip from '../components/Tooltip'

const formatCompact = (v) => {
  if (!v) return '$0'
  if (v >= 1e9) return '$' + (v / 1e9).toFixed(1) + 'B'
  if (v >= 1e6) return '$' + (v / 1e6).toFixed(1) + 'M'
  if (v >= 1e3) return '$' + (v / 1e3).toFixed(0) + 'K'
  return '$' + v.toFixed(0)
}

export default function TickerDeepDive() {
  const { ticker: paramTicker } = useParams()
  const navigate = useNavigate()
  const [searchInput, setSearchInput] = useState(paramTicker || '')
  const [ticker, setTicker] = useState(paramTicker || '')
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (paramTicker) {
      setTicker(paramTicker.toUpperCase())
      setSearchInput(paramTicker.toUpperCase())
    }
  }, [paramTicker])

  useEffect(() => {
    if (!ticker) return
    setLoading(true)
    setError(null)
    getTickerActivity(ticker)
      .then(setData)
      .catch(err => {
        console.error(err)
        setError('Failed to load ticker data')
        setData(null)
      })
      .finally(() => setLoading(false))
  }, [ticker])

  const handleSearch = (e) => {
    e.preventDefault()
    if (searchInput.trim()) {
      navigate(`/ticker/${searchInput.trim().toUpperCase()}`)
    }
  }

  // Build timeline data
  const timelineData = data ? (() => {
    const dayMap = {}
    data.transactions.forEach(tx => {
      const date = tx.transaction_date
      if (!dayMap[date]) dayMap[date] = { date, buy: 0, sell: 0 }
      if (tx.transaction_type === 'buy') dayMap[date].buy += tx.total_value || 0
      else dayMap[date].sell += tx.total_value || 0
    })
    return Object.values(dayMap).sort((a, b) => a.date.localeCompare(b.date))
  })() : []

  // Politicians/committees involved
  const politicians = data ? data.transactions
    .filter(tx => tx.filer_type === 'politician')
    .reduce((acc, tx) => {
      const key = tx.filer_name
      if (!acc[key]) acc[key] = { name: tx.filer_name, party: tx.party, committee: tx.committee, count: 0 }
      acc[key].count++
      return acc
    }, {}) : {}

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold">Ticker Deep Dive</h2>

      {/* Search */}
      <form onSubmit={handleSearch} className="flex gap-2">
        <input
          type="text"
          value={searchInput}
          onChange={e => setSearchInput(e.target.value.toUpperCase())}
          placeholder="Enter ticker symbol (e.g., NVDA)"
          className="flex-1 max-w-xs bg-dark-card border border-dark-border rounded-lg px-4 py-2 text-white placeholder-gray-600 focus:border-blue-500 focus:outline-none"
        />
        <button
          type="submit"
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm transition-colors"
        >
          Search
        </button>
      </form>

      {loading && (
        <div className="space-y-4">
          <LoadingSkeleton type="cards" />
          <LoadingSkeleton type="chart" />
        </div>
      )}

      {error && <p className="text-red-400 text-sm">{error}</p>}

      {data && !loading && (
        <>
          {/* Header */}
          <div className="flex items-center gap-4">
            <span className="text-3xl font-bold font-mono text-white">{data.ticker}</span>
            {data.company_name && <span className="text-gray-400">{data.company_name}</span>}
            {data.sector && (
              <span className="text-xs bg-dark-hover text-gray-400 px-2 py-1 rounded">{data.sector}</span>
            )}
          </div>

          {/* Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            <MetricCard
              label="Cluster Score"
              value={`${data.cluster_score}/10`}
              color={data.cluster_score >= 5 ? 'text-green-400' : 'text-gray-300'}
              tooltip="Unique insiders buying in last 30 days (1-10)"
            />
            <MetricCard
              label="Conviction Score"
              value={data.conviction_score.toFixed(1)}
              color={data.conviction_score >= 5 ? 'text-green-400' : 'text-gray-300'}
              tooltip="Weighted by filer seniority + trade size"
            />
            <MetricCard
              label="Committee Exposure"
              value={data.committee_exposure}
              color={data.committee_exposure > 0 ? 'text-yellow-400' : 'text-gray-300'}
              tooltip="Politicians with committee oversight buying"
            />
            <MetricCard
              label="Avg Delay"
              value={`${data.delay_score}d`}
              color={data.delay_score > 30 ? 'text-red-400' : 'text-gray-300'}
              tooltip="Average days between trade and filing"
            />
            <MetricCard
              label="Buy/Sell Ratio"
              value={data.buy_sell_ratio.toFixed(1)}
              color={data.buy_sell_ratio > 1 ? 'text-green-400' : 'text-red-400'}
              tooltip="Ratio of buy value to sell value"
            />
          </div>

          {/* Timeline Chart */}
          {timelineData.length > 0 && (
            <div className="bg-dark-card border border-dark-border rounded-lg p-4">
              <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-3">
                Buy/Sell Activity Timeline
              </h3>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={timelineData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2a2d3a" />
                  <XAxis dataKey="date" stroke="#6b7280" fontSize={10} />
                  <YAxis stroke="#6b7280" fontSize={11} tickFormatter={formatCompact} />
                  <RTooltip
                    contentStyle={{ background: '#1a1d29', border: '1px solid #2a2d3a', borderRadius: '8px' }}
                    labelStyle={{ color: '#e5e7eb' }}
                    formatter={(v) => [formatCompact(v)]}
                  />
                  <Bar dataKey="buy" name="Buy" fill="#22c55e" radius={[2, 2, 0, 0]} />
                  <Bar dataKey="sell" name="Sell" fill="#ef4444" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Politicians & Committees */}
          {Object.keys(politicians).length > 0 && (
            <div className="bg-dark-card border border-dark-border rounded-lg p-4">
              <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-3">
                Politicians / Committees Involved
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                {Object.values(politicians).map(p => (
                  <div key={p.name} className="flex items-center justify-between bg-dark-bg rounded p-2">
                    <div>
                      <span className="text-sm text-gray-200">{p.name}</span>
                      {p.party && (
                        <span className={`ml-2 text-xs font-bold ${p.party === 'D' ? 'text-blue-400' : 'text-red-400'}`}>
                          ({p.party})
                        </span>
                      )}
                    </div>
                    <div className="text-right">
                      {p.committee && <span className="text-xs text-gray-500">{p.committee}</span>}
                      <span className="text-xs text-gray-400 ml-2">{p.count} trades</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* All Transactions */}
          <div className="bg-dark-card border border-dark-border rounded-lg p-4">
            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-3">
              All Transactions ({data.transactions.length})
            </h3>
            <TransactionTable transactions={data.transactions} />
          </div>
        </>
      )}

      {!data && !loading && !error && (
        <div className="text-center py-20 text-gray-500">
          <p className="text-lg">Search for a ticker to see detailed activity</p>
          <p className="text-sm mt-2">Try NVDA, AAPL, JPM, or LMT</p>
        </div>
      )}
    </div>
  )
}
