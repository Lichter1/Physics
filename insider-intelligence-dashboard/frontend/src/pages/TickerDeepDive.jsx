import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  BarChart, Bar, LineChart, Line, Area, AreaChart, XAxis, YAxis, CartesianGrid,
  Tooltip as RTooltip, ResponsiveContainer, ReferenceDot, ComposedChart, Scatter
} from 'recharts'
import { getTickerActivity, getTickerPrices } from '../api'
import TransactionTable from '../components/TransactionTable'
import MetricCard from '../components/MetricCard'
import LoadingSkeleton from '../components/LoadingSkeleton'

const formatCompact = (v) => {
  if (!v) return '$0'
  if (v >= 1e9) return '$' + (v / 1e9).toFixed(1) + 'B'
  if (v >= 1e6) return '$' + (v / 1e6).toFixed(1) + 'M'
  if (v >= 1e3) return '$' + (v / 1e3).toFixed(0) + 'K'
  return '$' + v.toFixed(0)
}

function SignalScoreRing({ score }) {
  const circumference = 2 * Math.PI * 36
  const offset = circumference - (score / 100) * circumference
  const color = score >= 70 ? '#22c55e' : score >= 50 ? '#84cc16' : score >= 30 ? '#eab308' : '#6b7280'
  return (
    <div className="relative w-24 h-24">
      <svg className="w-24 h-24 -rotate-90" viewBox="0 0 80 80">
        <circle cx="40" cy="40" r="36" fill="none" stroke="#1f2937" strokeWidth="5" />
        <circle cx="40" cy="40" r="36" fill="none" stroke={color} strokeWidth="5"
          strokeDasharray={circumference} strokeDashoffset={offset} strokeLinecap="round" />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-2xl font-bold" style={{ color }}>{Math.round(score)}</span>
        <span className="text-[9px] text-gray-500 uppercase">Signal</span>
      </div>
    </div>
  )
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
      <span className="text-gray-500 w-20 shrink-0">{label}</span>
      <div className="flex-1 bg-gray-800 rounded-full h-1.5">
        <div className={`h-1.5 rounded-full ${getColor(score)}`} style={{ width: `${Math.min(score, 100)}%` }} />
      </div>
      <span className="text-gray-400 w-8 text-right">{Math.round(score)}</span>
    </div>
  )
}

export default function TickerDeepDive() {
  const { ticker: paramTicker } = useParams()
  const navigate = useNavigate()
  const [searchInput, setSearchInput] = useState(paramTicker || '')
  const [ticker, setTicker] = useState(paramTicker || '')
  const [data, setData] = useState(null)
  const [prices, setPrices] = useState([])
  const [pricePeriod, setPricePeriod] = useState('6m')
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
    Promise.all([
      getTickerActivity(ticker),
      getTickerPrices(ticker, pricePeriod).catch(() => []),
    ])
      .then(([activityData, priceData]) => {
        setData(activityData)
        setPrices(priceData)
      })
      .catch(err => {
        console.error(err)
        setError('Failed to load ticker data')
        setData(null)
      })
      .finally(() => setLoading(false))
  }, [ticker, pricePeriod])

  const handleSearch = (e) => {
    e.preventDefault()
    if (searchInput.trim()) {
      navigate(`/ticker/${searchInput.trim().toUpperCase()}`)
    }
  }

  // Build timeline data from transactions
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

  // Build price chart data with insider markers
  const priceChartData = prices.map(p => {
    const txDay = timelineData.find(t => t.date === p.date)
    return {
      ...p,
      insiderBuy: txDay?.buy || null,
      insiderSell: txDay?.sell || null,
    }
  })

  // Politicians/committees
  const politicians = data ? data.transactions
    .filter(tx => tx.filer_type === 'politician')
    .reduce((acc, tx) => {
      if (!acc[tx.filer_name]) acc[tx.filer_name] = { name: tx.filer_name, party: tx.party, committee: tx.committee, count: 0 }
      acc[tx.filer_name].count++
      return acc
    }, {}) : {}

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold">Ticker Deep Dive</h2>

      <form onSubmit={handleSearch} className="flex gap-2">
        <input
          type="text" value={searchInput}
          onChange={e => setSearchInput(e.target.value.toUpperCase())}
          placeholder="Enter ticker symbol (e.g., NVDA)"
          className="flex-1 max-w-xs bg-dark-card border border-dark-border rounded-lg px-4 py-2 text-white placeholder-gray-600 focus:border-blue-500 focus:outline-none"
        />
        <button type="submit" className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm transition-colors">
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
          {/* Header + Signal Score */}
          <div className="flex items-center gap-6">
            {data.signal_score != null && <SignalScoreRing score={data.signal_score} />}
            <div>
              <div className="flex items-center gap-3">
                <span className="text-3xl font-bold font-mono text-white">{data.ticker}</span>
                {data.company_name && <span className="text-gray-400">{data.company_name}</span>}
                {data.sector && (
                  <span className="text-xs bg-dark-hover text-gray-400 px-2 py-1 rounded">{data.sector}</span>
                )}
              </div>
              {data.price_data?.current && (
                <div className="flex items-center gap-4 mt-2 text-sm">
                  <span className="text-gray-200 font-mono">${data.price_data.current.toFixed(2)}</span>
                  {data.price_data.low_52w && (
                    <span className="text-gray-500">
                      52w: ${data.price_data.low_52w.toFixed(0)} - ${data.price_data.high_52w.toFixed(0)}
                    </span>
                  )}
                  {data.price_data.pct_from_low != null && (
                    <span className="text-xs text-gray-500">
                      ({data.price_data.pct_from_low.toFixed(0)}% from 52w low)
                    </span>
                  )}
                </div>
              )}
              {data.forward_returns && (
                <div className="flex gap-3 mt-2">
                  {Object.entries(data.forward_returns).map(([period, ret]) => (
                    ret != null && (
                      <span key={period} className={`text-xs font-mono ${ret >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {period}: {ret >= 0 ? '+' : ''}{ret}%
                      </span>
                    )
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Signal Components */}
          {data.signal_components && (
            <div className="bg-dark-card border border-dark-border rounded-lg p-4">
              <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-3">
                Signal Score Breakdown
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-x-6 gap-y-2">
                <ScoreBar score={data.signal_components.cluster} label="Cluster" />
                <ScoreBar score={data.signal_components.conviction} label="Conviction" />
                <ScoreBar score={data.signal_components.seniority} label="Seniority" />
                <ScoreBar score={data.signal_components.political_convergence} label="Political" />
                <ScoreBar score={data.signal_components.timeliness} label="Timeliness" />
                <ScoreBar score={data.signal_components.price_context} label="Price Ctx" />
                <ScoreBar score={data.signal_components.filer_accuracy} label="Filer Acc." />
              </div>
            </div>
          )}

          {/* Classic Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            <MetricCard label="Cluster Score" value={`${data.cluster_score}/10`}
              color={data.cluster_score >= 5 ? 'text-green-400' : 'text-gray-300'}
              tooltip="Unique insiders buying in last 30 days" />
            <MetricCard label="Conviction" value={data.conviction_score.toFixed(1)}
              color={data.conviction_score >= 5 ? 'text-green-400' : 'text-gray-300'}
              tooltip="Weighted by filer seniority + trade size" />
            <MetricCard label="Committee" value={data.committee_exposure}
              color={data.committee_exposure > 0 ? 'text-yellow-400' : 'text-gray-300'}
              tooltip="Politicians with committee oversight buying" />
            <MetricCard label="Avg Delay" value={`${data.delay_score}d`}
              color={data.delay_score > 30 ? 'text-red-400' : 'text-gray-300'}
              tooltip="Average days between trade and filing" />
            <MetricCard label="Buy/Sell" value={data.buy_sell_ratio.toFixed(1)}
              color={data.buy_sell_ratio > 1 ? 'text-green-400' : 'text-red-400'}
              tooltip="Ratio of buy value to sell value" />
          </div>

          {/* Price Chart with Insider Markers */}
          {priceChartData.length > 0 && (
            <div className="bg-dark-card border border-dark-border rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide">
                  Price + Insider Activity
                </h3>
                <div className="flex gap-1">
                  {['1m', '3m', '6m', '1y', '2y'].map(p => (
                    <button key={p} onClick={() => setPricePeriod(p)}
                      className={`px-2 py-0.5 text-xs rounded ${pricePeriod === p ? 'bg-blue-600 text-white' : 'text-gray-500 hover:text-gray-300'}`}>
                      {p}
                    </button>
                  ))}
                </div>
              </div>
              <ResponsiveContainer width="100%" height={300}>
                <ComposedChart data={priceChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2a2d3a" />
                  <XAxis dataKey="date" stroke="#6b7280" fontSize={10}
                    tickFormatter={d => d.slice(5)} interval="preserveStartEnd" />
                  <YAxis yAxisId="price" stroke="#6b7280" fontSize={11}
                    domain={['auto', 'auto']} tickFormatter={v => `$${v}`} />
                  <YAxis yAxisId="volume" orientation="right" stroke="#6b7280" fontSize={10} hide />
                  <RTooltip
                    contentStyle={{ background: '#1a1d29', border: '1px solid #2a2d3a', borderRadius: '8px' }}
                    labelStyle={{ color: '#e5e7eb' }}
                    formatter={(v, name) => {
                      if (name === 'close') return [`$${v.toFixed(2)}`, 'Price']
                      if (name === 'insiderBuy') return [formatCompact(v), 'Insider Buy']
                      if (name === 'insiderSell') return [formatCompact(v), 'Insider Sell']
                      return [v, name]
                    }}
                  />
                  <Area yAxisId="price" type="monotone" dataKey="close" stroke="#3b82f6"
                    fill="#3b82f6" fillOpacity={0.1} strokeWidth={1.5} dot={false} />
                  <Bar yAxisId="volume" dataKey="insiderBuy" fill="#22c55e" opacity={0.8}
                    radius={[2, 2, 0, 0]} barSize={4} />
                  <Bar yAxisId="volume" dataKey="insiderSell" fill="#ef4444" opacity={0.8}
                    radius={[2, 2, 0, 0]} barSize={4} />
                </ComposedChart>
              </ResponsiveContainer>
              <div className="flex gap-4 mt-2 text-xs text-gray-500 justify-center">
                <span><span className="inline-block w-3 h-0.5 bg-blue-500 mr-1"></span>Price</span>
                <span><span className="inline-block w-2 h-2 bg-green-500 rounded-sm mr-1"></span>Insider Buy</span>
                <span><span className="inline-block w-2 h-2 bg-red-500 rounded-sm mr-1"></span>Insider Sell</span>
              </div>
            </div>
          )}

          {/* Buy/Sell Activity Timeline */}
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
