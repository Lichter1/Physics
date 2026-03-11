import { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { getTickerActivity, getTickerPrices } from '../api'
import MetricCard from '../components/MetricCard'
import LoadingSkeleton from '../components/LoadingSkeleton'

const STORAGE_KEY = 'insider_intel_watchlist'

const formatCompact = (v) => {
  if (!v) return '$0'
  if (v >= 1e9) return '$' + (v / 1e9).toFixed(1) + 'B'
  if (v >= 1e6) return '$' + (v / 1e6).toFixed(1) + 'M'
  if (v >= 1e3) return '$' + (v / 1e3).toFixed(0) + 'K'
  return '$' + v.toFixed(0)
}

function loadWatchlist() {
  try {
    const data = localStorage.getItem(STORAGE_KEY)
    return data ? JSON.parse(data) : []
  } catch {
    return []
  }
}

function saveWatchlist(list) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(list))
}

export default function Watchlist() {
  const navigate = useNavigate()
  const [watchlist, setWatchlist] = useState(loadWatchlist)
  const [tickerData, setTickerData] = useState({})
  const [loading, setLoading] = useState(false)
  const [addInput, setAddInput] = useState('')
  const [addingDate, setAddingDate] = useState(new Date().toISOString().slice(0, 10))
  const [addingPrice, setAddingPrice] = useState('')

  const fetchAllData = useCallback(async () => {
    if (watchlist.length === 0) return
    setLoading(true)
    const data = {}
    for (const item of watchlist) {
      try {
        const activity = await getTickerActivity(item.ticker)
        data[item.ticker] = activity
      } catch (err) {
        console.error(`Failed to fetch ${item.ticker}:`, err)
      }
    }
    setTickerData(data)
    setLoading(false)
  }, [watchlist])

  useEffect(() => {
    fetchAllData()
  }, [fetchAllData])

  const addToWatchlist = (e) => {
    e.preventDefault()
    const ticker = addInput.trim().toUpperCase()
    if (!ticker || watchlist.find(w => w.ticker === ticker)) return

    const entry = {
      ticker,
      added_at: new Date().toISOString(),
      entry_date: addingDate,
      entry_price: addingPrice ? parseFloat(addingPrice) : null,
    }
    const updated = [...watchlist, entry]
    setWatchlist(updated)
    saveWatchlist(updated)
    setAddInput('')
    setAddingPrice('')
  }

  const removeFromWatchlist = (ticker) => {
    const updated = watchlist.filter(w => w.ticker !== ticker)
    setWatchlist(updated)
    saveWatchlist(updated)
  }

  // Compute portfolio stats
  const totalPnl = watchlist.reduce((sum, item) => {
    const data = tickerData[item.ticker]
    if (!data?.price_data?.current || !item.entry_price) return sum
    const pct = ((data.price_data.current - item.entry_price) / item.entry_price) * 100
    return sum + pct
  }, 0)
  const avgPnl = watchlist.filter(w => w.entry_price && tickerData[w.ticker]?.price_data?.current).length > 0
    ? totalPnl / watchlist.filter(w => w.entry_price && tickerData[w.ticker]?.price_data?.current).length
    : 0
  const avgSignal = watchlist.length > 0
    ? watchlist.reduce((s, w) => s + (tickerData[w.ticker]?.signal_score || 0), 0) / watchlist.length
    : 0

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold">Watchlist & Portfolio Simulator</h2>
      <p className="text-xs text-gray-500 -mt-4">
        Track signals and simulate P&L. Data stored locally in your browser.
      </p>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard label="Watching" value={watchlist.length} tooltip="Tickers in watchlist" />
        <MetricCard
          label="Avg Signal"
          value={Math.round(avgSignal)}
          color={avgSignal >= 50 ? 'text-green-400' : 'text-gray-300'}
          tooltip="Average signal score across watchlist"
        />
        <MetricCard
          label="Avg P&L"
          value={`${avgPnl >= 0 ? '+' : ''}${avgPnl.toFixed(1)}%`}
          color={avgPnl >= 0 ? 'text-green-400' : 'text-red-400'}
          tooltip="Average simulated return since entry"
        />
        <MetricCard
          label="Positions"
          value={watchlist.filter(w => w.entry_price).length}
          subtext="with entry price"
          tooltip="Watchlist items with an entry price set"
        />
      </div>

      {/* Add Ticker */}
      <form onSubmit={addToWatchlist} className="bg-dark-card border border-dark-border rounded-lg p-4">
        <div className="flex gap-3 items-end flex-wrap">
          <div>
            <label className="text-xs text-gray-500 block mb-1">Ticker</label>
            <input
              type="text"
              value={addInput}
              onChange={e => setAddInput(e.target.value.toUpperCase())}
              placeholder="NVDA"
              className="bg-dark-bg border border-dark-border rounded px-3 py-1.5 text-sm text-white placeholder-gray-600 w-28"
            />
          </div>
          <div>
            <label className="text-xs text-gray-500 block mb-1">Entry Date</label>
            <input
              type="date"
              value={addingDate}
              onChange={e => setAddingDate(e.target.value)}
              className="bg-dark-bg border border-dark-border rounded px-3 py-1.5 text-sm text-gray-300"
            />
          </div>
          <div>
            <label className="text-xs text-gray-500 block mb-1">Entry Price (optional)</label>
            <input
              type="number"
              step="0.01"
              value={addingPrice}
              onChange={e => setAddingPrice(e.target.value)}
              placeholder="150.00"
              className="bg-dark-bg border border-dark-border rounded px-3 py-1.5 text-sm text-gray-300 placeholder-gray-600 w-32"
            />
          </div>
          <button
            type="submit"
            className="px-4 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded transition-colors"
          >
            Add to Watchlist
          </button>
        </div>
      </form>

      {/* Watchlist Table */}
      {watchlist.length === 0 ? (
        <div className="text-center py-16 text-gray-500">
          <p className="text-lg">Watchlist is empty</p>
          <p className="text-sm mt-1">
            Add tickers above, or click the bookmark icon on Trade Ideas to track signals
          </p>
        </div>
      ) : (
        <div className="bg-dark-card border border-dark-border rounded-lg overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-dark-border text-gray-500 text-xs uppercase">
                <th className="text-left py-3 px-4">Ticker</th>
                <th className="text-right py-3 px-4">Signal</th>
                <th className="text-right py-3 px-4">Entry Price</th>
                <th className="text-right py-3 px-4">Current</th>
                <th className="text-right py-3 px-4">P&L</th>
                <th className="text-right py-3 px-4">Buy/Sell</th>
                <th className="text-left py-3 px-4">Entry Date</th>
                <th className="text-center py-3 px-4">Actions</th>
              </tr>
            </thead>
            <tbody>
              {watchlist.map(item => {
                const data = tickerData[item.ticker]
                const current = data?.price_data?.current
                const pnl = item.entry_price && current
                  ? ((current - item.entry_price) / item.entry_price * 100).toFixed(1)
                  : null

                return (
                  <tr key={item.ticker} className="border-b border-dark-border/50 hover:bg-dark-hover">
                    <td className="py-2.5 px-4">
                      <button
                        onClick={() => navigate(`/ticker/${item.ticker}`)}
                        className="text-blue-400 hover:text-blue-300 font-mono font-bold"
                      >
                        {item.ticker}
                      </button>
                      {data?.company_name && (
                        <span className="text-xs text-gray-500 ml-2">{data.company_name}</span>
                      )}
                    </td>
                    <td className="py-2.5 px-4 text-right">
                      {data?.signal_score != null ? (
                        <span className={`font-mono font-bold ${
                          data.signal_score >= 60 ? 'text-green-400' :
                          data.signal_score >= 40 ? 'text-yellow-400' : 'text-gray-400'
                        }`}>
                          {Math.round(data.signal_score)}
                        </span>
                      ) : (
                        <span className="text-gray-600">-</span>
                      )}
                    </td>
                    <td className="py-2.5 px-4 text-right text-gray-300 font-mono text-xs">
                      {item.entry_price ? `$${item.entry_price.toFixed(2)}` : '-'}
                    </td>
                    <td className="py-2.5 px-4 text-right text-gray-300 font-mono text-xs">
                      {current ? `$${current.toFixed(2)}` : '-'}
                    </td>
                    <td className="py-2.5 px-4 text-right font-mono text-xs">
                      {pnl !== null ? (
                        <span className={parseFloat(pnl) >= 0 ? 'text-green-400' : 'text-red-400'}>
                          {parseFloat(pnl) >= 0 ? '+' : ''}{pnl}%
                        </span>
                      ) : (
                        <span className="text-gray-600">-</span>
                      )}
                    </td>
                    <td className="py-2.5 px-4 text-right font-mono text-xs">
                      {data?.buy_sell_ratio != null ? (
                        <span className={data.buy_sell_ratio > 1 ? 'text-green-400' : 'text-red-400'}>
                          {data.buy_sell_ratio.toFixed(1)}x
                        </span>
                      ) : '-'}
                    </td>
                    <td className="py-2.5 px-4 text-gray-400 text-xs">{item.entry_date}</td>
                    <td className="py-2.5 px-4 text-center">
                      <button
                        onClick={() => removeFromWatchlist(item.ticker)}
                        className="text-gray-600 hover:text-red-400 text-xs transition-colors"
                      >
                        Remove
                      </button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      {loading && <LoadingSkeleton rows={3} />}
    </div>
  )
}
