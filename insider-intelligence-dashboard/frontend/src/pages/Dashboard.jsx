import { useState, useEffect, useCallback } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { getTransactions, getSectorSummary, getAlerts } from '../api'
import AlertCard from '../components/AlertCard'
import SectorHeatmap from '../components/SectorHeatmap'
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

export default function Dashboard() {
  const [alerts, setAlerts] = useState([])
  const [sectors, setSectors] = useState([])
  const [recentTx, setRecentTx] = useState([])
  const [topTickers, setTopTickers] = useState([])
  const [loading, setLoading] = useState(true)
  const [lastUpdated, setLastUpdated] = useState(null)

  const fetchData = useCallback(async () => {
    try {
      const [alertData, sectorData, txData] = await Promise.all([
        getAlerts({ limit: 8 }),
        getSectorSummary(),
        getTransactions({ limit: 15, transaction_type: '' }),
      ])
      setAlerts(alertData)
      setSectors(sectorData)
      setRecentTx(txData.transactions)

      // Compute top tickers from transactions
      const tickerMap = {}
      txData.transactions.forEach(tx => {
        if (tx.ticker && tx.transaction_type === 'buy') {
          if (!tickerMap[tx.ticker]) tickerMap[tx.ticker] = { ticker: tx.ticker, value: 0 }
          tickerMap[tx.ticker].value += tx.total_value || 0
        }
      })
      const sorted = Object.values(tickerMap).sort((a, b) => b.value - a.value).slice(0, 10)
      setTopTickers(sorted)
      setLastUpdated(new Date())
    } catch (err) {
      console.error('Dashboard fetch error:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 10 * 60 * 1000) // 10 min auto-refresh
    return () => clearInterval(interval)
  }, [fetchData])

  // Compute quick stats
  const totalTx = recentTx.length
  const totalBuy = recentTx.filter(t => t.transaction_type === 'buy').reduce((s, t) => s + (t.total_value || 0), 0)
  const totalSell = recentTx.filter(t => t.transaction_type === 'sell').reduce((s, t) => s + (t.total_value || 0), 0)
  const highAlerts = alerts.filter(a => a.severity === 'HIGH').length

  if (loading) {
    return (
      <div className="space-y-6">
        <h2 className="text-xl font-bold">Dashboard</h2>
        <LoadingSkeleton type="cards" />
        <LoadingSkeleton type="chart" />
        <LoadingSkeleton rows={8} />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold">Dashboard</h2>
        {lastUpdated && (
          <span className="text-xs text-gray-500">
            Updated {lastUpdated.toLocaleTimeString()}
          </span>
        )}
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          label="Transactions"
          value={totalTx}
          subtext="recent"
          tooltip="Total transactions in current view"
        />
        <MetricCard
          label="Buy Volume"
          value={formatCompact(totalBuy)}
          color="text-green-400"
          tooltip="Total buy value of recent transactions"
        />
        <MetricCard
          label="Sell Volume"
          value={formatCompact(totalSell)}
          color="text-red-400"
          tooltip="Total sell value of recent transactions"
        />
        <MetricCard
          label="Active Alerts"
          value={highAlerts}
          color={highAlerts > 0 ? 'text-red-400' : 'text-gray-400'}
          subtext="high severity"
          tooltip="Number of HIGH severity alerts currently active"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Active Alerts */}
        <div className="bg-dark-card border border-dark-border rounded-lg p-4">
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-3">Active Alerts</h3>
          <div className="space-y-2 max-h-[300px] overflow-y-auto">
            {alerts.length > 0 ? (
              alerts.slice(0, 6).map(alert => (
                <AlertCard key={alert.id} alert={alert} compact />
              ))
            ) : (
              <p className="text-gray-500 text-sm">No active alerts</p>
            )}
          </div>
        </div>

        {/* Sector Heatmap */}
        <div className="bg-dark-card border border-dark-border rounded-lg p-4">
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-3">
            Sector Buy/Sell Pressure
          </h3>
          <SectorHeatmap sectors={sectors} />
        </div>
      </div>

      {/* Top Tickers Chart */}
      {topTickers.length > 0 && (
        <div className="bg-dark-card border border-dark-border rounded-lg p-4">
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-3">
            Top Bought Tickers
          </h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={topTickers}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2d3a" />
              <XAxis dataKey="ticker" stroke="#6b7280" fontSize={12} />
              <YAxis stroke="#6b7280" fontSize={11} tickFormatter={v => formatCompact(v)} />
              <Tooltip
                contentStyle={{ background: '#1a1d29', border: '1px solid #2a2d3a', borderRadius: '8px' }}
                labelStyle={{ color: '#e5e7eb' }}
                formatter={(v) => [formatCompact(v), 'Buy Value']}
              />
              <Bar dataKey="value" fill="#22c55e" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Recent Transactions */}
      <div className="bg-dark-card border border-dark-border rounded-lg p-4">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-3">
          Recent Transactions
        </h3>
        <TransactionTable transactions={recentTx} compact />
      </div>
    </div>
  )
}
