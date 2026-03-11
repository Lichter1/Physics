import { useState, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
  LineChart, Line
} from 'recharts'
import { getSectorSummary, getWeeklyTrends } from '../api'
import LoadingSkeleton from '../components/LoadingSkeleton'

const formatCompact = (v) => {
  if (!v) return '$0'
  if (v >= 1e9) return '$' + (v / 1e9).toFixed(1) + 'B'
  if (v >= 1e6) return '$' + (v / 1e6).toFixed(1) + 'M'
  if (v >= 1e3) return '$' + (v / 1e3).toFixed(0) + 'K'
  return '$' + v.toFixed(0)
}

const SECTOR_COLORS = {
  'Defense & Aerospace': '#ef4444',
  'AI & Semiconductors': '#8b5cf6',
  'Energy': '#f97316',
  'Healthcare & Pharma': '#06b6d4',
  'Financials & Banking': '#22c55e',
  'Technology': '#3b82f6',
  'Real Estate': '#eab308',
  'Consumer Discretionary': '#ec4899',
  'Industrials': '#6b7280',
}

export default function SectorAnalysis() {
  const [sectors, setSectors] = useState([])
  const [trends, setTrends] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function fetchData() {
      try {
        const [sectorData, trendData] = await Promise.all([
          getSectorSummary(),
          getWeeklyTrends(),
        ])
        setSectors(sectorData)
        setTrends(trendData)
      } catch (err) {
        console.error('Sector analysis fetch error:', err)
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [])

  if (loading) {
    return (
      <div className="space-y-6">
        <h2 className="text-xl font-bold">Sector Analysis</h2>
        <LoadingSkeleton type="chart" />
        <LoadingSkeleton type="chart" />
      </div>
    )
  }

  // Prepare bar chart data
  const barData = sectors.map(s => ({
    sector: s.sector.replace(' & ', '\n& '),
    sectorFull: s.sector,
    buy: s.total_buy_value,
    sell: s.total_sell_value,
  }))

  // Prepare trend line data - group by week
  const weekMap = {}
  trends.forEach(t => {
    if (!weekMap[t.week]) weekMap[t.week] = { week: t.week }
    weekMap[t.week][t.sector] = t.buy_value
  })
  const trendLineData = Object.values(weekMap)

  // Get unique sectors from trends
  const trendSectors = [...new Set(trends.map(t => t.sector))]

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold">Sector Analysis</h2>

      {/* Buy vs Sell Volume */}
      <div className="bg-dark-card border border-dark-border rounded-lg p-4">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-4">
          Buy vs Sell Volume by Sector (Last 30 Days)
        </h3>
        <ResponsiveContainer width="100%" height={350}>
          <BarChart data={barData} layout="vertical" margin={{ left: 120 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2d3a" />
            <XAxis type="number" stroke="#6b7280" fontSize={11} tickFormatter={formatCompact} />
            <YAxis type="category" dataKey="sectorFull" stroke="#6b7280" fontSize={11} width={110} />
            <Tooltip
              contentStyle={{ background: '#1a1d29', border: '1px solid #2a2d3a', borderRadius: '8px' }}
              labelStyle={{ color: '#e5e7eb' }}
              formatter={(v) => [formatCompact(v)]}
            />
            <Legend />
            <Bar dataKey="buy" name="Buy Volume" fill="#22c55e" radius={[0, 4, 4, 0]} />
            <Bar dataKey="sell" name="Sell Volume" fill="#ef4444" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Activity Trend */}
      {trendLineData.length > 0 && (
        <div className="bg-dark-card border border-dark-border rounded-lg p-4">
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-4">
            Sector Buy Activity Trend (4-Week Rolling)
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={trendLineData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2d3a" />
              <XAxis dataKey="week" stroke="#6b7280" fontSize={12} />
              <YAxis stroke="#6b7280" fontSize={11} tickFormatter={formatCompact} />
              <Tooltip
                contentStyle={{ background: '#1a1d29', border: '1px solid #2a2d3a', borderRadius: '8px' }}
                labelStyle={{ color: '#e5e7eb' }}
                formatter={(v) => [formatCompact(v)]}
              />
              <Legend />
              {trendSectors.map(sector => (
                <Line
                  key={sector}
                  type="monotone"
                  dataKey={sector}
                  stroke={SECTOR_COLORS[sector] || '#6b7280'}
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  name={sector}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Sector Details Table */}
      <div className="bg-dark-card border border-dark-border rounded-lg p-4">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-3">
          Sector Details
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-dark-border text-gray-500 text-xs uppercase">
                <th className="text-left py-2 px-2">Sector</th>
                <th className="text-right py-2 px-2">Buy Volume</th>
                <th className="text-right py-2 px-2">Sell Volume</th>
                <th className="text-right py-2 px-2">B/S Ratio</th>
                <th className="text-right py-2 px-2">Filers</th>
                <th className="text-right py-2 px-2">Transactions</th>
              </tr>
            </thead>
            <tbody>
              {sectors.map(s => (
                <tr key={s.sector} className="border-b border-dark-border/50 hover:bg-dark-hover transition-colors">
                  <td className="py-2 px-2 text-gray-200">{s.sector}</td>
                  <td className="py-2 px-2 text-right text-green-400 font-mono text-xs">
                    {formatCompact(s.total_buy_value)}
                  </td>
                  <td className="py-2 px-2 text-right text-red-400 font-mono text-xs">
                    {formatCompact(s.total_sell_value)}
                  </td>
                  <td className="py-2 px-2 text-right font-mono text-xs">
                    <span className={s.buy_sell_ratio > 1 ? 'text-green-400' : 'text-red-400'}>
                      {s.buy_sell_ratio.toFixed(2)}
                    </span>
                  </td>
                  <td className="py-2 px-2 text-right text-gray-400">{s.unique_filers}</td>
                  <td className="py-2 px-2 text-right text-gray-400">{s.transaction_count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
