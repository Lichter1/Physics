import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { getFilerLeaderboard, getTransactions } from '../api'
import LoadingSkeleton from '../components/LoadingSkeleton'
import TransactionTable from '../components/TransactionTable'

const formatCompact = (v) => {
  if (!v) return '$0'
  if (v >= 1e9) return '$' + (v / 1e9).toFixed(1) + 'B'
  if (v >= 1e6) return '$' + (v / 1e6).toFixed(1) + 'M'
  if (v >= 1e3) return '$' + (v / 1e3).toFixed(0) + 'K'
  return '$' + v.toFixed(0)
}

export default function FilerProfiles() {
  const [filers, setFilers] = useState([])
  const [loading, setLoading] = useState(true)
  const [selectedFiler, setSelectedFiler] = useState(null)
  const [filerTx, setFilerTx] = useState([])
  const [loadingTx, setLoadingTx] = useState(false)
  const [filterType, setFilterType] = useState('')

  useEffect(() => {
    setLoading(true)
    getFilerLeaderboard({ filer_type: filterType || undefined, limit: 30 })
      .then(setFilers)
      .catch(err => console.error(err))
      .finally(() => setLoading(false))
  }, [filterType])

  const handleSelectFiler = async (filerName) => {
    if (selectedFiler === filerName) {
      setSelectedFiler(null)
      setFilerTx([])
      return
    }
    setSelectedFiler(filerName)
    setLoadingTx(true)
    try {
      // We don't have a filer-specific endpoint, so we'll use transactions with name matching
      // This is a simplification - in production you'd have a proper endpoint
      const data = await getTransactions({ limit: 100 })
      setFilerTx(data.transactions.filter(tx => tx.filer_name === filerName))
    } catch (err) {
      console.error(err)
    } finally {
      setLoadingTx(false)
    }
  }

  if (loading) {
    return (
      <div className="space-y-6">
        <h2 className="text-xl font-bold">Filer Profiles</h2>
        <LoadingSkeleton rows={15} />
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold">Filer Profiles</h2>

      {/* Filter */}
      <div className="flex gap-2">
        {['', 'politician', 'ceo', 'director'].map(type => (
          <button
            key={type}
            onClick={() => setFilterType(type)}
            className={`px-3 py-1.5 text-xs rounded transition-colors ${
              filterType === type
                ? 'bg-blue-600 text-white'
                : 'bg-dark-card border border-dark-border text-gray-400 hover:text-gray-200'
            }`}
          >
            {type || 'All'}
          </button>
        ))}
      </div>

      {/* Leaderboard */}
      <div className="bg-dark-card border border-dark-border rounded-lg overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-dark-border text-gray-500 text-xs uppercase">
              <th className="text-left py-3 px-4">#</th>
              <th className="text-left py-3 px-4">Filer</th>
              <th className="text-left py-3 px-4">Type</th>
              <th className="text-left py-3 px-4">Party</th>
              <th className="text-right py-3 px-4">Transactions</th>
              <th className="text-right py-3 px-4">Buy Volume</th>
              <th className="text-right py-3 px-4">Sell Volume</th>
              <th className="text-right py-3 px-4">Tickers</th>
              <th className="text-left py-3 px-4">Top Sector</th>
            </tr>
          </thead>
          <tbody>
            {filers.map((filer, idx) => {
              const isSelected = selectedFiler === filer.filer_name
              return (
                <>
                  <tr
                    key={filer.filer_name}
                    onClick={() => handleSelectFiler(filer.filer_name)}
                    className={`border-b border-dark-border/50 cursor-pointer transition-colors ${
                      isSelected ? 'bg-blue-900/20' : 'hover:bg-dark-hover'
                    }`}
                  >
                    <td className="py-2.5 px-4 text-gray-500 text-xs">{idx + 1}</td>
                    <td className="py-2.5 px-4 text-gray-200 font-medium">{filer.filer_name}</td>
                    <td className="py-2.5 px-4">
                      <span className="text-xs px-1.5 py-0.5 rounded bg-dark-bg text-gray-400">
                        {filer.filer_type}
                      </span>
                    </td>
                    <td className="py-2.5 px-4">
                      {filer.party && (
                        <span className={`text-xs font-bold ${filer.party === 'D' ? 'text-blue-400' : 'text-red-400'}`}>
                          {filer.party}
                        </span>
                      )}
                    </td>
                    <td className="py-2.5 px-4 text-right text-gray-300">{filer.total_transactions}</td>
                    <td className="py-2.5 px-4 text-right text-green-400 font-mono text-xs">
                      {formatCompact(filer.total_buy_value)}
                    </td>
                    <td className="py-2.5 px-4 text-right text-red-400 font-mono text-xs">
                      {formatCompact(filer.total_sell_value)}
                    </td>
                    <td className="py-2.5 px-4 text-right text-gray-400">{filer.unique_tickers}</td>
                    <td className="py-2.5 px-4 text-gray-400 text-xs truncate max-w-[150px]">
                      {filer.favorite_sector || '-'}
                    </td>
                  </tr>
                  {isSelected && (
                    <tr key={`${filer.filer_name}-detail`}>
                      <td colSpan={9} className="bg-dark-bg p-4">
                        <h4 className="text-sm font-semibold text-gray-400 mb-2">
                          Transaction History - {filer.filer_name}
                        </h4>
                        {loadingTx ? (
                          <LoadingSkeleton rows={5} />
                        ) : filerTx.length > 0 ? (
                          <TransactionTable transactions={filerTx} compact />
                        ) : (
                          <p className="text-gray-500 text-sm">No recent transactions in current view</p>
                        )}
                      </td>
                    </tr>
                  )}
                </>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
