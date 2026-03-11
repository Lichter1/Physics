import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { getFilerLeaderboard, getTransactions, getFilerTrackRecord } from '../api'
import LoadingSkeleton from '../components/LoadingSkeleton'
import TransactionTable from '../components/TransactionTable'

const formatCompact = (v) => {
  if (!v) return '$0'
  if (v >= 1e9) return '$' + (v / 1e9).toFixed(1) + 'B'
  if (v >= 1e6) return '$' + (v / 1e6).toFixed(1) + 'M'
  if (v >= 1e3) return '$' + (v / 1e3).toFixed(0) + 'K'
  return '$' + v.toFixed(0)
}

function AccuracyBadge({ pct }) {
  if (pct == null) return <span className="text-gray-600 text-xs">-</span>
  const color = pct >= 65 ? 'text-green-400 bg-green-400/10' :
                pct >= 50 ? 'text-yellow-400 bg-yellow-400/10' :
                'text-red-400 bg-red-400/10'
  return (
    <span className={`text-xs font-mono font-bold px-1.5 py-0.5 rounded ${color}`}>
      {pct.toFixed(0)}%
    </span>
  )
}

export default function FilerProfiles() {
  const [filers, setFilers] = useState([])
  const [loading, setLoading] = useState(true)
  const [selectedFiler, setSelectedFiler] = useState(null)
  const [filerTx, setFilerTx] = useState([])
  const [filerRecord, setFilerRecord] = useState(null)
  const [loadingDetail, setLoadingDetail] = useState(false)
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
      setFilerRecord(null)
      return
    }
    setSelectedFiler(filerName)
    setLoadingDetail(true)
    try {
      const [txData, record] = await Promise.all([
        getTransactions({ limit: 100 }),
        getFilerTrackRecord(filerName).catch(() => null),
      ])
      setFilerTx(txData.transactions.filter(tx => tx.filer_name === filerName))
      setFilerRecord(record)
    } catch (err) {
      console.error(err)
    } finally {
      setLoadingDetail(false)
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
      <p className="text-xs text-gray-500 -mt-2">
        Accuracy = % of buys that preceded price gains within 60 days
      </p>

      {/* Filter */}
      <div className="flex gap-2 flex-wrap">
        {['', 'politician', 'ceo', 'director', 'officer', '10% owner'].map(type => (
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
              <th className="text-right py-3 px-4">Trades</th>
              <th className="text-right py-3 px-4">Buy Vol</th>
              <th className="text-right py-3 px-4">Sell Vol</th>
              <th className="text-right py-3 px-4">Accuracy</th>
              <th className="text-right py-3 px-4">Avg Ret</th>
              <th className="text-left py-3 px-4">Top Sector</th>
            </tr>
          </thead>
          <tbody>
            {filers.map((filer, idx) => {
              const isSelected = selectedFiler === filer.filer_name
              return (
                <tbody key={filer.filer_name}>
                  <tr
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
                    <td className="py-2.5 px-4 text-right">
                      <AccuracyBadge pct={filer.accuracy_pct} />
                    </td>
                    <td className="py-2.5 px-4 text-right">
                      {filer.avg_return_pct != null ? (
                        <span className={`text-xs font-mono ${filer.avg_return_pct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {filer.avg_return_pct >= 0 ? '+' : ''}{filer.avg_return_pct.toFixed(1)}%
                        </span>
                      ) : <span className="text-gray-600 text-xs">-</span>}
                    </td>
                    <td className="py-2.5 px-4 text-gray-400 text-xs truncate max-w-[150px]">
                      {filer.favorite_sector || '-'}
                    </td>
                  </tr>
                  {isSelected && (
                    <tr>
                      <td colSpan={10} className="bg-dark-bg p-4">
                        {/* Track Record Detail */}
                        {filerRecord && filerRecord.tracked_buys > 0 && (
                          <div className="mb-4 grid grid-cols-2 md:grid-cols-4 gap-3">
                            <div className="bg-dark-card border border-dark-border rounded p-3">
                              <p className="text-xs text-gray-500">Total Buys</p>
                              <p className="text-lg font-bold text-white">{filerRecord.total_buys}</p>
                            </div>
                            <div className="bg-dark-card border border-dark-border rounded p-3">
                              <p className="text-xs text-gray-500">30d Accuracy</p>
                              <p className="text-lg font-bold"><AccuracyBadge pct={filerRecord.accuracy_30d} /></p>
                              {filerRecord.avg_return_30d != null && (
                                <p className={`text-xs font-mono mt-1 ${filerRecord.avg_return_30d >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                  avg {filerRecord.avg_return_30d >= 0 ? '+' : ''}{filerRecord.avg_return_30d}%
                                </p>
                              )}
                            </div>
                            <div className="bg-dark-card border border-dark-border rounded p-3">
                              <p className="text-xs text-gray-500">60d Accuracy</p>
                              <p className="text-lg font-bold"><AccuracyBadge pct={filerRecord.accuracy_60d} /></p>
                              {filerRecord.avg_return_60d != null && (
                                <p className={`text-xs font-mono mt-1 ${filerRecord.avg_return_60d >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                  avg {filerRecord.avg_return_60d >= 0 ? '+' : ''}{filerRecord.avg_return_60d}%
                                </p>
                              )}
                            </div>
                            <div className="bg-dark-card border border-dark-border rounded p-3">
                              <p className="text-xs text-gray-500">90d Accuracy</p>
                              <p className="text-lg font-bold"><AccuracyBadge pct={filerRecord.accuracy_90d} /></p>
                              {filerRecord.avg_return_90d != null && (
                                <p className={`text-xs font-mono mt-1 ${filerRecord.avg_return_90d >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                  avg {filerRecord.avg_return_90d >= 0 ? '+' : ''}{filerRecord.avg_return_90d}%
                                </p>
                              )}
                            </div>
                          </div>
                        )}
                        <h4 className="text-sm font-semibold text-gray-400 mb-2">
                          Transaction History - {filer.filer_name}
                        </h4>
                        {loadingDetail ? (
                          <LoadingSkeleton rows={5} />
                        ) : filerTx.length > 0 ? (
                          <TransactionTable transactions={filerTx} compact />
                        ) : (
                          <p className="text-gray-500 text-sm">No recent transactions in current view</p>
                        )}
                      </td>
                    </tr>
                  )}
                </tbody>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
