import { useState, useEffect, useCallback } from 'react'
import { getTransactions } from '../api'
import TransactionTable from '../components/TransactionTable'
import LoadingSkeleton from '../components/LoadingSkeleton'

const SECTORS = [
  '', 'Defense & Aerospace', 'AI & Semiconductors', 'Energy',
  'Healthcare & Pharma', 'Financials & Banking', 'Technology',
  'Real Estate', 'Consumer Discretionary', 'Industrials'
]

const FILER_TYPES = ['', 'politician', 'ceo', 'director', 'officer', '10% owner']

export default function Transactions() {
  const [transactions, setTransactions] = useState([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [page, setPage] = useState(0)
  const [filters, setFilters] = useState({
    sector: '', filer_type: '', transaction_type: '', ticker: '',
    date_from: '', date_to: '', min_value: '',
  })

  const limit = 50

  const fetchData = useCallback(async () => {
    setLoading(true)
    try {
      const params = { ...filters, limit, offset: page * limit }
      const data = await getTransactions(params)
      setTransactions(data.transactions)
      setTotal(data.total)
    } catch (err) {
      console.error('Transactions fetch error:', err)
    } finally {
      setLoading(false)
    }
  }, [filters, page])

  useEffect(() => { fetchData() }, [fetchData])

  const handleFilter = (key, value) => {
    setFilters(prev => ({ ...prev, [key]: value }))
    setPage(0)
  }

  const exportCSV = () => {
    const headers = ['Date', 'Filer', 'Type', 'Party', 'Ticker', 'Sector', 'Action', 'Value', 'Delay', 'Source']
    const rows = transactions.map(tx => [
      tx.transaction_date, tx.filer_name, tx.filer_type, tx.party || '',
      tx.ticker || '', tx.sector || '', tx.transaction_type,
      tx.total_value || '', tx.delay_days || '', tx.source
    ])
    const csv = [headers, ...rows].map(r => r.map(c => `"${c}"`).join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `insider_transactions_${new Date().toISOString().slice(0, 10)}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  const totalPages = Math.ceil(total / limit)

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold">Transactions Explorer</h2>
        <button
          onClick={exportCSV}
          className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded transition-colors"
        >
          Export CSV
        </button>
      </div>

      {/* Filters */}
      <div className="bg-dark-card border border-dark-border rounded-lg p-4">
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
          <select
            value={filters.sector}
            onChange={e => handleFilter('sector', e.target.value)}
            className="bg-dark-bg border border-dark-border rounded px-2 py-1.5 text-sm text-gray-300"
          >
            <option value="">All Sectors</option>
            {SECTORS.filter(Boolean).map(s => <option key={s} value={s}>{s}</option>)}
          </select>

          <select
            value={filters.filer_type}
            onChange={e => handleFilter('filer_type', e.target.value)}
            className="bg-dark-bg border border-dark-border rounded px-2 py-1.5 text-sm text-gray-300"
          >
            <option value="">All Filer Types</option>
            {FILER_TYPES.filter(Boolean).map(t => <option key={t} value={t}>{t}</option>)}
          </select>

          <select
            value={filters.transaction_type}
            onChange={e => handleFilter('transaction_type', e.target.value)}
            className="bg-dark-bg border border-dark-border rounded px-2 py-1.5 text-sm text-gray-300"
          >
            <option value="">Buy & Sell</option>
            <option value="buy">Buy Only</option>
            <option value="sell">Sell Only</option>
          </select>

          <input
            type="text"
            placeholder="Ticker..."
            value={filters.ticker}
            onChange={e => handleFilter('ticker', e.target.value.toUpperCase())}
            className="bg-dark-bg border border-dark-border rounded px-2 py-1.5 text-sm text-gray-300 placeholder-gray-600"
          />

          <input
            type="date"
            value={filters.date_from}
            onChange={e => handleFilter('date_from', e.target.value)}
            className="bg-dark-bg border border-dark-border rounded px-2 py-1.5 text-sm text-gray-300"
          />

          <input
            type="date"
            value={filters.date_to}
            onChange={e => handleFilter('date_to', e.target.value)}
            className="bg-dark-bg border border-dark-border rounded px-2 py-1.5 text-sm text-gray-300"
          />

          <input
            type="number"
            placeholder="Min value..."
            value={filters.min_value}
            onChange={e => handleFilter('min_value', e.target.value)}
            className="bg-dark-bg border border-dark-border rounded px-2 py-1.5 text-sm text-gray-300 placeholder-gray-600"
          />
        </div>
      </div>

      {/* Results */}
      <div className="bg-dark-card border border-dark-border rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs text-gray-500">{total} transactions found</span>
          <div className="flex items-center gap-2">
            <button
              disabled={page === 0}
              onClick={() => setPage(p => p - 1)}
              className="px-2 py-1 text-xs bg-dark-bg border border-dark-border rounded disabled:opacity-30 hover:bg-dark-hover"
            >
              Prev
            </button>
            <span className="text-xs text-gray-500">
              Page {page + 1} of {totalPages || 1}
            </span>
            <button
              disabled={page >= totalPages - 1}
              onClick={() => setPage(p => p + 1)}
              className="px-2 py-1 text-xs bg-dark-bg border border-dark-border rounded disabled:opacity-30 hover:bg-dark-hover"
            >
              Next
            </button>
          </div>
        </div>

        {loading ? (
          <LoadingSkeleton rows={10} />
        ) : (
          <TransactionTable transactions={transactions} />
        )}
      </div>
    </div>
  )
}
