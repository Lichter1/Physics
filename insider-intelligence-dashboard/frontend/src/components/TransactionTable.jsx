import { useNavigate } from 'react-router-dom'

const formatValue = (v) => {
  if (!v) return '-'
  return '$' + v.toLocaleString(undefined, { maximumFractionDigits: 0 })
}

export default function TransactionTable({ transactions, compact = false, showFilters = false }) {
  const navigate = useNavigate()

  if (!transactions || transactions.length === 0) {
    return <p className="text-gray-500 text-sm py-4 text-center">No transactions found</p>
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-dark-border text-gray-500 text-xs uppercase">
            <th className="text-left py-2 px-2">Date</th>
            <th className="text-left py-2 px-2">Filer</th>
            {!compact && <th className="text-left py-2 px-2">Type</th>}
            {!compact && <th className="text-left py-2 px-2">Party</th>}
            <th className="text-left py-2 px-2">Ticker</th>
            {!compact && <th className="text-left py-2 px-2">Sector</th>}
            <th className="text-left py-2 px-2">Action</th>
            <th className="text-right py-2 px-2">Value</th>
            {!compact && <th className="text-right py-2 px-2">Delay</th>}
            {!compact && <th className="text-left py-2 px-2">Flags</th>}
          </tr>
        </thead>
        <tbody>
          {transactions.map(tx => {
            const isBuy = tx.transaction_type === 'buy'
            const hasFlags = tx.flags && tx.flags.length > 0
            const rowColor = hasFlags
              ? 'bg-yellow-900/10 hover:bg-yellow-900/20'
              : isBuy
                ? 'hover:bg-green-900/10'
                : 'hover:bg-red-900/10'

            return (
              <tr key={tx.id} className={`border-b border-dark-border/50 ${rowColor} transition-colors`}>
                <td className="py-2 px-2 text-gray-400 text-xs whitespace-nowrap">
                  {tx.transaction_date}
                </td>
                <td className="py-2 px-2 text-gray-200 truncate max-w-[150px]">
                  {tx.filer_name}
                </td>
                {!compact && (
                  <td className="py-2 px-2">
                    <span className="text-xs px-1.5 py-0.5 rounded bg-dark-hover text-gray-400">
                      {tx.filer_type}
                    </span>
                  </td>
                )}
                {!compact && (
                  <td className="py-2 px-2">
                    {tx.party && (
                      <span className={`text-xs font-bold ${tx.party === 'D' ? 'text-blue-400' : 'text-red-400'}`}>
                        {tx.party}
                      </span>
                    )}
                  </td>
                )}
                <td className="py-2 px-2">
                  {tx.ticker ? (
                    <button
                      onClick={() => navigate(`/ticker/${tx.ticker}`)}
                      className="font-mono font-bold text-blue-400 hover:text-blue-300 hover:underline"
                    >
                      {tx.ticker}
                    </button>
                  ) : (
                    <span className="text-gray-600">-</span>
                  )}
                </td>
                {!compact && (
                  <td className="py-2 px-2 text-gray-400 text-xs truncate max-w-[120px]">
                    {tx.sector || '-'}
                  </td>
                )}
                <td className="py-2 px-2">
                  <span className={`text-xs font-bold ${isBuy ? 'text-green-400' : 'text-red-400'}`}>
                    {tx.transaction_type.toUpperCase()}
                  </span>
                </td>
                <td className="py-2 px-2 text-right font-mono text-xs">
                  {formatValue(tx.total_value)}
                </td>
                {!compact && (
                  <td className="py-2 px-2 text-right text-xs">
                    {tx.delay_days !== null ? (
                      <span className={tx.delay_days > 30 ? 'text-yellow-400' : 'text-gray-500'}>
                        {tx.delay_days}d
                      </span>
                    ) : '-'}
                  </td>
                )}
                {!compact && (
                  <td className="py-2 px-2">
                    {hasFlags && tx.flags.map(flag => (
                      <span key={flag} className="text-[10px] bg-yellow-500/20 text-yellow-400 px-1 py-0.5 rounded mr-1">
                        {flag}
                      </span>
                    ))}
                  </td>
                )}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
