const severityStyles = {
  HIGH: 'border-red-500/50 bg-red-500/10 text-red-400',
  MEDIUM: 'border-orange-500/50 bg-orange-500/10 text-orange-400',
  LOW: 'border-yellow-500/50 bg-yellow-500/10 text-yellow-400',
}

const severityLabels = {
  HIGH: 'HIGH',
  MEDIUM: 'MED',
  LOW: 'LOW',
}

export default function AlertCard({ alert, compact = false }) {
  const styles = severityStyles[alert.severity] || severityStyles.LOW

  if (compact) {
    return (
      <div className={`border rounded px-3 py-2 ${styles}`}>
        <div className="flex items-center justify-between">
          <span className="text-xs font-mono font-bold">{severityLabels[alert.severity]}</span>
          {alert.ticker && (
            <span className="text-xs font-mono bg-white/10 px-1.5 py-0.5 rounded">{alert.ticker}</span>
          )}
        </div>
        <p className="text-xs mt-1 opacity-80 line-clamp-2">{alert.description}</p>
      </div>
    )
  }

  return (
    <div className={`border rounded-lg p-4 ${styles}`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-white/10">
            {alert.severity}
          </span>
          <span className="text-xs font-mono opacity-60">{alert.alert_type}</span>
        </div>
        {alert.ticker && (
          <span className="text-sm font-mono font-bold">{alert.ticker}</span>
        )}
      </div>
      <p className="text-sm">{alert.description}</p>
      <p className="text-xs opacity-50 mt-2">
        {new Date(alert.triggered_at).toLocaleString()}
      </p>
    </div>
  )
}
