import { useState, useEffect } from 'react'
import { getAlerts } from '../api'
import AlertCard from '../components/AlertCard'
import LoadingSkeleton from '../components/LoadingSkeleton'

const ALERT_TYPES = [
  '', 'CLUSTER_BUY', 'COMMITTEE_MATCH', 'LARGE_INSIDER_BUY',
  'REPEATED_FILER', 'FILING_DELAY', 'SECTOR_SURGE', 'CROSS_SOURCE_CONVERGENCE'
]

const SEVERITIES = ['', 'HIGH', 'MEDIUM', 'LOW']

export default function AlertsCenter() {
  const [alerts, setAlerts] = useState([])
  const [loading, setLoading] = useState(true)
  const [severity, setSeverity] = useState('')
  const [alertType, setAlertType] = useState('')

  useEffect(() => {
    setLoading(true)
    getAlerts({
      severity: severity || undefined,
      alert_type: alertType || undefined,
      limit: 100,
    })
      .then(setAlerts)
      .catch(err => console.error(err))
      .finally(() => setLoading(false))
  }, [severity, alertType])

  const highCount = alerts.filter(a => a.severity === 'HIGH').length
  const medCount = alerts.filter(a => a.severity === 'MEDIUM').length
  const lowCount = alerts.filter(a => a.severity === 'LOW').length

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold">Alerts Center</h2>

      {/* Summary */}
      <div className="flex gap-4">
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-2">
          <span className="text-red-400 font-bold text-lg">{highCount}</span>
          <span className="text-red-400/70 text-xs ml-2">HIGH</span>
        </div>
        <div className="bg-orange-500/10 border border-orange-500/30 rounded-lg px-4 py-2">
          <span className="text-orange-400 font-bold text-lg">{medCount}</span>
          <span className="text-orange-400/70 text-xs ml-2">MEDIUM</span>
        </div>
        <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg px-4 py-2">
          <span className="text-yellow-400 font-bold text-lg">{lowCount}</span>
          <span className="text-yellow-400/70 text-xs ml-2">LOW</span>
        </div>
      </div>

      {/* Filters */}
      <div className="flex gap-3 flex-wrap">
        <div className="flex gap-1">
          {SEVERITIES.map(s => (
            <button
              key={s}
              onClick={() => setSeverity(s)}
              className={`px-3 py-1.5 text-xs rounded transition-colors ${
                severity === s
                  ? 'bg-blue-600 text-white'
                  : 'bg-dark-card border border-dark-border text-gray-400 hover:text-gray-200'
              }`}
            >
              {s || 'All Severity'}
            </button>
          ))}
        </div>
        <select
          value={alertType}
          onChange={e => setAlertType(e.target.value)}
          className="bg-dark-card border border-dark-border rounded px-2 py-1.5 text-xs text-gray-300"
        >
          <option value="">All Types</option>
          {ALERT_TYPES.filter(Boolean).map(t => (
            <option key={t} value={t}>{t.replace(/_/g, ' ')}</option>
          ))}
        </select>
      </div>

      {/* Alert List */}
      {loading ? (
        <div className="space-y-3">
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="skeleton h-24 rounded-lg" />
          ))}
        </div>
      ) : alerts.length > 0 ? (
        <div className="space-y-3">
          {alerts.map(alert => (
            <AlertCard key={alert.id} alert={alert} />
          ))}
        </div>
      ) : (
        <div className="text-center py-12 text-gray-500">
          <p className="text-lg">No alerts matching filters</p>
          <p className="text-sm mt-1">Try adjusting your severity or type filter</p>
        </div>
      )}
    </div>
  )
}
