export default function SectorHeatmap({ sectors }) {
  if (!sectors || sectors.length === 0) return null

  const maxRatio = Math.max(...sectors.map(s => s.buy_sell_ratio), 1)

  return (
    <div className="grid grid-cols-3 gap-2">
      {sectors.map(sector => {
        const ratio = sector.buy_sell_ratio
        let bgColor = 'bg-gray-800'
        let textColor = 'text-gray-400'

        if (ratio > 2) {
          bgColor = 'bg-green-900/60'
          textColor = 'text-green-400'
        } else if (ratio > 1.2) {
          bgColor = 'bg-green-900/30'
          textColor = 'text-green-300'
        } else if (ratio < 0.5) {
          bgColor = 'bg-red-900/60'
          textColor = 'text-red-400'
        } else if (ratio < 0.8) {
          bgColor = 'bg-red-900/30'
          textColor = 'text-red-300'
        } else {
          bgColor = 'bg-yellow-900/20'
          textColor = 'text-yellow-300'
        }

        return (
          <div
            key={sector.sector}
            className={`${bgColor} rounded-lg p-3 border border-dark-border hover:border-gray-600 transition-colors cursor-default`}
          >
            <p className="text-xs text-gray-400 truncate">{sector.sector}</p>
            <p className={`text-lg font-bold ${textColor} mt-1`}>
              {ratio.toFixed(1)}x
            </p>
            <div className="flex justify-between mt-1">
              <span className="text-[10px] text-green-500">${formatCompact(sector.total_buy_value)}</span>
              <span className="text-[10px] text-red-500">${formatCompact(sector.total_sell_value)}</span>
            </div>
            <p className="text-[10px] text-gray-500 mt-1">{sector.unique_filers} filers</p>
          </div>
        )
      })}
    </div>
  )
}

function formatCompact(value) {
  if (!value) return '0'
  if (value >= 1e9) return (value / 1e9).toFixed(1) + 'B'
  if (value >= 1e6) return (value / 1e6).toFixed(1) + 'M'
  if (value >= 1e3) return (value / 1e3).toFixed(0) + 'K'
  return value.toFixed(0)
}
