export default function LoadingSkeleton({ rows = 5, type = 'table' }) {
  if (type === 'cards') {
    return (
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="bg-dark-card border border-dark-border rounded-lg p-4">
            <div className="skeleton h-3 w-16 rounded mb-2"></div>
            <div className="skeleton h-7 w-24 rounded mb-1"></div>
            <div className="skeleton h-2 w-20 rounded"></div>
          </div>
        ))}
      </div>
    )
  }

  if (type === 'chart') {
    return (
      <div className="bg-dark-card border border-dark-border rounded-lg p-4">
        <div className="skeleton h-4 w-32 rounded mb-4"></div>
        <div className="skeleton h-48 w-full rounded"></div>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="flex gap-4">
          <div className="skeleton h-4 w-20 rounded"></div>
          <div className="skeleton h-4 w-32 rounded"></div>
          <div className="skeleton h-4 w-16 rounded"></div>
          <div className="skeleton h-4 flex-1 rounded"></div>
          <div className="skeleton h-4 w-20 rounded"></div>
        </div>
      ))}
    </div>
  )
}
