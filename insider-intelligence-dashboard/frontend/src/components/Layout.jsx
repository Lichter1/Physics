import { NavLink } from 'react-router-dom'

const navItems = [
  { path: '/', label: 'Dashboard', icon: '📊' },
  { path: '/ideas', label: 'Trade Ideas', icon: '💡' },
  { path: '/transactions', label: 'Transactions', icon: '📋' },
  { path: '/sectors', label: 'Sectors', icon: '📈' },
  { path: '/ticker', label: 'Ticker Dive', icon: '🔍' },
  { path: '/filers', label: 'Filers', icon: '👤' },
  { path: '/alerts', label: 'Alerts', icon: '🔔' },
  { path: '/watchlist', label: 'Watchlist', icon: '⭐' },
]

export default function Layout({ children }) {
  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <nav className="w-56 bg-dark-card border-r border-dark-border flex flex-col shrink-0">
        <div className="p-4 border-b border-dark-border">
          <h1 className="text-lg font-bold text-white">Insider Intel</h1>
          <p className="text-xs text-gray-500 mt-1">Market Intelligence Dashboard</p>
        </div>
        <div className="flex-1 py-2">
          {navItems.map(item => (
            <NavLink
              key={item.path}
              to={item.path}
              end={item.path === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-2.5 text-sm transition-colors ${
                  isActive
                    ? 'bg-blue-600/20 text-blue-400 border-r-2 border-blue-400'
                    : 'text-gray-400 hover:bg-dark-hover hover:text-gray-200'
                }`
              }
            >
              <span>{item.icon}</span>
              <span>{item.label}</span>
            </NavLink>
          ))}
        </div>
        <div className="p-3 border-t border-dark-border">
          <p className="text-[10px] text-gray-600 leading-tight">
            For informational purposes only. Not financial advice.
          </p>
        </div>
      </nav>

      {/* Main content */}
      <main className="flex-1 overflow-auto">
        <div className="p-6 max-w-[1600px] mx-auto">
          {children}
        </div>
      </main>
    </div>
  )
}
