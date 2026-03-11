import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Transactions from './pages/Transactions'
import SectorAnalysis from './pages/SectorAnalysis'
import TickerDeepDive from './pages/TickerDeepDive'
import FilerProfiles from './pages/FilerProfiles'
import AlertsCenter from './pages/AlertsCenter'
import TradeIdeas from './pages/TradeIdeas'
import Watchlist from './pages/Watchlist'

export default function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/ideas" element={<TradeIdeas />} />
          <Route path="/transactions" element={<Transactions />} />
          <Route path="/sectors" element={<SectorAnalysis />} />
          <Route path="/ticker/:ticker" element={<TickerDeepDive />} />
          <Route path="/ticker" element={<TickerDeepDive />} />
          <Route path="/filers" element={<FilerProfiles />} />
          <Route path="/alerts" element={<AlertsCenter />} />
          <Route path="/watchlist" element={<Watchlist />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  )
}
