import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Transactions from './pages/Transactions'
import SectorAnalysis from './pages/SectorAnalysis'
import TickerDeepDive from './pages/TickerDeepDive'
import FilerProfiles from './pages/FilerProfiles'
import AlertsCenter from './pages/AlertsCenter'

export default function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/transactions" element={<Transactions />} />
          <Route path="/sectors" element={<SectorAnalysis />} />
          <Route path="/ticker/:ticker" element={<TickerDeepDive />} />
          <Route path="/ticker" element={<TickerDeepDive />} />
          <Route path="/filers" element={<FilerProfiles />} />
          <Route path="/alerts" element={<AlertsCenter />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  )
}
