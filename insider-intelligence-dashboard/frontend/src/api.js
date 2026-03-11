const BASE_URL = '/api';

async function fetchApi(endpoint, options = {}) {
  const url = `${BASE_URL}${endpoint}`;
  try {
    const response = await fetch(url, {
      headers: { 'Content-Type': 'application/json', ...options.headers },
      ...options,
    });
    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error(`Failed to fetch ${url}:`, error);
    throw error;
  }
}

function buildQuery(params) {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== null && value !== undefined && value !== '') {
      query.set(key, value);
    }
  });
  const qs = query.toString();
  return qs ? '?' + qs : '';
}

export async function getTransactions(params = {}) {
  return fetchApi(`/transactions${buildQuery(params)}`);
}

export async function getSectorSummary() {
  return fetchApi('/sectors/summary');
}

export async function getAlerts(params = {}) {
  return fetchApi(`/alerts${buildQuery(params)}`);
}

export async function getFilerLeaderboard(params = {}) {
  return fetchApi(`/filers/leaderboard${buildQuery(params)}`);
}

export async function getTickerActivity(ticker) {
  return fetchApi(`/tickers/${ticker}/activity`);
}

export async function getWeeklyTrends() {
  return fetchApi('/trends/weekly');
}

export async function triggerRefresh() {
  return fetchApi('/refresh', { method: 'POST' });
}

// --- New v2 endpoints ---

export async function getTradeIdeas(params = {}) {
  return fetchApi(`/trade-ideas${buildQuery(params)}`);
}

export async function getTickerPrices(ticker, period = '6m') {
  return fetchApi(`/prices/${ticker}?period=${period}`);
}

export async function getPriceContext(ticker) {
  return fetchApi(`/prices/${ticker}/context`);
}

export async function getTradeImpact(ticker, fromDate) {
  return fetchApi(`/prices/${ticker}/impact?from_date=${fromDate}`);
}

export async function getFilerTrackRecord(filerName) {
  return fetchApi(`/filers/${encodeURIComponent(filerName)}/track-record`);
}

export async function startBackfill(months = 24, source = 'all') {
  return fetchApi(`/backfill?months=${months}&source=${source}`, { method: 'POST' });
}

export async function getBackfillStatus() {
  return fetchApi('/backfill/status');
}

export async function refreshPrices() {
  return fetchApi('/prices/refresh', { method: 'POST' });
}

export async function recomputeTrackRecords() {
  return fetchApi('/filers/recompute-track-records', { method: 'POST' });
}
