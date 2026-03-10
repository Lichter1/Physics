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

export async function getTransactions(params = {}) {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== null && value !== undefined && value !== '') {
      query.set(key, value);
    }
  });
  const qs = query.toString();
  return fetchApi(`/transactions${qs ? '?' + qs : ''}`);
}

export async function getSectorSummary() {
  return fetchApi('/sectors/summary');
}

export async function getAlerts(params = {}) {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value) query.set(key, value);
  });
  const qs = query.toString();
  return fetchApi(`/alerts${qs ? '?' + qs : ''}`);
}

export async function getFilerLeaderboard(params = {}) {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value) query.set(key, value);
  });
  const qs = query.toString();
  return fetchApi(`/filers/leaderboard${qs ? '?' + qs : ''}`);
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
