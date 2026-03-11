"""Stock price data service using yfinance for historical and current prices."""

import logging
import asyncio
import aiosqlite
from datetime import datetime, timedelta
from database import get_db, DB_PATH

logger = logging.getLogger(__name__)

# Sector mapping for tickers (expanded reference data)
TICKER_SECTORS = {
    "AAPL": ("Apple Inc", "Technology", "Consumer Electronics"),
    "MSFT": ("Microsoft", "Technology", "Software"),
    "GOOGL": ("Alphabet", "Technology", "Internet Services"),
    "META": ("Meta Platforms", "Technology", "Social Media"),
    "AMZN": ("Amazon", "Technology", "E-Commerce"),
    "NVDA": ("NVIDIA Corp", "AI & Semiconductors", "Semiconductors"),
    "AMD": ("AMD Inc", "AI & Semiconductors", "Semiconductors"),
    "AVGO": ("Broadcom", "AI & Semiconductors", "Semiconductors"),
    "INTC": ("Intel Corp", "AI & Semiconductors", "Semiconductors"),
    "TSM": ("Taiwan Semi", "AI & Semiconductors", "Semiconductors"),
    "JPM": ("JPMorgan Chase", "Financials & Banking", "Banking"),
    "BAC": ("Bank of America", "Financials & Banking", "Banking"),
    "GS": ("Goldman Sachs", "Financials & Banking", "Investment Banking"),
    "MS": ("Morgan Stanley", "Financials & Banking", "Investment Banking"),
    "LMT": ("Lockheed Martin", "Defense & Aerospace", "Defense"),
    "RTX": ("RTX Corp", "Defense & Aerospace", "Defense"),
    "NOC": ("Northrop Grumman", "Defense & Aerospace", "Defense"),
    "GD": ("General Dynamics", "Defense & Aerospace", "Defense"),
    "XOM": ("Exxon Mobil", "Energy", "Oil & Gas"),
    "CVX": ("Chevron", "Energy", "Oil & Gas"),
    "COP": ("ConocoPhillips", "Energy", "Oil & Gas"),
    "NEE": ("NextEra Energy", "Energy", "Renewables"),
    "JNJ": ("Johnson & Johnson", "Healthcare & Pharma", "Pharmaceuticals"),
    "PFE": ("Pfizer", "Healthcare & Pharma", "Pharmaceuticals"),
    "UNH": ("UnitedHealth", "Healthcare & Pharma", "Health Insurance"),
    "ABBV": ("AbbVie", "Healthcare & Pharma", "Pharmaceuticals"),
    "AMT": ("American Tower", "Real Estate", "REITs"),
    "PLD": ("Prologis", "Real Estate", "REITs"),
    "SPG": ("Simon Property", "Real Estate", "REITs"),
    "TSLA": ("Tesla Inc", "Consumer Discretionary", "Automotive"),
    "NKE": ("Nike Inc", "Consumer Discretionary", "Apparel"),
    "SBUX": ("Starbucks", "Consumer Discretionary", "Restaurants"),
    "MCD": ("McDonalds", "Consumer Discretionary", "Restaurants"),
    "CAT": ("Caterpillar", "Industrials", "Machinery"),
    "BA": ("Boeing", "Industrials", "Aerospace"),
    "HON": ("Honeywell", "Industrials", "Conglomerates"),
    "UPS": ("UPS Inc", "Industrials", "Logistics"),
    # Additional high-interest tickers
    "PLTR": ("Palantir", "Technology", "Data Analytics"),
    "CRM": ("Salesforce", "Technology", "Cloud Software"),
    "ORCL": ("Oracle", "Technology", "Enterprise Software"),
    "NFLX": ("Netflix", "Technology", "Streaming"),
    "DIS": ("Disney", "Consumer Discretionary", "Entertainment"),
    "V": ("Visa", "Financials & Banking", "Payments"),
    "MA": ("Mastercard", "Financials & Banking", "Payments"),
    "WMT": ("Walmart", "Consumer Discretionary", "Retail"),
    "HD": ("Home Depot", "Consumer Discretionary", "Retail"),
    "PG": ("Procter & Gamble", "Consumer Discretionary", "Consumer Goods"),
    "KO": ("Coca-Cola", "Consumer Discretionary", "Beverages"),
    "PEP": ("PepsiCo", "Consumer Discretionary", "Beverages"),
    "COST": ("Costco", "Consumer Discretionary", "Retail"),
    "ABNB": ("Airbnb", "Technology", "Travel Tech"),
    "UBER": ("Uber", "Technology", "Ride-sharing"),
    "SQ": ("Block Inc", "Financials & Banking", "Fintech"),
    "COIN": ("Coinbase", "Financials & Banking", "Crypto Exchange"),
    "RIVN": ("Rivian", "Consumer Discretionary", "EV"),
    "LCID": ("Lucid Group", "Consumer Discretionary", "EV"),
    "SOFI": ("SoFi Technologies", "Financials & Banking", "Fintech"),
    "ARM": ("ARM Holdings", "AI & Semiconductors", "Semiconductors"),
    "SMCI": ("Super Micro", "AI & Semiconductors", "Servers"),
    "SNOW": ("Snowflake", "Technology", "Cloud Data"),
    "NET": ("Cloudflare", "Technology", "Cybersecurity"),
    "CRWD": ("CrowdStrike", "Technology", "Cybersecurity"),
    "LLY": ("Eli Lilly", "Healthcare & Pharma", "Pharmaceuticals"),
    "MRK": ("Merck", "Healthcare & Pharma", "Pharmaceuticals"),
    "TMO": ("Thermo Fisher", "Healthcare & Pharma", "Life Sciences"),
    "ABT": ("Abbott Labs", "Healthcare & Pharma", "Medical Devices"),
}


def _run_sync(func, *args):
    """Run a blocking function in executor to avoid blocking the event loop."""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, func, *args)


def _fetch_yfinance_prices(ticker: str, period: str = "2y") -> list[dict]:
    """Synchronous yfinance fetch - runs in thread executor."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            return []

        rows = []
        for date, row in hist.iterrows():
            rows.append({
                "ticker": ticker,
                "date": date.strftime("%Y-%m-%d"),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
            })
        return rows
    except Exception as e:
        logger.error(f"yfinance error for {ticker}: {e}")
        return []


def _fetch_yfinance_info(ticker: str) -> dict | None:
    """Fetch company info from yfinance."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info or "symbol" not in info:
            return None
        return {
            "ticker": ticker,
            "name": info.get("longName") or info.get("shortName", ticker),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "market_cap": info.get("marketCap", 0),
            "exchange": info.get("exchange", ""),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "pe_ratio": info.get("trailingPE"),
            "avg_volume": info.get("averageVolume"),
        }
    except Exception as e:
        logger.error(f"yfinance info error for {ticker}: {e}")
        return None


async def fetch_and_store_prices(ticker: str, period: str = "2y") -> int:
    """Fetch price history for a ticker and store in DB. Returns row count."""
    rows = await _run_sync(_fetch_yfinance_prices, ticker, period)
    if not rows:
        return 0

    db = await get_db()
    try:
        count = 0
        for row in rows:
            await db.execute(
                """INSERT OR IGNORE INTO price_history
                   (ticker, date, open, high, low, close, volume)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (row["ticker"], row["date"], row["open"], row["high"],
                 row["low"], row["close"], row["volume"])
            )
            count += 1
        await db.commit()
        logger.info(f"Stored {count} price rows for {ticker}")
        return count
    finally:
        await db.close()


async def fetch_and_store_company_info(ticker: str) -> bool:
    """Fetch company info and update companies table."""
    info = await _run_sync(_fetch_yfinance_info, ticker)
    if not info:
        return False

    db = await get_db()
    try:
        await db.execute(
            """INSERT OR REPLACE INTO companies
               (ticker, name, sector, industry, market_cap)
               VALUES (?, ?, ?, ?, ?)""",
            (info["ticker"], info["name"], info["sector"],
             info["industry"], info["market_cap"])
        )
        await db.commit()
        return True
    finally:
        await db.close()


async def update_prices_for_tracked_tickers():
    """Update prices for all tickers that appear in transactions."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT DISTINCT ticker FROM transactions WHERE ticker IS NOT NULL"
        )
        tickers = [row[0] for row in await cursor.fetchall()]
    finally:
        await db.close()

    total = 0
    for ticker in tickers:
        try:
            count = await fetch_and_store_prices(ticker, "3mo")
            total += count
            await asyncio.sleep(0.5)  # Rate limit yfinance
        except Exception as e:
            logger.error(f"Price update failed for {ticker}: {e}")

    logger.info(f"Updated prices for {len(tickers)} tickers, {total} total rows")
    return total


async def get_price_at_date(db, ticker: str, date: str) -> float | None:
    """Get closing price for a ticker on or near a given date."""
    cursor = await db.execute(
        """SELECT close FROM price_history
           WHERE ticker = ? AND date <= ?
           ORDER BY date DESC LIMIT 1""",
        (ticker, date)
    )
    row = await cursor.fetchone()
    return row[0] if row else None


async def get_price_range(db, ticker: str, start_date: str, end_date: str) -> list[dict]:
    """Get price history for a ticker within a date range."""
    cursor = await db.execute(
        """SELECT date, open, high, low, close, volume
           FROM price_history
           WHERE ticker = ? AND date >= ? AND date <= ?
           ORDER BY date ASC""",
        (ticker, start_date, end_date)
    )
    rows = await cursor.fetchall()
    return [{"date": r[0], "open": r[1], "high": r[2], "low": r[3],
             "close": r[4], "volume": r[5]} for r in rows]


async def compute_forward_returns(db, ticker: str, from_date: str) -> dict:
    """Compute forward returns from a given date (7d, 30d, 60d, 90d)."""
    base_price = await get_price_at_date(db, ticker, from_date)
    if not base_price or base_price == 0:
        return {"7d": None, "30d": None, "60d": None, "90d": None}

    results = {}
    for label, days in [("7d", 7), ("30d", 30), ("60d", 60), ("90d", 90)]:
        target_date = (datetime.strptime(from_date, "%Y-%m-%d") + timedelta(days=days)).strftime("%Y-%m-%d")
        future_price = await get_price_at_date(db, ticker, target_date)
        if future_price:
            results[label] = round((future_price - base_price) / base_price * 100, 2)
        else:
            results[label] = None

    return results


async def get_52w_range(db, ticker: str) -> dict:
    """Get 52-week high/low and current price for context."""
    one_year_ago = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    cursor = await db.execute(
        """SELECT MIN(low), MAX(high),
                  (SELECT close FROM price_history WHERE ticker = ? ORDER BY date DESC LIMIT 1)
           FROM price_history
           WHERE ticker = ? AND date >= ?""",
        (ticker, ticker, one_year_ago)
    )
    row = await cursor.fetchone()
    if not row or row[0] is None:
        return {"low_52w": None, "high_52w": None, "current": None, "pct_from_low": None}

    low, high, current = row[0], row[1], row[2]
    pct_from_low = round((current - low) / low * 100, 1) if low and current else None

    return {
        "low_52w": low,
        "high_52w": high,
        "current": current,
        "pct_from_low": pct_from_low,
    }


async def enrich_companies_from_reference():
    """Populate companies table from built-in reference data + fill missing sectors on transactions."""
    db = await get_db()
    try:
        for ticker, (name, sector, industry) in TICKER_SECTORS.items():
            await db.execute(
                """INSERT OR IGNORE INTO companies (ticker, name, sector, industry, market_cap)
                   VALUES (?, ?, ?, ?, NULL)""",
                (ticker, name, sector, industry)
            )

        # Fill missing sectors on transactions
        await db.execute("""
            UPDATE transactions SET sector = (
                SELECT c.sector FROM companies c WHERE c.ticker = transactions.ticker
            ) WHERE sector IS NULL AND ticker IS NOT NULL
              AND EXISTS (SELECT 1 FROM companies c WHERE c.ticker = transactions.ticker AND c.sector IS NOT NULL)
        """)

        await db.commit()
        logger.info(f"Enriched companies table with {len(TICKER_SECTORS)} reference tickers")
    finally:
        await db.close()
