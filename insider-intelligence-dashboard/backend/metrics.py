"""Computed metrics for tickers and sectors."""

from datetime import datetime, timedelta


async def compute_ticker_metrics(db, ticker: str) -> dict:
    """Compute all metrics for a given ticker."""
    now = datetime.now()
    thirty_days_ago = (now - timedelta(days=30)).strftime("%Y-%m-%d")
    fourteen_days_ago = (now - timedelta(days=14)).strftime("%Y-%m-%d")

    # Cluster Score: unique insiders buying in last 30 days (1-10 scale)
    cursor = await db.execute("""
        SELECT COUNT(DISTINCT filer_name) FROM transactions
        WHERE ticker = ? AND transaction_type = 'buy' AND transaction_date >= ?
    """, (ticker, thirty_days_ago))
    unique_buyers = (await cursor.fetchone())[0]
    cluster_score = min(unique_buyers, 10)

    # Conviction Score: weighted by filer seniority + trade size
    cursor = await db.execute("""
        SELECT filer_type, total_value FROM transactions
        WHERE ticker = ? AND transaction_type = 'buy' AND transaction_date >= ?
    """, (ticker, thirty_days_ago))
    rows = await cursor.fetchall()
    seniority_weights = {"ceo": 3.0, "director": 2.0, "politician": 2.5, "officer": 1.5, "10% owner": 2.0}
    conviction_raw = 0
    for row in rows:
        filer_type, value = row
        weight = seniority_weights.get(filer_type, 1.0)
        conviction_raw += (value or 0) * weight
    # Normalize to 0-10 scale (log scale)
    import math
    conviction_score = min(10, math.log10(max(conviction_raw, 1)) - 3) if conviction_raw > 1000 else 0
    conviction_score = max(0, round(conviction_score, 1))

    # Committee Exposure: how many committee-relevant politicians are buying
    cursor = await db.execute("""
        SELECT COUNT(DISTINCT filer_name) FROM transactions
        WHERE ticker = ? AND filer_type = 'politician' AND committee IS NOT NULL
          AND transaction_type = 'buy' AND transaction_date >= ?
    """, (ticker, thirty_days_ago))
    committee_exposure = (await cursor.fetchone())[0]

    # Delay Score: average filing delay (lower = more transparent)
    cursor = await db.execute("""
        SELECT AVG(delay_days) FROM transactions
        WHERE ticker = ? AND delay_days IS NOT NULL AND transaction_date >= ?
    """, (ticker, thirty_days_ago))
    delay_score = (await cursor.fetchone())[0] or 0
    delay_score = round(delay_score, 1)

    # Buy/Sell Ratio
    cursor = await db.execute("""
        SELECT
            COALESCE(SUM(CASE WHEN transaction_type='buy' THEN total_value ELSE 0 END), 0),
            COALESCE(SUM(CASE WHEN transaction_type='sell' THEN total_value ELSE 0 END), 0)
        FROM transactions WHERE ticker = ? AND transaction_date >= ?
    """, (ticker, thirty_days_ago))
    row = await cursor.fetchone()
    total_buy = row[0]
    total_sell = row[1]
    buy_sell_ratio = round(total_buy / total_sell, 2) if total_sell > 0 else (10.0 if total_buy > 0 else 0)

    return {
        "cluster_score": cluster_score,
        "conviction_score": conviction_score,
        "committee_exposure": committee_exposure,
        "delay_score": delay_score,
        "buy_sell_ratio": buy_sell_ratio,
        "total_buy_value": total_buy,
        "total_sell_value": total_sell,
    }


async def compute_sector_metrics(db, sector: str) -> dict:
    """Compute metrics for a sector."""
    now = datetime.now()
    this_week = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    last_week = (now - timedelta(days=14)).strftime("%Y-%m-%d")
    thirty_days = (now - timedelta(days=30)).strftime("%Y-%m-%d")

    # Momentum: week-over-week change in buy activity
    cursor = await db.execute("""
        SELECT
            COALESCE(SUM(CASE WHEN transaction_date >= ? THEN total_value ELSE 0 END), 0),
            COALESCE(SUM(CASE WHEN transaction_date >= ? AND transaction_date < ? THEN total_value ELSE 0 END), 0)
        FROM transactions
        WHERE sector = ? AND transaction_type = 'buy' AND total_value IS NOT NULL
    """, (this_week, last_week, this_week, sector))
    row = await cursor.fetchone()
    this_week_val = row[0]
    last_week_val = row[1]
    momentum = round((this_week_val - last_week_val) / last_week_val * 100, 1) if last_week_val > 0 else 0

    # Political Interest Score
    cursor = await db.execute("""
        SELECT COUNT(DISTINCT filer_name) FROM transactions
        WHERE sector = ? AND filer_type = 'politician' AND transaction_date >= ?
    """, (sector, thirty_days))
    political_interest = (await cursor.fetchone())[0]

    # Insider Confidence: ratio of CEO buys to sells
    cursor = await db.execute("""
        SELECT
            COALESCE(SUM(CASE WHEN transaction_type='buy' THEN 1 ELSE 0 END), 0),
            COALESCE(SUM(CASE WHEN transaction_type='sell' THEN 1 ELSE 0 END), 0)
        FROM transactions
        WHERE sector = ? AND filer_type = 'ceo' AND transaction_date >= ?
    """, (sector, thirty_days))
    row = await cursor.fetchone()
    ceo_buys = row[0]
    ceo_sells = row[1]
    insider_confidence = round(ceo_buys / ceo_sells, 2) if ceo_sells > 0 else (5.0 if ceo_buys > 0 else 0)

    return {
        "momentum": momentum,
        "political_interest": political_interest,
        "insider_confidence": insider_confidence,
    }
