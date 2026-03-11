"""Composite Insider Signal Score — the flagship feature.

Computes a 0-100 score per ticker that answers:
"How strong is the insider buying signal right now?"
"""

import math
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Weights for the composite score
WEIGHTS = {
    "cluster": 0.25,       # Multiple insiders buying = consensus
    "conviction": 0.20,    # Trade size relative to norms
    "seniority": 0.15,     # C-suite > director > officer
    "political": 0.15,     # Political convergence with corporate insiders
    "timeliness": 0.10,    # Quick filings = transparent
    "price_context": 0.10, # Buying near lows = more conviction
    "accuracy": 0.05,      # Filer's historical hit rate
}

SENIORITY_SCORES = {
    "ceo": 10,
    "10% owner": 8,
    "politician": 7,
    "director": 6,
    "officer": 4,
}


async def compute_signal_score(db, ticker: str) -> dict:
    """Compute the composite Insider Signal Score (0-100) for a ticker."""
    now = datetime.now()
    thirty_days_ago = (now - timedelta(days=30)).strftime("%Y-%m-%d")
    fourteen_days_ago = (now - timedelta(days=14)).strftime("%Y-%m-%d")

    # --- Cluster component (0-100) ---
    cursor = await db.execute("""
        SELECT COUNT(DISTINCT filer_name) FROM transactions
        WHERE ticker = ? AND transaction_type = 'buy' AND transaction_date >= ?
    """, (ticker, fourteen_days_ago))
    unique_buyers_14d = (await cursor.fetchone())[0]
    cluster_raw = min(unique_buyers_14d / 5.0, 1.0) * 100  # 5+ buyers = max

    # --- Conviction component (0-100) ---
    cursor = await db.execute("""
        SELECT total_value FROM transactions
        WHERE ticker = ? AND transaction_type = 'buy' AND transaction_date >= ?
          AND total_value IS NOT NULL
        ORDER BY total_value DESC
    """, (ticker, thirty_days_ago))
    buy_values = [row[0] for row in await cursor.fetchall()]
    if buy_values:
        total_buy = sum(buy_values)
        max_single = max(buy_values)
        # Large buys score higher; $1M+ = very high conviction
        conviction_raw = min(math.log10(max(total_buy, 1)) / 7.0, 1.0) * 100
        # Bonus for single large trades
        if max_single > 500000:
            conviction_raw = min(conviction_raw + 15, 100)
    else:
        conviction_raw = 0

    # --- Seniority component (0-100) ---
    cursor = await db.execute("""
        SELECT filer_type, COUNT(*) FROM transactions
        WHERE ticker = ? AND transaction_type = 'buy' AND transaction_date >= ?
        GROUP BY filer_type
    """, (ticker, thirty_days_ago))
    filer_types = await cursor.fetchall()
    if filer_types:
        weighted_seniority = sum(
            SENIORITY_SCORES.get(ft[0], 3) * ft[1] for ft in filer_types
        )
        total_count = sum(ft[1] for ft in filer_types)
        avg_seniority = weighted_seniority / total_count
        seniority_raw = min(avg_seniority / 10.0, 1.0) * 100
    else:
        seniority_raw = 0

    # --- Political convergence (0-100) ---
    cursor = await db.execute("""
        SELECT
            SUM(CASE WHEN filer_type = 'politician' THEN 1 ELSE 0 END),
            SUM(CASE WHEN filer_type != 'politician' THEN 1 ELSE 0 END)
        FROM transactions
        WHERE ticker = ? AND transaction_type = 'buy' AND transaction_date >= ?
    """, (ticker, thirty_days_ago))
    row = await cursor.fetchone()
    pol_buys = row[0] or 0
    corp_buys = row[1] or 0
    if pol_buys > 0 and corp_buys > 0:
        political_raw = min((pol_buys + corp_buys) / 6.0, 1.0) * 100
    elif pol_buys > 0 or corp_buys > 0:
        political_raw = min((pol_buys + corp_buys) / 8.0, 1.0) * 50
    else:
        political_raw = 0

    # --- Timeliness component (0-100) ---
    cursor = await db.execute("""
        SELECT AVG(delay_days) FROM transactions
        WHERE ticker = ? AND transaction_date >= ? AND delay_days IS NOT NULL
    """, (ticker, thirty_days_ago))
    avg_delay = (await cursor.fetchone())[0]
    if avg_delay is not None:
        timeliness_raw = max(0, min(100, (30 - avg_delay) / 30.0 * 100))
    else:
        timeliness_raw = 50  # Neutral if unknown

    # --- Price context (0-100) ---
    # Buying near 52-week low is more bullish
    price_context_raw = 50  # Default neutral
    try:
        cursor = await db.execute("""
            SELECT MIN(low), MAX(high),
                   (SELECT close FROM price_history WHERE ticker = ? ORDER BY date DESC LIMIT 1)
            FROM price_history
            WHERE ticker = ? AND date >= ?
        """, (ticker, ticker, (now - timedelta(days=365)).strftime("%Y-%m-%d")))
        prow = await cursor.fetchone()
        if prow and prow[0] and prow[1] and prow[2]:
            low_52w, high_52w, current = prow[0], prow[1], prow[2]
            if high_52w > low_52w:
                # How far from the low (0 = at low = most bullish for insider buys)
                pct_from_low = (current - low_52w) / (high_52w - low_52w)
                price_context_raw = max(0, (1 - pct_from_low) * 100)
    except Exception:
        pass

    # --- Accuracy component (0-100) ---
    # Check if filers who bought this ticker have good track records
    accuracy_raw = 50  # Default to neutral
    cursor = await db.execute("""
        SELECT DISTINCT filer_name FROM transactions
        WHERE ticker = ? AND transaction_type = 'buy' AND transaction_date >= ?
    """, (ticker, thirty_days_ago))
    recent_buyers = [row[0] for row in await cursor.fetchall()]
    if recent_buyers:
        accuracies = []
        for filer in recent_buyers[:10]:  # Limit to avoid slow queries
            cursor = await db.execute("""
                SELECT accuracy_pct FROM filer_track_record WHERE filer_name = ?
            """, (filer,))
            acc_row = await cursor.fetchone()
            if acc_row and acc_row[0] is not None:
                accuracies.append(acc_row[0])
        if accuracies:
            accuracy_raw = sum(accuracies) / len(accuracies)

    # --- Composite Score ---
    composite = (
        WEIGHTS["cluster"] * cluster_raw +
        WEIGHTS["conviction"] * conviction_raw +
        WEIGHTS["seniority"] * seniority_raw +
        WEIGHTS["political"] * political_raw +
        WEIGHTS["timeliness"] * timeliness_raw +
        WEIGHTS["price_context"] * price_context_raw +
        WEIGHTS["accuracy"] * accuracy_raw
    )
    composite = round(min(max(composite, 0), 100), 1)

    return {
        "signal_score": composite,
        "components": {
            "cluster": round(cluster_raw, 1),
            "conviction": round(conviction_raw, 1),
            "seniority": round(seniority_raw, 1),
            "political_convergence": round(political_raw, 1),
            "timeliness": round(timeliness_raw, 1),
            "price_context": round(price_context_raw, 1),
            "filer_accuracy": round(accuracy_raw, 1),
        },
        "unique_buyers_14d": unique_buyers_14d,
        "total_buy_value_30d": sum(buy_values) if buy_values else 0,
    }


async def generate_trade_ideas(db, min_score: float = 20, limit: int = 50) -> list[dict]:
    """Generate ranked trade ideas based on signal scores."""
    # Get all tickers with recent buying activity
    cursor = await db.execute("""
        SELECT DISTINCT t.ticker, c.name, c.sector
        FROM transactions t
        LEFT JOIN companies c ON t.ticker = c.ticker
        WHERE t.transaction_type = 'buy'
          AND t.transaction_date >= ?
          AND t.ticker IS NOT NULL
        ORDER BY t.transaction_date DESC
    """, ((datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),))
    tickers = await cursor.fetchall()

    ideas = []
    for row in tickers:
        ticker, company_name, sector = row[0], row[1], row[2]
        try:
            score_data = await compute_signal_score(db, ticker)
            if score_data["signal_score"] < min_score:
                continue

            # Build thesis string
            thesis = await _build_thesis(db, ticker, score_data)

            # Get price context
            price_info = {}
            try:
                pcursor = await db.execute("""
                    SELECT close FROM price_history
                    WHERE ticker = ? ORDER BY date DESC LIMIT 1
                """, (ticker,))
                prow = await pcursor.fetchone()
                if prow:
                    price_info["current_price"] = prow[0]
            except Exception:
                pass

            ideas.append({
                "ticker": ticker,
                "company_name": company_name,
                "sector": sector,
                "signal_score": score_data["signal_score"],
                "components": score_data["components"],
                "thesis": thesis,
                "unique_buyers": score_data["unique_buyers_14d"],
                "total_buy_value": score_data["total_buy_value_30d"],
                "current_price": price_info.get("current_price"),
            })
        except Exception as e:
            logger.debug(f"Signal score failed for {ticker}: {e}")
            continue

    ideas.sort(key=lambda x: x["signal_score"], reverse=True)
    return ideas[:limit]


async def _build_thesis(db, ticker: str, score_data: dict) -> str:
    """Build a concise thesis string for a trade idea."""
    parts = []
    buyers = score_data["unique_buyers_14d"]
    total_val = score_data["total_buy_value_30d"]

    if buyers >= 3:
        parts.append(f"{buyers} insiders buying")
    elif buyers > 0:
        parts.append(f"{buyers} insider{'s' if buyers > 1 else ''} buying")

    if total_val >= 1_000_000:
        parts.append(f"${total_val/1e6:.1f}M total")
    elif total_val >= 1_000:
        parts.append(f"${total_val/1e3:.0f}K total")

    comp = score_data["components"]
    if comp["political_convergence"] > 50:
        parts.append("political + corporate convergence")
    if comp["price_context"] > 70:
        parts.append("near 52-week low")
    if comp["seniority"] > 70:
        parts.append("C-suite buying")
    if comp["filer_accuracy"] > 65:
        parts.append("high-accuracy filers")

    return " | ".join(parts) if parts else "Active insider buying detected"
