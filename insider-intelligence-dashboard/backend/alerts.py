import logging
from datetime import datetime, timedelta
from committees import get_sectors_for_committee

logger = logging.getLogger(__name__)


async def generate_all_alerts(db) -> int:
    """Run all alert detection functions and store results. Returns count of new alerts."""
    total = 0
    detectors = [
        detect_cluster_buys,
        detect_committee_match,
        detect_large_insider_buy,
        detect_repeated_filer,
        detect_filing_delay,
        detect_sector_surge,
        detect_cross_source_convergence,
        detect_sector_rotation,
        detect_high_signal_score,
        detect_anomalous_trade,
    ]
    for detector in detectors:
        try:
            count = await detector(db)
            total += count
        except Exception as e:
            logger.error(f"Alert detector {detector.__name__} failed: {e}")
    return total


async def _insert_alert(db, alert_type: str, ticker: str, description: str,
                        severity: str, related_ids: list[int] = None) -> bool:
    """Insert an alert if a similar one doesn't already exist in the last 24h."""
    cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
    cursor = await db.execute(
        """SELECT id FROM alerts WHERE alert_type = ? AND ticker = ? AND triggered_at > ?""",
        (alert_type, ticker, cutoff)
    )
    if await cursor.fetchone():
        return False

    ids_str = ",".join(str(i) for i in related_ids) if related_ids else None
    await db.execute(
        """INSERT INTO alerts (alert_type, ticker, description, severity, related_transaction_ids)
           VALUES (?, ?, ?, ?, ?)""",
        (alert_type, ticker, description, severity, ids_str)
    )
    await db.commit()
    return True


async def detect_cluster_buys(db) -> int:
    """3+ different filers buying same ticker within 14 days → HIGH."""
    count = 0
    cutoff = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")

    cursor = await db.execute("""
        SELECT ticker, COUNT(DISTINCT filer_name) as filer_count,
               GROUP_CONCAT(DISTINCT filer_name) as filers,
               GROUP_CONCAT(id) as ids
        FROM transactions
        WHERE transaction_type = 'buy'
          AND transaction_date >= ?
          AND ticker IS NOT NULL
        GROUP BY ticker
        HAVING filer_count >= 3
    """, (cutoff,))

    rows = await cursor.fetchall()
    for row in rows:
        ticker = row[0]
        filer_count = row[1]
        filers = row[2]
        ids = [int(x) for x in row[3].split(",")]
        desc = f"CLUSTER BUY: {filer_count} different filers bought {ticker} in last 14 days ({filers})"
        if await _insert_alert(db, "CLUSTER_BUY", ticker, desc, "HIGH", ids):
            count += 1
    return count


async def detect_committee_match(db) -> int:
    """Politician buys stock in sector their committee oversees → HIGH."""
    count = 0
    cutoff = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    cursor = await db.execute("""
        SELECT id, filer_name, committee, ticker, sector, total_value
        FROM transactions
        WHERE filer_type = 'politician'
          AND committee IS NOT NULL
          AND sector IS NOT NULL
          AND transaction_type = 'buy'
          AND transaction_date >= ?
    """, (cutoff,))

    rows = await cursor.fetchall()
    for row in rows:
        tx_id, name, committee, ticker, sector, value = row
        overseen_sectors = get_sectors_for_committee(committee)
        if sector in overseen_sectors:
            desc = (f"COMMITTEE MATCH: {name} ({committee} committee) bought {ticker} "
                    f"in {sector} sector — committee has oversight")
            if await _insert_alert(db, "COMMITTEE_MATCH", ticker, desc, "HIGH", [tx_id]):
                count += 1
    return count


async def detect_large_insider_buy(db) -> int:
    """Corporate insider buy > $500,000 → MEDIUM."""
    count = 0
    cutoff = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")

    cursor = await db.execute("""
        SELECT id, filer_name, filer_type, ticker, total_value
        FROM transactions
        WHERE filer_type IN ('ceo', 'director', 'officer', '10% owner')
          AND transaction_type = 'buy'
          AND total_value > 500000
          AND transaction_date >= ?
    """, (cutoff,))

    rows = await cursor.fetchall()
    for row in rows:
        tx_id, name, filer_type, ticker, value = row
        desc = f"LARGE INSIDER BUY: {name} ({filer_type}) bought ${value:,.0f} of {ticker}"
        if await _insert_alert(db, "LARGE_INSIDER_BUY", ticker, desc, "MEDIUM", [tx_id]):
            count += 1
    return count


async def detect_repeated_filer(db) -> int:
    """Same person trades same ticker 3+ times in 60 days → MEDIUM."""
    count = 0
    cutoff = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

    cursor = await db.execute("""
        SELECT filer_name, ticker, COUNT(*) as tx_count,
               GROUP_CONCAT(id) as ids
        FROM transactions
        WHERE transaction_date >= ?
          AND ticker IS NOT NULL
        GROUP BY filer_name, ticker
        HAVING tx_count >= 3
    """, (cutoff,))

    rows = await cursor.fetchall()
    for row in rows:
        name, ticker, tx_count, ids_str = row
        ids = [int(x) for x in ids_str.split(",")]
        desc = f"REPEATED FILER: {name} traded {ticker} {tx_count} times in 60 days"
        if await _insert_alert(db, "REPEATED_FILER", ticker, desc, "MEDIUM", ids):
            count += 1
    return count


async def detect_filing_delay(db) -> int:
    """Transaction filed >30 days after trade date → LOW."""
    count = 0
    cutoff = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

    cursor = await db.execute("""
        SELECT id, filer_name, ticker, delay_days, transaction_date, filing_date
        FROM transactions
        WHERE delay_days > 30
          AND transaction_date >= ?
    """, (cutoff,))

    rows = await cursor.fetchall()
    for row in rows:
        tx_id, name, ticker, delay, tx_date, filing_date = row
        desc = (f"FILING DELAY: {name} filed {ticker} trade {delay} days late "
                f"(traded {tx_date}, filed {filing_date})")
        if await _insert_alert(db, "FILING_DELAY", ticker, desc, "LOW", [tx_id]):
            count += 1
    return count


async def detect_sector_surge(db) -> int:
    """Sector sees 2x normal weekly buy volume → MEDIUM."""
    count = 0
    now = datetime.now()
    this_week_start = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    prev_weeks_start = (now - timedelta(days=35)).strftime("%Y-%m-%d")

    # Get this week's buy volume per sector
    cursor = await db.execute("""
        SELECT sector,
               SUM(CASE WHEN transaction_date >= ? THEN total_value ELSE 0 END) as this_week,
               SUM(CASE WHEN transaction_date < ? AND transaction_date >= ? THEN total_value ELSE 0 END) / 4.0 as avg_week
        FROM transactions
        WHERE transaction_type = 'buy'
          AND sector IS NOT NULL
          AND total_value IS NOT NULL
        GROUP BY sector
    """, (this_week_start, this_week_start, prev_weeks_start))

    rows = await cursor.fetchall()
    for row in rows:
        sector, this_week, avg_week = row
        if avg_week and avg_week > 0 and this_week > avg_week * 2:
            ratio = this_week / avg_week
            desc = f"SECTOR SURGE: {sector} buy volume is {ratio:.1f}x normal (${this_week:,.0f} vs avg ${avg_week:,.0f})"
            if await _insert_alert(db, "SECTOR_SURGE", sector, desc, "MEDIUM"):
                count += 1
    return count


async def detect_cross_source_convergence(db) -> int:
    """Both politicians AND corporate insiders buying same ticker → HIGH."""
    count = 0
    cutoff = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")

    cursor = await db.execute("""
        SELECT ticker,
               SUM(CASE WHEN filer_type = 'politician' THEN 1 ELSE 0 END) as pol_count,
               SUM(CASE WHEN filer_type != 'politician' THEN 1 ELSE 0 END) as insider_count,
               GROUP_CONCAT(id) as ids
        FROM transactions
        WHERE transaction_type = 'buy'
          AND transaction_date >= ?
          AND ticker IS NOT NULL
        GROUP BY ticker
        HAVING pol_count > 0 AND insider_count > 0
    """, (cutoff,))

    rows = await cursor.fetchall()
    for row in rows:
        ticker, pol_count, insider_count, ids_str = row
        ids = [int(x) for x in ids_str.split(",")]
        desc = (f"CROSS-SOURCE CONVERGENCE: {pol_count} politician(s) AND {insider_count} "
                f"corporate insider(s) buying {ticker} simultaneously")
        if await _insert_alert(db, "CROSS_SOURCE_CONVERGENCE", ticker, desc, "HIGH", ids):
            count += 1
    return count


async def detect_sector_rotation(db) -> int:
    """Detect when capital is flowing out of one sector into another → HIGH."""
    count = 0
    now = datetime.now()
    this_period = (now - timedelta(days=14)).strftime("%Y-%m-%d")
    prev_period_start = (now - timedelta(days=42)).strftime("%Y-%m-%d")
    prev_period_end = (now - timedelta(days=14)).strftime("%Y-%m-%d")

    cursor = await db.execute("""
        SELECT sector,
               SUM(CASE WHEN transaction_date >= ? THEN
                   CASE WHEN transaction_type='buy' THEN total_value ELSE -total_value END
               ELSE 0 END) as recent_net,
               SUM(CASE WHEN transaction_date >= ? AND transaction_date < ? THEN
                   CASE WHEN transaction_type='buy' THEN total_value ELSE -total_value END
               ELSE 0 END) as prev_net
        FROM transactions
        WHERE sector IS NOT NULL AND total_value IS NOT NULL
        GROUP BY sector
        HAVING recent_net IS NOT NULL AND prev_net IS NOT NULL
    """, (this_period, prev_period_start, prev_period_end))

    rows = await cursor.fetchall()
    outflows = []
    inflows = []
    for row in rows:
        sector, recent_net, prev_net = row
        if recent_net < 0 and (prev_net is None or prev_net >= 0 or recent_net < prev_net * 1.5):
            outflows.append((sector, recent_net))
        if recent_net > 0 and (prev_net is None or prev_net <= 0 or recent_net > prev_net * 1.5):
            inflows.append((sector, recent_net))

    if outflows and inflows:
        outflow_sectors = ", ".join(f"{s[0]}" for s in sorted(outflows, key=lambda x: x[1])[:2])
        inflow_sectors = ", ".join(f"{s[0]}" for s in sorted(inflows, key=lambda x: -x[1])[:2])
        desc = f"SECTOR ROTATION: Capital flowing OUT of {outflow_sectors} and INTO {inflow_sectors}"
        if await _insert_alert(db, "SECTOR_ROTATION", "ROTATION", desc, "HIGH"):
            count += 1

    return count


async def detect_high_signal_score(db) -> int:
    """Tickers with exceptionally high signal scores → HIGH."""
    count = 0
    try:
        from signal_score import compute_signal_score

        cutoff = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
        cursor = await db.execute("""
            SELECT DISTINCT ticker FROM transactions
            WHERE transaction_type = 'buy' AND transaction_date >= ? AND ticker IS NOT NULL
        """, (cutoff,))
        tickers = [row[0] for row in await cursor.fetchall()]

        for ticker in tickers[:30]:  # Limit to avoid slow queries
            try:
                score_data = await compute_signal_score(db, ticker)
                if score_data["signal_score"] >= 70:
                    desc = (f"HIGH SIGNAL: {ticker} has signal score of {score_data['signal_score']:.0f}/100 — "
                            f"{score_data['unique_buyers_14d']} buyers in 14 days")
                    if await _insert_alert(db, "HIGH_SIGNAL", ticker, desc, "HIGH"):
                        count += 1
            except Exception:
                continue
    except ImportError:
        pass

    return count


async def detect_anomalous_trade(db) -> int:
    """Flag transactions where the value is >3x a filer's average → MEDIUM."""
    count = 0
    cutoff = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")

    cursor = await db.execute("""
        SELECT t.id, t.filer_name, t.ticker, t.total_value, t.transaction_type,
               avg_t.avg_value
        FROM transactions t
        JOIN (
            SELECT filer_name, AVG(total_value) as avg_value
            FROM transactions
            WHERE total_value IS NOT NULL
            GROUP BY filer_name
            HAVING COUNT(*) >= 3
        ) avg_t ON t.filer_name = avg_t.filer_name
        WHERE t.transaction_date >= ?
          AND t.total_value IS NOT NULL
          AND t.total_value > avg_t.avg_value * 3
    """, (cutoff,))

    rows = await cursor.fetchall()
    for row in rows:
        tx_id, name, ticker, value, tx_type, avg_val = row
        ratio = value / avg_val if avg_val else 0
        desc = (f"ANOMALOUS TRADE: {name} {tx_type} ${value:,.0f} of {ticker} — "
                f"{ratio:.1f}x their average trade size (${avg_val:,.0f})")
        if await _insert_alert(db, "ANOMALOUS_TRADE", ticker, desc, "MEDIUM", [tx_id]):
            count += 1

    return count
