"""Filer track record and accuracy computation.

Computes what % of a filer's historical buys led to price gains
at various time horizons (30d, 60d, 90d).
"""

import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


async def compute_filer_accuracy(db, filer_name: str) -> dict:
    """Compute accuracy metrics for a specific filer."""
    cursor = await db.execute("""
        SELECT id, ticker, transaction_date, total_value
        FROM transactions
        WHERE filer_name = ? AND transaction_type = 'buy'
          AND ticker IS NOT NULL AND transaction_date IS NOT NULL
        ORDER BY transaction_date ASC
    """, (filer_name,))
    buys = await cursor.fetchall()

    if not buys:
        return {"total_buys": 0, "tracked_buys": 0, "accuracy_30d": None,
                "accuracy_60d": None, "accuracy_90d": None, "avg_return_30d": None,
                "avg_return_60d": None, "avg_return_90d": None}

    results = {"30d": [], "60d": [], "90d": []}

    for buy in buys:
        tx_id, ticker, tx_date, value = buy
        # Get price at transaction date
        cursor = await db.execute(
            """SELECT close FROM price_history
               WHERE ticker = ? AND date <= ? ORDER BY date DESC LIMIT 1""",
            (ticker, tx_date)
        )
        base_row = await cursor.fetchone()
        if not base_row or not base_row[0]:
            continue

        base_price = base_row[0]

        for label, days in [("30d", 30), ("60d", 60), ("90d", 90)]:
            target_date = (datetime.strptime(tx_date, "%Y-%m-%d") + timedelta(days=days)).strftime("%Y-%m-%d")
            cursor = await db.execute(
                """SELECT close FROM price_history
                   WHERE ticker = ? AND date <= ? ORDER BY date DESC LIMIT 1""",
                (ticker, target_date)
            )
            future_row = await cursor.fetchone()
            if future_row and future_row[0]:
                ret = (future_row[0] - base_price) / base_price * 100
                results[label].append(ret)

    total_buys = len(buys)
    tracked = max(len(results["30d"]), len(results["60d"]), len(results["90d"]))

    def calc_accuracy(returns, threshold=0):
        if not returns:
            return None
        wins = sum(1 for r in returns if r > threshold)
        return round(wins / len(returns) * 100, 1)

    def calc_avg(returns):
        if not returns:
            return None
        return round(sum(returns) / len(returns), 2)

    return {
        "total_buys": total_buys,
        "tracked_buys": tracked,
        "accuracy_30d": calc_accuracy(results["30d"]),
        "accuracy_60d": calc_accuracy(results["60d"]),
        "accuracy_90d": calc_accuracy(results["90d"]),
        "avg_return_30d": calc_avg(results["30d"]),
        "avg_return_60d": calc_avg(results["60d"]),
        "avg_return_90d": calc_avg(results["90d"]),
    }


async def update_all_filer_track_records(db):
    """Recompute track records for all filers and store in filer_track_record table."""
    cursor = await db.execute(
        "SELECT DISTINCT filer_name FROM transactions WHERE transaction_type = 'buy'"
    )
    filers = [row[0] for row in await cursor.fetchall()]

    count = 0
    for filer_name in filers:
        try:
            record = await compute_filer_accuracy(db, filer_name)
            # Use 60d accuracy as the primary metric
            accuracy_pct = record["accuracy_60d"]
            avg_return = record["avg_return_60d"]

            await db.execute("""
                INSERT OR REPLACE INTO filer_track_record
                (filer_name, total_buys, tracked_buys, accuracy_pct, avg_return_pct,
                 accuracy_30d, accuracy_60d, accuracy_90d,
                 avg_return_30d, avg_return_60d, avg_return_90d, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                filer_name, record["total_buys"], record["tracked_buys"],
                accuracy_pct, avg_return,
                record["accuracy_30d"], record["accuracy_60d"], record["accuracy_90d"],
                record["avg_return_30d"], record["avg_return_60d"], record["avg_return_90d"],
                datetime.now().isoformat()
            ))
            count += 1
        except Exception as e:
            logger.debug(f"Track record failed for {filer_name}: {e}")

    await db.commit()
    logger.info(f"Updated track records for {count} filers")
    return count
