"""Transaction classifier — Smart Money vs Routine.

Classifies each transaction as:
- conviction: Open-market, discretionary, significant size
- routine: Planned trades (10b5-1), option exercises, small amounts
- suspicious: Unusual timing, late filing, pre-event
"""

import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Thresholds
MIN_CONVICTION_VALUE = 100_000  # $100k+ for conviction
LATE_FILING_DAYS = 10           # >10 days = suspicious
VERY_LATE_FILING_DAYS = 30      # >30 days = very suspicious


def classify_transaction(tx: dict) -> str:
    """Classify a single transaction. Returns 'conviction', 'routine', or 'suspicious'."""
    value = tx.get("total_value") or 0
    delay = tx.get("delay_days")
    filer_type = tx.get("filer_type", "")
    tx_type = tx.get("transaction_type", "")

    # Exchange transactions are typically option exercises = routine
    if tx_type == "exchange":
        return "routine"

    # Check for suspicious indicators
    suspicious_flags = 0
    if delay is not None and delay > VERY_LATE_FILING_DAYS:
        suspicious_flags += 2
    elif delay is not None and delay > LATE_FILING_DAYS:
        suspicious_flags += 1

    if suspicious_flags >= 2:
        return "suspicious"

    # Conviction criteria
    if tx_type == "buy":
        if value >= MIN_CONVICTION_VALUE:
            return "conviction"
        if filer_type in ("ceo", "10% owner") and value >= 50_000:
            return "conviction"
        if filer_type == "politician" and value >= 50_000:
            return "conviction"

    if tx_type == "sell":
        if value >= 500_000:
            return "conviction"
        if filer_type in ("ceo", "10% owner") and value >= 250_000:
            return "conviction"

    # Everything else is routine
    return "routine"


async def classify_all_transactions(db):
    """Classify all unclassified transactions and update the classification column."""
    cursor = await db.execute("""
        SELECT id, transaction_type, filer_type, total_value, delay_days
        FROM transactions WHERE classification IS NULL
    """)
    rows = await cursor.fetchall()

    count = 0
    for row in rows:
        tx = {
            "transaction_type": row[1],
            "filer_type": row[2],
            "total_value": row[3],
            "delay_days": row[4],
        }
        classification = classify_transaction(tx)
        await db.execute(
            "UPDATE transactions SET classification = ? WHERE id = ?",
            (classification, row[0])
        )
        count += 1

    await db.commit()
    logger.info(f"Classified {count} transactions")
    return count
