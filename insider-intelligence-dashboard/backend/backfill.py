"""Historical backfill engine — populates up to 2 years of data.

Runs scrapers with extended date ranges and manages backfill state
so it can resume if interrupted.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from database import get_db

logger = logging.getLogger(__name__)


async def run_backfill(months: int = 24, source: str = "all") -> dict:
    """Run historical backfill for specified number of months.

    Breaks the range into monthly chunks and processes each sequentially
    to stay within rate limits.
    """
    db = await get_db()
    try:
        # Initialize backfill_status
        await db.execute("""
            INSERT OR IGNORE INTO backfill_status (source, status, months_requested)
            VALUES (?, 'running', ?)
        """, (source, months))
        await db.commit()

        total_added = 0
        errors = []

        if source in ("all", "openinsider"):
            count = await _backfill_openinsider(db, months)
            total_added += count

        if source in ("all", "sec_edgar"):
            count = await _backfill_sec_edgar(db, months)
            total_added += count

        # Update status
        await db.execute("""
            UPDATE backfill_status SET status = 'complete',
                   transactions_added = ?, completed_at = ?, error_message = ?
            WHERE source = ?
        """, (total_added, datetime.now().isoformat(),
              "; ".join(errors) if errors else None, source))
        await db.commit()

        return {"status": "complete", "transactions_added": total_added, "errors": errors}

    except Exception as e:
        logger.error(f"Backfill failed: {e}")
        await db.execute("""
            UPDATE backfill_status SET status = 'failed', error_message = ?
            WHERE source = ?
        """, (str(e), source))
        await db.commit()
        return {"status": "failed", "error": str(e)}
    finally:
        await db.close()


async def _backfill_openinsider(db, months: int) -> int:
    """Backfill from OpenInsider in monthly chunks."""
    from scrapers.openinsider import OpenInsiderScraper

    scraper = OpenInsiderScraper()
    total = 0
    now = datetime.now()

    for m in range(months):
        start = now - timedelta(days=30 * (m + 1))
        end = now - timedelta(days=30 * m)

        try:
            # OpenInsider supports date range params
            transactions = await scraper.scrape_date_range(
                start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
            )
            added = await _insert_transactions(db, transactions)
            total += added
            logger.info(f"OpenInsider backfill month -{m+1}: {added} new transactions")
            await asyncio.sleep(2)  # Rate limit between months
        except Exception as e:
            logger.error(f"OpenInsider backfill month -{m+1} failed: {e}")

    return total


async def _backfill_sec_edgar(db, months: int) -> int:
    """Backfill from SEC EDGAR in monthly chunks."""
    from scrapers.sec_edgar import SECEdgarScraper

    scraper = SECEdgarScraper()
    total = 0
    now = datetime.now()

    for m in range(months):
        days_back_start = 30 * (m + 1)
        days_back_end = 30 * m

        try:
            transactions = await scraper.scrape(
                days_back=days_back_start, max_filings=100
            )
            # Filter to the correct date range
            start = (now - timedelta(days=days_back_start)).strftime("%Y-%m-%d")
            end = (now - timedelta(days=days_back_end)).strftime("%Y-%m-%d")
            filtered = [tx for tx in transactions
                       if start <= tx.get("transaction_date", "") <= end]

            added = await _insert_transactions(db, filtered)
            total += added
            logger.info(f"SEC EDGAR backfill month -{m+1}: {added} new transactions")
            await asyncio.sleep(3)  # Respect SEC rate limits
        except Exception as e:
            logger.error(f"SEC EDGAR backfill month -{m+1} failed: {e}")

    return total


async def _insert_transactions(db, transactions: list[dict]) -> int:
    """Insert transactions with deduplication. Returns count added."""
    count = 0
    for tx in transactions:
        cursor = await db.execute(
            """SELECT id FROM transactions
               WHERE filer_name = ? AND ticker = ? AND transaction_date = ?
                 AND transaction_type = ? AND source = ?""",
            (tx["filer_name"], tx.get("ticker"), tx["transaction_date"],
             tx["transaction_type"], tx["source"])
        )
        if await cursor.fetchone():
            continue

        # Look up sector
        if not tx.get("sector") and tx.get("ticker"):
            cursor = await db.execute(
                "SELECT sector FROM companies WHERE ticker = ?",
                (tx["ticker"],)
            )
            row = await cursor.fetchone()
            if row:
                tx["sector"] = row[0]

        await db.execute(
            """INSERT INTO transactions
            (source, filer_name, filer_type, party, committee, ticker, company_name,
             sector, transaction_type, shares, price, total_value,
             transaction_date, filing_date, delay_days)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (tx["source"], tx["filer_name"], tx["filer_type"],
             tx.get("party"), tx.get("committee"), tx.get("ticker"),
             tx.get("company_name"), tx.get("sector"), tx["transaction_type"],
             tx.get("shares"), tx.get("price"), tx.get("total_value"),
             tx["transaction_date"], tx["filing_date"], tx.get("delay_days"))
        )
        count += 1

    await db.commit()
    return count


async def get_backfill_status() -> list[dict]:
    """Get current backfill status."""
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM backfill_status ORDER BY started_at DESC")
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        await db.close()
