from fastapi import APIRouter
from database import get_db
from models import RefreshResponse
from scrapers import SECEdgarScraper, HouseDisclosureScraper, SenateDisclosureScraper, OpenInsiderScraper
from alerts import generate_all_alerts
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


async def run_scrapers_and_store() -> dict:
    """Run all scrapers and store results in the database."""
    db = await get_db()
    total_added = 0

    scrapers = [
        ("SEC EDGAR", SECEdgarScraper()),
        ("House Disclosures", HouseDisclosureScraper()),
        ("Senate Disclosures", SenateDisclosureScraper()),
        ("OpenInsider", OpenInsiderScraper()),
    ]

    try:
        for name, scraper in scrapers:
            try:
                logger.info(f"Running {name} scraper...")
                transactions = await scraper.scrape()

                for tx in transactions:
                    # Skip if we already have this transaction
                    cursor = await db.execute(
                        """SELECT id FROM transactions
                           WHERE filer_name = ? AND ticker = ? AND transaction_date = ?
                             AND transaction_type = ? AND source = ?""",
                        (tx["filer_name"], tx.get("ticker"), tx["transaction_date"],
                         tx["transaction_type"], tx["source"])
                    )
                    if await cursor.fetchone():
                        continue

                    # Look up sector from companies table if not provided
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
                    total_added += 1

                await db.commit()
                logger.info(f"{name}: added {len(transactions)} transactions")

            except Exception as e:
                logger.error(f"{name} scraper failed: {e}")
                continue

        # Update sector summaries
        await _update_sector_summaries(db)

        # Generate alerts
        alerts_count = await generate_all_alerts(db)

        return {"transactions_added": total_added, "alerts_generated": alerts_count}

    finally:
        await db.close()


async def _update_sector_summaries(db):
    """Recompute sector summary table."""
    from datetime import datetime

    cursor = await db.execute("""
        SELECT sector,
            COALESCE(SUM(CASE WHEN transaction_type='buy' THEN total_value ELSE 0 END), 0),
            COALESCE(SUM(CASE WHEN transaction_type='sell' THEN total_value ELSE 0 END), 0),
            COUNT(DISTINCT filer_name),
            COUNT(*)
        FROM transactions
        WHERE sector IS NOT NULL
        GROUP BY sector
    """)
    rows = await cursor.fetchall()

    for row in rows:
        await db.execute(
            """INSERT OR REPLACE INTO sector_summary
            (sector, total_buy_value, total_sell_value, unique_filers, transaction_count, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (row[0], row[1], row[2], row[3], row[4], datetime.now().isoformat())
        )
    await db.commit()


@router.post("/refresh", response_model=RefreshResponse)
async def refresh_data():
    """Trigger a manual data refresh."""
    try:
        result = await run_scrapers_and_store()
        return RefreshResponse(
            status="success",
            message=f"Refresh complete. Added {result['transactions_added']} transactions, generated {result['alerts_generated']} alerts.",
            transactions_added=result["transactions_added"],
            alerts_generated=result["alerts_generated"],
        )
    except Exception as e:
        logger.error(f"Refresh failed: {e}")
        return RefreshResponse(
            status="error",
            message=f"Refresh failed: {str(e)}",
        )
