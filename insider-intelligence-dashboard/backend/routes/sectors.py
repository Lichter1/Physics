from fastapi import APIRouter
from database import get_db
from models import SectorSummary, WeeklyTrend
from metrics import compute_sector_metrics
from datetime import datetime, timedelta

router = APIRouter()


@router.get("/sectors/summary", response_model=list[SectorSummary])
async def get_sector_summary():
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT sector, total_buy_value, total_sell_value, unique_filers, transaction_count, last_updated FROM sector_summary ORDER BY total_buy_value DESC"
        )
        rows = await cursor.fetchall()

        summaries = []
        for row in rows:
            buy = row[1] or 0
            sell = row[2] or 0
            ratio = round(buy / sell, 2) if sell > 0 else (10.0 if buy > 0 else 0)
            summaries.append(SectorSummary(
                sector=row[0],
                total_buy_value=buy,
                total_sell_value=sell,
                buy_sell_ratio=ratio,
                unique_filers=row[3] or 0,
                transaction_count=row[4] or 0,
                last_updated=row[5] or datetime.now().isoformat(),
            ))

        # Add computed metrics
        for s in summaries:
            metrics = await compute_sector_metrics(db, s.sector)
            # Attach as extra attributes won't break Pydantic serialization
            # but client can compute from raw data

        return summaries
    finally:
        await db.close()


@router.get("/trends/weekly", response_model=list[WeeklyTrend])
async def get_weekly_trends():
    db = await get_db()
    try:
        trends = []
        now = datetime.now()

        for week_offset in range(4):
            week_end = now - timedelta(days=week_offset * 7)
            week_start = week_end - timedelta(days=7)
            week_label = week_start.strftime("%m/%d")

            cursor = await db.execute("""
                SELECT sector,
                    COALESCE(SUM(CASE WHEN transaction_type='buy' THEN total_value ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN transaction_type='sell' THEN total_value ELSE 0 END), 0),
                    COUNT(*)
                FROM transactions
                WHERE transaction_date BETWEEN ? AND ?
                  AND sector IS NOT NULL
                GROUP BY sector
            """, (week_start.strftime("%Y-%m-%d"), week_end.strftime("%Y-%m-%d")))

            rows = await cursor.fetchall()
            for row in rows:
                buy = row[1]
                sell = row[2]
                # Calculate momentum vs previous week
                prev_start = week_start - timedelta(days=7)
                prev_cursor = await db.execute("""
                    SELECT COALESCE(SUM(CASE WHEN transaction_type='buy' THEN total_value ELSE 0 END), 0)
                    FROM transactions
                    WHERE sector = ? AND transaction_date BETWEEN ? AND ?
                """, (row[0], prev_start.strftime("%Y-%m-%d"), week_start.strftime("%Y-%m-%d")))
                prev_buy = (await prev_cursor.fetchone())[0]
                momentum = round((buy - prev_buy) / prev_buy * 100, 1) if prev_buy > 0 else 0

                trends.append(WeeklyTrend(
                    sector=row[0],
                    week=week_label,
                    buy_value=buy,
                    sell_value=sell,
                    transaction_count=row[3],
                    momentum=momentum,
                ))

        return trends
    finally:
        await db.close()
