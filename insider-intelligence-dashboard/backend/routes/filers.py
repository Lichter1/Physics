from fastapi import APIRouter, Query
from database import get_db
from models import FilerLeaderboard

router = APIRouter()


@router.get("/filers/leaderboard", response_model=list[FilerLeaderboard])
async def get_filer_leaderboard(
    limit: int = Query(default=20, le=100),
    filer_type: str = None,
):
    db = await get_db()
    try:
        conditions = []
        params = []

        if filer_type:
            conditions.append("filer_type = ?")
            params.append(filer_type)

        where = " WHERE " + " AND ".join(conditions) if conditions else ""

        cursor = await db.execute(f"""
            SELECT
                t.filer_name,
                t.filer_type,
                t.party,
                COUNT(*) as total_transactions,
                COALESCE(SUM(CASE WHEN t.transaction_type='buy' THEN t.total_value ELSE 0 END), 0) as total_buy,
                COALESCE(SUM(CASE WHEN t.transaction_type='sell' THEN t.total_value ELSE 0 END), 0) as total_sell,
                COUNT(DISTINCT t.ticker) as unique_tickers,
                (SELECT sector FROM transactions t2 WHERE t2.filer_name = t.filer_name
                 GROUP BY sector ORDER BY COUNT(*) DESC LIMIT 1) as fav_sector,
                ftr.accuracy_pct,
                ftr.avg_return_pct
            FROM transactions t
            LEFT JOIN filer_track_record ftr ON t.filer_name = ftr.filer_name
            {where}
            GROUP BY t.filer_name
            ORDER BY (total_buy + total_sell) DESC
            LIMIT ?
        """, params + [limit])

        rows = await cursor.fetchall()
        return [
            FilerLeaderboard(
                filer_name=row[0], filer_type=row[1], party=row[2],
                total_transactions=row[3], total_buy_value=row[4],
                total_sell_value=row[5], unique_tickers=row[6],
                favorite_sector=row[7],
                accuracy_pct=row[8], avg_return_pct=row[9],
            )
            for row in rows
        ]
    finally:
        await db.close()
