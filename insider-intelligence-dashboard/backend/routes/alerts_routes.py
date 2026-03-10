from fastapi import APIRouter, Query
from typing import Optional
from database import get_db
from models import Alert

router = APIRouter()


@router.get("/alerts", response_model=list[Alert])
async def get_alerts(
    severity: Optional[str] = None,
    alert_type: Optional[str] = None,
    ticker: Optional[str] = None,
    limit: int = Query(default=50, le=200),
):
    db = await get_db()
    try:
        conditions = []
        params = []

        if severity:
            conditions.append("severity = ?")
            params.append(severity.upper())
        if alert_type:
            conditions.append("alert_type = ?")
            params.append(alert_type.upper())
        if ticker:
            conditions.append("ticker = ?")
            params.append(ticker.upper())

        where = " WHERE " + " AND ".join(conditions) if conditions else ""

        cursor = await db.execute(
            f"""SELECT id, alert_type, ticker, description, triggered_at, severity, related_transaction_ids
                FROM alerts{where}
                ORDER BY
                    CASE severity WHEN 'HIGH' THEN 1 WHEN 'MEDIUM' THEN 2 WHEN 'LOW' THEN 3 END,
                    triggered_at DESC
                LIMIT ?""",
            params + [limit]
        )
        rows = await cursor.fetchall()

        return [
            Alert(
                id=row[0], alert_type=row[1], ticker=row[2],
                description=row[3], triggered_at=row[4],
                severity=row[5], related_transaction_ids=row[6]
            )
            for row in rows
        ]
    finally:
        await db.close()
