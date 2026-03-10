from fastapi import APIRouter, Query
from typing import Optional
from database import get_db
from models import TransactionList, Transaction

router = APIRouter()


@router.get("/transactions", response_model=TransactionList)
async def get_transactions(
    sector: Optional[str] = None,
    filer_type: Optional[str] = None,
    transaction_type: Optional[str] = None,
    ticker: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    min_value: Optional[float] = None,
    limit: int = Query(default=50, le=500),
    offset: int = Query(default=0, ge=0),
):
    db = await get_db()
    try:
        conditions = []
        params = []

        if sector:
            conditions.append("sector = ?")
            params.append(sector)
        if filer_type:
            conditions.append("filer_type = ?")
            params.append(filer_type)
        if transaction_type:
            conditions.append("transaction_type = ?")
            params.append(transaction_type)
        if ticker:
            conditions.append("ticker = ?")
            params.append(ticker.upper())
        if date_from:
            conditions.append("transaction_date >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("transaction_date <= ?")
            params.append(date_to)
        if min_value:
            conditions.append("total_value >= ?")
            params.append(min_value)

        where = " WHERE " + " AND ".join(conditions) if conditions else ""

        # Get total count
        cursor = await db.execute(f"SELECT COUNT(*) FROM transactions{where}", params)
        total = (await cursor.fetchone())[0]

        # Get paginated results
        cursor = await db.execute(
            f"SELECT * FROM transactions{where} ORDER BY transaction_date DESC LIMIT ? OFFSET ?",
            params + [limit, offset]
        )
        rows = await cursor.fetchall()

        transactions = []
        for row in rows:
            tx = Transaction(
                id=row[0], source=row[1], filer_name=row[2], filer_type=row[3],
                party=row[4], committee=row[5], ticker=row[6], company_name=row[7],
                sector=row[8], transaction_type=row[9], shares=row[10], price=row[11],
                total_value=row[12], transaction_date=row[13], filing_date=row[14],
                delay_days=row[15]
            )

            # Add flags
            flags = []
            if tx.delay_days and tx.delay_days > 30:
                flags.append("LATE_FILING")
            if tx.total_value and tx.total_value > 500000 and tx.filer_type != "politician":
                flags.append("LARGE_TRADE")
            tx.flags = flags

            transactions.append(tx)

        return TransactionList(transactions=transactions, total=total, limit=limit, offset=offset)
    finally:
        await db.close()
