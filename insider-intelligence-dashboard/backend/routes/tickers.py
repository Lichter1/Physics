from fastapi import APIRouter
from database import get_db
from models import TickerActivity, Transaction
from metrics import compute_ticker_metrics
from signal_score import compute_signal_score
from price_service import get_52w_range, compute_forward_returns

router = APIRouter()


@router.get("/tickers/{ticker}/activity", response_model=TickerActivity)
async def get_ticker_activity(ticker: str):
    ticker = ticker.upper()
    db = await get_db()
    try:
        # Get company info
        cursor = await db.execute(
            "SELECT name, sector FROM companies WHERE ticker = ?", (ticker,)
        )
        company_row = await cursor.fetchone()
        company_name = company_row[0] if company_row else None
        sector = company_row[1] if company_row else None

        # Get all transactions for this ticker
        cursor = await db.execute(
            "SELECT * FROM transactions WHERE ticker = ? ORDER BY transaction_date DESC",
            (ticker,)
        )
        rows = await cursor.fetchall()

        transactions = []
        col_names = [d[0] for d in cursor.description] if cursor.description else []
        has_classification = "classification" in col_names

        for row in rows:
            tx = Transaction(
                id=row[0], source=row[1], filer_name=row[2], filer_type=row[3],
                party=row[4], committee=row[5], ticker=row[6], company_name=row[7],
                sector=row[8], transaction_type=row[9], shares=row[10], price=row[11],
                total_value=row[12], transaction_date=row[13], filing_date=row[14],
                delay_days=row[15],
                classification=row[16] if has_classification and len(row) > 16 else None,
            )
            transactions.append(tx)

        # Compute metrics
        metrics = await compute_ticker_metrics(db, ticker)

        # Compute signal score
        signal_data = await compute_signal_score(db, ticker)

        # Get price context
        price_data = await get_52w_range(db, ticker)

        # Get forward returns for most recent buy
        forward_returns = None
        recent_buy = next(
            (tx for tx in transactions if tx.transaction_type == "buy"), None
        )
        if recent_buy:
            forward_returns = await compute_forward_returns(
                db, ticker, recent_buy.transaction_date
            )

        return TickerActivity(
            ticker=ticker,
            company_name=company_name,
            sector=sector,
            transactions=transactions,
            signal_score=signal_data["signal_score"],
            signal_components=signal_data["components"],
            price_data=price_data,
            forward_returns=forward_returns,
            **metrics,
        )
    finally:
        await db.close()
