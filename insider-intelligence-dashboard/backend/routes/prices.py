from fastapi import APIRouter, Query
from database import get_db
from models import PricePoint
from price_service import (
    fetch_and_store_prices, get_price_range, get_52w_range,
    compute_forward_returns, update_prices_for_tracked_tickers
)
from datetime import datetime, timedelta

router = APIRouter()


@router.get("/prices/{ticker}", response_model=list[PricePoint])
async def get_ticker_prices(
    ticker: str,
    period: str = Query("6m", regex="^(1m|3m|6m|1y|2y)$"),
):
    """Get price history for a ticker."""
    ticker = ticker.upper()
    days_map = {"1m": 30, "3m": 90, "6m": 180, "1y": 365, "2y": 730}
    days = days_map.get(period, 180)

    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    db = await get_db()
    try:
        prices = await get_price_range(db, ticker, start_date, end_date)

        # If no data, try fetching from yfinance
        if not prices:
            await fetch_and_store_prices(ticker, period)
            db2 = await get_db()
            try:
                prices = await get_price_range(db2, ticker, start_date, end_date)
            finally:
                await db2.close()

        return prices
    finally:
        await db.close()


@router.get("/prices/{ticker}/context")
async def get_price_context(ticker: str):
    """Get 52-week range and current price context."""
    ticker = ticker.upper()
    db = await get_db()
    try:
        context = await get_52w_range(db, ticker)
        return context
    finally:
        await db.close()


@router.get("/prices/{ticker}/impact")
async def get_trade_impact(
    ticker: str,
    from_date: str = Query(..., description="Transaction date YYYY-MM-DD"),
):
    """Get forward returns from a specific transaction date."""
    ticker = ticker.upper()
    db = await get_db()
    try:
        returns = await compute_forward_returns(db, ticker, from_date)
        return returns
    finally:
        await db.close()


@router.post("/prices/refresh")
async def refresh_all_prices():
    """Fetch/update prices for all tracked tickers."""
    total = await update_prices_for_tracked_tickers()
    return {"status": "success", "total_price_rows_updated": total}
