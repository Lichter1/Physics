from fastapi import APIRouter, Query
from database import get_db
from models import TradeIdea
from signal_score import generate_trade_ideas

router = APIRouter()


@router.get("/trade-ideas", response_model=list[TradeIdea])
async def get_trade_ideas(
    min_score: float = Query(20, ge=0, le=100),
    sector: str = Query(None),
    limit: int = Query(30, le=100),
):
    """Get ranked trade ideas based on Insider Signal Score."""
    db = await get_db()
    try:
        ideas = await generate_trade_ideas(db, min_score=min_score, limit=limit)
        if sector:
            ideas = [i for i in ideas if i.get("sector") == sector]
        return ideas
    finally:
        await db.close()
