from fastapi import APIRouter, Query, BackgroundTasks
from models import BackfillStatus
from backfill import run_backfill, get_backfill_status

router = APIRouter()


@router.post("/backfill")
async def start_backfill(
    background_tasks: BackgroundTasks,
    months: int = Query(24, ge=1, le=24),
    source: str = Query("all"),
):
    """Start a historical backfill job (runs in background)."""
    background_tasks.add_task(run_backfill, months, source)
    return {"status": "started", "months": months, "source": source,
            "message": "Backfill running in background. Check /api/backfill/status for progress."}


@router.get("/backfill/status", response_model=list[BackfillStatus])
async def backfill_status():
    """Get the current backfill job status."""
    return await get_backfill_status()
