from fastapi import APIRouter, BackgroundTasks
from database import get_db
from models import FilerTrackRecord
from track_record import compute_filer_accuracy, update_all_filer_track_records

router = APIRouter()


@router.get("/filers/{filer_name}/track-record", response_model=FilerTrackRecord)
async def get_filer_track_record(filer_name: str):
    """Get accuracy and return metrics for a specific filer."""
    db = await get_db()
    try:
        # Check cache first
        cursor = await db.execute(
            "SELECT * FROM filer_track_record WHERE filer_name = ?", (filer_name,)
        )
        cached = await cursor.fetchone()
        if cached:
            return dict(cached)

        # Compute on demand
        record = await compute_filer_accuracy(db, filer_name)
        record["filer_name"] = filer_name
        return record
    finally:
        await db.close()


@router.post("/filers/recompute-track-records")
async def recompute_track_records(background_tasks: BackgroundTasks):
    """Recompute track records for all filers (runs in background)."""
    async def _run():
        db = await get_db()
        try:
            await update_all_filer_track_records(db)
        finally:
            await db.close()

    background_tasks.add_task(_run)
    return {"status": "started", "message": "Recomputing all filer track records in background."}
