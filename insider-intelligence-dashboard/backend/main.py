import sys
import os
import logging

# Add backend dir to path so imports work
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from database import init_db
from routes import api_router
from routes.refresh import run_scrapers_and_store
from alerts import generate_all_alerts
from database import get_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()


async def scheduled_refresh():
    """Background job: scrape data and generate alerts every 6 hours."""
    logger.info("Scheduled refresh starting...")
    try:
        result = await run_scrapers_and_store()
        logger.info(f"Scheduled refresh complete: {result}")
    except Exception as e:
        logger.error(f"Scheduled refresh failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing database...")
    await init_db()

    # Generate initial alerts from seed data
    db = await get_db()
    try:
        await generate_all_alerts(db)
    finally:
        await db.close()

    # Start scheduler
    scheduler.add_job(scheduled_refresh, "interval", hours=6, id="data_refresh")
    scheduler.start()
    logger.info("Scheduler started (6-hour refresh cycle)")

    yield

    # Shutdown
    scheduler.shutdown()
    logger.info("Scheduler stopped")


app = FastAPI(
    title="Insider Intelligence Dashboard API",
    description="Aggregates and analyzes insider trading and political stock trading data",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/")
async def root():
    return {
        "name": "Insider Intelligence Dashboard API",
        "version": "1.0.0",
        "disclaimer": "For informational purposes only. Not financial advice.",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
