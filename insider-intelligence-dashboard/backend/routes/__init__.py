from fastapi import APIRouter
from .transactions import router as transactions_router
from .sectors import router as sectors_router
from .alerts_routes import router as alerts_router
from .filers import router as filers_router
from .tickers import router as tickers_router
from .refresh import router as refresh_router

api_router = APIRouter(prefix="/api")
api_router.include_router(transactions_router)
api_router.include_router(sectors_router)
api_router.include_router(alerts_router)
api_router.include_router(filers_router)
api_router.include_router(tickers_router)
api_router.include_router(refresh_router)
