from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class Transaction(BaseModel):
    id: int
    source: str
    filer_name: str
    filer_type: str
    party: Optional[str] = None
    committee: Optional[str] = None
    ticker: Optional[str] = None
    company_name: Optional[str] = None
    sector: Optional[str] = None
    transaction_type: str
    shares: Optional[float] = None
    price: Optional[float] = None
    total_value: Optional[float] = None
    transaction_date: str
    filing_date: str
    delay_days: Optional[int] = None
    flags: list[str] = []


class TransactionList(BaseModel):
    transactions: list[Transaction]
    total: int
    limit: int
    offset: int


class SectorSummary(BaseModel):
    sector: str
    total_buy_value: float
    total_sell_value: float
    buy_sell_ratio: float
    unique_filers: int
    transaction_count: int
    last_updated: str


class Alert(BaseModel):
    id: int
    alert_type: str
    ticker: Optional[str] = None
    description: str
    triggered_at: str
    severity: str
    related_transaction_ids: Optional[str] = None


class FilerLeaderboard(BaseModel):
    filer_name: str
    filer_type: str
    party: Optional[str] = None
    total_transactions: int
    total_buy_value: float
    total_sell_value: float
    unique_tickers: int
    favorite_sector: Optional[str] = None


class TickerActivity(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    sector: Optional[str] = None
    transactions: list[Transaction]
    cluster_score: float
    conviction_score: float
    committee_exposure: int
    delay_score: float
    buy_sell_ratio: float
    total_buy_value: float
    total_sell_value: float


class WeeklyTrend(BaseModel):
    sector: str
    week: str
    buy_value: float
    sell_value: float
    transaction_count: int
    momentum: float


class RefreshResponse(BaseModel):
    status: str
    message: str
    transactions_added: int = 0
    alerts_generated: int = 0
