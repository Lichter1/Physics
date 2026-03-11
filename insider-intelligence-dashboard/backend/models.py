from pydantic import BaseModel
from typing import Optional


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
    classification: Optional[str] = None
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
    accuracy_pct: Optional[float] = None
    avg_return_pct: Optional[float] = None


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
    signal_score: Optional[float] = None
    signal_components: Optional[dict] = None
    price_data: Optional[dict] = None
    forward_returns: Optional[dict] = None


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


class TradeIdea(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    sector: Optional[str] = None
    signal_score: float
    components: dict
    thesis: str
    unique_buyers: int
    total_buy_value: float
    current_price: Optional[float] = None


class FilerTrackRecord(BaseModel):
    filer_name: str
    total_buys: int
    tracked_buys: int
    accuracy_pct: Optional[float] = None
    avg_return_pct: Optional[float] = None
    accuracy_30d: Optional[float] = None
    accuracy_60d: Optional[float] = None
    accuracy_90d: Optional[float] = None
    avg_return_30d: Optional[float] = None
    avg_return_60d: Optional[float] = None
    avg_return_90d: Optional[float] = None


class PricePoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class BackfillStatus(BaseModel):
    source: str
    status: str
    months_requested: Optional[int] = None
    transactions_added: int = 0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
