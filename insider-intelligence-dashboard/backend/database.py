import aiosqlite
import os
from datetime import datetime, timedelta
import random

DB_PATH = os.path.join(os.path.dirname(__file__), "insider_intel.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    filer_name TEXT NOT NULL,
    filer_type TEXT NOT NULL CHECK(filer_type IN ('politician','ceo','director','officer','10% owner')),
    party TEXT,
    committee TEXT,
    ticker TEXT,
    company_name TEXT,
    sector TEXT,
    transaction_type TEXT NOT NULL CHECK(transaction_type IN ('buy','sell','exchange')),
    shares REAL,
    price REAL,
    total_value REAL,
    transaction_date TEXT NOT NULL,
    filing_date TEXT NOT NULL,
    delay_days INTEGER,
    classification TEXT CHECK(classification IN ('conviction','routine','suspicious')),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS companies (
    ticker TEXT PRIMARY KEY,
    name TEXT,
    sector TEXT,
    industry TEXT,
    market_cap REAL
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_type TEXT NOT NULL,
    ticker TEXT,
    description TEXT NOT NULL,
    triggered_at TEXT DEFAULT CURRENT_TIMESTAMP,
    severity TEXT NOT NULL CHECK(severity IN ('HIGH','MEDIUM','LOW')),
    related_transaction_ids TEXT
);

CREATE TABLE IF NOT EXISTS sector_summary (
    sector TEXT PRIMARY KEY,
    total_buy_value REAL DEFAULT 0,
    total_sell_value REAL DEFAULT 0,
    unique_filers INTEGER DEFAULT 0,
    transaction_count INTEGER DEFAULT 0,
    last_updated TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS price_history (
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS filer_track_record (
    filer_name TEXT PRIMARY KEY,
    total_buys INTEGER DEFAULT 0,
    tracked_buys INTEGER DEFAULT 0,
    accuracy_pct REAL,
    avg_return_pct REAL,
    accuracy_30d REAL,
    accuracy_60d REAL,
    accuracy_90d REAL,
    avg_return_30d REAL,
    avg_return_60d REAL,
    avg_return_90d REAL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS backfill_status (
    source TEXT PRIMARY KEY,
    status TEXT DEFAULT 'pending',
    months_requested INTEGER,
    transactions_added INTEGER DEFAULT 0,
    started_at TEXT DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_transactions_ticker ON transactions(ticker);
CREATE INDEX IF NOT EXISTS idx_transactions_filer ON transactions(filer_name);
CREATE INDEX IF NOT EXISTS idx_transactions_sector ON transactions(sector);
CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(transaction_date);
CREATE INDEX IF NOT EXISTS idx_transactions_type ON transactions(transaction_type);
CREATE INDEX IF NOT EXISTS idx_transactions_class ON transactions(classification);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(alert_type);
CREATE INDEX IF NOT EXISTS idx_price_ticker_date ON price_history(ticker, date);
"""


async def get_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    return db


async def init_db():
    db = await get_db()
    try:
        await db.executescript(SCHEMA)
        await db.commit()
        # Check if we need seed data
        cursor = await db.execute("SELECT COUNT(*) FROM transactions")
        count = (await cursor.fetchone())[0]
        if count == 0:
            await seed_demo_data(db)
    finally:
        await db.close()


async def seed_demo_data(db: aiosqlite.Connection):
    """Populate with realistic demo data so dashboard works immediately."""
    sectors = [
        "Defense & Aerospace", "AI & Semiconductors", "Energy",
        "Healthcare & Pharma", "Financials & Banking", "Technology",
        "Real Estate", "Consumer Discretionary", "Industrials"
    ]

    tickers_by_sector = {
        "Defense & Aerospace": [("LMT", "Lockheed Martin"), ("RTX", "RTX Corp"), ("NOC", "Northrop Grumman"), ("GD", "General Dynamics")],
        "AI & Semiconductors": [("NVDA", "NVIDIA Corp"), ("AMD", "AMD Inc"), ("AVGO", "Broadcom"), ("INTC", "Intel Corp"), ("TSM", "Taiwan Semi")],
        "Energy": [("XOM", "Exxon Mobil"), ("CVX", "Chevron"), ("COP", "ConocoPhillips"), ("NEE", "NextEra Energy")],
        "Healthcare & Pharma": [("JNJ", "Johnson & Johnson"), ("PFE", "Pfizer"), ("UNH", "UnitedHealth"), ("ABBV", "AbbVie")],
        "Financials & Banking": [("JPM", "JPMorgan Chase"), ("BAC", "Bank of America"), ("GS", "Goldman Sachs"), ("MS", "Morgan Stanley")],
        "Technology": [("AAPL", "Apple Inc"), ("MSFT", "Microsoft"), ("GOOGL", "Alphabet"), ("META", "Meta Platforms"), ("AMZN", "Amazon")],
        "Real Estate": [("AMT", "American Tower"), ("PLD", "Prologis"), ("SPG", "Simon Property")],
        "Consumer Discretionary": [("TSLA", "Tesla Inc"), ("NKE", "Nike Inc"), ("SBUX", "Starbucks"), ("MCD", "McDonalds")],
        "Industrials": [("CAT", "Caterpillar"), ("BA", "Boeing"), ("HON", "Honeywell"), ("UPS", "UPS Inc")],
    }

    politicians = [
        ("Nancy Pelosi", "D", "Financial Services"),
        ("Dan Crenshaw", "R", "Energy and Commerce"),
        ("Tommy Tuberville", "R", "Armed Services"),
        ("Mark Green", "R", "Homeland Security"),
        ("Josh Gottheimer", "D", "Financial Services"),
        ("Michael McCaul", "R", "Foreign Affairs"),
        ("Ro Khanna", "D", "Armed Services"),
        ("Virginia Foxx", "R", "Education and the Workforce"),
        ("Debbie Wasserman Schultz", "D", "Appropriations"),
        ("Pat Fallon", "R", "Armed Services"),
    ]

    executives = [
        ("Jensen Huang", "ceo"), ("Lisa Su", "ceo"), ("Tim Cook", "ceo"),
        ("Satya Nadella", "ceo"), ("Jamie Dimon", "ceo"), ("Mary Barra", "ceo"),
        ("David Solomon", "ceo"), ("Brian Moynihan", "ceo"),
        ("John Smith", "director"), ("Jane Williams", "director"),
        ("Robert Chen", "officer"), ("Sarah Johnson", "officer"),
        ("Michael Brown", "10% owner"), ("Emily Davis", "10% owner"),
    ]

    # Insert companies
    for sector, tickers in tickers_by_sector.items():
        for ticker, name in tickers:
            await db.execute(
                "INSERT OR IGNORE INTO companies (ticker, name, sector, industry, market_cap) VALUES (?, ?, ?, ?, ?)",
                (ticker, name, sector, sector, random.uniform(10e9, 3e12))
            )

    # Generate transactions over last 60 days
    now = datetime.now()
    transactions = []

    for _ in range(300):
        is_politician = random.random() < 0.4
        sector = random.choice(sectors)
        ticker, company = random.choice(tickers_by_sector[sector])
        tx_date = now - timedelta(days=random.randint(0, 60))
        delay = random.randint(1, 45)
        filing_date = tx_date + timedelta(days=delay)
        tx_type = random.choices(["buy", "sell"], weights=[0.6, 0.4])[0]
        shares = random.randint(100, 50000)
        price = random.uniform(20, 800)
        total_value = shares * price

        if is_politician:
            name, party, committee = random.choice(politicians)
            filer_type = "politician"
            source = random.choice(["house_disclosures", "senate_disclosures"])
            # Politicians report in ranges
            total_value = random.choice([15001, 50001, 100001, 250001, 500001, 1000001])
        else:
            name, filer_type = random.choice(executives)
            party = None
            committee = None
            source = "sec_edgar"
            total_value = round(total_value, 2)

        transactions.append((
            source, name, filer_type, party, committee, ticker, company,
            sector, tx_type, shares, round(price, 2), total_value,
            tx_date.strftime("%Y-%m-%d"), filing_date.strftime("%Y-%m-%d"), delay
        ))

    await db.executemany(
        """INSERT INTO transactions
        (source, filer_name, filer_type, party, committee, ticker, company_name,
         sector, transaction_type, shares, price, total_value,
         transaction_date, filing_date, delay_days)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        transactions
    )

    # Update sector summaries
    for sector in sectors:
        cursor = await db.execute(
            """SELECT
                COALESCE(SUM(CASE WHEN transaction_type='buy' THEN total_value ELSE 0 END), 0),
                COALESCE(SUM(CASE WHEN transaction_type='sell' THEN total_value ELSE 0 END), 0),
                COUNT(DISTINCT filer_name),
                COUNT(*)
            FROM transactions WHERE sector = ?""",
            (sector,)
        )
        row = await cursor.fetchone()
        await db.execute(
            """INSERT OR REPLACE INTO sector_summary
            (sector, total_buy_value, total_sell_value, unique_filers, transaction_count, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (sector, row[0], row[1], row[2], row[3], datetime.now().isoformat())
        )

    await db.commit()
