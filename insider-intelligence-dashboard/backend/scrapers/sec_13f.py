"""SEC EDGAR 13F filing scraper — institutional holdings (hedge funds, mutual funds).

13F filings are quarterly reports by institutional investment managers
with over $100M AUM. This reveals what major funds are buying/selling.
"""

import logging
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from .base import BaseScraper

logger = logging.getLogger(__name__)

EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"

# Well-known institutional filers to track
NOTABLE_FILERS = {
    "BERKSHIRE HATHAWAY": "Warren Buffett / Berkshire Hathaway",
    "SOROS FUND": "George Soros / Soros Fund",
    "BRIDGEWATER": "Ray Dalio / Bridgewater",
    "RENAISSANCE": "Renaissance Technologies",
    "CITADEL": "Ken Griffin / Citadel",
    "PERSHING SQUARE": "Bill Ackman / Pershing Square",
    "APPALOOSA": "David Tepper / Appaloosa",
    "ICAHN": "Carl Icahn / Icahn Enterprises",
    "DRUCKENMILLER": "Stanley Druckenmiller / Duquesne",
    "DUQUESNE": "Stanley Druckenmiller / Duquesne",
    "TIGER GLOBAL": "Tiger Global Management",
    "COATUE": "Coatue Management",
    "ARK INVEST": "Cathie Wood / ARK Invest",
    "BAUPOST": "Seth Klarman / Baupost Group",
    "GREENLIGHT": "David Einhorn / Greenlight Capital",
    "THIRD POINT": "Dan Loeb / Third Point",
    "ELLIOT": "Paul Singer / Elliott Management",
    "VIKING GLOBAL": "Viking Global Investors",
    "D1 CAPITAL": "D1 Capital Partners",
    "LONE PINE": "Lone Pine Capital",
}


class SEC13FScraper(BaseScraper):
    """Scrape SEC EDGAR 13F filings for institutional holdings changes."""

    def __init__(self):
        super().__init__(rate_limit=1.0)
        self.headers["User-Agent"] = "InsiderIntelligenceDashboard research@example.com"

    async def scrape(self, max_filings: int = 30) -> list[dict]:
        """Fetch recent 13F filings."""
        transactions = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)  # 13F is quarterly

        try:
            params = {
                "q": '"13F"',
                "dateRange": "custom",
                "startdt": start_date.strftime("%Y-%m-%d"),
                "enddt": end_date.strftime("%Y-%m-%d"),
                "forms": "13F-HR",
            }

            response = await self.fetch(EDGAR_SEARCH_URL, params=params)
            if not response:
                logger.warning("EDGAR 13F search returned no response")
                return transactions

            data = response.json()
            hits = data.get("hits", {}).get("hits", [])[:max_filings]

            for hit in hits:
                try:
                    source = hit.get("_source", {})
                    company_name = source.get("display_names", [""])[0] if source.get("display_names") else ""
                    filing_date = source.get("file_date", end_date.strftime("%Y-%m-%d"))
                    file_url = source.get("file_url", "")

                    # Check if this is a notable filer
                    notable_name = None
                    for key, name in NOTABLE_FILERS.items():
                        if key.lower() in company_name.lower():
                            notable_name = name
                            break

                    if not notable_name:
                        continue  # Only track notable institutional filers

                    # Try to parse the 13F information table
                    if file_url:
                        holdings = await self._parse_13f_table(file_url)
                        for holding in holdings:
                            transactions.append({
                                "source": "sec_13f",
                                "filer_name": notable_name,
                                "filer_type": "10% owner",  # Institutional
                                "party": None,
                                "committee": None,
                                "ticker": holding.get("ticker"),
                                "company_name": holding.get("name"),
                                "sector": None,
                                "transaction_type": "buy",  # 13F shows holdings; new position = buy signal
                                "shares": holding.get("shares"),
                                "price": None,
                                "total_value": holding.get("value"),
                                "transaction_date": filing_date[:10],
                                "filing_date": filing_date[:10],
                                "delay_days": 0,
                            })

                except Exception as e:
                    logger.debug(f"Error processing 13F filing: {e}")
                    continue

        except Exception as e:
            logger.error(f"13F scraping failed: {e}")

        logger.info(f"SEC 13F: found {len(transactions)} institutional holdings")
        return transactions

    async def _parse_13f_table(self, url: str) -> list[dict]:
        """Parse a 13F information table for top holdings."""
        response = await self.fetch(url)
        if not response:
            return []

        holdings = []
        try:
            soup = BeautifulSoup(response.text, "lxml-xml")

            # 13F info tables can be XML or HTML
            entries = soup.find_all("infoTable") or soup.find_all("infotable")

            for entry in entries[:50]:  # Top 50 holdings only
                try:
                    name_tag = entry.find("nameOfIssuer") or entry.find("nameofissuer")
                    name = name_tag.get_text().strip() if name_tag else ""

                    # Try to find CUSIP and map to ticker (simplified)
                    cusip_tag = entry.find("cusip")
                    cusip = cusip_tag.get_text().strip() if cusip_tag else ""

                    value_tag = entry.find("value")
                    value = float(value_tag.get_text().strip()) * 1000 if value_tag else None  # 13F values are in thousands

                    shares_tag = entry.find("sshPrnamt") or entry.find("sshprnamt")
                    shares = float(shares_tag.get_text().strip().replace(",", "")) if shares_tag else None

                    # We'll store the company name; ticker lookup happens during insert
                    if name and value:
                        holdings.append({
                            "name": name,
                            "ticker": None,  # Would need CUSIP-to-ticker mapping
                            "value": value,
                            "shares": shares,
                        })

                except Exception as e:
                    logger.debug(f"Error parsing 13F entry: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error parsing 13F table: {e}")

        return holdings
