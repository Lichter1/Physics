import logging
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from .base import BaseScraper

logger = logging.getLogger(__name__)

SENATE_SEARCH_URL = "https://efts.senate.gov/LATEST/search-index"
SENATE_DISCLOSURES_URL = "https://efts.senate.gov/LATEST/search-disp"


class SenateDisclosureScraper(BaseScraper):
    """Scrape Senate financial disclosures."""

    def __init__(self):
        super().__init__(rate_limit=2.0)

    async def scrape(self, days_back: int = 90) -> list[dict]:
        """Fetch Senate periodic transaction reports."""
        transactions = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        try:
            # Try the Senate eFTS search
            params = {
                "q": "",
                "dateRange": "custom",
                "startdt": start_date.strftime("%m/%d/%Y"),
                "enddt": end_date.strftime("%m/%d/%Y"),
            }

            response = await self.fetch(SENATE_SEARCH_URL, params=params)
            if not response:
                logger.warning("Senate eFTS not accessible, trying alternate URL")
                return await self._scrape_alternate()

            # Try to parse as JSON first
            try:
                data = response.json()
                hits = data.get("hits", {}).get("hits", [])
                for hit in hits[:50]:
                    source = hit.get("_source", {})
                    tx = self._parse_senate_hit(source)
                    if tx:
                        transactions.append(tx)
            except Exception:
                # Parse as HTML
                soup = BeautifulSoup(response.text, "html.parser")
                transactions.extend(self._parse_senate_html(soup))

        except Exception as e:
            logger.error(f"Senate disclosure scraping failed: {e}")

        logger.info(f"Senate Disclosures: scraped {len(transactions)} transactions")
        return transactions

    async def _scrape_alternate(self) -> list[dict]:
        """Fallback scraping method for Senate disclosures."""
        transactions = []
        url = "https://www.senate.gov/legislative/public_disclosure/electronic_filing_search.htm"
        response = await self.fetch(url)
        if not response:
            return transactions

        try:
            soup = BeautifulSoup(response.text, "html.parser")
            transactions.extend(self._parse_senate_html(soup))
        except Exception as e:
            logger.error(f"Senate alternate scraping failed: {e}")

        return transactions

    def _parse_senate_hit(self, source: dict) -> dict | None:
        """Parse a Senate eFTS search result."""
        try:
            filer_name = source.get("first_name", "") + " " + source.get("last_name", "")
            filer_name = filer_name.strip()
            if not filer_name:
                return None

            return {
                "source": "senate_disclosures",
                "filer_name": filer_name,
                "filer_type": "politician",
                "party": source.get("party", None),
                "committee": source.get("committee", None),
                "ticker": source.get("ticker", None),
                "company_name": source.get("asset_description", None),
                "sector": None,
                "transaction_type": "buy" if source.get("type", "").lower() in ("purchase", "buy") else "sell",
                "shares": self._parse_shares(str(source.get("shares", ""))),
                "price": None,
                "total_value": self._parse_amount_range(source.get("amount", "")),
                "transaction_date": source.get("transaction_date", datetime.now().strftime("%Y-%m-%d")),
                "filing_date": source.get("filing_date", datetime.now().strftime("%Y-%m-%d")),
                "delay_days": None,
            }
        except Exception:
            return None

    def _parse_senate_html(self, soup: BeautifulSoup) -> list[dict]:
        """Parse Senate disclosure HTML page."""
        transactions = []
        table = soup.find("table", class_="data-table") or soup.find("table")
        if not table:
            return transactions

        rows = table.find_all("tr")[1:]
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 4:
                continue

            try:
                name = cols[0].get_text(strip=True)
                filing_type = cols[1].get_text(strip=True) if len(cols) > 1 else ""
                date_str = cols[2].get_text(strip=True) if len(cols) > 2 else ""

                if "periodic" not in filing_type.lower() and "transaction" not in filing_type.lower():
                    continue

                transactions.append({
                    "source": "senate_disclosures",
                    "filer_name": name,
                    "filer_type": "politician",
                    "party": None,
                    "committee": None,
                    "ticker": None,
                    "company_name": None,
                    "sector": None,
                    "transaction_type": "buy",
                    "shares": None,
                    "price": None,
                    "total_value": None,
                    "transaction_date": date_str or datetime.now().strftime("%Y-%m-%d"),
                    "filing_date": datetime.now().strftime("%Y-%m-%d"),
                    "delay_days": None,
                })
            except Exception as e:
                logger.debug(f"Error parsing Senate row: {e}")
                continue

        return transactions

    def _parse_amount_range(self, text: str) -> float | None:
        """Parse amount range strings."""
        if not text:
            return None
        text = str(text).replace("$", "").replace(",", "").strip()
        if "-" in text or "–" in text:
            parts = text.replace("–", "-").split("-")
            try:
                low = float(parts[0].strip())
                high = float(parts[1].strip())
                return (low + high) / 2
            except (ValueError, IndexError):
                return None
        return self._parse_value(text)
