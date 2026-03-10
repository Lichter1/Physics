import logging
from datetime import datetime
from bs4 import BeautifulSoup
from .base import BaseScraper

logger = logging.getLogger(__name__)

HOUSE_URL = "https://disclosures-clerk.house.gov/FinancialDisclosure/ViewMemberSearchResult"


class HouseDisclosureScraper(BaseScraper):
    """Scrape House of Representatives financial disclosures."""

    def __init__(self):
        super().__init__(rate_limit=2.0)

    async def scrape(self, year: int = None) -> list[dict]:
        """Fetch House periodic transaction reports."""
        if year is None:
            year = datetime.now().year

        transactions = []

        try:
            # POST to search for periodic transaction reports
            params = {
                "FilingYear": str(year),
                "State": "",
                "District": "",
                "FilingType": "P",  # Periodic Transaction Reports
            }

            response = await self.fetch(
                f"https://disclosures-clerk.house.gov/FinancialDisclosure",
                params=params
            )
            if not response:
                logger.warning("House disclosures page not accessible")
                return transactions

            soup = BeautifulSoup(response.text, "html.parser")

            # Parse the results table
            table = soup.find("table", class_="library-table") or soup.find("table")
            if not table:
                logger.info("No results table found on House disclosures page")
                return transactions

            rows = table.find_all("tr")[1:]  # Skip header

            for row in rows:
                try:
                    cols = row.find_all("td")
                    if len(cols) < 5:
                        continue

                    member_name = cols[0].get_text(strip=True)
                    office = cols[1].get_text(strip=True) if len(cols) > 1 else ""
                    filing_year = cols[2].get_text(strip=True) if len(cols) > 2 else str(year)
                    filing_type = cols[3].get_text(strip=True) if len(cols) > 3 else ""

                    # Try to get the PDF/detail link
                    link = cols[0].find("a") or row.find("a")
                    detail_url = link.get("href", "") if link else ""

                    if detail_url and "pdf" not in detail_url.lower():
                        detail_txs = await self._parse_detail_page(detail_url, member_name, office)
                        transactions.extend(detail_txs)
                    else:
                        # Create a basic transaction entry from the row
                        state = ""
                        party = ""
                        if office:
                            parts = office.split("-")
                            if len(parts) >= 1:
                                state = parts[0].strip()

                        transactions.append({
                            "source": "house_disclosures",
                            "filer_name": member_name,
                            "filer_type": "politician",
                            "party": party or None,
                            "committee": None,
                            "ticker": None,
                            "company_name": None,
                            "sector": None,
                            "transaction_type": "buy",
                            "shares": None,
                            "price": None,
                            "total_value": None,
                            "transaction_date": f"{filing_year}-01-01",
                            "filing_date": datetime.now().strftime("%Y-%m-%d"),
                            "delay_days": None,
                        })

                except Exception as e:
                    logger.debug(f"Error parsing House disclosure row: {e}")
                    continue

        except Exception as e:
            logger.error(f"House disclosures scraping failed: {e}")

        logger.info(f"House Disclosures: scraped {len(transactions)} transactions")
        return transactions

    async def _parse_detail_page(self, url: str, member_name: str, office: str) -> list[dict]:
        """Parse a disclosure detail page for individual transactions."""
        if not url.startswith("http"):
            url = f"https://disclosures-clerk.house.gov{url}"

        response = await self.fetch(url)
        if not response:
            return []

        transactions = []
        try:
            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table")
            if not table:
                return transactions

            rows = table.find_all("tr")[1:]
            for row in rows:
                cols = row.find_all("td")
                if len(cols) < 6:
                    continue

                try:
                    asset_desc = cols[1].get_text(strip=True) if len(cols) > 1 else ""
                    tx_type_text = cols[2].get_text(strip=True).lower() if len(cols) > 2 else ""
                    tx_date = cols[3].get_text(strip=True) if len(cols) > 3 else ""
                    amount_range = cols[4].get_text(strip=True) if len(cols) > 4 else ""

                    # Try to extract ticker from asset description (usually in parentheses)
                    ticker = None
                    if "(" in asset_desc and ")" in asset_desc:
                        potential_ticker = asset_desc.split("(")[-1].split(")")[0].strip()
                        if potential_ticker.isalpha() and len(potential_ticker) <= 5:
                            ticker = potential_ticker.upper()

                    tx_type = "buy" if "purchase" in tx_type_text or "buy" in tx_type_text else "sell"

                    # Parse amount range
                    total_value = self._parse_amount_range(amount_range)

                    transactions.append({
                        "source": "house_disclosures",
                        "filer_name": member_name,
                        "filer_type": "politician",
                        "party": None,
                        "committee": None,
                        "ticker": ticker,
                        "company_name": asset_desc if not ticker else None,
                        "sector": None,
                        "transaction_type": tx_type,
                        "shares": None,
                        "price": None,
                        "total_value": total_value,
                        "transaction_date": tx_date or datetime.now().strftime("%Y-%m-%d"),
                        "filing_date": datetime.now().strftime("%Y-%m-%d"),
                        "delay_days": None,
                    })
                except Exception as e:
                    logger.debug(f"Error parsing transaction detail: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error parsing House detail page: {e}")

        return transactions

    def _parse_amount_range(self, text: str) -> float | None:
        """Parse House disclosure amount ranges like '$1,001 - $15,000'."""
        if not text:
            return None
        text = text.replace("$", "").replace(",", "").strip()
        if "-" in text or "–" in text:
            parts = text.replace("–", "-").split("-")
            try:
                low = float(parts[0].strip())
                high = float(parts[1].strip())
                return (low + high) / 2
            except (ValueError, IndexError):
                return None
        return self._parse_value(text)
