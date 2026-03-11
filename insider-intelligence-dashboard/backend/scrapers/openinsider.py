import logging
from datetime import datetime
from bs4 import BeautifulSoup
from .base import BaseScraper

logger = logging.getLogger(__name__)

OPENINSIDER_URL = "http://openinsider.com/screener"


class OpenInsiderScraper(BaseScraper):
    """Fallback scraper using OpenInsider public data."""

    def __init__(self):
        super().__init__(rate_limit=3.0)  # Be extra respectful

    async def scrape(self, min_value: int = 100000, max_results: int = 100) -> list[dict]:
        """Scrape OpenInsider for recent insider transactions."""
        transactions = []

        try:
            params = {
                "s": "",
                "o": "",
                "pl": "",
                "ph": "",
                "ll": "",
                "lh": "",
                "fd": "30",  # Last 30 days
                "fdr": "",
                "td": "0",
                "tdr": "",
                "fdlyl": "",
                "fdlyh": "",
                "dtefrom": "",
                "dteto": "",
                "xp": "1",  # Exclude planned trades
                "vl": str(min_value),
                "vh": "",
                "ocl": "",
                "och": "",
                "session_id": "",
                "cnt": str(max_results),
                "page": "1",
            }

            response = await self.fetch(OPENINSIDER_URL, params=params)
            if not response:
                logger.warning("OpenInsider not accessible")
                return transactions

            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table", class_="tinytable")
            if not table:
                logger.info("No data table found on OpenInsider")
                return transactions

            rows = table.find_all("tr")[1:]  # Skip header

            for row in rows:
                cols = row.find_all("td")
                if len(cols) < 12:
                    continue

                try:
                    filing_date = cols[1].get_text(strip=True)
                    trade_date = cols[2].get_text(strip=True)
                    ticker = cols[3].get_text(strip=True).upper()
                    company_name = cols[4].get_text(strip=True)
                    insider_name = cols[5].get_text(strip=True)
                    title = cols[6].get_text(strip=True).lower()
                    trade_type = cols[7].get_text(strip=True)
                    price = self._parse_value(cols[8].get_text(strip=True))
                    qty = self._parse_shares(cols[9].get_text(strip=True))
                    owned = cols[10].get_text(strip=True)
                    value = self._parse_value(cols[11].get_text(strip=True))

                    # Determine filer type from title
                    filer_type = "officer"
                    if "ceo" in title or "chief exec" in title:
                        filer_type = "ceo"
                    elif "dir" in title:
                        filer_type = "director"
                    elif "10%" in title:
                        filer_type = "10% owner"

                    tx_type = "buy" if "P" in trade_type or "purchase" in trade_type.lower() else "sell"

                    # Calculate delay
                    delay = None
                    try:
                        td = datetime.strptime(trade_date, "%Y-%m-%d")
                        fd = datetime.strptime(filing_date, "%Y-%m-%d")
                        delay = (fd - td).days
                    except ValueError:
                        pass

                    transactions.append({
                        "source": "openinsider",
                        "filer_name": insider_name,
                        "filer_type": filer_type,
                        "party": None,
                        "committee": None,
                        "ticker": ticker,
                        "company_name": company_name,
                        "sector": None,
                        "transaction_type": tx_type,
                        "shares": qty,
                        "price": price,
                        "total_value": value,
                        "transaction_date": trade_date,
                        "filing_date": filing_date,
                        "delay_days": delay,
                    })

                except Exception as e:
                    logger.debug(f"Error parsing OpenInsider row: {e}")
                    continue

        except Exception as e:
            logger.error(f"OpenInsider scraping failed: {e}")

        logger.info(f"OpenInsider: scraped {len(transactions)} transactions")
        return transactions

    async def scrape_date_range(self, start_date: str, end_date: str,
                                 min_value: int = 100000, max_results: int = 200) -> list[dict]:
        """Scrape OpenInsider for a specific date range (for historical backfill)."""
        transactions = []

        try:
            params = {
                "s": "", "o": "", "pl": "", "ph": "", "ll": "", "lh": "",
                "fd": "0",
                "fdr": f"{start_date}+-+{end_date}",
                "td": "0", "tdr": "",
                "fdlyl": "", "fdlyh": "",
                "dtefrom": start_date, "dteto": end_date,
                "xp": "1", "vl": str(min_value), "vh": "",
                "ocl": "", "och": "", "session_id": "",
                "cnt": str(max_results), "page": "1",
            }

            response = await self.fetch(OPENINSIDER_URL, params=params)
            if not response:
                return transactions

            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table", class_="tinytable")
            if not table:
                return transactions

            rows = table.find_all("tr")[1:]
            for row in rows:
                cols = row.find_all("td")
                if len(cols) < 12:
                    continue
                try:
                    filing_date = cols[1].get_text(strip=True)
                    trade_date = cols[2].get_text(strip=True)
                    ticker = cols[3].get_text(strip=True).upper()
                    company_name = cols[4].get_text(strip=True)
                    insider_name = cols[5].get_text(strip=True)
                    title = cols[6].get_text(strip=True).lower()
                    trade_type = cols[7].get_text(strip=True)
                    price = self._parse_value(cols[8].get_text(strip=True))
                    qty = self._parse_shares(cols[9].get_text(strip=True))
                    value = self._parse_value(cols[11].get_text(strip=True))

                    filer_type = "officer"
                    if "ceo" in title or "chief exec" in title:
                        filer_type = "ceo"
                    elif "dir" in title:
                        filer_type = "director"
                    elif "10%" in title:
                        filer_type = "10% owner"

                    tx_type = "buy" if "P" in trade_type or "purchase" in trade_type.lower() else "sell"

                    delay = None
                    try:
                        td = datetime.strptime(trade_date, "%Y-%m-%d")
                        fd = datetime.strptime(filing_date, "%Y-%m-%d")
                        delay = (fd - td).days
                    except ValueError:
                        pass

                    transactions.append({
                        "source": "openinsider",
                        "filer_name": insider_name, "filer_type": filer_type,
                        "party": None, "committee": None,
                        "ticker": ticker, "company_name": company_name,
                        "sector": None, "transaction_type": tx_type,
                        "shares": qty, "price": price, "total_value": value,
                        "transaction_date": trade_date, "filing_date": filing_date,
                        "delay_days": delay,
                    })
                except Exception as e:
                    logger.debug(f"Error parsing backfill row: {e}")
                    continue

        except Exception as e:
            logger.error(f"OpenInsider backfill failed: {e}")

        logger.info(f"OpenInsider backfill ({start_date} to {end_date}): {len(transactions)} transactions")
        return transactions
