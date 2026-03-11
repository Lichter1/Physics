import logging
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from .base import BaseScraper

logger = logging.getLogger(__name__)

# SEC EDGAR full-text search for Form 4 filings
EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
EDGAR_FILING_URL = "https://www.sec.gov/Archives/edgar/data"


class SECEdgarScraper(BaseScraper):
    """Scrape SEC EDGAR Form 4 filings for insider transactions."""

    def __init__(self):
        super().__init__(rate_limit=1.0)
        self.headers["User-Agent"] = "InsiderIntelligenceDashboard research@example.com"

    async def scrape(self, days_back: int = 30, max_filings: int = 50) -> list[dict]:
        """Fetch recent Form 4 filings from EDGAR full-text search."""
        transactions = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        try:
            # Use EDGAR full-text search API
            params = {
                "q": '"form 4"',
                "dateRange": "custom",
                "startdt": start_date.strftime("%Y-%m-%d"),
                "enddt": end_date.strftime("%Y-%m-%d"),
                "forms": "4",
            }

            response = await self.fetch(EDGAR_SEARCH_URL, params=params)
            if not response:
                logger.warning("EDGAR search returned no response, using EDGAR filing index")
                return await self._scrape_filing_index(days_back, max_filings)

            data = response.json()
            hits = data.get("hits", {}).get("hits", [])[:max_filings]

            for hit in hits:
                try:
                    source = hit.get("_source", {})
                    filing_url = source.get("file_url", "")
                    if filing_url:
                        tx = await self._parse_form4_filing(filing_url, source)
                        if tx:
                            transactions.extend(tx)
                except Exception as e:
                    logger.debug(f"Error parsing EDGAR hit: {e}")
                    continue

        except Exception as e:
            logger.error(f"EDGAR search failed: {e}")
            return await self._scrape_filing_index(days_back, max_filings)

        logger.info(f"SEC EDGAR: scraped {len(transactions)} transactions")
        return transactions

    async def _scrape_filing_index(self, days_back: int, max_filings: int) -> list[dict]:
        """Fallback: scrape EDGAR recent filings index."""
        transactions = []
        url = "https://www.sec.gov/cgi-bin/browse-edgar"
        params = {
            "action": "getcurrent",
            "type": "4",
            "dateb": "",
            "owner": "include",
            "count": str(min(max_filings, 40)),
            "search_text": "",
            "start": "0",
            "output": "atom",
        }

        response = await self.fetch(url, params=params)
        if not response:
            return transactions

        try:
            soup = BeautifulSoup(response.text, "lxml-xml")
            entries = soup.find_all("entry")

            for entry in entries[:max_filings]:
                try:
                    title = entry.find("title")
                    if not title:
                        continue
                    title_text = title.get_text()

                    # Parse "4 - Company Name (Ticker) (Filer Name)"
                    link_tag = entry.find("link")
                    filing_href = link_tag.get("href", "") if link_tag else ""

                    updated = entry.find("updated")
                    filing_date = updated.get_text()[:10] if updated else datetime.now().strftime("%Y-%m-%d")

                    # Try to get the actual filing document
                    if filing_href:
                        doc_txs = await self._parse_form4_page(filing_href, filing_date)
                        transactions.extend(doc_txs)

                except Exception as e:
                    logger.debug(f"Error parsing EDGAR entry: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error parsing EDGAR index: {e}")

        return transactions

    async def _parse_form4_filing(self, url: str, metadata: dict) -> list[dict]:
        """Parse a Form 4 XML filing document."""
        response = await self.fetch(url)
        if not response:
            return []

        transactions = []
        try:
            soup = BeautifulSoup(response.text, "lxml-xml")

            # Get reporting owner
            owner = soup.find("reportingOwner")
            owner_name = ""
            if owner:
                name_tag = owner.find("rptOwnerName")
                owner_name = name_tag.get_text().strip() if name_tag else ""

            # Get issuer info
            issuer = soup.find("issuer")
            ticker = ""
            company_name = ""
            if issuer:
                ticker_tag = issuer.find("issuerTradingSymbol")
                ticker = ticker_tag.get_text().strip().upper() if ticker_tag else ""
                name_tag = issuer.find("issuerName")
                company_name = name_tag.get_text().strip() if name_tag else ""

            # Determine filer type from relationship
            relationship = soup.find("reportingOwnerRelationship")
            filer_type = "officer"
            if relationship:
                if relationship.find("isDirector") and relationship.find("isDirector").get_text() == "1":
                    filer_type = "director"
                elif relationship.find("isOfficer") and relationship.find("isOfficer").get_text() == "1":
                    title_tag = relationship.find("officerTitle")
                    title = title_tag.get_text().lower() if title_tag else ""
                    filer_type = "ceo" if "ceo" in title or "chief executive" in title else "officer"
                elif relationship.find("isTenPercentOwner") and relationship.find("isTenPercentOwner").get_text() == "1":
                    filer_type = "10% owner"

            # Parse non-derivative transactions
            for tx in soup.find_all("nonDerivativeTransaction"):
                try:
                    date_tag = tx.find("transactionDate")
                    tx_date = date_tag.find("value").get_text() if date_tag else ""

                    code_tag = tx.find("transactionCode")
                    tx_code = code_tag.get_text().strip() if code_tag else ""
                    tx_type = "buy" if tx_code in ("P", "A") else "sell" if tx_code in ("S", "D") else "exchange"

                    shares_tag = tx.find("transactionShares")
                    shares = self._parse_shares(shares_tag.find("value").get_text()) if shares_tag else None

                    price_tag = tx.find("transactionPricePerShare")
                    price = self._parse_value(price_tag.find("value").get_text()) if price_tag and price_tag.find("value") else None

                    total_value = (shares * price) if shares and price else None

                    filing_date = metadata.get("file_date", datetime.now().strftime("%Y-%m-%d"))
                    delay = None
                    if tx_date and filing_date:
                        try:
                            td = datetime.strptime(tx_date, "%Y-%m-%d")
                            fd = datetime.strptime(filing_date[:10], "%Y-%m-%d")
                            delay = (fd - td).days
                        except ValueError:
                            pass

                    transactions.append({
                        "source": "sec_edgar",
                        "filer_name": owner_name,
                        "filer_type": filer_type,
                        "party": None,
                        "committee": None,
                        "ticker": ticker,
                        "company_name": company_name,
                        "sector": None,
                        "transaction_type": tx_type,
                        "shares": shares,
                        "price": price,
                        "total_value": total_value,
                        "transaction_date": tx_date,
                        "filing_date": filing_date[:10] if filing_date else "",
                        "delay_days": delay,
                    })
                except Exception as e:
                    logger.debug(f"Error parsing transaction element: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error parsing Form 4 XML: {e}")

        return transactions

    async def _parse_form4_page(self, url: str, filing_date: str) -> list[dict]:
        """Parse a Form 4 filing detail page to find the XML document link."""
        response = await self.fetch(url)
        if not response:
            return []

        try:
            soup = BeautifulSoup(response.text, "html.parser")
            # Look for the XML document link
            for link in soup.find_all("a"):
                href = link.get("href", "")
                if href.endswith(".xml") and "primary_doc" not in href:
                    xml_url = f"https://www.sec.gov{href}" if href.startswith("/") else href
                    return await self._parse_form4_filing(xml_url, {"file_date": filing_date})
        except Exception as e:
            logger.debug(f"Error finding Form 4 XML link: {e}")

        return []
