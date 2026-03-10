import httpx
import asyncio
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    """Base scraper with rate limiting, retry logic, and error handling."""

    def __init__(self, rate_limit: float = 1.0, max_retries: int = 3):
        self.rate_limit = rate_limit  # seconds between requests
        self.max_retries = max_retries
        self._last_request_time = 0
        self.headers = {
            "User-Agent": "InsiderIntelligenceDashboard research@example.com",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

    async def _rate_limit_wait(self):
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_request_time
        if elapsed < self.rate_limit:
            await asyncio.sleep(self.rate_limit - elapsed)
        self._last_request_time = asyncio.get_event_loop().time()

    async def fetch(self, url: str, params: dict = None) -> httpx.Response | None:
        """Fetch a URL with rate limiting and retry logic."""
        for attempt in range(self.max_retries):
            try:
                await self._rate_limit_wait()
                async with httpx.AsyncClient(
                    headers=self.headers,
                    timeout=30.0,
                    follow_redirects=True
                ) as client:
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    return response
            except httpx.HTTPStatusError as e:
                logger.warning(f"HTTP {e.response.status_code} for {url} (attempt {attempt + 1})")
                if e.response.status_code == 429:
                    await asyncio.sleep(2 ** (attempt + 1))
                elif e.response.status_code >= 500:
                    await asyncio.sleep(2 ** attempt)
                else:
                    break
            except (httpx.ConnectError, httpx.ReadTimeout) as e:
                logger.warning(f"Connection error for {url}: {e} (attempt {attempt + 1})")
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Unexpected error fetching {url}: {e}")
                break

        logger.error(f"Failed to fetch {url} after {self.max_retries} attempts")
        return None

    @abstractmethod
    async def scrape(self) -> list[dict]:
        """Scrape data and return list of transaction dicts."""
        pass

    def _parse_value(self, text: str) -> float | None:
        """Parse a dollar value string to float."""
        if not text:
            return None
        text = text.strip().replace("$", "").replace(",", "")
        try:
            return float(text)
        except ValueError:
            return None

    def _parse_shares(self, text: str) -> float | None:
        """Parse a share count string to float."""
        if not text:
            return None
        text = text.strip().replace(",", "")
        try:
            return float(text)
        except ValueError:
            return None
