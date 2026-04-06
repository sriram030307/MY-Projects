"""
Project 2: Real-Time Sentiment & Market Signal Pipeline
Ingestion layer — fetches financial news and processes it
"""

import os
import json
import time
import logging
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsIngester:
    """Fetches financial news from NewsAPI."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NEWS_API_KEY", "demo")
        self.base_url = "https://newsapi.org/v2/everything"

    def fetch(self, query: str, days_back: int = 1, max_articles: int = 50) -> List[Dict]:
        from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        params = {
            "q": query,
            "from": from_date,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": max_articles,
            "apiKey": self.api_key
        }
        try:
            res = requests.get(self.base_url, params=params, timeout=10)
            res.raise_for_status()
            articles = res.json().get("articles", [])
            logger.info(f"Fetched {len(articles)} articles for '{query}'")
            return [
                {
                    "title": a["title"],
                    "description": a.get("description", ""),
                    "content": a.get("content", ""),
                    "published_at": a["publishedAt"],
                    "source": a["source"]["name"],
                    "url": a["url"],
                    "ticker": query.upper()
                }
                for a in articles if a.get("title")
            ]
        except Exception as e:
            logger.warning(f"News fetch failed: {e}. Returning sample data.")
            return self._sample_data(query)

    def _sample_data(self, ticker: str) -> List[Dict]:
        """Returns sample data when API is unavailable."""
        samples = [
            f"{ticker} reports record quarterly earnings beating analyst expectations.",
            f"Analysts upgrade {ticker} stock to buy following strong revenue growth.",
            f"{ticker} faces regulatory scrutiny over data privacy practices.",
            f"Market volatility impacts {ticker} amid broader economic concerns.",
            f"{ticker} announces new product launch, stock surges in after-hours trading.",
        ]
        return [
            {
                "title": s,
                "description": s,
                "content": s,
                "published_at": datetime.utcnow().isoformat(),
                "source": "Sample News",
                "url": "",
                "ticker": ticker
            }
            for s in samples
        ]


class StockDataFetcher:
    """Fetches stock price data using yfinance."""

    def fetch_prices(self, ticker: str, period: str = "5d") -> List[Dict]:
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            return [
                {
                    "date": str(idx.date()),
                    "open": round(row["Open"], 2),
                    "close": round(row["Close"], 2),
                    "volume": int(row["Volume"]),
                    "pct_change": round((row["Close"] - row["Open"]) / row["Open"] * 100, 2)
                }
                for idx, row in hist.iterrows()
            ]
        except Exception as e:
            logger.warning(f"Stock fetch failed: {e}. Returning sample prices.")
            return [
                {"date": str((datetime.utcnow() - timedelta(days=i)).date()),
                 "open": 150.0 + i, "close": 152.0 + i,
                 "volume": 1000000, "pct_change": round(1.3 - i * 0.1, 2)}
                for i in range(5)
            ]


if __name__ == "__main__":
    ingester = NewsIngester()
    fetcher = StockDataFetcher()

    articles = ingester.fetch("AAPL", days_back=1)
    prices = fetcher.fetch_prices("AAPL")

    print(f"\nFetched {len(articles)} articles")
    print(f"Sample: {articles[0]['title'][:80]}")
    print(f"\nStock prices (last 5 days):")
    for p in prices:
        print(f"  {p['date']}: close={p['close']}, change={p['pct_change']}%")
