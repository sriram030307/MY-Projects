"""
Sentiment Analysis using FinBERT + correlation with stock price movement
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """FinBERT-based financial sentiment analyzer."""

    def __init__(self, use_finbert: bool = True):
        self.model = None
        self.tokenizer = None
        self.use_finbert = use_finbert
        self._load_model()

    def _load_model(self):
        try:
            from transformers import pipeline
            model_name = "ProsusAI/finbert" if self.use_finbert else "distilbert-base-uncased-finetuned-sst-2-english"
            logger.info(f"Loading model: {model_name}")
            self.pipe = pipeline("text-classification", model=model_name, truncation=True, max_length=512)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.warning(f"Model load failed ({e}). Using rule-based fallback.")
            self.pipe = None

    def _rule_based_sentiment(self, text: str) -> Dict:
        """Simple keyword-based fallback."""
        positive_words = ["surge", "beat", "upgrade", "growth", "record", "strong", "gain", "profit"]
        negative_words = ["scrutiny", "concern", "decline", "loss", "regulatory", "volatility", "risk"]
        text_lower = text.lower()
        pos = sum(1 for w in positive_words if w in text_lower)
        neg = sum(1 for w in negative_words if w in text_lower)
        if pos > neg:
            return {"label": "positive", "score": min(0.5 + pos * 0.1, 0.95)}
        elif neg > pos:
            return {"label": "negative", "score": min(0.5 + neg * 0.1, 0.95)}
        return {"label": "neutral", "score": 0.6}

    def analyze(self, text: str) -> Dict:
        """Returns sentiment label and confidence score."""
        if not text or len(text.strip()) < 5:
            return {"label": "neutral", "score": 0.5}

        if self.pipe:
            try:
                result = self.pipe(text[:512])[0]
                label = result["label"].lower()
                if label not in ["positive", "negative", "neutral"]:
                    label = "neutral"
                return {"label": label, "score": round(result["score"], 4)}
            except Exception as e:
                logger.warning(f"Inference failed: {e}")

        return self._rule_based_sentiment(text)

    def batch_analyze(self, articles: List[Dict]) -> List[Dict]:
        """Adds sentiment fields to each article."""
        results = []
        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}".strip()
            sentiment = self.analyze(text)
            results.append({
                **article,
                "sentiment_label": sentiment["label"],
                "sentiment_score": sentiment["score"],
                "sentiment_numeric": {"positive": 1, "neutral": 0, "negative": -1}[sentiment["label"]]
            })
        return results


class SignalCorrelator:
    """Correlates NLP sentiment signals with price movements."""

    def compute_daily_sentiment(self, articles: List[Dict]) -> Dict[str, float]:
        """Aggregates sentiment scores by date."""
        from collections import defaultdict
        daily = defaultdict(list)
        for a in articles:
            date = a.get("published_at", "")[:10]
            daily[date].append(a.get("sentiment_numeric", 0))
        return {date: round(np.mean(scores), 4) for date, scores in daily.items()}

    def correlate(self, sentiment_by_date: Dict[str, float], prices: List[Dict]) -> Dict:
        """Pearson correlation between sentiment and next-day price change."""
        price_map = {p["date"]: p["pct_change"] for p in prices}
        dates = sorted(sentiment_by_date.keys())

        paired_sentiment, paired_price = [], []
        for date in dates:
            if date in price_map:
                paired_sentiment.append(sentiment_by_date[date])
                paired_price.append(price_map[date])

        if len(paired_sentiment) < 2:
            return {"correlation": None, "n_pairs": len(paired_sentiment), "message": "Not enough data"}

        correlation = float(np.corrcoef(paired_sentiment, paired_price)[0, 1])
        return {
            "correlation": round(correlation, 4),
            "n_pairs": len(paired_sentiment),
            "interpretation": (
                "Strong positive" if correlation > 0.5 else
                "Moderate positive" if correlation > 0.2 else
                "Weak/no correlation" if correlation > -0.2 else
                "Negative correlation"
            )
        }


if __name__ == "__main__":
    from ingestion import NewsIngester, StockDataFetcher

    ingester = NewsIngester()
    fetcher = StockDataFetcher()
    analyzer = SentimentAnalyzer(use_finbert=False)
    correlator = SignalCorrelator()

    articles = ingester.fetch("AAPL")
    articles_with_sentiment = analyzer.batch_analyze(articles)
    prices = fetcher.fetch_prices("AAPL")

    print("\n=== Sentiment Results ===")
    for a in articles_with_sentiment[:3]:
        print(f"  [{a['sentiment_label'].upper():8s}] {a['title'][:70]}")

    daily_sentiment = correlator.compute_daily_sentiment(articles_with_sentiment)
    correlation = correlator.correlate(daily_sentiment, prices)
    print(f"\n=== Correlation ===")
    print(f"  Sentiment ↔ Price: {correlation}")
