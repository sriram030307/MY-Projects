"""
Sentiment & Market Signal Dashboard
Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from ingestion import NewsIngester, StockDataFetcher
from sentiment_model import SentimentAnalyzer, SignalCorrelator

st.set_page_config(page_title="Market Sentiment Dashboard", layout="wide", page_icon="📈")
st.title("📈 Real-Time Sentiment & Market Signal Pipeline")

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()
days = st.sidebar.slider("Days of news", 1, 7, 2)
use_finbert = st.sidebar.checkbox("Use FinBERT (slower but accurate)", value=False)

if st.sidebar.button("🔄 Analyze"):
    with st.spinner("Fetching data and running sentiment analysis..."):
        ingester = NewsIngester()
        fetcher = StockDataFetcher()
        analyzer = SentimentAnalyzer(use_finbert=use_finbert)
        correlator = SignalCorrelator()

        articles = ingester.fetch(ticker, days_back=days)
        articles = analyzer.batch_analyze(articles)
        prices = fetcher.fetch_prices(ticker)
        daily_sentiment = correlator.compute_daily_sentiment(articles)
        correlation = correlator.correlate(daily_sentiment, prices)

    st.session_state["articles"] = articles
    st.session_state["prices"] = prices
    st.session_state["daily_sentiment"] = daily_sentiment
    st.session_state["correlation"] = correlation
    st.session_state["ticker"] = ticker

# ── Display Results ───────────────────────────────────────────
if "articles" in st.session_state:
    articles = st.session_state["articles"]
    prices = st.session_state["prices"]
    correlation = st.session_state["correlation"]
    ticker_name = st.session_state["ticker"]

    # Metrics
    pos = sum(1 for a in articles if a["sentiment_label"] == "positive")
    neg = sum(1 for a in articles if a["sentiment_label"] == "negative")
    neu = sum(1 for a in articles if a["sentiment_label"] == "neutral")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Articles", len(articles))
    col2.metric("🟢 Positive", pos)
    col3.metric("🔴 Negative", neg)
    col4.metric("⚪ Neutral", neu)

    # Sentiment Distribution
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sentiment Distribution")
        fig = px.pie(
            values=[pos, neg, neu],
            names=["Positive", "Negative", "Neutral"],
            color_discrete_map={"Positive": "#00cc96", "Negative": "#ef553b", "Neutral": "#636efa"}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(f"{ticker_name} Price Movement")
        df_prices = pd.DataFrame(prices)
        if not df_prices.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_prices["date"], y=df_prices["pct_change"],
                                  marker_color=["green" if x > 0 else "red" for x in df_prices["pct_change"]]))
            fig.update_layout(yaxis_title="% Change", xaxis_title="Date")
            st.plotly_chart(fig, use_container_width=True)

    # Correlation
    st.subheader("📊 Sentiment ↔ Price Correlation")
    if correlation.get("correlation") is not None:
        col1, col2 = st.columns(2)
        col1.metric("Pearson Correlation", correlation["correlation"])
        col2.metric("Interpretation", correlation["interpretation"])
    else:
        st.info(correlation.get("message", "Not enough data"))

    # News Table
    st.subheader("📰 Recent News with Sentiment")
    df = pd.DataFrame(articles)[["title", "source", "sentiment_label", "sentiment_score", "published_at"]]
    df["sentiment_score"] = df["sentiment_score"].round(3)
    st.dataframe(df, use_container_width=True)

else:
    st.info("👈 Configure your ticker and click **Analyze** to start.")
