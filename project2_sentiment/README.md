# 📈 Project 2: Real-Time Sentiment & Market Signal Pipeline

## Overview
Streams financial news, runs FinBERT sentiment analysis, and correlates signals with stock price movements in a live dashboard.

## Features
- NewsAPI financial article ingestion
- FinBERT (or lightweight) sentiment classification
- yfinance stock price fetching
- Pearson correlation between sentiment and price change
- Interactive Streamlit dashboard

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add: NEWS_API_KEY=your_key (get free key at newsapi.org)

streamlit run dashboard.py
```

## Project Structure
```
project2_sentiment/
├── ingestion.py         # News + stock data fetching
├── sentiment_model.py   # FinBERT sentiment + correlation
├── dashboard.py         # Streamlit UI
├── requirements.txt
└── README.md
```

## Resume Line
> Engineered a real-time NLP pipeline using FinBERT to process financial articles and correlate sentiment signals with stock price movements.
