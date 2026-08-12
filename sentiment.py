"""
Phase 2: NLP Sentiment Stream
==============================
FinBERT sentiment extraction from financial news,
aligned with market data by date and ticker.
"""

import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from newsapi import NewsApiClient
from dotenv import load_dotenv

load_dotenv()

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────
DB_PATH = "sqlite:///data/mmis.db"
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
MODEL_NAME = "ProsusAI/finbert"

TICKER_KEYWORDS = {
    "AAPL": "Apple stock",
    "GOOGL": "Google Alphabet stock",
    "MSFT": "Microsoft stock",
    "TSLA": "Tesla stock",
    "AMZN": "Amazon stock",
    "SPY": "S&P 500 market"
}

# ── Load FinBERT ──────────────────────────────────────────────
def load_finbert():
    logger.info("Loading FinBERT model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    logger.info("✅ FinBERT loaded")
    return tokenizer, model

# ── Get Sentiment Vector ──────────────────────────────────────
def get_sentiment(texts: list, tokenizer, model) -> np.ndarray:
    """Returns mean sentiment vector [negative, neutral, positive]."""
    if not texts:
        return np.array([0.0, 1.0, 0.0])  # Default: neutral

    vectors = []
    for text in texts[:10]:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        )
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze().numpy()
        vectors.append(probs)

    return np.mean(vectors, axis=0)

# ── Fetch News Headlines ──────────────────────────────────────
def fetch_headlines(ticker: str, date: str, client: NewsApiClient) -> list:
    """Fetch headlines for a ticker on a specific date ±1 day window."""
    query = TICKER_KEYWORDS.get(ticker, ticker)
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    from_date = (date_obj - timedelta(days=1)).strftime("%Y-%m-%d")
    to_date = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        response = client.get_everything(
            q=query,
            from_param=from_date,
            to=to_date,
            language="en",
            sort_by="relevancy",
            page_size=10
        )
        headlines = [a["title"] for a in response.get("articles", [])]
        return headlines
    except Exception as e:
        logger.warning(f"News fetch failed for {ticker} on {date}: {e}")
        return []

# ── Build Sentiment Table ─────────────────────────────────────
def run_sentiment_pipeline():
    engine = create_engine(DB_PATH)
    tokenizer, model = load_finbert()
    client = NewsApiClient(api_key=NEWS_API_KEY)

    # Create sentiment table
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sentiment_data (
                date TIMESTAMP,
                ticker TEXT,
                sentiment_neg REAL,
                sentiment_neu REAL,
                sentiment_pos REAL,
                sentiment_score REAL,
                headline_count INTEGER,
                PRIMARY KEY (date, ticker)
            )
        """))
        conn.commit()

    # Load recent dates from market data (last 30 days — free API limit)
    with engine.connect() as conn:
        df = pd.read_sql("""
            SELECT DISTINCT date, ticker
            FROM market_data
            WHERE date >= date('now', '-30 days')
            ORDER BY date DESC
        """, conn)

    logger.info(f"Processing sentiment for {len(df)} date-ticker pairs...")

    results = []
    for _, row in df.iterrows():
        date_str = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
        ticker = row["ticker"]

        headlines = fetch_headlines(ticker, date_str, client)
        sentiment = get_sentiment(headlines, tokenizer, model)

        results.append({
            "date": row["date"],
            "ticker": ticker,
            "sentiment_neg": float(sentiment[0]),
            "sentiment_neu": float(sentiment[1]),
            "sentiment_pos": float(sentiment[2]),
            "sentiment_score": float(sentiment[2] - sentiment[0]),
            "headline_count": len(headlines)
        })

        logger.info(f"  {ticker} {date_str} | score: {float(sentiment[2] - sentiment[0]):.3f} | headlines: {len(headlines)}")
        time.sleep(0.2)  # Rate limiting

    sentiment_df = pd.DataFrame(results)

    # ── Fail-closed guard (see docs/STATE_REPORT.md D8): the write below DROPs
    # and recreates sentiment_data. Its 2026-05-12..2026-06-10 rows came from a
    # NewsAPI free tier serving only ~30 days and can never be re-fetched.
    import sqlite3 as _sqlite3
    try:
        _con = _sqlite3.connect("file:data/mmis.db?mode=ro", uri=True)
        _existing = _con.execute("SELECT COUNT(*) FROM sentiment_data").fetchone()[0]
        _con.close()
    except _sqlite3.Error:
        _existing = 0
    if _existing > 0 and os.environ.get("MMIS_ALLOW_DESTRUCTIVE") != "1":
        raise RuntimeError(
            f"REFUSING TO RUN: would DROP sentiment_data, destroying {_existing} rows spanning "
            "2026-05-12 to 2026-06-10. That NewsAPI free-tier window can no longer be re-fetched, "
            "so the loss is PERMANENT. Set MMIS_ALLOW_DESTRUCTIVE=1 to override."
        )

    sentiment_df.to_sql("sentiment_data", con=engine, if_exists="replace", index=False)

    logger.info(f"\n✅ Sentiment pipeline complete")
    logger.info(f"   Rows saved: {len(sentiment_df)}")
    logger.info(f"   Avg sentiment score: {sentiment_df['sentiment_score'].mean():.4f}")
    logger.info(f"\nSample:")
    print(sentiment_df[["date", "ticker", "sentiment_score", "headline_count"]].head(10))

    return sentiment_df

if __name__ == "__main__":
    run_sentiment_pipeline()
    