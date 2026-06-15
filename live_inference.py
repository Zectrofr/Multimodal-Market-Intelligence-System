"""
live_inference.py — Layer 6: on-demand, live single-ticker prediction
=====================================================================
Enter a ticker → fetch the latest OHLCV (yfinance) → compute indicators →
render a fresh candlestick chart → live EfficientNet-B0 visual features →
live FinBERT sentiment on current headlines → HMM regime → cross-modal fusion →
MC-Dropout (mean + uncertainty) → isotonic-calibrated probability.

This is the serving path that REPLACES the frozen-feature/DB path (inference.py).
All transforms reuse the exact artifacts fit during training:
  - models/best_fusion_model.pt   (fusion network)
  - models/feature_scalers.pkl    (market+visual StandardScalers, temperature)
  - models/calibrator.pkl         (isotonic P(up) calibration)
  - models/hmm_<TICKER>.pkl        (per-ticker regime HMM)

⚠️ Honest disclaimer surfaced on every prediction: out-of-sample this model has
no demonstrated directional edge (≈ coin-flip). For research / decision-support
and as an interpretability demo only — NOT financial advice.
"""

import os
import logging
import pickle
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from ta.trend import MACD, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

import yfinance as yf

from fusion import (
    CrossModalAttention, prepare_features,
    SENTIMENT_DIM, VISUAL_DIM, FUSION_DIM, NUM_HEADS, DROPOUT, SCALER_PATH,
)
from inference import mc_dropout_inference, MODEL_PATH, CALIBRATOR_PATH, MC_PASSES
import vision
import sentiment as senti
from regime import MarketRegimeHMM

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

CLASS_NAMES = {0: "DOWN", 1: "FLAT", 2: "UP"}
SUPPORTED_TICKERS = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "SPY"]
DISCLAIMER = ("Research / interpretability demo only — NOT financial advice. "
              "Out-of-sample this model shows no directional edge (~coin-flip).")


def compute_indicators_live(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the same technical indicators as ingest.add_indicators, but keep
    the most recent row (ingest drops it because next-day target is NaN). No
    lookahead: every feature uses only data up to and including its own row."""
    df = df.copy()
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    df["rsi_14"] = RSIIndicator(close=close, window=14).rsi()
    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"], df["macd_signal"], df["macd_diff"] = macd.macd(), macd.macd_signal(), macd.macd_diff()
    df["ema_20"] = EMAIndicator(close=close, window=20).ema_indicator()
    df["ema_50"] = EMAIndicator(close=close, window=50).ema_indicator()

    bb = BollingerBands(close=close, window=20, window_dev=2)
    df["bb_upper"], df["bb_lower"], df["bb_mid"] = bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_mavg()
    df["bb_bandwidth"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    df["atr_14"] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()

    df["returns_1d"] = close.pct_change(1)
    df["returns_5d"] = close.pct_change(5)
    df["returns_10d"] = close.pct_change(10)
    df["returns_20d"] = close.pct_change(20)
    df["volume_ma_20"] = volume.rolling(20).mean()
    df["volume_ratio"] = volume / df["volume_ma_20"]
    df["volume_change_1d"] = volume.pct_change(1)

    df["body_size"] = abs(close - df["open"])
    df["upper_shadow"] = high - df[["close", "open"]].max(axis=1)
    df["lower_shadow"] = df[["close", "open"]].min(axis=1) - low
    df["high_low_range"] = high - low

    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Drop rows where indicators are still warming up (NOT the last row).
    feature_cols = ["rsi_14", "macd", "ema_50", "bb_upper", "atr_14", "returns_20d"]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    return df


class LiveMMIS:
    """Loads every model once and serves on-demand predictions. Reused across
    API requests so the heavy models (EfficientNet, FinBERT, fusion) load once."""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        logger.info("Loading models for live inference...")

        # Scalers + temperature + calibrator
        with open(SCALER_PATH, "rb") as f:
            self.scalers = pickle.load(f)
        self.temperature = float(self.scalers.get("temperature", 1.0))
        with open(CALIBRATOR_PATH, "rb") as f:
            self.calibrator = pickle.load(f)

        # Fusion model (market_dim = 31 numeric + 3 regime one-hot = 34)
        self.fusion = CrossModalAttention(
            market_dim=34, sentiment_dim=SENTIMENT_DIM, visual_dim=VISUAL_DIM,
            fusion_dim=FUSION_DIM, num_heads=NUM_HEADS, dropout=DROPOUT)
        self.fusion.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.fusion.to(self.device)

        # Vision + NLP encoders
        self.efficientnet = vision.load_efficientnet().to(self.device)
        self.img_transform = vision.get_transform()
        self.finbert_tok, self.finbert = senti.load_finbert()

        # NewsAPI client (optional — graceful neutral fallback)
        self.news_client = None
        if senti.NEWS_API_KEY:
            try:
                from newsapi import NewsApiClient
                self.news_client = NewsApiClient(api_key=senti.NEWS_API_KEY)
            except Exception as e:
                logger.warning(f"NewsAPI client init failed: {e}")
        logger.info("✅ Live MMIS ready")

    # ── data ────────────────────────────────────────────────────
    def _fetch(self, ticker: str) -> pd.DataFrame:
        start = (datetime.today() - timedelta(days=420)).strftime("%Y-%m-%d")
        df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        if df is None or len(df) == 0:
            raise ValueError(f"No price data returned for '{ticker}'.")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df = df.dropna()
        df["ticker"] = ticker
        df = df.reset_index()
        df.rename(columns={df.columns[0]: "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def _regime(self, ticker: str, df: pd.DataFrame) -> str:
        path = Path(f"models/hmm_{ticker}.pkl")
        if not path.exists():
            return "trending"
        try:
            hmm = MarketRegimeHMM.load(str(path))
            return hmm.label(df)["regime"].iloc[-1]
        except Exception as e:
            logger.warning(f"Regime labelling failed ({e}); defaulting to 'trending'.")
            return "trending"

    def _sentiment(self, ticker: str) -> tuple:
        headlines = []
        if self.news_client is not None:
            today = datetime.today().strftime("%Y-%m-%d")
            headlines = senti.fetch_headlines(ticker, today, self.news_client)
        vec = senti.get_sentiment(headlines, self.finbert_tok, self.finbert)
        return vec, headlines

    # ── prediction ──────────────────────────────────────────────
    def predict(self, ticker: str) -> dict:
        ticker = ticker.strip().upper()
        logger.info(f"Live prediction for {ticker}...")

        raw = self._fetch(ticker)
        feat = compute_indicators_live(raw)
        if len(feat) < vision.WINDOW + 1:
            raise ValueError(f"Not enough history for {ticker} after indicators.")
        as_of = feat.iloc[-1]["date"]

        # Visual: render the latest 20-day chart and extract EfficientNet features
        chart_path = vision.generate_chart(feat, len(feat) - 1, ticker)
        if chart_path is None:
            raise ValueError("Chart generation failed.")
        from PIL import Image
        img_tensor = self.img_transform(
            Image.open(chart_path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            visual_vec = self.efficientnet(img_tensor.to(self.device)).squeeze().cpu().numpy()

        # Sentiment + regime
        sent_vec, headlines = self._sentiment(ticker)
        regime = self._regime(ticker, feat)

        # Build the single prediction row and reuse the training feature pipeline
        row = feat.iloc[[-1]].copy()
        row["sentiment_neg"], row["sentiment_neu"], row["sentiment_pos"] = sent_vec
        row["regime"] = regime
        row["feature_vector"] = [visual_vec.astype(np.float32).tobytes()]
        row["target"] = 1  # dummy (unused at inference)

        market, sentiment_arr, visual, _, _, _ = prepare_features(
            row, market_scaler=self.scalers["market"],
            visual_scaler=self.scalers["visual"])

        market_t = torch.tensor(market, dtype=torch.float32)
        sentiment_t = torch.tensor(sentiment_arr, dtype=torch.float32)
        visual_t = torch.tensor(visual, dtype=torch.float32)

        # MC-Dropout: mean 3-class probability + predictive uncertainty
        mc = mc_dropout_inference(self.fusion, market_t, sentiment_t, visual_t,
                                  n_passes=MC_PASSES, device=str(self.device),
                                  temperature=self.temperature)
        proba = mc["mean_proba"][0]                       # [down, flat, up]
        pred_class = int(np.argmax(proba))
        uncertainty = float(mc["uncertainty"][0])
        up_calibrated = float(self.calibrator.predict([proba[2]])[0])

        return {
            "ticker": ticker,
            "as_of_date": str(pd.Timestamp(as_of).date()),
            "last_close": round(float(row["close"].iloc[0]), 2),
            "direction": CLASS_NAMES[pred_class],
            "direction_class": pred_class,
            "probabilities": {"down": round(float(proba[0]), 4),
                              "flat": round(float(proba[1]), 4),
                              "up": round(float(proba[2]), 4)},
            "up_probability_calibrated": round(up_calibrated, 4),
            "uncertainty": round(uncertainty, 6),
            "regime": regime,
            "sentiment": {"neg": round(float(sent_vec[0]), 3),
                          "neu": round(float(sent_vec[1]), 3),
                          "pos": round(float(sent_vec[2]), 3),
                          "headline_count": len(headlines)},
            "headlines": headlines[:5],
            "chart_path": str(chart_path),
            "disclaimer": DISCLAIMER,
            # internal artifacts for Grad-CAM (not JSON-serialised by the API)
            "_artifacts": {"img_tensor": img_tensor, "market_t": market_t,
                           "sentiment_t": sentiment_t, "window_df": feat.tail(vision.WINDOW)},
        }


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="MMIS live single-ticker prediction")
    parser.add_argument("ticker", nargs="?", default="AAPL")
    args = parser.parse_args()

    engine = LiveMMIS()
    result = engine.predict(args.ticker)
    result.pop("_artifacts", None)
    print(json.dumps(result, indent=2))
