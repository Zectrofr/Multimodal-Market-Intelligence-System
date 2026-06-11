import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import MACD, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from sqlalchemy import create_engine, text
import mplfinance as mpf
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Tuple
import argparse

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────
TICKERS: List[str] = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "SPY"]
START_DATE: str = "2020-01-01"
END_DATE: str = datetime.today().strftime("%Y-%m-%d")
DB_PATH: str = "sqlite:///data/mmis.db"
CHART_DIR: Path = Path("data/images/charts")
FLAT_THRESHOLD: float = 0.005  # 0.5% — within this = Flat class

# Ensure directories exist
Path("data").mkdir(exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

# ── Fetch raw OHLCV data ────────────────────────────────────
def fetch_ohlcv(ticker: str, start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    """Download OHLCV data with quality checks."""
    logger.info(f"Fetching {ticker}...")
    
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    
    # Flatten MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    
    # Standardize column names
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    
    # Quality checks — drop nulls instead of asserting
    df = df.dropna()
    df = df[df["high"] >= df["low"]].copy()
    df = df[df["volume"] > 0].copy()
    
    df["ticker"] = ticker
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    
    logger.info(f"  ✓ {len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()}")
    return df

# ── Add technical indicators ────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators and TFT-ready features."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    
    # ── Trend Indicators ─────────────────────────────────────
    # RSI
    df["rsi_14"] = RSIIndicator(close=close, window=14).rsi()
    
    # MACD
    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()
    
    # EMA
    df["ema_20"] = EMAIndicator(close=close, window=20).ema_indicator()
    df["ema_50"] = EMAIndicator(close=close, window=50).ema_indicator()
    
    # ── Volatility Indicators ────────────────────────────────
    # Bollinger Bands
    bb = BollingerBands(close=close, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_bandwidth"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    
    # ATR
    df["atr_14"] = AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range()
    
    # ── Price/Volume Features ──────────────────────────────
    # Returns (lag features for TFT autoregressive inputs)
    df["returns_1d"] = close.pct_change(1)
    df["returns_5d"] = close.pct_change(5)
    df["returns_10d"] = close.pct_change(10)
    df["returns_20d"] = close.pct_change(20)
    
    # Volume features
    df["volume_ma_20"] = volume.rolling(window=20).mean()
    df["volume_ratio"] = volume / df["volume_ma_20"]
    df["volume_change_1d"] = volume.pct_change(1)
    
    # Candlestick features
    df["body_size"] = abs(close - df["open"])
    df["upper_shadow"] = high - df[["close", "open"]].max(axis=1)
    df["lower_shadow"] = df[["close", "open"]].min(axis=1) - low
    df["high_low_range"] = high - low
    
    # ── Cyclical Time Features (critical for TFT) ────────────
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # ── TFT Required Columns ───────────────────────────────────
    # Integer time index (sequential, no gaps)
    df["time_idx"] = np.arange(len(df))
    
    # Group identifier (for multi-ticker support)
    df["group"] = df["ticker"]
    
    # ── 3-Class Target (Up / Flat / Down) ────────────────────
    next_return = close.shift(-1) / close - 1
    
    # 0 = Down, 1 = Flat, 2 = Up
    df["target"] = 1  # Default Flat
    df.loc[next_return > FLAT_THRESHOLD, "target"] = 2   # Up
    df.loc[next_return < -FLAT_THRESHOLD, "target"] = 0  # Down
    
    # Drop NaN from rolling indicators and last row (no target)
    df = df.dropna().reset_index(drop=True)
    
    # Recompute time_idx after dropna (critical!)
    df["time_idx"] = np.arange(len(df))
    
    return df

# ── Generate Candlestick Charts ─────────────────────────────
def generate_chart(df: pd.DataFrame, ticker: str, date_idx: int, 
                   window: int = 20, size: Tuple[int, int] = (224, 224)) -> Path:
    """Generate 224x224 candlestick chart for a single date."""
    start_idx = max(0, date_idx - window + 1)
    window_df = df.iloc[start_idx:date_idx + 1].copy()
    
    if len(window_df) < window * 0.8:
        return None
    
    # Set DatetimeIndex for mplfinance
    window_df.index = pd.to_datetime(window_df["date"])
    
    # Moving averages for visual context
    window_df["ma5"] = window_df["close"].rolling(window=5).mean()
    window_df["ma20"] = window_df["close"].rolling(window=20).mean()
    
    ap = [
        mpf.make_addplot(window_df["ma5"], color="blue", width=0.8),
        mpf.make_addplot(window_df["ma20"], color="orange", width=0.8)
    ]
    
    figsize = (size[0] / 100, size[1] / 100)
    
    fig, axes = mpf.plot(
    window_df,
    type="candle",
    style="yahoo",
    volume=True,
    addplot=ap,
    figsize=figsize,
    returnfig=True,
    axisoff=True,
    scale_padding=dict(left=0, right=0, top=0.05, bottom=0.05),
    tight_layout=True
)
    
    date_str = df.iloc[date_idx]["date"].strftime("%Y-%m-%d")
    path = CHART_DIR / f"{ticker}_{date_str}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    
    return path

# ── Database Schema ──────────────────────────────────────────
def init_database(engine):
    """Create tables with proper schema for multimodal data."""
    with engine.connect() as conn:
        # Market data table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS market_data (
                date TIMESTAMP,
                ticker TEXT,
                open REAL, high REAL, low REAL, close REAL, volume REAL,
                rsi_14 REAL, macd REAL, macd_signal REAL, macd_diff REAL,
                ema_20 REAL, ema_50 REAL,
                bb_upper REAL, bb_lower REAL, bb_mid REAL, 
                bb_bandwidth REAL, bb_position REAL,
                atr_14 REAL,
                returns_1d REAL, returns_5d REAL, returns_10d REAL, returns_20d REAL,
                volume_ma_20 REAL, volume_ratio REAL, volume_change_1d REAL,
                body_size REAL, upper_shadow REAL, lower_shadow REAL, high_low_range REAL,
                day_of_week INTEGER, month INTEGER, day_of_year INTEGER,
                day_of_year_sin REAL, day_of_year_cos REAL,
                month_sin REAL, month_cos REAL,
                time_idx INTEGER,
                group_id TEXT,
                target INTEGER,
                split TEXT DEFAULT 'train',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (date, ticker)
            )
        """))
        
        # Chart images table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS chart_images (
                date TIMESTAMP,
                ticker TEXT,
                image_path TEXT,
                window_size INTEGER,
                PRIMARY KEY (date, ticker),
                FOREIGN KEY (date, ticker) REFERENCES market_data(date, ticker)
            )
        """))
        
        conn.commit()
    logger.info("✅ Database initialized")

# ── Save to Database ─────────────────────────────────────────
def save_to_db(df: pd.DataFrame, engine, ticker: str):
    """Save DataFrame to SQLite with validation."""
    # Add train/val split marker (80/20 time-based split)
    split_idx = int(len(df) * 0.8)
    df["split"] = "train"
    df.iloc[split_idx:, df.columns.get_loc("split")] = "validation"
    
    # Rename 'group' to 'group_id' for SQL compatibility
    df = df.rename(columns={"group": "group_id"})
    
    # Save
    df.to_sql("market_data", con=engine, if_exists="append", index=False)
    logger.info(f"  ✓ Saved {len(df)} rows for {ticker}")

# ── Main Pipeline ───────────────────────────────────────────
def run_pipeline(tickers: List[str] = TICKERS, generate_charts: bool = True):
    """Run full Phase 1 pipeline."""
    engine = create_engine(DB_PATH)
    init_database(engine)
    
    all_data = []
    
    for ticker in tickers:
        try:
            # Fetch
            df = fetch_ohlcv(ticker)
            
            # Indicators
            df = add_indicators(df)
            
            # Generate charts (skip first 20 rows for window history)
            if generate_charts:
                logger.info(f"  Generating charts for {ticker}...")
                chart_count = 0
                for idx in range(20, len(df)):
                    path = generate_chart(df, ticker, idx)
                    if path:
                        chart_count += 1
                logger.info(f"  ✓ Generated {chart_count} charts")
            
            # Save to DB
            save_to_db(df, engine, ticker)
            all_data.append(df)
            
        except Exception as e:
            logger.error(f"❌ Failed for {ticker}: {e}")
            continue
    
    # Summary
    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"\n{'='*50}")
    logger.info(f"PIPELINE COMPLETE")
    logger.info(f"{'='*50}")
    logger.info(f"Total rows: {len(combined):,}")
    logger.info(f"Tickers: {combined['ticker'].nunique()}")
    logger.info(f"Date range: {combined['date'].min().date()} to {combined['date'].max().date()}")
    logger.info(f"Features: {len([c for c in combined.columns if c not in ['date', 'ticker', 'target', 'split']])}")
    logger.info(f"Class distribution:")
    for cls, name in [(0, "Down"), (1, "Flat"), (2, "Up")]:
        count = (combined["target"] == cls).sum()
        pct = count / len(combined) * 100
        logger.info(f"  {name}: {count:,} ({pct:.1f}%)")
    
    # Sample output
    logger.info(f"\nSample data:")
    print(combined[["date", "ticker", "close", "rsi_14", "macd", "atr_14", "target"]].tail(5))
    
    return combined

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Data Pipeline")
    parser.add_argument("--tickers", nargs="+", default=TICKERS, help="Tickers to process")
    parser.add_argument("--no-charts", action="store_true", help="Skip chart generation")
    args = parser.parse_args()
    
    run_pipeline(tickers=args.tickers, generate_charts=not args.no_charts)

