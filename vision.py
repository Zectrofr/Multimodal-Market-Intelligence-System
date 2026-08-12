"""
Phase 3: Computer Vision Stream
================================
EfficientNet-B0 visual feature extraction from candlestick charts.
Generates 1280-dim visual embeddings per ticker per date.
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sqlalchemy import create_engine, text
import mplfinance as mpf
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
from tqdm import tqdm
from db_guard import assert_safe_to_replace

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────
DB_PATH = "sqlite:///data/mmis.db"
CHART_DIR = Path("data/images/charts")
CHART_DIR.mkdir(parents=True, exist_ok=True)
WINDOW = 20
IMG_SIZE = 224

# ── Load EfficientNet ─────────────────────────────────────────
def load_efficientnet():
    logger.info("Loading EfficientNet-B0...")
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)

    # Remove classification head — we want 1280-dim features
    model.classifier = torch.nn.Identity()
    model.eval()
    logger.info("✅ EfficientNet-B0 loaded")
    return model

# ── Image Transform ───────────────────────────────────────────
def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# ── Generate Single Candlestick Chart ────────────────────────
def generate_chart(df: pd.DataFrame, date_idx: int, ticker: str) -> Path:
    """Generate a 20-day candlestick chart ending at date_idx."""
    start_idx = max(0, date_idx - WINDOW + 1)
    window_df = df.iloc[start_idx:date_idx + 1].copy()

    if len(window_df) < int(WINDOW * 0.8):
        return None

    window_df.index = pd.to_datetime(window_df["date"])

    # Add MAs for visual context
    window_df["ma5"] = window_df["close"].rolling(5).mean()
    window_df["ma20"] = window_df["close"].rolling(20).mean()

    # Drop NaN MAs for addplot
    valid = window_df.dropna(subset=["ma5", "ma20"])
    ap = []
    if len(valid) == len(window_df):
        ap = [
            mpf.make_addplot(window_df["ma5"], color="blue", width=0.8),
            mpf.make_addplot(window_df["ma20"], color="orange", width=0.8)
        ]

    date_str = pd.to_datetime(df.iloc[date_idx]["date"]).strftime("%Y-%m-%d")
    path = CHART_DIR / f"{ticker}_{date_str}.png"

    # Skip if already generated
    if path.exists():
        return path

    try:
        fig, _ = mpf.plot(
            window_df,
            type="candle",
            style="yahoo",
            volume=True,
            **({'addplot': ap} if ap else {}),
            figsize=(2.24, 2.24),
            returnfig=True,
            axisoff=True,
            tight_layout=True
        )
        fig.savefig(path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return path
    except Exception as e:
        logger.warning(f"Chart generation failed for {ticker} {date_str}: {e}")
        return None

# ── Extract Visual Features ───────────────────────────────────
def extract_features(image_path: Path, model, transform) -> np.ndarray:
    """Extract 1280-dim feature vector from a chart image."""
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)  # Add batch dim

    with torch.no_grad():
        features = model(tensor).squeeze().numpy()

    return features  # Shape: (1280,)

# ── Main Vision Pipeline ──────────────────────────────────────
def run_vision_pipeline():
    # ── Fail-closed guard, FIRST statement in the pipeline (see docs/STATE_REPORT.md).
    # The write at the end of this function DROPs and recreates visual_features, so the
    # refusal must happen before EfficientNet-B0 is loaded and before any chart is rendered.
    assert_safe_to_replace(
        "visual_features",
        db_path="data/mmis.db",
        human_reason=(
            "Regenerating those ~9,292 EfficientNet-B0 feature vectors means re-rendering "
            "charts and running a full CPU-only forward pass over every one of them "
            "(torch 2.12.0+cpu, cuda unavailable) — expensive, on the order of an hour, "
            "but NOT irreversible."
        ),
    )

    engine = create_engine(DB_PATH)
    model = load_efficientnet()
    transform = get_transform()

    # Create visual features table
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS visual_features (
                date TIMESTAMP,
                ticker TEXT,
                image_path TEXT,
                feature_vector BLOB,
                PRIMARY KEY (date, ticker)
            )
        """))
        conn.commit()

    # Load market data
    with engine.connect() as conn:
        df_all = pd.read_sql("SELECT * FROM market_data ORDER BY ticker, date", conn)

    logger.info(f"Processing visual features for {df_all['ticker'].nunique()} tickers...")

    results = []

    for ticker in df_all["ticker"].unique():
        df = df_all[df_all["ticker"] == ticker].reset_index(drop=True)
        logger.info(f"  Generating charts + features for {ticker}...")

        ticker_results = []
        for idx in tqdm(range(WINDOW, len(df)), desc=ticker, leave=False):
            # Generate chart
            path = generate_chart(df, idx, ticker)
            if path is None:
                continue

            # Extract features
            features = extract_features(path, model, transform)

            ticker_results.append({
                "date": df.iloc[idx]["date"],
                "ticker": ticker,
                "image_path": str(path),
                "feature_vector": features.tobytes()  # Store as binary
            })

        logger.info(f"  ✓ {len(ticker_results)} feature vectors for {ticker}")
        results.extend(ticker_results)

    # Save to DB
    visual_df = pd.DataFrame(results)

    visual_df.to_sql("visual_features", con=engine, if_exists="replace", index=False)

    logger.info(f"\n✅ Vision pipeline complete")
    logger.info(f"   Total feature vectors: {len(visual_df)}")
    logger.info(f"   Feature dimensions: 1280")
    logger.info(f"   Charts saved to: {CHART_DIR}")
    logger.info(f"\nSample:")
    print(visual_df[["date", "ticker", "image_path"]].head(6))

    return visual_df

if __name__ == "__main__":
    run_vision_pipeline()