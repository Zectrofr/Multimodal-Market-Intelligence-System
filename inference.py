"""
inference.py — Bridge between Phase 4/5 (fusion model) and Phase 6 (evaluation)
=================================================================================
Loads models/best_fusion_model.pt, re-builds the same feature pipeline as
fusion.py (incl. regime one-hot), runs Monte Carlo Dropout (50 passes with
dropout active), and exports results/final_predictions.csv ready for:

    python evaluation.py --csv results/final_predictions.csv --uncertainty-filter

This REPLACES the dummy_model placeholder used in regime.py's pipeline.
"""

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sqlalchemy import create_engine
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Reuse exact same feature pipeline as fusion.py — import directly
from fusion import (
    CrossModalAttention,
    load_aligned_data,
    prepare_features,
    DB_PATH,
    SENTIMENT_DIM,
    VISUAL_DIM,
    FUSION_DIM,
    NUM_HEADS,
    DROPOUT,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

MC_PASSES = 50
UNCERTAINTY_THRESHOLD = 0.02
MODEL_PATH = "models/best_fusion_model.pt"
OUTPUT_CSV = "results/final_predictions.csv"

# UP class index in the 3-class target (0=Down, 1=Flat, 2=Up — from ingest.py)
UP_CLASS = 2


def mc_dropout_inference(
    model: nn.Module,
    market: torch.Tensor,
    sentiment: torch.Tensor,
    visual: torch.Tensor,
    n_passes: int = MC_PASSES,
    batch_size: int = 256,
    device: str = "cpu",
) -> dict:
    """
    Run N stochastic forward passes with dropout ACTIVE (model.train()).
    Batches internally to avoid OOM on 9k rows x 1280-dim visual features.

    Returns:
        mean_proba   : [N, 3]  — averaged softmax across passes
        uncertainty  : [N]     — predictive variance (mean over classes)
        pred_direction: [N]    — 1 if UP class has highest mean proba, else 0
        up_proba     : [N]     — mean_proba[:, UP_CLASS]
    """
    model.train()  # keep dropout ON — this is the MC Dropout trick
    model.to(device)

    n = market.shape[0]
    n_classes = 3

    sum_proba   = torch.zeros(n, n_classes)
    sum_sq_proba = torch.zeros(n, n_classes)

    with torch.no_grad():
        for p in range(n_passes):
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)

                m_batch = market[start:end].to(device)
                s_batch = sentiment[start:end].to(device)
                v_batch = visual[start:end].to(device)

                logits, _, _ = model(m_batch, s_batch, v_batch)
                proba = torch.softmax(logits, dim=-1).cpu()

                sum_proba[start:end]    += proba
                sum_sq_proba[start:end] += proba ** 2

            if (p + 1) % 10 == 0 or p == 0:
                logger.info(f"  MC Dropout pass {p + 1}/{n_passes}")

    mean_proba = sum_proba / n_passes
    # Var(X) = E[X^2] - E[X]^2
    var_proba  = (sum_sq_proba / n_passes) - mean_proba ** 2
    var_proba  = torch.clamp(var_proba, min=0.0)  # numerical safety
    uncertainty = var_proba.mean(dim=-1)          # [N]

    pred_direction = (mean_proba.argmax(dim=-1) == UP_CLASS).long()
    up_proba = mean_proba[:, UP_CLASS]

    return {
        "mean_proba":     mean_proba.numpy(),
        "uncertainty":    uncertainty.numpy(),
        "pred_direction": pred_direction.numpy(),
        "up_proba":       up_proba.numpy(),
    }


def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    engine = create_engine(DB_PATH)

    # ── 1. Rebuild the exact same feature pipeline as fusion.py ──────
    logger.info("Loading & preparing features (same pipeline as fusion.py)...")
    df = load_aligned_data(engine)
    market_data, sentiment_data, visual_data, targets, scaler = prepare_features(df)

    market_dim = market_data.shape[1]
    logger.info(f"Market dim (incl. regime one-hot): {market_dim}")

    # ── 2. Load trained model ──────────────────────────────────────
    model = CrossModalAttention(
        market_dim=market_dim,
        sentiment_dim=SENTIMENT_DIM,
        visual_dim=VISUAL_DIM,
        fusion_dim=FUSION_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
    )

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run `python fusion.py` first to train the model."
        )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    logger.info(f"✅ Loaded {MODEL_PATH}")

    # ── 3. Convert to tensors ──────────────────────────────────────
    market_t    = torch.tensor(market_data, dtype=torch.float32)
    sentiment_t = torch.tensor(sentiment_data, dtype=torch.float32)
    visual_t    = torch.tensor(visual_data, dtype=torch.float32)

    # ── 4. MC Dropout inference ─────────────────────────────────────
    logger.info(f"Running MC Dropout: {MC_PASSES} passes over {len(df)} rows...")
    mc = mc_dropout_inference(
        model, market_t, sentiment_t, visual_t,
        n_passes=MC_PASSES, device=device
    )

    # ── 5. Build evaluation.py-compatible dataframe ─────────────────
    out = df[["date", "ticker", "close", "regime"]].copy()
    out["date"] = pd.to_datetime(out["date"])

    # actual_return: next-day return, computed per-ticker (avoids
    # cross-ticker boundary bug from the regime.py CSV)
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    out["actual_return"] = out.groupby("ticker")["close"].pct_change().shift(-1)

    out["pred_proba"]      = mc["up_proba"]
    out["pred_direction"]  = mc["pred_direction"]
    out["uncertainty"]     = mc["uncertainty"]
    out["high_uncertainty"] = (out["uncertainty"] > UNCERTAINTY_THRESHOLD).astype(int)

    # Drop last row per ticker (no actual_return available)
    out = out.dropna(subset=["actual_return"]).reset_index(drop=True)

    # ── 6. Save ──────────────────────────────────────────────────────
    Path("results").mkdir(exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"\n✅ Predictions saved → {OUTPUT_CSV}")
    logger.info(f"   Rows: {len(out)}")
    logger.info(f"   Columns: {list(out.columns)}")

    # ── 7. Quick summary ────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"  Inference Summary (MC Dropout, {MC_PASSES} passes)")
    print(f"{'═'*55}")
    print(f"  Mean up_proba      : {out['pred_proba'].mean():.4f}")
    print(f"  Mean uncertainty   : {out['uncertainty'].mean():.5f}")
    print(f"  High-uncertainty % : {out['high_uncertainty'].mean():.1%}")
    print(f"  Pred UP rate       : {out['pred_direction'].mean():.1%}")
    print(f"\n  Regime distribution:")
    for reg, grp in out.groupby("regime"):
        print(f"    {reg:<18}: {len(grp):>5} rows | "
              f"unc={grp['uncertainty'].mean():.5f} | "
              f"pred_up={grp['pred_direction'].mean():.1%}")

    print(f"\n  Next step:")
    print(f"    python evaluation.py --csv {OUTPUT_CSV} --uncertainty-filter")
    print(f"{'═'*55}\n")

    return out


if __name__ == "__main__":
    run_inference()