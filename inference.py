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
import pickle
from sklearn.isotonic import IsotonicRegression
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
    SCALER_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

MC_PASSES = 50
UNCERTAINTY_THRESHOLD = 0.02
MODEL_PATH = "models/best_fusion_model.pt"
CALIBRATOR_PATH = "models/calibrator.pkl"
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
    temperature: float = 1.0,
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
                proba = torch.softmax(logits / temperature, dim=-1).cpu()

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
    # Seed the stochastic MC-Dropout passes so predictions are reproducible.
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    engine = create_engine(DB_PATH)

    # ── 1. Rebuild the exact same feature pipeline as fusion.py ──────
    logger.info("Loading & preparing features (same pipeline as fusion.py)...")
    df = load_aligned_data(engine)

    # Load the scalers fit on the TRAIN split during fusion.py training, so
    # inference applies the identical transform (no refitting / no leakage).
    if not Path(SCALER_PATH).exists():
        raise FileNotFoundError(
            f"{SCALER_PATH} not found. Run `python fusion.py` first to fit & save scalers."
        )
    with open(SCALER_PATH, "rb") as f:
        scalers = pickle.load(f)
    temperature = float(scalers.get("temperature", 1.0))
    logger.info(f"Calibration temperature: {temperature:.4f}")
    market_data, sentiment_data, visual_data, targets, _, _ = prepare_features(
        df, market_scaler=scalers["market"], visual_scaler=scalers["visual"]
    )

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
        n_passes=MC_PASSES, device=device, temperature=temperature
    )

    # ── 5. Build evaluation.py-compatible dataframe ─────────────────
    out = df[["date", "ticker", "close", "regime"]].copy()
    out["date"] = pd.to_datetime(out["date"])

    # CRITICAL: attach the MC-Dropout outputs while `out` is still in df order
    # (the mc arrays are aligned to df rows). Doing this AFTER a re-sort assigns
    # them positionally to the wrong (date,ticker) and scrambles every
    # prediction against its actual outcome.
    out["pred_proba"]      = mc["up_proba"]
    out["pred_direction"]  = mc["pred_direction"]
    out["uncertainty"]     = mc["uncertainty"]

    # Now sort by ticker,date (all columns move together) and compute the
    # next-day return per ticker.
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    out["actual_return"] = out.groupby("ticker")["close"].pct_change().shift(-1)
    # Flag the most-uncertain 20% by PERCENTILE — MC-Dropout variance is tiny
    # (~1e-3), so a fixed absolute threshold flags nothing.
    unc_cut = float(np.quantile(out["uncertainty"], 0.80))
    out["high_uncertainty"] = (out["uncertainty"] > unc_cut).astype(int)
    logger.info(f"High-uncertainty cutoff (80th pct): {unc_cut:.6f}")

    # Drop last row per ticker (no actual_return available)
    out = out.dropna(subset=["actual_return"]).reset_index(drop=True)

    # ── 5b. Isotonic probability calibration ───────────────────────────
    # The raw UP-probability is poorly calibrated (class-weighted training +
    # temperature scaling don't fix the binary up/down calibration). Fit an
    # isotonic map pred_proba -> P(actual up) on the TRAINING period and apply
    # to all rows. Fitting on TRAIN (not val) keeps the validation ECE an HONEST
    # held-out test — fitting on val would make its ECE trivially ~0. This
    # changes only the probability, NOT pred_direction, so precision/returns
    # are unaffected; only calibration (ECE) improves.
    val_start = out["date"].quantile(0.80)
    train_mask = out["date"] < val_start
    y_up = (out["actual_return"] > 0).astype(int).values
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(out.loc[train_mask, "pred_proba"].values, y_up[train_mask])
    out["pred_proba_uncalibrated"] = out["pred_proba"]
    out["pred_proba"] = iso.predict(out["pred_proba"].values)
    with open(CALIBRATOR_PATH, "wb") as f:
        pickle.dump(iso, f)
    logger.info(f"Isotonic calibrator fit on training (< {val_start.date()}) "
                f"→ saved {CALIBRATOR_PATH}")

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