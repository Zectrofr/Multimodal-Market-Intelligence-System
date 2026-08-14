"""
Phase 4: Cross-Modal Attention Fusion
=======================================
Custom multi-head attention mechanism that dynamically combines
market data (TFT), news sentiment (FinBERT), and visual features (EfficientNet).
This is the core differentiator of the entire system.

Phase 5 update: regime label (from regime.py) is now one-hot encoded
and concatenated into the market feature vector.
"""

import logging
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import mlflow
import mlflow.pytorch
import pickle
from pathlib import Path
import warnings
# Boundary IMPORTED, never restated, so this file cannot drift from
# regime_causal / rv_target. Transitively pulls in regime.py and hmmlearn.
from regime_causal import TRAIN_END_DATE
warnings.filterwarnings("ignore")

# ── Logging ──────────────────────────────────────────────────
# force=True so our config wins even if an imported lib (mlflow) already
# called basicConfig — otherwise INFO logs (incl. per-epoch metrics) vanish.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)

SCALER_PATH = "models/feature_scalers.pkl"
EARLY_STOP_PATIENCE = 7
# Default seed 42; override with MMIS_SEED env var for robustness sweeps.
SEED = int(os.environ.get("MMIS_SEED", "42"))


def set_seed(seed: int = SEED):
    """Make training reproducible. Without this, walk-forward results swing
    materially between retrains because the signal is weak (edge ~ noise)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

# ── Configuration ─────────────────────────────────────────────
DB_PATH = "sqlite:///data/mmis.db"
MARKET_DIM = 37        # Technical indicator features
SENTIMENT_DIM = 3      # FinBERT: [neg, neu, pos]
VISUAL_DIM = 1280      # EfficientNet-B0 features
FUSION_DIM = 256       # Output fusion dimension
NUM_HEADS = 8          # Attention heads
NUM_CLASSES = 3        # Down / Flat / Up
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-4
DROPOUT = 0.3
MLFLOW_EXPERIMENT = "mmis_fusion"

# ── Embargo at the train/validation seam ──────────────────────
# The last EMBARGO_DAYS training DATES are dropped, because their labels are
# drawn from dates that sit in validation. The correct value is the forward
# horizon of the TARGET:
#   * market_data.target (CURRENT)  — next-day direction  -> 1 day.
#   * market_data.vol_target (T5)   — 5-day forward RV    -> T5 MUST set 5.
# Leaving this at 1 after switching the target to vol_target would leak four
# days of validation outcomes into the training labels. See docs/STATE_REPORT.md
# D1 for the one-day overlap this closes, and results/T5_PREP_SPLIT_PIN.md.
EMBARGO_DAYS = 1

# Fixed order for regime one-hot encoding
REGIME_ORDER = ["mean_reverting", "trending", "high_vol"]

# ── Cross-Modal Attention Module ──────────────────────────────
class CrossModalAttention(nn.Module):
    """
    Custom multi-head cross-modal attention.
    Query: market temporal embedding
    Keys/Values: sentiment + visual embeddings
    Dynamically weights: 'How much does news/chart matter given this market state?'
    """
    def __init__(self, market_dim, sentiment_dim, visual_dim,
                 fusion_dim, num_heads, dropout):
        super().__init__()

        # Project each modality to fusion_dim
        self.market_proj = nn.Sequential(
            nn.Linear(market_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.sentiment_proj = nn.Sequential(
            nn.Linear(sentiment_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Multi-head attention: market queries sentiment+visual
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Self-attention on fused representation
        self.self_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward fusion head
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Final classifier
        self.classifier = nn.Linear(fusion_dim // 2, NUM_CLASSES)

        # Modality importance weights (learnable)
        self.modality_weights = nn.Parameter(torch.ones(3) / 3)

    def forward(self, market, sentiment, visual):
        # Project to fusion space
        m = self.market_proj(market).unsqueeze(1)      # (B, 1, D)
        s = self.sentiment_proj(sentiment).unsqueeze(1) # (B, 1, D)
        v = self.visual_proj(visual).unsqueeze(1)       # (B, 1, D)

        # Cross-modal attention: market attends to sentiment + visual
        kv = torch.cat([s, v], dim=1)                  # (B, 2, D)
        attended, attn_weights = self.cross_attention(
            query=m, key=kv, value=kv
        )                                               # (B, 1, D)

        # Self-attention across all modalities
        all_modalities = torch.cat([m, s, v], dim=1)   # (B, 3, D)
        self_attended, _ = self.self_attention(
            query=all_modalities,
            key=all_modalities,
            value=all_modalities
        )                                               # (B, 3, D)

        # Weighted combination using learnable modality weights
        weights = torch.softmax(self.modality_weights, dim=0)
        weighted = (
            weights[0] * self_attended[:, 0, :] +
            weights[1] * self_attended[:, 1, :] +
            weights[2] * self_attended[:, 2, :]
        )                                               # (B, D)

        # Concatenate attended + weighted for rich representation
        fused = torch.cat([
            attended.squeeze(1),
            weighted,
            m.squeeze(1)
        ], dim=-1)                                      # (B, 3D)

        # Final fusion + classification
        out = self.fusion_head(fused)
        logits = self.classifier(out)

        return logits, attn_weights, weights

# ── Dataset ───────────────────────────────────────────────────
class MultimodalDataset(Dataset):
    def __init__(self, market_data, sentiment_data, visual_data, targets):
        self.market = torch.FloatTensor(market_data)
        self.sentiment = torch.FloatTensor(sentiment_data)
        self.visual = torch.FloatTensor(visual_data)
        self.targets = torch.LongTensor(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            self.market[idx],
            self.sentiment[idx],
            self.visual[idx],
            self.targets[idx]
        )

# ── Load and Align Data ───────────────────────────────────────
def load_aligned_data(engine):
    """Load and align all three modalities by date and ticker.
    Now also pulls 'regime' column (written by regime.py)."""
    logger.info("Loading data from database...")

    # Market data (now includes regime, duplicates removed)
    market_df = pd.read_sql("""
        SELECT date, ticker, target, regime,
               rsi_14, macd, macd_signal, macd_diff,
               ema_20, ema_50, bb_upper, bb_lower, bb_mid,
               bb_bandwidth, bb_position, atr_14,
               returns_1d, returns_5d, returns_10d, returns_20d,
               volume_ratio, volume_change_1d,
               body_size, upper_shadow, lower_shadow, high_low_range,
               day_of_year_sin, day_of_year_cos, month_sin, month_cos,
               open, high, low, close, volume
        FROM market_data
    """, engine)

    # Sentiment data
    sentiment_df = pd.read_sql("""
        SELECT date, ticker,
               sentiment_neg, sentiment_neu, sentiment_pos
        FROM sentiment_data
    """, engine)

    # Visual features
    visual_df = pd.read_sql("""
        SELECT date, ticker, feature_vector
        FROM visual_features
    """, engine)

    logger.info(f"Market rows: {len(market_df)}")
    logger.info(f"Sentiment rows: {len(sentiment_df)}")
    logger.info(f"Visual rows: {len(visual_df)}")

    # Normalize dates
    market_df["date"] = pd.to_datetime(market_df["date"]).dt.date
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"]).dt.date
    visual_df["date"] = pd.to_datetime(visual_df["date"]).dt.date

    # Merge all three modalities
    df = market_df.merge(sentiment_df, on=["date", "ticker"], how="left")
    df = df.merge(visual_df, on=["date", "ticker"], how="left")

    # Fill missing sentiment with neutral
    df["sentiment_neg"] = df["sentiment_neg"].fillna(0.0)
    df["sentiment_neu"] = df["sentiment_neu"].fillna(1.0)
    df["sentiment_pos"] = df["sentiment_pos"].fillna(0.0)

    # Fill missing regime (safety fallback — shouldn't trigger after
    # `python regime.py --db data/mmis.db --all`)
    df["regime"] = df["regime"].fillna("trending")

    # Drop rows without visual features
    df = df.dropna(subset=["feature_vector"])

    # Sort by date so the downstream 80/20 split is a TRUE temporal split
    # (most-recent 20% = validation). Without this the split relies on
    # physical DB row order — which flips to per-ticker blocks after a
    # re-ingest, silently causing lookahead leakage / single-ticker val.
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    logger.info(f"Aligned rows: {len(df)}")

    return df

# ── Prepare Features ──────────────────────────────────────────
MARKET_COLS = [
    "rsi_14", "macd", "macd_signal", "macd_diff",
    "ema_20", "ema_50", "bb_upper", "bb_lower", "bb_mid",
    "bb_bandwidth", "bb_position", "atr_14",
    "returns_1d", "returns_5d", "returns_10d", "returns_20d",
    "volume_ratio", "volume_change_1d",
    "body_size", "upper_shadow", "lower_shadow", "high_low_range",
    "day_of_year_sin", "day_of_year_cos", "month_sin", "month_cos",
    "open", "high", "low", "close", "volume"
]


def prepare_features(df, market_scaler=None, visual_scaler=None):
    """Extract and scale features for each modality.
    Market features include a 3-dim one-hot regime encoding (appended AFTER
    scaling, so the one-hot stays 0/1).

    Scaler handling (prevents train→val leakage):
      - If `market_scaler`/`visual_scaler` are None, NEW scalers are fit on
        `df` and returned (use this on the TRAIN split only).
      - If fitted scalers are passed in, they are only applied via transform
        (use this on the VAL split and at inference time).
    """
    market_cols = list(dict.fromkeys(MARKET_COLS))

    # Market numeric features
    market_numeric = df[market_cols].fillna(0).values
    if market_scaler is None:
        market_scaler = StandardScaler().fit(market_numeric)
    market_numeric = market_scaler.transform(market_numeric)

    # Regime one-hot (3 extra columns) — appended to market features (unscaled)
    regime_onehot = pd.get_dummies(df["regime"]).reindex(
        columns=REGIME_ORDER, fill_value=0
    ).values.astype(np.float32)
    market_data = np.concatenate([market_numeric, regime_onehot], axis=1)

    # Sentiment features
    sentiment_data = df[["sentiment_neg", "sentiment_neu", "sentiment_pos"]].values

    # Visual features — decode from bytes
    visual_data = np.stack([
        np.frombuffer(row, dtype=np.float32)
        for row in df["feature_vector"]
    ])
    if visual_scaler is None:
        visual_scaler = StandardScaler().fit(visual_data)
    visual_data = visual_scaler.transform(visual_data)

    targets = df["target"].values

    logger.info(f"Market shape (incl. regime one-hot): {market_data.shape}")
    logger.info(f"Sentiment shape: {sentiment_data.shape}")
    logger.info(f"Visual shape: {visual_data.shape}")
    logger.info(f"Target distribution: {np.bincount(targets)}")

    return market_data, sentiment_data, visual_data, targets, market_scaler, visual_scaler

# ── Train / Eval ──────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for market, sentiment, visual, targets in loader:
        market = market.to(device)
        sentiment = sentiment.to(device)
        visual = visual.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits, _, _ = model(market, sentiment, visual)
        loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += len(targets)

    return total_loss / len(loader), correct / total

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for market, sentiment, visual, targets in loader:
            market = market.to(device)
            sentiment = sentiment.to(device)
            visual = visual.to(device)
            targets = targets.to(device)

            logits, _, _ = model(market, sentiment, visual)
            loss = criterion(logits, targets)

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += len(targets)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    return total_loss / len(loader), correct / total, all_preds, all_targets

# ── Probability Calibration (temperature scaling) ─────────────
def collect_logits(model, loader, device):
    """Return (logits, labels) tensors over a loader — for calibration."""
    model.eval()
    logits_all, labels_all = [], []
    with torch.no_grad():
        for market, sentiment, visual, targets in loader:
            logits, _, _ = model(market.to(device), sentiment.to(device),
                                 visual.to(device))
            logits_all.append(logits.cpu())
            labels_all.append(targets)
    return torch.cat(logits_all), torch.cat(labels_all)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Fit a single scalar temperature T that minimises NLL on the validation
    logits (Guo et al. 2017). Dividing logits by T>1 softens overconfident
    probabilities — fixes calibration (ECE) WITHOUT changing argmax/predictions.
    """
    T = torch.nn.Parameter(torch.ones(1))
    nll = nn.CrossEntropyLoss()
    opt = optim.LBFGS([T], lr=0.01, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = nll(logits / T.clamp(min=1e-3), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().clamp(min=1e-3).item())

# ── Temporal Split ────────────────────────────────────────────
def temporal_split(df, train_end_date=TRAIN_END_DATE, embargo_days=EMBARGO_DAYS):
    """Cut train/validation on a fixed DATE, never on a row-count fraction.

    Replaces `int(len(df) * 0.8)`, which MOVED whenever visual_features gained
    or lost rows, and which cut INSIDE a date — 2025-03-17 put five tickers in
    train and TSLA in validation, decided by alphabetical order. Here every row
    of a date lands on the same side. Index/order inherited from `df`.
    """
    dates = pd.to_datetime(df["date"]).dt.normalize()
    boundary = pd.Timestamp(train_end_date).normalize()
    train_mask, val_mask = dates <= boundary, dates > boundary

    # Whole DATES are embargoed, not rows, so the six tickers stay aligned.
    embargoed = []
    if embargo_days > 0:
        embargoed = list(np.sort(dates[train_mask].unique())[-embargo_days:])
        train_mask &= ~dates.isin(embargoed)

    train_df, val_df = df[train_mask], df[val_mask]
    if train_df.empty or val_df.empty:
        raise ValueError(f"empty side at {boundary.date()}: "
                         f"{len(train_df)} train / {len(val_df)} val")

    # Fail loudly rather than train on a leaky split — the two properties the
    # old row-count cut could not guarantee.
    tr_d = pd.to_datetime(train_df["date"]).dt.normalize()
    va_d = pd.to_datetime(val_df["date"]).dt.normalize()
    if tr_d.max() >= va_d.min():
        raise ValueError(f"overlap: last train {tr_d.max().date()} "
                         f">= first val {va_d.min().date()}")
    shared = set(tr_d.unique()) & set(va_d.unique())
    if shared:
        raise ValueError(f"{len(shared)} date(s) in BOTH splits: {sorted(shared)[:5]}")

    logger.info(
        f"Temporal split pinned at {boundary.date()} (embargo {embargo_days}d, "
        f"{len(embargoed)} date(s) dropped) — train: {tr_d.min().date()}.."
        f"{tr_d.max().date()} ({len(train_df)} rows) | val: {va_d.min().date()}.."
        f"{va_d.max().date()} ({len(val_df)} rows)"
    )
    return train_df, val_df

# ── Main Training Pipeline ────────────────────────────────────
def run_fusion_pipeline():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} | seed: {SEED}")

    engine = create_engine(DB_PATH)
    Path("models").mkdir(exist_ok=True)

    # Load data (already sorted by date in load_aligned_data)
    df = load_aligned_data(engine)

    # ── Temporal split BEFORE scaling (no train→val leakage) ────────
    train_df, val_df = temporal_split(df)

    # Fit scalers on TRAIN only, then transform both splits with them
    (m_tr, s_tr, v_tr, y_tr,
     market_scaler, visual_scaler) = prepare_features(train_df)
    m_va, s_va, v_va, y_va, _, _ = prepare_features(
        val_df, market_scaler=market_scaler, visual_scaler=visual_scaler
    )

    # Persist scalers so inference uses the EXACT same transform
    with open(SCALER_PATH, "wb") as f:
        pickle.dump({"market": market_scaler, "visual": visual_scaler}, f)
    logger.info(f"Saved feature scalers → {SCALER_PATH}")

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(
        MultimodalDataset(m_tr, s_tr, v_tr, y_tr),
        batch_size=BATCH_SIZE, shuffle=True, generator=g
    )
    val_loader = DataLoader(
        MultimodalDataset(m_va, s_va, v_va, y_va), batch_size=BATCH_SIZE
    )

    # Model
    model = CrossModalAttention(
        market_dim=m_tr.shape[1],
        sentiment_dim=SENTIMENT_DIM,
        visual_dim=VISUAL_DIM,
        fusion_dim=FUSION_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT
    ).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Class-weighted loss — counters the UP-class imbalance that otherwise
    # collapses the model into predicting UP ~80% of the time.
    class_counts = np.bincount(y_tr, minlength=NUM_CLASSES).astype(float)
    class_weights = len(y_tr) / (NUM_CLASSES * np.clip(class_counts, 1, None))
    weight_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
    logger.info(f"Train class counts: {class_counts.astype(int).tolist()} "
                f"→ weights: {np.round(class_weights, 3).tolist()}")
    criterion = nn.CrossEntropyLoss(weight=weight_t)

    # MLflow tracking
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="cross_modal_attention_v2"):
        mlflow.log_params({
            "fusion_dim": FUSION_DIM,
            "num_heads": NUM_HEADS,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "dropout": DROPOUT,
            "market_dim": m_tr.shape[1],
            "regime_conditioning": True,
            "class_weighted_loss": True,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "model_selection": "macro_f1",
            "scaler_fit": "train_only",
        })

        # Model selection by macro-F1 (balanced across classes) — NOT raw
        # accuracy, which rewards the degenerate all-UP predictor.
        best_val_f1 = -1.0
        best_epoch = 0
        epochs_no_improve = 0

        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_acc, preds, targets_val = eval_epoch(
                model, val_loader, criterion, device
            )
            scheduler.step()

            val_f1 = f1_score(targets_val, preds, average="macro", zero_division=0)

            mlflow.log_metrics({
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc, "val_macro_f1": val_f1,
            }, step=epoch)

            logger.info(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(model.state_dict(), "models/best_fusion_model.pt")
                logger.info(f"  ✅ New best model saved (val_macro_f1={val_f1:.4f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOP_PATIENCE:
                    logger.info(
                        f"  ⏹  Early stop at epoch {epoch} "
                        f"(no F1 improvement for {EARLY_STOP_PATIENCE} epochs)"
                    )
                    break

        # Final evaluation on the best checkpoint
        model.load_state_dict(torch.load("models/best_fusion_model.pt"))
        _, final_acc, final_preds, final_targets = eval_epoch(
            model, val_loader, criterion, device
        )

        # Fit calibration temperature on the validation set and persist it
        # alongside the scalers so inference produces calibrated probabilities.
        val_logits, val_labels = collect_logits(model, val_loader, device)
        temperature = fit_temperature(val_logits, val_labels)
        logger.info(f"Fitted calibration temperature: {temperature:.4f}")
        with open(SCALER_PATH, "wb") as f:
            pickle.dump({"market": market_scaler, "visual": visual_scaler,
                         "temperature": temperature}, f)
        mlflow.log_metric("calibration_temperature", temperature)

        report = classification_report(
            final_targets, final_preds,
            target_names=["Down", "Flat", "Up"], zero_division=0
        )
        logger.info(f"\nClassification Report (best epoch {best_epoch}):\n{report}")
        mlflow.log_text(report, "classification_report.txt")
        mlflow.log_metric("best_val_macro_f1", best_val_f1)
        mlflow.log_metric("best_val_acc", final_acc)
        mlflow.pytorch.log_model(model, "fusion_model")

        logger.info(f"\n✅ Fusion pipeline complete")
        logger.info(f"   Best epoch: {best_epoch} | val macro-F1: {best_val_f1:.4f} "
                    f"| val acc: {final_acc:.4f}")
        logger.info(f"   Model saved to: models/best_fusion_model.pt")

    return model, best_val_f1

if __name__ == "__main__":
    run_fusion_pipeline()
  