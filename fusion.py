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
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import mlflow
import mlflow.pytorch
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

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
    logger.info(f"Aligned rows: {len(df)}")

    return df

# ── Prepare Features ──────────────────────────────────────────
def prepare_features(df):
    """Extract and scale features for each modality.
    Market features now include a 3-dim one-hot regime encoding."""
    market_cols = [
        "rsi_14", "macd", "macd_signal", "macd_diff",
        "ema_20", "ema_50", "bb_upper", "bb_lower", "bb_mid",
        "bb_bandwidth", "bb_position", "atr_14",
        "returns_1d", "returns_5d", "returns_10d", "returns_20d",
        "volume_ratio", "volume_change_1d",
        "body_size", "upper_shadow", "lower_shadow", "high_low_range",
        "day_of_year_sin", "day_of_year_cos", "month_sin", "month_cos",
        "open", "high", "low", "close", "volume"
    ]

    # Remove duplicates
    market_cols = list(dict.fromkeys(market_cols))

    # Market features
    market_data = df[market_cols].fillna(0).values
    scaler = StandardScaler()
    market_data = scaler.fit_transform(market_data)

    # Regime one-hot (3 extra columns) — appended to market features
    regime_onehot = pd.get_dummies(df["regime"]).reindex(
        columns=REGIME_ORDER, fill_value=0
    ).values.astype(np.float32)

    market_data = np.concatenate([market_data, regime_onehot], axis=1)

    # Sentiment features
    sentiment_data = df[["sentiment_neg", "sentiment_neu", "sentiment_pos"]].values

    # Visual features — decode from bytes
    visual_data = np.stack([
        np.frombuffer(row, dtype=np.float32)
        for row in df["feature_vector"]
    ])

    # Normalize visual features
    visual_scaler = StandardScaler()
    visual_data = visual_scaler.fit_transform(visual_data)

    targets = df["target"].values

    logger.info(f"Market shape (incl. regime one-hot): {market_data.shape}")
    logger.info(f"Sentiment shape: {sentiment_data.shape}")
    logger.info(f"Visual shape: {visual_data.shape}")
    logger.info(f"Target distribution: {np.bincount(targets)}")
    logger.info(f"Regime distribution:\n{df['regime'].value_counts().to_string()}")

    return market_data, sentiment_data, visual_data, targets, scaler

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

# ── Main Training Pipeline ────────────────────────────────────
def run_fusion_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    engine = create_engine(DB_PATH)

    # Load data
    df = load_aligned_data(engine)
    market_data, sentiment_data, visual_data, targets, scaler = prepare_features(df)

    # Train/val split (80/20 time-based)
    split = int(len(targets) * 0.8)
    train_dataset = MultimodalDataset(
        market_data[:split], sentiment_data[:split],
        visual_data[:split], targets[:split]
    )
    val_dataset = MultimodalDataset(
        market_data[split:], sentiment_data[split:],
        visual_data[split:], targets[split:]
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Model
    model = CrossModalAttention(
        market_dim=market_data.shape[1],
        sentiment_dim=SENTIMENT_DIM,
        visual_dim=VISUAL_DIM,
        fusion_dim=FUSION_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT
    ).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    # MLflow tracking
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="cross_modal_attention_v1"):
        mlflow.log_params({
            "fusion_dim": FUSION_DIM,
            "num_heads": NUM_HEADS,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "dropout": DROPOUT,
            "market_dim": market_data.shape[1],
            "regime_conditioning": True
        })

        best_val_acc = 0
        Path("models").mkdir(exist_ok=True)

        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_acc, preds, targets_val = eval_epoch(
                model, val_loader, criterion, device
            )
            scheduler.step()

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            }, step=epoch)

            logger.info(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "models/best_fusion_model.pt")
                logger.info(f"  ✅ New best model saved (val_acc={val_acc:.4f})")

        # Final evaluation
        model.load_state_dict(torch.load("models/best_fusion_model.pt"))
        _, _, final_preds, final_targets = eval_epoch(
            model, val_loader, criterion, device
        )

        report = classification_report(
            final_targets, final_preds,
            target_names=["Down", "Flat", "Up"]
        )
        logger.info(f"\nClassification Report:\n{report}")
        mlflow.log_text(report, "classification_report.txt")
        mlflow.log_metric("best_val_acc", best_val_acc)
        mlflow.pytorch.log_model(model, "fusion_model")

        logger.info(f"\n✅ Fusion pipeline complete")
        logger.info(f"   Best validation accuracy: {best_val_acc:.4f}")
        logger.info(f"   Model saved to: models/best_fusion_model.pt")

    return model, best_val_acc

if __name__ == "__main__":
    run_fusion_pipeline()
  