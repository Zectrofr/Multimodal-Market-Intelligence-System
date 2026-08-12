"""
regime.py — Phase 5: Regime Conditioning & Uncertainty Quantification
Fincorp / Multimodal Market Intelligence System

Reads directly from data/mmis.db (market_data table from ingest.py).

What this does:
  1. HMM Regime Detection   — 3 states: trending / mean_reverting / high_vol
  2. Regime-conditioned output heads — separate prediction weights per regime
  3. Monte Carlo Dropout    — 50 forward passes → variance = uncertainty
  4. Uncertainty trading filter — skip trade when variance > threshold
  5. Saves regime labels back to mmis.db (market_data table)
  6. Exports regime-tagged CSV ready for evaluation.py

Usage:
    # Label regimes + run MC Dropout on your model predictions:
    python regime.py --db data/mmis.db --ticker AAPL

    # All tickers:
    python regime.py --db data/mmis.db --all

    # Demo mode (no DB needed):
    python regime.py --demo
"""

import numpy as np
import pandas as pd
import sqlite3
import pickle
import json
import warnings
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── Deps (install if missing) ──────────────────────────────
try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    raise ImportError("Run: pip install hmmlearn")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not found — MC Dropout will use numpy simulation mode.")

# ── Constants ──────────────────────────────────────────────
DB_PATH          = "data/mmis.db"
N_REGIMES        = 3
HMM_ITERATIONS   = 200
MC_PASSES        = 50          # number of forward passes for uncertainty
MC_DROPOUT_RATE  = 0.3
UNCERTAINTY_THRESHOLD = 0.02   # variance above this → skip trade
MODEL_DIR        = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

REGIME_NAMES = {
    0: "mean_reverting",
    1: "trending",
    2: "high_vol",
}


# ══════════════════════════════════════════════════════════
#  PART 1 — FEATURE ENGINEERING FOR HMM
# ══════════════════════════════════════════════════════════

def build_hmm_features(df: pd.DataFrame) -> np.ndarray:
    """
    Build the observation sequence the HMM is trained on.
    Uses 4 features that best discriminate market regimes:
      1. log_return          — direction signal
      2. rolling_vol_5d      — short-term volatility
      3. rolling_vol_20d     — medium-term volatility
      4. vol_ratio           — vol regime indicator (short/long vol)

    All pulled from columns already in mmis.db via ingest.py.
    """
    close = df["close"].values.astype(float)

    log_ret = np.log(close[1:] / close[:-1])
    log_ret = np.concatenate([[0.0], log_ret])  # pad first row

    # Rolling vol using pandas for clean window handling
    s = pd.Series(log_ret)
    vol_5   = s.rolling(5,  min_periods=1).std().fillna(0).values
    vol_20  = s.rolling(20, min_periods=1).std().fillna(0).values
    vol_ratio = np.where(vol_20 > 1e-8, vol_5 / vol_20, 1.0)

    features = np.column_stack([log_ret, vol_5, vol_20, vol_ratio])

    # Standardise — HMM is sensitive to scale
    means = features.mean(axis=0)
    stds  = features.std(axis=0)
    stds  = np.where(stds < 1e-8, 1.0, stds)
    features = (features - means) / stds

    return features.astype(np.float64)


# ══════════════════════════════════════════════════════════
#  PART 2 — HMM TRAINING & REGIME LABELLING
# ══════════════════════════════════════════════════════════

@dataclass
class RegimeStats:
    regime_id: int
    regime_name: str
    mean_return: float
    mean_vol: float
    n_days: int
    pct_of_total: float
    avg_duration_days: float


class MarketRegimeHMM:
    """
    3-state Gaussian HMM trained on per-ticker price features.
    Maps hidden states to interpretable regime names based on
    the learned emission means (vol + return characteristics).

    After fit(), call label(df) to get a 'regime' column.
    """

    def __init__(self, n_regimes: int = N_REGIMES, random_state: int = 42):
        self.n_regimes    = n_regimes
        self.random_state = random_state
        self.model        = None
        self.state_map    = {}   # HMM state int → regime name str
        self._fitted      = False

    def fit(self, df: pd.DataFrame) -> "MarketRegimeHMM":
        """Train HMM on OHLCV dataframe (single ticker, sorted by date)."""
        logger.info(f"  Training HMM on {len(df)} rows...")

        features = build_hmm_features(df)
        lengths  = [len(features)]   # single sequence

        self.model = GaussianHMM(
            n_components   = self.n_regimes,
            covariance_type= "diag",
            n_iter         = HMM_ITERATIONS,
            random_state   = self.random_state,
            tol            = 1e-4,
        )
        self.model.fit(features, lengths)
        logger.info(f"  HMM converged: {self.model.monitor_.converged}")

        # Map states to interpretable names via emission means
        self._build_state_map(df, features)
        self._fitted = True
        return self

    def _build_state_map(self, df: pd.DataFrame, features: np.ndarray):
        """
        Assign human-readable regime names to HMM states.
        State with highest vol   → high_vol
        State with lowest vol    → mean_reverting
        Remaining state          → trending
        """
        raw_states = self.model.predict(features)
        close      = df["close"].values.astype(float)
        log_ret    = np.concatenate([[0.0], np.log(close[1:] / close[:-1])])

        vol_per_state  = {}
        ret_per_state  = {}
        for s in range(self.n_regimes):
            mask = raw_states == s
            if mask.sum() == 0:
                vol_per_state[s] = 0.0
                ret_per_state[s] = 0.0
            else:
                vol_per_state[s] = float(np.std(log_ret[mask]))
                ret_per_state[s] = float(np.mean(np.abs(log_ret[mask])))

        sorted_by_vol = sorted(vol_per_state.items(), key=lambda x: x[1])

        self.state_map = {
            sorted_by_vol[0][0]: "mean_reverting",  # lowest vol
            sorted_by_vol[1][0]: "trending",         # mid vol
            sorted_by_vol[2][0]: "high_vol",         # highest vol
        }
        logger.info(f"  State map: {self.state_map}")

    def label(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add 'regime' and 'regime_id' columns to df.
        Returns a copy — does not modify original.
        """
        assert self._fitted, "Call fit() before label()"
        df   = df.copy()
        feat = build_hmm_features(df)
        raw  = self.model.predict(feat)
        df["regime_id"] = raw
        df["regime"]    = [self.state_map.get(int(s), "unknown") for s in raw]
        return df

    def regime_stats(self, df: pd.DataFrame) -> list[RegimeStats]:
        """Return summary statistics per regime."""
        assert "regime" in df.columns, "Call label() first"
        close   = df["close"].values.astype(float)
        log_ret = np.concatenate([[0.0], np.log(close[1:] / close[:-1])])
        df      = df.copy()
        df["_log_ret"] = log_ret

        stats = []
        for regime_name in REGIME_NAMES.values():
            mask = df["regime"] == regime_name
            if mask.sum() == 0:
                continue

            sub = df[mask]

            # Average run length (consecutive days in same regime)
            runs, cur_len = [], 1
            for i in range(1, len(df)):
                if df["regime"].iloc[i] == df["regime"].iloc[i - 1]:
                    cur_len += 1
                else:
                    if df["regime"].iloc[i - 1] == regime_name:
                        runs.append(cur_len)
                    cur_len = 1
            avg_dur = float(np.mean(runs)) if runs else float(mask.sum())

            stats.append(RegimeStats(
                regime_id        = int(df.loc[mask, "regime_id"].iloc[0]),
                regime_name      = regime_name,
                mean_return      = round(float(sub["_log_ret"].mean()), 6),
                mean_vol         = round(float(sub["_log_ret"].std()), 6),
                n_days           = int(mask.sum()),
                pct_of_total     = round(float(mask.sum() / len(df) * 100), 2),
                avg_duration_days= round(avg_dur, 1),
            ))

        return stats

    def save(self, path: str = "models/hmm_model.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"  HMM saved → {path}")

    @staticmethod
    def load(path: str = "models/hmm_model.pkl") -> "MarketRegimeHMM":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"  HMM loaded ← {path}")
        return obj


# ══════════════════════════════════════════════════════════
#  PART 3 — REGIME-CONDITIONED OUTPUT HEADS
# ══════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    class RegimeConditionedHead(nn.Module):
        """
        Separate linear prediction head per regime.
        Replaces the single output head in your fusion model.

        In fusion.py (Phase 4), after your cross-modal attention gives
        you a 256-dim fused_repr, pass it through this module:

            head = RegimeConditionedHead(input_dim=256, n_regimes=3)
            logits = head(fused_repr, regime_ids)  # regime_ids: LongTensor [B]

        Each regime gets its own weights → model learns different
        prediction strategies per market condition.
        """

        def __init__(
            self,
            input_dim:  int = 256,
            hidden_dim: int = 128,
            output_dim: int = 3,    # UP / FLAT / DOWN
            n_regimes:  int = 3,
            dropout:    float = MC_DROPOUT_RATE,
        ):
            super().__init__()
            self.n_regimes = n_regimes

            # One head per regime
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),          # ← MC Dropout lives here
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, output_dim),
                )
                for _ in range(n_regimes)
            ])

            # Regime embedding (optional soft-routing)
            self.regime_embed = nn.Embedding(n_regimes, input_dim)

        def forward(
            self,
            fused_repr: "torch.Tensor",    # [B, input_dim]
            regime_ids: "torch.Tensor",    # [B] LongTensor (0/1/2)
        ) -> "torch.Tensor":               # [B, output_dim]

            # Soft-condition: add regime embedding to input
            regime_context = self.regime_embed(regime_ids)   # [B, input_dim]
            conditioned     = fused_repr + regime_context     # residual add

            # Route each sample through its regime's head
            B      = fused_repr.shape[0]
            output = torch.zeros(B, self.heads[0][-1].out_features,
                                 device=fused_repr.device)

            for r in range(self.n_regimes):
                mask = (regime_ids == r)
                if mask.any():
                    output[mask] = self.heads[r](conditioned[mask])

            return output


# ══════════════════════════════════════════════════════════
#  PART 4 — MONTE CARLO DROPOUT UNCERTAINTY
# ══════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    def mc_dropout_predict(
        model:       nn.Module,
        x:           "torch.Tensor",
        regime_ids:  "torch.Tensor",
        n_passes:    int   = MC_PASSES,
        threshold:   float = UNCERTAINTY_THRESHOLD,
    ) -> dict:
        """
        Run N stochastic forward passes with dropout ACTIVE.
        Returns mean prediction + variance (uncertainty).

        The key trick: model.train() keeps dropout on during inference.
        This turns dropout into a Bayesian approximation.

        Args:
            model:      Your RegimeConditionedHead (or full fusion model)
            x:          Input features [B, input_dim]
            regime_ids: Regime labels  [B]
            n_passes:   Number of MC samples (50 is standard)
            threshold:  Variance above this → flag as high uncertainty

        Returns dict:
            mean_proba     : [B, n_classes] — mean softmax across passes
            uncertainty    : [B] — predictive variance (scalar per sample)
            pred_direction : [B] — argmax of mean_proba
            high_uncertainty: [B] bool — flag for trading filter
        """
        model.train()  # KEEP DROPOUT ACTIVE — this is the key

        all_logits = []
        with torch.no_grad():
            for _ in range(n_passes):
                logits = model(x, regime_ids)          # [B, n_classes]
                proba  = torch.softmax(logits, dim=-1) # [B, n_classes]
                all_logits.append(proba.unsqueeze(0))  # [1, B, n_classes]

        stacked    = torch.cat(all_logits, dim=0)      # [n_passes, B, n_classes]
        mean_proba = stacked.mean(dim=0)               # [B, n_classes]

        # Predictive variance: mean of per-class variances
        variance   = stacked.var(dim=0).mean(dim=-1)   # [B]

        pred_direction  = mean_proba.argmax(dim=-1)    # [B]
        high_uncertainty = variance > threshold

        return {
            "mean_proba":      mean_proba.cpu().numpy(),
            "uncertainty":     variance.cpu().numpy(),
            "pred_direction":  pred_direction.cpu().numpy(),
            "high_uncertainty": high_uncertainty.cpu().numpy(),
        }


def mc_dropout_numpy(
    pred_proba_fn,          # callable(x) → np.ndarray [B, n_classes]
    x:           np.ndarray,
    n_passes:    int   = MC_PASSES,
    dropout_rate:float = MC_DROPOUT_RATE,
    threshold:   float = UNCERTAINTY_THRESHOLD,
) -> dict:
    """
    Pure NumPy MC Dropout simulation — works without PyTorch.
    Simulates dropout by randomly zeroing input features each pass.

    Use this for prototyping or when you have a sklearn-style model.
    """
    all_probas = []

    for _ in range(n_passes):
        # Simulate dropout: randomly zero features
        mask   = np.random.binomial(1, 1 - dropout_rate, size=x.shape).astype(float)
        x_drop = x * mask / (1 - dropout_rate + 1e-8)   # scale to preserve expectation

        proba = pred_proba_fn(x_drop)
        all_probas.append(proba)

    stacked    = np.stack(all_probas, axis=0)   # [n_passes, B, n_classes]
    mean_proba = stacked.mean(axis=0)           # [B, n_classes]
    variance   = stacked.var(axis=0).mean(-1)   # [B]

    pred_direction   = mean_proba.argmax(axis=-1)
    high_uncertainty = variance > threshold

    # UP probability = class 2 (matches ingest.py: 0=Down, 1=Flat, 2=Up)
    up_proba = mean_proba[:, 2] if mean_proba.shape[1] == 3 else mean_proba[:, 1]

    return {
        "mean_proba":       mean_proba,
        "uncertainty":      variance,
        "pred_direction":   pred_direction,
        "up_proba":         up_proba,
        "high_uncertainty": high_uncertainty,
    }


# ══════════════════════════════════════════════════════════
#  PART 5 — DATABASE I/O
# ══════════════════════════════════════════════════════════

def load_from_db(db_path: str, ticker: str) -> pd.DataFrame:
    """Load market_data for one ticker from mmis.db."""
    conn = sqlite3.connect(db_path)
    df   = pd.read_sql(
        "SELECT * FROM market_data WHERE ticker = ? ORDER BY date",
        conn, params=(ticker,)
    )
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    logger.info(f"  Loaded {len(df)} rows for {ticker}")
    return df


def load_all_tickers(db_path: str) -> dict[str, pd.DataFrame]:
    """Load all tickers from mmis.db."""
    conn    = sqlite3.connect(db_path)
    tickers = pd.read_sql("SELECT DISTINCT ticker FROM market_data", conn)["ticker"].tolist()
    conn.close()

    result = {}
    for t in tickers:
        result[t] = load_from_db(db_path, t)
    return result


def save_regimes_to_db(df: pd.DataFrame, db_path: str, ticker: str):
    """
    Write regime labels back into market_data table.
    Adds 'regime'/'regime_id' columns if missing, then UPDATEs by (date, ticker).

    IMPORTANT: dates are stored in the DB as text *with microseconds*
    (e.g. '2020-03-13 00:00:00.000000'), but str(timestamp) produces
    '2020-03-13 00:00:00' — so a naive `WHERE date=str(row['date'])`
    matches ZERO rows silently and the regime column stays NULL.
    To be robust we resolve each row to the EXACT stored date string via a
    normalized-date lookup, then verify the update count so a silent
    0-row write can never happen again.
    """
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    # Add columns if missing (SQLite ALTER TABLE)
    for col, decl in [("regime", "TEXT"), ("regime_id", "INTEGER")]:
        try:
            cur.execute(f"ALTER TABLE market_data ADD COLUMN {col} {decl}")
            logger.info(f"  Added {col} column to market_data")
        except Exception:
            pass  # column already exists
    conn.commit()

    # Map normalized date -> exact stored date string (handles any text format)
    stored = cur.execute(
        "SELECT date FROM market_data WHERE ticker=?", (ticker,)
    ).fetchall()
    date_lookup = {pd.Timestamp(d[0]).normalize(): d[0] for d in stored}

    params, missing = [], 0
    for _, row in df.iterrows():
        stored_date = date_lookup.get(pd.Timestamp(row["date"]).normalize())
        if stored_date is None:
            missing += 1
            continue
        params.append((row["regime"], int(row["regime_id"]), stored_date, ticker))

    before = conn.total_changes
    cur.executemany(
        "UPDATE market_data SET regime=?, regime_id=? WHERE date=? AND ticker=?",
        params,
    )
    conn.commit()
    updated = conn.total_changes - before
    conn.close()

    if updated == 0:
        logger.error(
            f"  ⚠️  0 rows updated for {ticker} — regime column stays NULL! "
            f"Check date matching."
        )
    else:
        msg = f"  ✅ Regime labels saved to DB for {ticker}: {updated} rows updated"
        if missing:
            msg += f" ({missing} dates not found in DB)"
        logger.info(msg)


# ══════════════════════════════════════════════════════════
#  PART 6 — CALIBRATION ANALYSIS
# ══════════════════════════════════════════════════════════

def calibration_analysis(
    uncertainty: np.ndarray,
    actual_correct: np.ndarray,   # bool array: was prediction correct?
    n_bins: int = 10,
) -> dict:
    """
    Uncertainty calibration: do high-uncertainty predictions
    actually have lower accuracy? They should.

    Returns bin data for plotting uncertainty vs accuracy.
    """
    bins   = np.percentile(uncertainty, np.linspace(0, 100, n_bins + 1))
    result = {"bin_upper": [], "avg_uncertainty": [], "accuracy": [], "count": []}

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask   = (uncertainty >= lo) & (uncertainty <= hi)
        if mask.sum() == 0:
            continue
        result["bin_upper"].append(round(float(hi), 6))
        result["avg_uncertainty"].append(round(float(uncertainty[mask].mean()), 6))
        result["accuracy"].append(round(float(actual_correct[mask].mean()), 4))
        result["count"].append(int(mask.sum()))

    # Ideal: accuracy should DECREASE as uncertainty INCREASES
    if len(result["accuracy"]) > 1:
        corr = np.corrcoef(result["avg_uncertainty"], result["accuracy"])[0, 1]
        result["uncertainty_accuracy_correlation"] = round(float(corr), 4)
        result["well_calibrated"] = corr < -0.3   # negative = good

    return result


# ══════════════════════════════════════════════════════════
#  PART 7 — FULL PIPELINE
# ══════════════════════════════════════════════════════════

def run_regime_pipeline(
    db_path:        str  = DB_PATH,
    ticker:         str  = None,
    all_tickers:    bool = False,
    save_to_db_flag:bool = True,
    export_csv:     bool = True,
    output_dir:     str  = "results",
    demo_dummy:     bool = False,
) -> dict:
    """
    Full Phase 5 pipeline:
      1. Load from mmis.db
      2. Fit HMM per ticker
      3. Label regimes
      4. Run MC Dropout uncertainty (numpy simulation on returns)
      5. Save regimes to DB
      6. Export tagged CSV for evaluation.py
      7. Print regime stats

    Returns dict of {ticker: labelled_dataframe}
    """
    Path(output_dir).mkdir(exist_ok=True)

    if all_tickers:
        ticker_dfs = load_all_tickers(db_path)
    else:
        assert ticker, "Provide --ticker or --all"
        ticker_dfs = {ticker: load_from_db(db_path, ticker)}

    results    = {}
    all_tagged = []

    for tkr, df in ticker_dfs.items():
        logger.info(f"\n{'─'*50}")
        logger.info(f"Processing {tkr}...")

        # 1. Fit HMM
        hmm = MarketRegimeHMM(n_regimes=N_REGIMES)
        hmm.fit(df)

        # 2. Label regimes
        df_labelled = hmm.label(df)

        # 3. Regime stats
        stats = hmm.regime_stats(df_labelled)
        _print_regime_stats(tkr, stats)

        # 4. MC Dropout uncertainty — SYNTHETIC placeholder, opt-in only.
        #    Gated behind --demo-dummy: this fabricates predictions from a
        #    hand-written formula, not from any trained model. Off by default.
        if demo_dummy:
            def dummy_model(x_dropped: np.ndarray) -> np.ndarray:
                """
                Placeholder model: replace with real model.predict_proba() in Phase 4+.
                For now simulates a slight directional edge using log returns.
                """
                n    = x_dropped.shape[0]
                # Use first feature (log return) as weak signal
                ret  = x_dropped[:, 0] if x_dropped.shape[1] > 0 else np.zeros(n)
                # Build 3-class softmax: [down, flat, up]
                up   = np.clip(0.35 + ret * 5, 0.05, 0.85)
                down = np.clip(0.35 - ret * 5, 0.05, 0.85)
                flat = np.clip(1.0 - up - down, 0.05, 0.90)
                proba = np.column_stack([down, flat, up])
                proba = proba / proba.sum(axis=1, keepdims=True)
                return proba

            features    = build_hmm_features(df_labelled)
            mc_result   = mc_dropout_numpy(
                pred_proba_fn = dummy_model,
                x             = features,
                n_passes      = MC_PASSES,
                dropout_rate  = MC_DROPOUT_RATE,
                threshold     = UNCERTAINTY_THRESHOLD,
            )

            df_labelled["uncertainty"]      = mc_result["uncertainty"]
            df_labelled["up_proba"]         = mc_result["up_proba"]
            df_labelled["pred_direction"]   = (mc_result["pred_direction"] == 2).astype(int)
            df_labelled["high_uncertainty"] = mc_result["high_uncertainty"].astype(int)

        # 5. Save regimes to DB
        if save_to_db_flag:
            save_regimes_to_db(df_labelled, db_path, tkr)

        # 6. Save model
        hmm_path = f"models/hmm_{tkr}.pkl"
        hmm.save(hmm_path)

        results[tkr] = df_labelled
        all_tagged.append(df_labelled)

        # Save per-ticker stats JSON
        stats_path = f"{output_dir}/regime_stats_{tkr}.json"
        with open(stats_path, "w") as f:
            json.dump([asdict(s) for s in stats], f, indent=2)
        logger.info(f"  Stats saved → {stats_path}")

    # 7. Export combined CSV — SYNTHETIC placeholder output, opt-in only.
    if demo_dummy and export_csv and all_tagged:
        combined = pd.concat(all_tagged, ignore_index=True)
        csv_path = f"{output_dir}/SYNTHETIC_DO_NOT_EVALUATE_regime_predictions.csv"

        # Build evaluation.py-compatible columns
        eval_df = combined[["date", "ticker", "close", "regime",
                             "uncertainty", "up_proba",
                             "pred_direction", "high_uncertainty"]].copy()
        eval_df["actual_return"] = combined.groupby("ticker")["close"].transform(
            lambda x: x.pct_change().shift(-1)
        )
        eval_df["pred_proba"] = eval_df["up_proba"]
        eval_df = eval_df.dropna(subset=["actual_return"])
        eval_df.insert(0, "IS_SYNTHETIC", 1)

        eval_df.to_csv(csv_path, index=False)
        logger.warning(
            f"\n⚠️  SYNTHETIC placeholder output written → {csv_path}\n"
            f"   Produced by the dummy_model formula, NOT by any trained model.\n"
            f"   This is NOT a result. Never pass it to evaluation.py or quote it."
        )

        _print_uncertainty_summary(eval_df)

    return results


def _print_regime_stats(ticker: str, stats: list):
    print(f"\n{'═'*55}")
    print(f"  Regime Stats — {ticker}")
    print(f"{'═'*55}")
    print(f"  {'Regime':<18} {'Days':>6} {'%':>6} {'AvgRet':>9} {'Vol':>9} {'AvgDur':>8}")
    print(f"  {'─'*60}")
    for s in stats:
        print(
            f"  {s.regime_name:<18} {s.n_days:>6} "
            f"{s.pct_of_total:>5.1f}% "
            f"{s.mean_return:>+9.5f} "
            f"{s.mean_vol:>9.5f} "
            f"{s.avg_duration_days:>7.1f}d"
        )
    print()


def _print_uncertainty_summary(df: pd.DataFrame):
    print(f"\n{'═'*55}")
    print(f"  MC Dropout Uncertainty Summary")
    print(f"{'═'*55}")
    unc = df["uncertainty"]
    print(f"  Mean uncertainty   : {unc.mean():.5f}")
    print(f"  Median uncertainty : {unc.median():.5f}")
    print(f"  P95 uncertainty    : {unc.quantile(0.95):.5f}")
    high = df["high_uncertainty"].mean()
    print(f"  High-uncertainty % : {high:.1%}  (would be filtered out)")
    print(f"  Remaining trades   : {(1-high):.1%} of total")

    print(f"\n  Regime distribution:")
    for reg, grp in df.groupby("regime"):
        print(f"    {reg:<18}: {len(grp):>5} rows | "
              f"unc={grp['uncertainty'].mean():.5f} | "
              f"filtered={grp['high_uncertainty'].mean():.1%}")
    print()


# ══════════════════════════════════════════════════════════
#  DEMO MODE (no DB needed)
# ══════════════════════════════════════════════════════════

def run_demo():
    """
    Full pipeline demo without needing mmis.db.
    Generates synthetic OHLCV, fits HMM, runs MC Dropout,
    prints everything, exports CSV.
    """
    logger.info("Running DEMO mode — no DB needed")

    rng  = np.random.default_rng(42)
    n    = 600

    # Synthetic price with 3 embedded regimes
    regimes_true = np.repeat([0, 1, 2, 1, 0, 2], n // 6)
    drift  = np.where(regimes_true == 1, 0.001,
             np.where(regimes_true == 2, 0.0, -0.0002))
    vol    = np.where(regimes_true == 2, 0.025,
             np.where(regimes_true == 1, 0.010, 0.008))

    log_ret = drift + vol * rng.standard_normal(n)
    close   = 100 * np.exp(np.cumsum(log_ret))
    dates   = pd.date_range("2021-01-01", periods=n, freq="B")

    df = pd.DataFrame({
        "date":   dates,
        "ticker": "DEMO",
        "open":   close * (1 + rng.normal(0, 0.002, n)),
        "high":   close * (1 + np.abs(rng.normal(0, 0.005, n))),
        "low":    close * (1 - np.abs(rng.normal(0, 0.005, n))),
        "close":  close,
        "volume": rng.integers(1_000_000, 10_000_000, n),
    })

    # Run HMM
    hmm         = MarketRegimeHMM()
    hmm.fit(df)
    df_labelled = hmm.label(df)
    stats       = hmm.regime_stats(df_labelled)
    _print_regime_stats("DEMO", stats)

    # MC Dropout
    features = build_hmm_features(df_labelled)

    def demo_model(x):
        n = x.shape[0]
        ret = x[:, 0]
        up   = np.clip(0.38 + ret * 6, 0.05, 0.85)
        down = np.clip(0.35 - ret * 6, 0.05, 0.85)
        flat = np.clip(1 - up - down,  0.05, 0.90)
        p    = np.column_stack([down, flat, up])
        return p / p.sum(axis=1, keepdims=True)

    mc = mc_dropout_numpy(demo_model, features, n_passes=MC_PASSES)
    df_labelled["uncertainty"]    = mc["uncertainty"]
    df_labelled["up_proba"]       = mc["up_proba"]
    df_labelled["pred_direction"] = (mc["pred_direction"] == 2).astype(int)
    df_labelled["high_uncertainty"] = mc["high_uncertainty"].astype(int)

    Path("results").mkdir(exist_ok=True)
    out_path = "results/demo_regime_output.csv"
    df_labelled.to_csv(out_path, index=False)
    logger.info(f"\n✅ Demo output saved → {out_path}")
    logger.info(f"   Columns: {list(df_labelled.columns)}")

    _print_uncertainty_summary(df_labelled)

    # Quick calibration check
    actual_ret = df_labelled["close"].pct_change().shift(-1).dropna()
    preds      = df_labelled["pred_direction"].iloc[:len(actual_ret)].values
    correct    = ((actual_ret.values > 0).astype(int) == preds)
    cal        = calibration_analysis(
        df_labelled["uncertainty"].iloc[:len(actual_ret)].values, correct
    )
    print(f"\n  Uncertainty calibration correlation: "
          f"{cal.get('uncertainty_accuracy_correlation', 'N/A')}")
    print(f"  Well calibrated: {cal.get('well_calibrated', 'N/A')}")

    return df_labelled


# ══════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MMIS Phase 5 — Regime + Uncertainty")
    parser.add_argument("--db",     type=str, default=DB_PATH,
                        help="Path to mmis.db")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Single ticker to process (e.g. AAPL)")
    parser.add_argument("--all",    action="store_true",
                        help="Process all tickers in DB")
    parser.add_argument("--demo",   action="store_true",
                        help="Run demo mode (no DB required)")
    parser.add_argument("--no-save",action="store_true",
                        help="Don't write regime labels back to DB")
    parser.add_argument("--out",    type=str, default="results",
                        help="Output directory for CSVs and stats")
    parser.add_argument("--demo-dummy", action="store_true",
                        help="Also emit SYNTHETIC placeholder predictions from dummy_model. "
                             "These are fabricated, NOT results, and must never be evaluated.")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.all:
        run_regime_pipeline(
            db_path=args.db,
            all_tickers=True,
            save_to_db_flag=not args.no_save,
            output_dir=args.out,
            demo_dummy=args.demo_dummy,
        )
    elif args.ticker:
        run_regime_pipeline(
            db_path=args.db,
            ticker=args.ticker,
            save_to_db_flag=not args.no_save,
            output_dir=args.out,
            demo_dummy=args.demo_dummy,
        )
    else:
        print("⚠️  Specify --ticker AAPL, --all, or --demo")
        print("    Example: python regime.py --demo")
        print("    Example: python regime.py --db data/mmis.db --all")