# Multimodal Market Intelligence System (MMIS)

An end-to-end, **interpretable** deep-learning pipeline that predicts next-day stock
direction by fusing three modalities — price/technical indicators, news sentiment,
and candlestick **chart images** — via cross-modal attention, conditioned on a
market-regime model, with uncertainty quantification.

> ⚠️ **Honest scope.** This is an **engineering / interpretability artifact**, not a
> profitable trading system. Built on *free* daily data (yfinance OHLCV, NewsAPI
> headlines), next-day direction is near-random out-of-sample. In the reproducible
> benchmark below the model's **walk-forward precision is ~50% (coin-flip)** and it
> **does not beat buy-and-hold**. The value is the correct, reproducible, multimodal
> MLOps stack and the explainability — *not* alpha. **Not financial advice.**

---

## Architecture

| Layer | Component | File |
|-------|-----------|------|
| 0 | Data ingestion: OHLCV + technical indicators + 224×224 candlestick charts | `ingest.py` |
| 1a | News sentiment (FinBERT) → 3-dim sentiment vector | `sentiment.py` |
| 1b | Chart images → EfficientNet-B0 → 1280-dim visual features | `vision.py` |
| 2 | **Cross-modal attention fusion** (market query attends to sentiment+visual) | `fusion.py` |
| 3 | **HMM market-regime** conditioning (mean-reverting / trending / high-vol), one-hot into fusion | `regime.py` |
| 4 | Output head: 3-class direction + **MC-Dropout uncertainty** + temperature-calibrated probabilities | `fusion.py`, `inference.py` |
| 5 | Evaluation: Sharpe / drawdown / **precision\@UP** / calibration / **walk-forward** | `evaluation.py` |
| 6 | *(planned)* FastAPI + Streamlit live demo with Grad-CAM | — |

Data lives in `data/mmis.db` (SQLite): `market_data`, `sentiment_data`,
`visual_features`. Universe: AAPL, GOOGL, MSFT, TSLA, AMZN, SPY (2020–2026,
~9,300 aligned rows).

---

## Setup

```bash
pip install -r requirements.txt
# Create .env with:  NEWS_API_KEY=<your key>
```

> **Windows note:** scripts print Unicode (✅, box chars). Run with
> `PYTHONIOENCODING=utf-8 PYTHONUTF8=1` to avoid `UnicodeEncodeError` on cp1252 consoles.

## Run the pipeline (in order)

```bash
python ingest.py                              # 1. fetch data, indicators, charts → DB
python sentiment.py                           # 2. FinBERT sentiment → DB
python vision.py                              # 3. EfficientNet visual features → DB
python regime.py --db data/mmis.db --all      # 4. HMM regimes → DB (writes `regime` column)
python fusion.py                              # 5. train cross-modal fusion model (seeded)
python inference.py                           # 6. MC-Dropout predictions → results/final_predictions.csv
python evaluation.py --csv results/final_predictions.csv --uncertainty-filter
```

Everything is reproducible (`SEED = 42` in `fusion.py`; MC-Dropout seeded in `inference.py`).

---

## Reproducible benchmark (seed = 42)

Validation = most recent ~15 months (2025-03 → 2026-06), strict temporal split.

| Metric | Model | Buy & Hold |
|---|---|---|
| Total return (full sample) | −10.8% | **+373%** |
| Sharpe | −0.10 | **1.09** |
| Max drawdown | 32% | 42% |
| Precision\@UP (in-sample) | 53.8% | — |
| **Walk-forward precision\@UP (24 folds)** | **50.0%** | — |
| Calibration (ECE) | 0.20 (poor) | — |

**Interpretation:** out-of-sample the model is at coin-flip and underperforms simply
holding the basket. The only mildly interesting signal is per-regime: the model
behaves best in the `high_vol` regime. This is the expected result for free-data
next-day prediction and is reported honestly rather than hidden.

---

## Engineering correctness (bugs found & fixed)

This repo previously produced *meaningless* headline numbers. Fixed:

1. **Regime layer was 100% dead.** `save_regimes_to_db` matched dates as
   `'…00:00:00'` while the DB stored `'…00:00:00.000000'`, so every `UPDATE` hit
   **0 rows** silently — the `regime` column was all-NULL and `fusion.py` filled it
   with a constant `"trending"`. Now writes by exact stored-date string and verifies
   the row count.
2. **Train/val split was temporal only by accident** (relied on physical DB row
   order, no `ORDER BY`). Now an explicit `sort_values(["date","ticker"])`.
3. **Evaluation return engine was broken** — it compounded all 6 tickers' daily rows
   sequentially as separate all-in bets and charged transaction cost *every bar even
   for Buy & Hold*, reporting an impossible **−99.7%** B&H. Rewritten as a proper
   equal-weight **portfolio** backtest with **turnover-based** costs (B&H now +373%).
4. **Train→val leakage:** `StandardScaler` was fit on the full dataset. Now fit on
   **train only**, persisted (`models/feature_scalers.pkl`), and reused at inference.
5. **Degenerate UP-bias:** unweighted loss + accuracy-based model selection picked an
   all-UP predictor. Now **class-weighted** loss, **macro-F1** model selection, and
   **early stopping**.
6. **Dead uncertainty filter / calibration:** MC-Dropout variance (~1e-3) never
   crossed the fixed 0.02 threshold. Filter is now **percentile-based**, and
   probabilities are **temperature-scaled** (calibration still poor — a known limit).

## Known limitations
- **No out-of-sample edge** (by design — free data).
- **Sentiment coverage ~1.3%** (124 / 9,412 rows): NewsAPI free tier is 30-day
  limited, so the FinBERT modality is neutral for ~99% of history.
- **Calibration remains poor** (ECE ~0.20): the class-weighted model's UP-probability
  is systematically low; temperature scaling on 3-class logits doesn't fully fix the
  binary up/down calibration.

## Roadmap
- **Layer 6 live demo:** `live_inference.py` (on-demand single ticker) → Grad-CAM on
  the live chart → FastAPI `/predict` → Streamlit dashboard. This is the
  highest-impact remaining work for a portfolio/interview piece.
