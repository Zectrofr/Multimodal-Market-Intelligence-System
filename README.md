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

Strict temporal split: train ≤ 2025-03, **validation = the most recent ~15 months
(2025-03 → 2026-06) the model never trained on.**

> ⚠️ **Read the out-of-sample column, not the full-sample one.** A backtest over the
> *full* history includes the ~80% of dates the model trained on and looks
> spectacular (Sharpe ~1.9, +467%) — that is **in-sample memorisation, not skill.**
> The honest number is the out-of-sample (validation) column.

| Metric | **Out-of-sample** (honest) | Full sample (in-sample, inflated) |
|---|---|---|
| Precision\@UP | **52.8%** (base rate 53.4% → edge **−0.6 pts**) | 57.9% |
| Sharpe | **0.95** (Buy & Hold 1.35) | 1.90 (BH 1.09) |
| Total return | +19% (BH +46%) | +467% (BH +373%) |
| Calibration (ECE, isotonic) | **0.020 (good)** | 0.004 |

**Interpretation:** out-of-sample the model has **no directional edge** (precision
*below* the base rate) and **underperforms simply holding the basket**. This is the
expected result for next-day prediction on free daily data, and it is now measured
*correctly* (see bug #7 below) and reported as the headline rather than buried under
the in-sample mirage. The value of the project is the rigorous, honest, interpretable
engineering — not alpha.

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
6. **Dead uncertainty filter:** MC-Dropout variance (~1e-3) never crossed the fixed
   0.02 threshold. Now **percentile-based**.
7. **🔴 Prediction–row misalignment (the big one).** `inference.py` assigned the
   MC-Dropout outputs to the prediction frame *after* re-sorting it by
   `["ticker","date"]`, while the arrays were in `["date","ticker"]` order — so
   **100% of predictions were attached to the wrong (date, ticker)** and every
   evaluation metric was computed on scrambled predictions (≈coin-flip by
   construction). Fixed by attaching outputs *before* the re-sort. This is what
   flipped the result from "looks like no signal at all" to "clear in-sample fit,
   honest out-of-sample ≈ coin-flip".
8. **Calibration:** probabilities are now **isotonic-calibrated** (fit on the
   training period, so the validation ECE is an honest held-out test). OOS ECE
   improved from ~0.20 → **0.020**. (Temperature scaling is also fit during
   training but isotonic does the heavy lifting for the binary up/down metric.)
9. **Reproducibility:** `SEED=42` (override via `MMIS_SEED` env var), seeded
   MC-Dropout, and a `scripts/robustness_sweep.py` that re-runs across seeds to
   separate signal from noise.

## Known limitations
- **No out-of-sample edge** (by design — free daily data).
- **Sentiment coverage ~1.3%** (124 / 9,412 rows): NewsAPI free tier is 30-day
  limited, so the FinBERT modality is neutral for ~99% of history.
- **Regime-label lookahead:** `regime.py` fits the HMM on each ticker's *full*
  history, so the regime label at day *t* peeks at the future. Harmless to current
  conclusions (there's no edge to inflate), but for a live/causal system the HMM
  must be fit rolling. Tracked for Layer 6.

## Roadmap
- **Layer 6 live demo:** `live_inference.py` (on-demand single ticker) → Grad-CAM on
  the live chart → FastAPI `/predict` → Streamlit dashboard. This is the
  highest-impact remaining work for a portfolio/interview piece.
