"""
evaluate.py — Phase 6: Rigorous Evaluation Engine
Fincorp / Multimodal Market Intelligence System

Completely standalone — works with:
  - Phase 1 (TFT baseline predictions)
  - Phase 2 (TFT + FinBERT predictions)
  - Phase 3 (+ EfficientNet)
  - Phase 4 (full fusion predictions)
  - Any model output that returns (predicted_direction, predicted_proba, actual_return)

Usage:
    from evaluate import Evaluator
    ev = Evaluator(predictions_df)
    report = ev.full_report()
"""

import numpy as np
import pandas as pd
import sqlite3
import json
import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  Data Contract
#  Your predictions_df must have these columns:
#    date         : pd.Timestamp or str
#    ticker       : str
#    actual_return: float  (next-day log return or % return)
#    pred_proba   : float  (model confidence for UP, 0-1)
#    pred_direction: int   (1 = UP, 0 = DOWN/FLAT)
#  Optional:
#    uncertainty  : float  (MC Dropout variance — Phase 5)
#    regime       : str    (HMM regime label — Phase 5)
# ─────────────────────────────────────────────


TRANSACTION_COST = 0.001   # 0.1% per trade
SLIPPAGE         = 0.0005  # 0.05% per trade
TRADING_DAYS     = 252


# ══════════════════════════════════════════════
#  Core Metric Functions (pure, testable)
# ══════════════════════════════════════════════

def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """Annualised Sharpe ratio."""
    excess = returns - risk_free / TRADING_DAYS
    if excess.std() == 0:
        return 0.0
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / excess.std())


def max_drawdown(cumulative_returns: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown (positive number = loss magnitude)."""
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / np.where(peak == 0, 1e-9, peak)
    return float(abs(drawdown.min()))


def calmar_ratio(annualised_return: float, mdd: float) -> float:
    """Calmar = annualised return / max drawdown."""
    if mdd == 0:
        return 0.0
    return float(annualised_return / mdd)


def precision_at_up(pred_direction: np.ndarray, actual_return: np.ndarray) -> float:
    """
    Of all days we predicted UP, what fraction actually went UP?
    This is the metric that matters for a long-only strategy.
    """
    up_mask = pred_direction == 1
    if up_mask.sum() == 0:
        return 0.0
    actual_up = (actual_return[up_mask] > 0).mean()
    return float(actual_up)


def hit_rate(pred_direction: np.ndarray, actual_return: np.ndarray) -> float:
    """Overall directional accuracy (not the primary metric — markets are flat 70% of time)."""
    actual_direction = (actual_return > 0).astype(int)
    return float((pred_direction == actual_direction).mean())


def expected_calibration_error(
    pred_proba: np.ndarray,
    actual_direction: np.ndarray,
    n_bins: int = 10
) -> tuple[float, dict]:
    """
    ECE: does 70% confidence = 70% win rate?
    Returns (ECE score, bin_data for plotting).
    Lower ECE = better calibrated model.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_data = {"bin_lower": [], "bin_upper": [], "avg_confidence": [],
                "avg_accuracy": [], "count": []}
    ece = 0.0
    n = len(pred_proba)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (pred_proba >= lo) & (pred_proba < hi)
        if mask.sum() == 0:
            continue
        avg_conf = pred_proba[mask].mean()
        avg_acc  = actual_direction[mask].mean()
        count    = mask.sum()
        ece     += (count / n) * abs(avg_conf - avg_acc)

        bin_data["bin_lower"].append(round(lo, 2))
        bin_data["bin_upper"].append(round(hi, 2))
        bin_data["avg_confidence"].append(round(float(avg_conf), 4))
        bin_data["avg_accuracy"].append(round(float(avg_acc), 4))
        bin_data["count"].append(int(count))

    return float(ece), bin_data


def sortino_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """Like Sharpe but only penalises downside volatility."""
    excess = returns - risk_free / TRADING_DAYS
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / downside.std())


def profit_factor(returns: np.ndarray) -> float:
    """Gross profit / gross loss. >1 is profitable."""
    gains  = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float("inf")
    return float(gains / losses)


# ══════════════════════════════════════════════
#  Backtesting Engine
# ══════════════════════════════════════════════

@dataclass
class BacktestResult:
    strategy_name: str
    total_return: float
    annualised_return: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    precision_at_up: float
    hit_rate: float
    profit_factor: float
    n_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    equity_curve: list = field(default_factory=list)  # excluded from summary print


def run_backtest(
    df: pd.DataFrame,
    strategy_name: str = "Model",
    use_uncertainty_filter: bool = False,
    uncertainty_threshold: float = 0.02,
) -> BacktestResult:
    """
    Equal-weight, long-only PORTFOLIO backtest.

      - Each ticker is sized at 1/N of capital (N = tickers in the universe).
      - Go long a ticker on days pred_direction == 1; otherwise hold cash (0).
      - Transaction cost + slippage are charged only on TURNOVER (entering or
        exiting a position), NOT every day — so Buy & Hold pays the cost once,
        not every bar.
      - All tickers' daily P&L is aggregated into ONE portfolio return series,
        then compounded. (The previous version compounded every ticker-day
        sequentially as if they were separate all-in bets — which made Buy &
        Hold bleed the per-bar cost ~9k times and report an impossible -99%.)

    If use_uncertainty_filter=True (Phase 5 MC Dropout), skip trades where
    uncertainty > threshold.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    cost_per_trade = TRANSACTION_COST + SLIPPAGE   # one-way cost per turnover unit

    # ── Build per-ticker net daily return legs ──────────────────────
    legs = []
    for _, g in df.groupby("ticker"):
        g = g.sort_values("date")

        pos = (g["pred_direction"] == 1)
        if use_uncertainty_filter and "uncertainty" in g.columns:
            pos = pos & (g["uncertainty"] <= uncertainty_threshold)
        pos = pos.astype(float)

        prev     = pos.shift(1).fillna(0.0)
        turnover = (pos - prev).abs()                 # 1.0 on each enter/exit
        gross    = pos * g["actual_return"].astype(float)
        net      = gross - turnover * cost_per_trade

        legs.append(pd.DataFrame({
            "date":    g["date"].values,
            "net":     net.values,
            "pos":     pos.values,
            "entries": ((pos == 1) & (prev == 0)).astype(int).values,
        }))

    legs = pd.concat(legs, ignore_index=True)
    n_universe = max(df["ticker"].nunique(), 1)

    # Equal-weight portfolio: daily return = mean of per-ticker net across the
    # universe (held tickers contribute their net; cash positions contribute 0).
    daily = (legs.groupby("date")["net"].sum() / n_universe).sort_index()
    returns_arr = daily.values
    equity_arr  = np.concatenate([[1.0], np.cumprod(1.0 + returns_arr)])

    # Win/loss stats over held ticker-days (matches prior "days in market" sense)
    traded_returns = legs.loc[legs["pos"] == 1, "net"].values
    n_trades       = int((legs["pos"] == 1).sum())

    total_ret      = float(equity_arr[-1] - 1)
    n_days         = len(returns_arr)
    ann_ret        = float((1 + total_ret) ** (TRADING_DAYS / max(n_days, 1)) - 1)
    mdd            = max_drawdown(equity_arr)

    wins  = traded_returns[traded_returns > 0]
    losses = traded_returns[traded_returns < 0]

    return BacktestResult(
        strategy_name     = strategy_name,
        total_return      = round(total_ret, 4),
        annualised_return = round(ann_ret, 4),
        sharpe            = round(sharpe_ratio(returns_arr), 4),
        sortino           = round(sortino_ratio(returns_arr), 4),
        max_drawdown      = round(mdd, 4),
        calmar            = round(calmar_ratio(ann_ret, mdd), 4),
        precision_at_up   = round(precision_at_up(
                                df["pred_direction"].values,
                                df["actual_return"].values), 4),
        hit_rate          = round(hit_rate(
                                df["pred_direction"].values,
                                df["actual_return"].values), 4),
        profit_factor     = round(profit_factor(traded_returns), 4),
        n_trades          = n_trades,
        win_rate          = round(float(len(wins) / max(n_trades, 1)), 4),
        avg_win           = round(float(wins.mean())   if len(wins)   > 0 else 0.0, 5),
        avg_loss          = round(float(losses.mean()) if len(losses) > 0 else 0.0, 5),
        equity_curve      = [round(v, 6) for v in equity_arr.tolist()],
    )


# ══════════════════════════════════════════════
#  Benchmark Strategies
# ══════════════════════════════════════════════

def benchmark_buy_and_hold(df: pd.DataFrame) -> BacktestResult:
    """Always long — baseline every strategy must beat."""
    bh = df.copy()
    bh["pred_direction"] = 1
    bh["pred_proba"]     = 0.5
    return run_backtest(bh, strategy_name="Buy & Hold")


def benchmark_random(df: pd.DataFrame, seed: int = 42) -> BacktestResult:
    """Random 50/50 signal — checks if model adds any real value."""
    rng = np.random.default_rng(seed)
    rand = df.copy()
    rand["pred_direction"] = rng.integers(0, 2, size=len(rand))
    rand["pred_proba"]     = 0.5
    return run_backtest(rand, strategy_name="Random")


def benchmark_momentum(df: pd.DataFrame, lookback: int = 5) -> BacktestResult:
    """
    Buy if last N days were positive (simple momentum).
    Requires df sorted by date per ticker.
    """
    mom = df.copy().sort_values("date").reset_index(drop=True)
    rolling_ret = mom.groupby("ticker")["actual_return"].transform(
        lambda x: x.shift(1).rolling(lookback).mean()
    )
    mom["pred_direction"] = (rolling_ret > 0).astype(int).fillna(0)
    mom["pred_proba"]     = 0.5
    return run_backtest(mom, strategy_name=f"Momentum-{lookback}d")


# ══════════════════════════════════════════════
#  Walk-Forward Validation
# ══════════════════════════════════════════════

@dataclass
class WalkForwardFold:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    sharpe: float
    precision_at_up: float
    n_test_days: int


def walk_forward_split(
    df: pd.DataFrame,
    train_months: int = 6,
    test_months: int = 1,
) -> list[WalkForwardFold]:
    """
    Expanding-window walk-forward validation.
    For now returns fold metadata + computes metrics on model predictions
    (assumes df already has pred_direction and pred_proba columns).

    6-month train / 1-month test as per roadmap.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    min_date = df["date"].min()
    max_date = df["date"].max()

    folds = []
    fold_idx = 1
    train_start = min_date

    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end   = test_start + pd.DateOffset(months=test_months)

        if test_end > max_date:
            break

        test_mask = (df["date"] >= test_start) & (df["date"] < test_end)
        test_df   = df[test_mask]

        if len(test_df) < 5:
            break

        s = sharpe_ratio(test_df["actual_return"].values * test_df["pred_direction"].values)
        p = precision_at_up(test_df["pred_direction"].values, test_df["actual_return"].values)

        folds.append(WalkForwardFold(
            fold          = fold_idx,
            train_start   = str(train_start.date()),
            train_end     = str(train_end.date()),
            test_start    = str(test_start.date()),
            test_end      = str(test_end.date()),
            sharpe        = round(s, 4),
            precision_at_up = round(p, 4),
            n_test_days   = len(test_df),
        ))

        # Expanding window: train_start stays fixed, only test window moves
        test_start = test_end
        fold_idx  += 1

        if fold_idx > 24:  # safety cap
            break

    return folds


# ══════════════════════════════════════════════
#  Main Evaluator Class
# ══════════════════════════════════════════════

class Evaluator:
    """
    One-stop evaluation for any MMIS model stage.

    Args:
        predictions_df: DataFrame with columns —
            date, ticker, actual_return, pred_proba, pred_direction
            (optional: uncertainty, regime)

    Example:
        ev = Evaluator(df)
        report = ev.full_report()
        ev.print_summary(report)
        ev.save_report(report, "results/phase1_tft_baseline.json")
    """

    def __init__(self, predictions_df: pd.DataFrame):
        required = {"date", "ticker", "actual_return", "pred_proba", "pred_direction"}
        missing  = required - set(predictions_df.columns)
        if missing:
            raise ValueError(f"predictions_df missing columns: {missing}")

        self.df = predictions_df.copy()
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df = self.df.sort_values("date").reset_index(drop=True)

    # ── Main evaluation ──────────────────────

    def full_report(
        self,
        model_name: str = "MMIS Model",
        use_uncertainty_filter: bool = False,
        uncertainty_threshold: float = 0.02,
        run_walk_forward: bool = True,
    ) -> dict:
        """
        Returns a full evaluation report as a dict.
        Includes: model backtest, benchmarks, calibration, walk-forward.
        """
        df = self.df

        # 1. Model backtest. When filtering is on, derive the threshold from the
        #    uncertainty DISTRIBUTION (drop the most-uncertain 20%) instead of a
        #    fixed absolute value — MC-Dropout variance is tiny (~1e-3), so any
        #    fixed 0.02-style cutoff would filter nothing.
        unc_thr = uncertainty_threshold
        if use_uncertainty_filter and "uncertainty" in df.columns:
            unc_thr = float(np.quantile(df["uncertainty"].values, 0.80))
            print(f"   Uncertainty filter: keeping predictions with "
                  f"variance <= {unc_thr:.6f} (lowest 80%)")
        model_bt = run_backtest(
            df,
            strategy_name=model_name,
            use_uncertainty_filter=use_uncertainty_filter,
            uncertainty_threshold=unc_thr,
        )

        # 2. Benchmarks
        bh_bt  = benchmark_buy_and_hold(df)
        rnd_bt = benchmark_random(df)
        mom_bt = benchmark_momentum(df)

        # 3. Calibration
        actual_dir = (df["actual_return"] > 0).astype(int).values
        ece, cal_bins = expected_calibration_error(df["pred_proba"].values, actual_dir)

        # 4. Per-ticker breakdown
        per_ticker = {}
        for ticker, tdf in df.groupby("ticker"):
            bt = run_backtest(tdf.copy(), strategy_name=ticker)
            per_ticker[ticker] = {
                "sharpe":         bt.sharpe,
                "precision_at_up": bt.precision_at_up,
                "total_return":   bt.total_return,
                "n_trades":       bt.n_trades,
            }

        # 5. Regime breakdown (Phase 5 — graceful if column absent)
        per_regime = {}
        if "regime" in df.columns:
            for regime, rdf in df.groupby("regime"):
                bt = run_backtest(rdf.copy(), strategy_name=f"regime_{regime}")
                per_regime[str(regime)] = {
                    "sharpe":         bt.sharpe,
                    "precision_at_up": bt.precision_at_up,
                    "total_return":   bt.total_return,
                }

        # 6. Uncertainty filter analysis (Phase 5) — percentile-based.
        #    Does keeping only the most-confident predictions improve results?
        #    That's the whole point of MC-Dropout. We sweep "keep lowest-X%
        #    uncertainty" so it's meaningful regardless of the variance scale.
        uncertainty_analysis = {}
        if "uncertainty" in df.columns:
            unc = df["uncertainty"].values
            for keep_frac in [0.5, 0.7, 0.8, 0.9, 1.0]:
                thr = float(np.quantile(unc, keep_frac))
                filtered = df[df["uncertainty"] <= thr]
                if len(filtered) > 10:
                    bt = run_backtest(filtered.copy(),
                                      strategy_name=f"keep_{int(keep_frac*100)}pct")
                    uncertainty_analysis[f"keep_lowest_{int(keep_frac*100)}pct"] = {
                        "threshold":       round(thr, 6),
                        "sharpe":          bt.sharpe,
                        "precision_at_up": bt.precision_at_up,
                        "n_trades":        bt.n_trades,
                        "coverage":        round(len(filtered) / len(df), 3),
                    }

        # 7. Walk-forward
        wf_folds = []
        if run_walk_forward and len(df) > 100:
            folds = walk_forward_split(df)
            wf_folds = [asdict(f) for f in folds]

        # Build final report
        def _bt_to_dict(bt: BacktestResult) -> dict:
            d = asdict(bt)
            d.pop("equity_curve")  # too large for report JSON
            return d

        report = {
            "model_name": model_name,
            "evaluation_date": str(pd.Timestamp.now().date()),
            "n_predictions": len(df),
            "n_tickers": df["ticker"].nunique(),
            "date_range": {
                "start": str(df["date"].min().date()),
                "end":   str(df["date"].max().date()),
            },
            "model": _bt_to_dict(model_bt),
            "benchmarks": {
                "buy_and_hold": _bt_to_dict(bh_bt),
                "random":       _bt_to_dict(rnd_bt),
                "momentum_5d":  _bt_to_dict(mom_bt),
            },
            "calibration": {
                "ece": round(ece, 5),
                "interpretation": _interpret_ece(ece),
                "bins": cal_bins,
            },
            "per_ticker": per_ticker,
            "per_regime": per_regime,
            "uncertainty_filter": uncertainty_analysis,
            "walk_forward": {
                "folds": wf_folds,
                "mean_sharpe": round(
                    np.mean([f["sharpe"] for f in wf_folds]), 4
                ) if wf_folds else None,
                "mean_precision_at_up": round(
                    np.mean([f["precision_at_up"] for f in wf_folds]), 4
                ) if wf_folds else None,
            },
            "vs_benchmarks": {
                "sharpe_vs_bh":        round(model_bt.sharpe - bh_bt.sharpe, 4),
                "sharpe_vs_random":    round(model_bt.sharpe - rnd_bt.sharpe, 4),
                "return_vs_bh":        round(model_bt.total_return - bh_bt.total_return, 4),
            },
        }

        return report

    # ── Equity curve (for plotting) ──────────

    def equity_curves(self) -> pd.DataFrame:
        """Returns equity curves for model + all benchmarks — ready to plot."""
        df = self.df

        model_bt = run_backtest(df.copy(), "Model")
        bh_bt    = benchmark_buy_and_hold(df.copy())
        rnd_bt   = benchmark_random(df.copy())

        min_len = min(len(model_bt.equity_curve),
                      len(bh_bt.equity_curve),
                      len(rnd_bt.equity_curve))

        return pd.DataFrame({
            "Model":        model_bt.equity_curve[:min_len],
            "Buy & Hold":   bh_bt.equity_curve[:min_len],
            "Random":       rnd_bt.equity_curve[:min_len],
        })

    # ── Print summary ────────────────────────

    @staticmethod
    def print_summary(report: dict):
        """Clean terminal output of the most important metrics."""
        m  = report["model"]
        bh = report["benchmarks"]["buy_and_hold"]
        vs = report["vs_benchmarks"]

        sep = "─" * 55

        print(f"\n{'═'*55}")
        print(f"  MMIS Evaluation Report — {report['model_name']}")
        print(f"  {report['date_range']['start']} → {report['date_range']['end']}")
        print(f"  {report['n_predictions']} predictions | {report['n_tickers']} tickers")
        print(f"{'═'*55}")

        print(f"\n📈  RETURNS")
        print(f"  Total Return     : {m['total_return']:>+.2%}  (BH: {bh['total_return']:>+.2%})")
        print(f"  Annualised Return: {m['annualised_return']:>+.2%}  (BH: {bh['annualised_return']:>+.2%})")

        print(f"\n⚡  RISK-ADJUSTED")
        print(f"  Sharpe Ratio   : {m['sharpe']:>6.3f}  (BH: {bh['sharpe']:>6.3f})  Δ {vs['sharpe_vs_bh']:>+.3f}")
        print(f"  Sortino Ratio  : {m['sortino']:>6.3f}")
        print(f"  Calmar Ratio   : {m['calmar']:>6.3f}")

        print(f"\n📉  RISK")
        print(f"  Max Drawdown   : {m['max_drawdown']:>6.2%}  (BH: {bh['max_drawdown']:>6.2%})")
        print(f"  Profit Factor  : {m['profit_factor']:>6.3f}")

        print(f"\n🎯  PREDICTION QUALITY")
        print(f"  Precision@UP   : {m['precision_at_up']:>6.2%}  ← primary metric")
        print(f"  Hit Rate       : {m['hit_rate']:>6.2%}  (not primary)")
        print(f"  Win Rate       : {m['win_rate']:>6.2%}")

        print(f"\n🔬  CALIBRATION")
        print(f"  ECE Score      : {report['calibration']['ece']:.5f}")
        print(f"  Interpretation : {report['calibration']['interpretation']}")

        print(f"\n📊  TRADING ACTIVITY")
        print(f"  # Trades       : {m['n_trades']}")
        print(f"  Avg Win        : {m['avg_win']:>+.4%}")
        print(f"  Avg Loss       : {m['avg_loss']:>+.4%}")

        if report["walk_forward"]["folds"]:
            wf = report["walk_forward"]
            print(f"\n🔄  WALK-FORWARD ({len(wf['folds'])} folds)")
            print(f"  Mean Sharpe    : {wf['mean_sharpe']:.4f}")
            print(f"  Mean Prec@UP   : {wf['mean_precision_at_up']:.2%}")

        if report["per_regime"]:
            print(f"\n🌊  REGIME BREAKDOWN")
            for reg, stats in report["per_regime"].items():
                print(f"  {reg:<18}: Sharpe={stats['sharpe']:>6.3f}  Prec@UP={stats['precision_at_up']:.2%}")

        print(f"\n{'═'*55}\n")

    # ── Save report ──────────────────────────

    @staticmethod
    def save_report(report: dict, path: str = "results/evaluation_report.json"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"✅  Report saved → {path}")

    # ── Load from DB ─────────────────────────

    @classmethod
    def from_sqlite(
        cls,
        db_path: str,
        predictions_table: str = "predictions",
    ) -> "Evaluator":
        """
        Load predictions directly from mmis.db.
        Expects a 'predictions' table with the required columns.
        Create this table when you save model outputs in later phases.
        """
        conn = sqlite3.connect(db_path)
        df   = pd.read_sql(f"SELECT * FROM {predictions_table}", conn)
        conn.close()
        return cls(df)


# ══════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════

def _interpret_ece(ece: float) -> str:
    if ece < 0.01:
        return "Excellent — near-perfect calibration"
    elif ece < 0.03:
        return "Good calibration"
    elif ece < 0.07:
        return "Moderate — model is somewhat overconfident"
    elif ece < 0.15:
        return "Poor — confidence scores don't reflect real win rates"
    else:
        return "Very poor calibration — probabilities are unreliable"


def mock_predictions(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generates realistic fake predictions for testing.
    Replace with real model output in later phases.
    """
    rng     = np.random.default_rng(seed)
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"]
    dates   = pd.date_range("2023-01-01", periods=n // len(tickers), freq="B")

    rows = []
    for ticker in tickers:
        for date in dates:
            actual_return  = rng.normal(0.0005, 0.015)   # realistic daily return
            signal_quality = rng.uniform(0.52, 0.62)     # model has slight edge
            pred_proba     = signal_quality + rng.normal(0, 0.05)
            pred_proba     = float(np.clip(pred_proba, 0.05, 0.95))
            pred_direction = 1 if pred_proba > 0.5 else 0
            uncertainty    = abs(rng.normal(0, 0.015))   # MC Dropout variance

            rows.append({
                "date":           date,
                "ticker":         ticker,
                "actual_return":  actual_return,
                "pred_proba":     pred_proba,
                "pred_direction": pred_direction,
                "uncertainty":    uncertainty,
                "regime":         rng.choice(["trending", "mean_reverting", "high_vol"]),
            })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════
#  CLI Entry Point
# ══════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MMIS Evaluation Engine")
    parser.add_argument("--db",    type=str, default=None,
                        help="Path to mmis.db (loads from 'predictions' table)")
    parser.add_argument("--csv",   type=str, default=None,
                        help="Path to predictions CSV file")
    parser.add_argument("--mock",  action="store_true",
                        help="Run on mock data (for testing)")
    parser.add_argument("--out",   type=str, default="results/evaluation_report.json",
                        help="Output path for JSON report")
    parser.add_argument("--name",  type=str, default="MMIS Model",
                        help="Model name shown in report")
    parser.add_argument("--uncertainty-filter", action="store_true",
                        help="Apply MC Dropout uncertainty filter (Phase 5)")
    args = parser.parse_args()

    # Load data
    if args.mock:
        print("🔧  Running on MOCK predictions (for testing)...")
        df = mock_predictions(n=600)
        ev = Evaluator(df)

    elif args.csv:
        print(f"📂  Loading predictions from {args.csv}...")
        df = pd.read_csv(args.csv)
        ev = Evaluator(df)

    elif args.db:
        print(f"🗄️  Loading predictions from {args.db}...")
        ev = Evaluator.from_sqlite(args.db)

    else:
        print("⚠️  No input specified. Running mock demo. Use --mock / --csv / --db")
        df = mock_predictions(n=600)
        ev = Evaluator(df)

    # Run evaluation
    report = ev.full_report(
        model_name=args.name,
        use_uncertainty_filter=args.uncertainty_filter,
    )

    # Print & save
    Evaluator.print_summary(report)
    Evaluator.save_report(report, args.out)

    # Also print equity curves head
    curves = ev.equity_curves()
    print("\n📈  Equity curve tail (last 5 rows):")
    print(curves.tail())