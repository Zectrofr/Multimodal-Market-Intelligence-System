"""
har_baseline.py — T4: classical baselines for the forward volatility target.

This repository has a carefully built target (T3) and, until now, no null model.
A volatility forecaster with no classical baseline is one a practitioner
dismisses in a single question, so this module builds the baseline the
multimodal stack will have to beat: HAR-RV, plus two cruder references.

Four forecasts of market_data.fwd_rv_5d, all scored on identical rows:

  1. HAR-RV (level)  — OLS on the level of forward RV. THE HEADLINE BASELINE.
  2. HAR-RV (log)    — OLS in log space, back-transformed. Volatility is
                       right-skewed, so this is the more standard specification;
                       reported as a secondary row.
  3. naive persistence — predict the backward 5-day RV. No fitting at all.
  4. constant median   — predict the training median. The "does HAR earn its
                       three parameters?" reference.

Nothing here trains, touches or evaluates the fusion model, and nothing here
writes to the database. The database is opened read-only, by URI.

──────────────────────────────────────────────────────────────────────────────
THE HAR SPECIFICATION, AND HOW IT IS ADAPTED

Corsi (2009), "A Simple Approximate Long-Memory Model of Realized Volatility",
regresses future realized volatility on realized volatility measured over three
horizons — daily, weekly, monthly — to mimic the heterogeneous horizons over
which different market participants operate. It is deliberately simple and it
is genuinely hard to beat.

THIS IS AN ADAPTATION, NOT THE ORIGINAL SPECIFICATION. Corsi builds a DAILY RV
from INTRADAY returns: with 5-minute bars, a single day contains ~78 returns, so
one day's realized volatility is estimable on its own. This dataset has daily
bars only. A one-day "realized volatility" from daily data would be the standard
deviation of a single return, which is not defined at ddof=1 and is meaningless
at ddof=0. The daily component therefore cannot be reproduced here.

What is used instead: backward-looking realized volatility over 5, 22 and 66
trading days, each ending at t INCLUSIVE. That mirrors the weekly / monthly /
quarterly structure of the original (Corsi's 1/5/22 shifted up one horizon),
which is the part of the model that carries the long-memory approximation. The
loss relative to the original is the shortest horizon, and with it some of the
model's responsiveness to very recent shocks.

Each component is computed with EXACTLY the target's formula — standard
deviation of log returns, ddof=1, annualised by sqrt(252) — so predictors and
target live on the same scale and a coefficient is directly interpretable.

──────────────────────────────────────────────────────────────────────────────
WHY THE PREDICTORS MAY INCLUDE ROW t'S OWN RETURN AND THE TARGET MAY NOT

The target (T3) is strictly forward: fwd_rv_5d[t] uses log_ret[t+1 .. t+5] and
excludes row t's own return, because a label must not be readable off its own
feature row.

The predictors here are the mirror image: every HAR component ends at t
INCLUSIVE, so log_ret[t] IS included. This asymmetry is correct and is the whole
point of the setup. A predictor is information available at decision time —
standing at the close of day t, having observed close[t], you know log_ret[t].
The target is the outcome that has not happened yet. Excluding log_ret[t] from
the predictors would throw away real, legitimately available information and
would handicap the baseline; including it in the target would be leakage. Same
row, opposite directions in time, opposite rules.

──────────────────────────────────────────────────────────────────────────────
THE EMBARGO — imported, never reimplemented

T3 established that "training row" cannot mean date <= TRAIN_END_DATE. A row
dated 2025-03-11 has a forward window running to 2025-03-18, which is validation
data; pooling such rows let post-boundary prices move a fitted parameter, and
T3 measured the leak. The eligibility rule is therefore imported from
rv_target.threshold_fit_mask rather than restated here, so the two cannot drift.

Training rows for this fit are the intersection of:
    * rv_target.threshold_fit_mask  (entire forward window <= TRAIN_END_DATE)
    * a complete 66-day backward lookback
    * a non-NULL target

──────────────────────────────────────────────────────────────────────────────
POOLED, NOT PER-TICKER

ONE model is fit across all six tickers, matching T3's pooled thresholds. A
pooled fit gives a single global baseline on the same scale the classes are
defined on, which is what a later model comparison needs.

The trade-off, stated rather than hidden: a pooled HAR forces one set of
coefficients onto SPY and TSLA alike, and their volatility levels differ by
roughly an order of magnitude in class terms (T3: SPY 5.6% turbulent, TSLA
78.3%). Six per-ticker fits would very likely score better. That would be a
different experiment, and a per-ticker baseline is not the thing a pooled
multimodal model must beat. Recorded as an open item rather than silently
chosen.

Usage:
    python har_baseline.py --all
"""

import argparse
import json
import logging
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

from regime import DB_PATH, MODEL_DIR
from regime_causal import TRAIN_END_DATE
from rv_target import (
    ANNUALISATION,
    DDOF,
    VOL_NAME_TO_ID,
    load_thresholds,
    threshold_fit_mask,
)

logger = logging.getLogger(__name__)

# Weekly / monthly / quarterly, in trading days. See the docstring for why the
# daily component of Corsi's original is absent.
HAR_WINDOWS = (5, 22, 66)
MAX_LOOKBACK = max(HAR_WINDOWS)

MODEL_PATH = MODEL_DIR / "har_baseline.json"

RESULTS_DIR = MODEL_DIR.parent / "results"
PREDICTIONS_CSV = RESULTS_DIR / "T4_BASELINE_PREDICTIONS.csv"

CLASS_NAMES = ["calm", "normal", "turbulent"]
CLASS_IDS = [VOL_NAME_TO_ID[n] for n in CLASS_NAMES]


# ══════════════════════════════════════════════════════════════════
#  Data — READ ONLY. This task writes nothing to the database.
# ══════════════════════════════════════════════════════════════════

def load_panel(db_path: str = DB_PATH) -> dict:
    """Load every ticker, date-ascending, through a READ-ONLY connection.

    Deliberately not regime.load_all_tickers, which opens the database
    read-write. T4 has no business holding a writable handle on the only copy
    of this data.
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        df = pd.read_sql(
            "SELECT date, ticker, close, fwd_rv_5d, vol_target, vol_target_id "
            "FROM market_data ORDER BY ticker, date",
            con,
        )
    finally:
        con.close()

    df["date"] = pd.to_datetime(df["date"])
    return {tkr: g.reset_index(drop=True) for tkr, g in df.groupby("ticker")}


# ══════════════════════════════════════════════════════════════════
#  B1 — HAR predictors
# ══════════════════════════════════════════════════════════════════

def backward_realized_vol(close, window: int) -> np.ndarray:
    """Annualised std-dev of the log returns over rows t-window+1 .. t INCLUSIVE.

    BACKWARD and inclusive of row t's own return — see the module docstring on
    why this asymmetry against the target is correct rather than sloppy.

    Index algebra, stated because it is the easy thing to get wrong by one:
        log_ret[i] = log(close[i] / close[i-1]);   log_ret[0] undefined
        r = log_ret[1:]      so r[j] = log_ret[j+1]
        row t wants           log_ret[t-window+1 .. t] == r[t-window .. t-1]
    which is the (t-window)-th window of a length-`window` sliding view over r.
    So out[t] is defined for t >= window, and rows 0 .. window-1 are NaN.

    Each window is reduced INDEPENDENTLY rather than through pandas'
    rolling().std(), for the bit-locality reason T3 recorded: rolling
    accumulates a running variance left to right, so its output at one index
    carries float error from every earlier row. Independent windows make each
    predictor an exact function of its own closes and nothing else.

    Rows without a complete lookback are NaN. Never 0.0, never forward-filled —
    the same rule the target follows, for the same reason (U3).
    """
    close = np.asarray(close, dtype=float)
    n = close.size

    out = np.full(n, np.nan, dtype=float)
    if n < window + 1:
        return out

    r = np.log(close[1:]) - np.log(close[:-1])          # r[j] = log_ret[j+1]
    windows = np.lib.stride_tricks.sliding_window_view(r, window)

    # windows[k] == r[k .. k+window-1] == log_ret[k+1 .. k+window]
    # Row t wants k = t - window, valid for t = window .. n-1.
    out[window:] = windows.std(axis=1, ddof=DDOF) * ANNUALISATION
    return out


def add_har_predictors(df: pd.DataFrame) -> pd.DataFrame:
    """Attach har_rv_5 / har_rv_22 / har_rv_66 to one ticker's frame."""
    dates = pd.to_datetime(df["date"])
    if not dates.is_monotonic_increasing:
        raise ValueError(
            "rows are not in ascending date order; a backward window computed "
            "over unsorted rows would silently look sideways, not backward."
        )

    out = df.copy()
    for w in HAR_WINDOWS:
        out[f"har_rv_{w}"] = backward_realized_vol(out["close"].values, w)
    return out


def predictor_columns() -> list:
    return [f"har_rv_{w}" for w in HAR_WINDOWS]


def complete_predictors(df: pd.DataFrame) -> np.ndarray:
    """Rows whose full 66-day lookback exists. Excluded from fit AND scoring."""
    cols = df[predictor_columns()].to_numpy(dtype=float)
    return np.isfinite(cols).all(axis=1)


def build_design(frames: dict) -> pd.DataFrame:
    """One pooled frame carrying predictors, target, and period membership."""
    parts = []
    for tkr in sorted(frames):
        d = add_har_predictors(frames[tkr])
        d["embargo_eligible"] = threshold_fit_mask(d["date"])
        d["has_predictors"] = complete_predictors(d)
        d["has_target"] = np.isfinite(d["fwd_rv_5d"].to_numpy(dtype=float))
        d["after_boundary"] = (
            pd.to_datetime(d["date"]) > pd.Timestamp(TRAIN_END_DATE)
        ).to_numpy()
        parts.append(d)

    panel = pd.concat(parts, ignore_index=True)
    panel["is_train"] = (
        panel["embargo_eligible"] & panel["has_predictors"] & panel["has_target"]
    )
    panel["is_oos"] = (
        panel["after_boundary"] & panel["has_predictors"] & panel["has_target"]
    )
    return panel


# ══════════════════════════════════════════════════════════════════
#  B2 — train-only OLS
# ══════════════════════════════════════════════════════════════════

def fit_ols(X: np.ndarray, y: np.ndarray) -> dict:
    """Plain OLS via np.linalg.lstsq, with textbook standard errors.

    statsmodels is not installed, so the inference is computed directly:
        sigma^2 = RSS / (n - k)
        Var(b)  = sigma^2 * (X'X)^-1
    lstsq is a single deterministic LAPACK call on an explicit design matrix,
    which keeps the fit bit-reproducible across runs.
    """
    n, k = X.shape
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    if rank < k:
        raise ValueError(f"design matrix is rank-deficient: rank {rank} < {k}")

    resid = y - X @ beta
    dof = n - k
    sigma2 = float(resid @ resid / dof)
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(sigma2 * np.diag(xtx_inv))
    tstat = beta / se
    pval = 2.0 * stats.t.sf(np.abs(tstat), df=dof)

    return {
        "coefficients": [float(b) for b in beta],
        "std_errors": [float(s) for s in se],
        "t_statistics": [float(t) for t in tstat],
        "p_values": [float(p) for p in pval],
        "n": int(n),
        "dof": int(dof),
        "residual_sigma2": sigma2,
        "rank": int(rank),
    }


def design_matrix(frame: pd.DataFrame, log_space: bool = False) -> np.ndarray:
    """[1, rv_5, rv_22, rv_66], or their logs."""
    x = frame[predictor_columns()].to_numpy(dtype=float)
    if log_space:
        x = np.log(x)
    return np.column_stack([np.ones(len(x)), x])


def fit_har(panel: pd.DataFrame) -> dict:
    """Fit both HAR variants on embargo-eligible training rows only."""
    train = panel[panel["is_train"]]
    if train.empty:
        raise ValueError("no eligible training rows; refusing to fit.")

    y = train["fwd_rv_5d"].to_numpy(dtype=float)
    if (y <= 0).any():
        raise ValueError("non-positive target values; log fit would be undefined.")

    level = fit_ols(design_matrix(train, log_space=False), y)
    log = fit_ols(design_matrix(train, log_space=True), np.log(y))

    # Back-transforming exp(X'b) yields the conditional MEDIAN of a lognormal,
    # not its mean, so the log model systematically under-predicts on a
    # squared-error scale. The Duan/smearing correction exp(sigma^2 / 2) is
    # recorded here so the log row is not silently handicapped; the headline
    # log prediction uses the plain exp, and both are reported.
    log["lognormal_correction_factor"] = float(np.exp(log["residual_sigma2"] / 2.0))

    per_ticker = train.groupby("ticker").size().to_dict()
    return {
        "level": level,
        "log": log,
        "predictor_names": ["intercept"] + predictor_columns(),
        "har_windows": list(HAR_WINDOWS),
        "train_end_date": TRAIN_END_DATE,
        "train_rows": int(len(train)),
        "train_rows_per_ticker": {k: int(v) for k, v in per_ticker.items()},
        "train_mean_fwd_rv": float(y.mean()),
        "train_median_fwd_rv": float(np.median(y)),
        "embargo_rule": (
            "rv_target.threshold_fit_mask — entire 5-day forward window on or "
            "before TRAIN_END_DATE — intersected with a complete 66-day lookback"
        ),
        "pooled_across_tickers": True,
        "annualisation_factor": float(ANNUALISATION),
        "ddof": DDOF,
        "fitted_at": datetime.now(timezone.utc).isoformat(),
    }


def save_model(model: dict, path=MODEL_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2, sort_keys=True)
    logger.info(f"  HAR model → {path}")


def load_model(path=MODEL_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════
#  B3 — the forecasts
# ══════════════════════════════════════════════════════════════════

def predict_har_level(frame: pd.DataFrame, model: dict) -> np.ndarray:
    return design_matrix(frame, log_space=False) @ np.asarray(
        model["level"]["coefficients"], dtype=float
    )


def predict_har_log(frame: pd.DataFrame, model: dict) -> np.ndarray:
    """Plain exp back-transform — the conditional median. See fit_har."""
    return np.exp(
        design_matrix(frame, log_space=True)
        @ np.asarray(model["log"]["coefficients"], dtype=float)
    )


def predict_naive(frame: pd.DataFrame) -> np.ndarray:
    """Persistence: tomorrow's 5-day RV looks like the last 5 days'. No fit."""
    return frame["har_rv_5"].to_numpy(dtype=float)


def predict_constant(frame: pd.DataFrame, model: dict) -> np.ndarray:
    return np.full(len(frame), model["train_median_fwd_rv"], dtype=float)


def all_predictions(frame: pd.DataFrame, model: dict) -> dict:
    return {
        "har_level": predict_har_level(frame, model),
        "har_log": predict_har_log(frame, model),
        "naive": predict_naive(frame),
        "constant": predict_constant(frame, model),
    }


# ══════════════════════════════════════════════════════════════════
#  B4 — metrics
# ══════════════════════════════════════════════════════════════════

def qlike(actual: np.ndarray, predicted: np.ndarray, floor: float) -> dict:
    """QLIKE in VARIANCE space.

        s2    = actual^2        (actual is annualised VOLATILITY, so variance
        s2hat = predicted^2      is its square — stated because QLIKE is not
                                 scale-free and the space matters)

        QLIKE = s2/s2hat - log(s2/s2hat) - 1

    Lower is better; it is 0 exactly when s2hat == s2 and strictly positive
    otherwise, since x - log(x) - 1 >= 0 with equality only at x = 1. Unlike
    MSE it is asymmetric, penalising under-prediction of variance far more
    heavily — which is the property that makes it the standard volatility loss.

    Non-positive predictions are undefined here (log of a non-positive number),
    so they are floored. The count is RETURNED, never silently absorbed.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    n_floored = int((predicted <= 0).sum())
    safe = np.where(predicted <= 0, floor, predicted)

    ratio = (actual ** 2) / (safe ** 2)
    values = ratio - np.log(ratio) - 1.0
    return {
        "qlike": float(values.mean()),
        "n_floored": n_floored,
        "floor_used": float(floor),
    }


def regression_metrics(actual: np.ndarray, predicted: np.ndarray,
                       train_mean: float) -> dict:
    """MSE / RMSE / MAE on the VOLATILITY level, plus R^2.

    R^2 is computed against the TRAINING mean, not the mean of the evaluation
    sample:
        R2 = 1 - SSE / sum((actual - train_mean)^2)
    Using the validation mean would let the held-out sample supply its own
    benchmark — a forecaster cannot know the validation mean in advance, and
    scoring against it would quietly leak. For the in-sample rows the two
    coincide by construction.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    err = actual - predicted
    sse = float(err @ err)
    sst = float(((actual - train_mean) ** 2).sum())

    return {
        "mse": sse / len(actual),
        "rmse": float(np.sqrt(sse / len(actual))),
        "mae": float(np.abs(err).mean()),
        "r2": 1.0 - sse / sst,
        "n": int(len(actual)),
    }


def bucket(predicted: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Class ids via the FROZEN thresholds. Never refit to the predictions.

    Convention identical to T3: rv < lower -> calm(0); lower <= rv < upper ->
    normal(1); rv >= upper -> turbulent(2). Each cut inclusive on its upper side.
    """
    predicted = np.asarray(predicted, dtype=float)
    out = np.full(predicted.shape, VOL_NAME_TO_ID["normal"], dtype=int)
    out[predicted < lower] = VOL_NAME_TO_ID["calm"]
    out[predicted >= upper] = VOL_NAME_TO_ID["turbulent"]
    return out


def classification_metrics(true_ids: np.ndarray, pred_ids: np.ndarray) -> dict:
    """Macro-F1, per-class precision/recall/F1, and a 3x3 confusion matrix.

    Macro-F1 rather than accuracy because the evaluation period is not balanced
    (T3 measured validation at 38.28 / 35.10 / 26.62), so accuracy rewards a
    forecaster for leaning on the majority class. Macro-F1 weights all three
    classes equally regardless of how often they occur.
    """
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        true_ids, pred_ids, labels=CLASS_IDS, zero_division=0
    )
    cm = confusion_matrix(true_ids, pred_ids, labels=CLASS_IDS)

    return {
        "macro_f1": float(np.mean(f1)),
        "accuracy": float((np.asarray(true_ids) == np.asarray(pred_ids)).mean()),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": cm.tolist(),
        "confusion_rows": "true",
        "confusion_labels": CLASS_NAMES,
    }


def score(frame: pd.DataFrame, predicted: np.ndarray, model: dict,
          thresholds: dict, floor: float) -> dict:
    actual = frame["fwd_rv_5d"].to_numpy(dtype=float)
    true_ids = frame["vol_target_id"].to_numpy(dtype=int)

    out = regression_metrics(actual, predicted, model["train_mean_fwd_rv"])
    out.update(qlike(actual, predicted, floor))
    out.update(
        classification_metrics(
            true_ids, bucket(predicted, thresholds["lower"], thresholds["upper"])
        )
    )
    return out


# ══════════════════════════════════════════════════════════════════
#  Pipeline
# ══════════════════════════════════════════════════════════════════

def run_baselines(db_path: str = DB_PATH, save: bool = True) -> dict:
    thresholds = load_thresholds()
    panel = build_design(load_panel(db_path))

    model = fit_har(panel)
    if save:
        save_model(model)

    train = panel[panel["is_train"]]
    oos = panel[panel["is_oos"]]
    logger.info(f"  train rows {len(train)}  |  out-of-sample rows {len(oos)}")

    # The QLIKE floor: the smallest forward RV actually observed in training.
    # Any prediction at or below zero is replaced by it, and the substitution
    # count is reported per model rather than absorbed.
    floor = float(train["fwd_rv_5d"].min())

    results = {"train": {}, "oos": {}}
    for period, frame in (("train", train), ("oos", oos)):
        for name, pred in all_predictions(frame, model).items():
            results[period][name] = score(frame, pred, model, thresholds, floor)

    return {
        "model": model,
        "thresholds": thresholds,
        "results": results,
        "panel": panel,
        "floor": floor,
        "n_train": int(len(train)),
        "n_oos": int(len(oos)),
    }


def write_predictions_csv(out: dict, path=PREDICTIONS_CSV) -> int:
    """The artefact a later model comparison joins against."""
    panel = out["panel"]
    thresholds = out["thresholds"]
    lo, hi = thresholds["lower"], thresholds["upper"]

    rows = panel[panel["is_train"] | panel["is_oos"]].copy()
    preds = all_predictions(rows, out["model"])

    frame = pd.DataFrame({
        "date": pd.to_datetime(rows["date"]).dt.strftime("%Y-%m-%d"),
        "ticker": rows["ticker"].values,
        "split": np.where(rows["is_train"], "train", "validation"),
        "fwd_rv_5d": rows["fwd_rv_5d"].values,
        "vol_target_id": rows["vol_target_id"].values.astype(int),
        "har_level_pred": preds["har_level"],
        "har_log_pred": preds["har_log"],
        "naive_pred": preds["naive"],
        "constant_pred": preds["constant"],
        "har_level_class": bucket(preds["har_level"], lo, hi),
        "har_log_class": bucket(preds["har_log"], lo, hi),
        "naive_class": bucket(preds["naive"], lo, hi),
        "constant_class": bucket(preds["constant"], lo, hi),
    }).sort_values(["ticker", "date"])

    frame.to_csv(path, index=False)
    logger.info(f"  Predictions → {path} ({len(frame)} rows)")
    return len(frame)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    parser = argparse.ArgumentParser(description="MMIS T4 — classical baselines")
    parser.add_argument("--db", type=str, default=DB_PATH, help="Path to mmis.db")
    parser.add_argument("--all", action="store_true", help="Fit and score everything")
    parser.add_argument("--no-save", action="store_true", help="Do not persist")
    args = parser.parse_args()

    if args.all:
        out = run_baselines(db_path=args.db, save=not args.no_save)
        if not args.no_save:
            write_predictions_csv(out)
        for period in ("train", "oos"):
            logger.info(f"\n── {period} ──")
            for name, m in out["results"][period].items():
                logger.info(
                    f"  {name:10s} QLIKE={m['qlike']:.6f} RMSE={m['rmse']:.6f} "
                    f"MAE={m['mae']:.6f} R2={m['r2']:.6f} macroF1={m['macro_f1']:.6f}"
                )
    else:
        print("⚠️  Specify --all")
