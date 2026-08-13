"""
rv_target.py — T3: forward realized volatility target.

Replaces the prediction problem, not the machinery. market_data.target is a
3-class NEXT-DAY DIRECTION label (ingest.py:135-140, FLAT_THRESHOLD = 0.005).
Daily directional prediction on liquid large-caps is arbitraged away; volatility
clustering is not. This module defines a 5-day forward realized volatility label
in three classes — calm / normal / turbulent — written to SHADOW columns so the
directional target, and everything that reads it, is untouched.

Nothing here trains, evaluates, or predicts. It builds a label and proves the
label cannot see the future.

──────────────────────────────────────────────────────────────────────────────
POOLED, NOT PER-TICKER THRESHOLDS

The terciles are fit on ONE pooled distribution across all six tickers, not six
per-ticker distributions. A single global definition of "turbulent" makes the
class comparable ACROSS tickers, which is what the cross-sectional regime
analysis this project exists to do requires: "AAPL is turbulent while SPY is
calm" is only a meaningful sentence under a shared scale.

The trade-off is real and is not a defect. SPY will be labelled turbulent far
less often than TSLA, and calm far more often, because SPY is a diversified
index and TSLA is a single high-beta name. That is a property of the market, not
an artefact of the labelling. Per-ticker terciles would force every ticker to be
turbulent exactly one third of the time, which would destroy precisely the
cross-sectional signal the pooled scale preserves.

──────────────────────────────────────────────────────────────────────────────
WHY THE THRESHOLD FIT EXCLUDES THE LAST 5 TRAINING ROWS

The training boundary is TRAIN_END_DATE, imported from regime_causal rather
than redefined so the two cannot drift apart.

A row dated on or before TRAIN_END_DATE is a training row. But its LABEL looks
5 trading days forward, so for the last 5 training rows that forward window
straddles the boundary and is computed from validation-period closes. Pooling
those 5 rows per ticker into the threshold fit would let post-boundary prices
move the thresholds — a fitted parameter that is then applied to the validation
period. That is exactly the leak this task exists to rule out.

So threshold eligibility is stricter than "date <= TRAIN_END_DATE": a row
contributes to the fit only if its ENTIRE forward window also lies on or before
TRAIN_END_DATE. This costs 5 rows per ticker (30 pooled) and is what makes the
thresholds provably independent of validation data.

The excluded rows still RECEIVE a label. A forward label on a training row is
supposed to depend on the following 5 days; that is what "forward" means. What
must not happen is those days informing a parameter applied to held-out data.

Consequence for a later training task, recorded here and not acted on: the last
5 training rows carry labels that peek past the boundary, so any model trained
on this target should EMBARGO them. This module does not enforce that — it only
declines to let them touch the thresholds.

Usage:
    python rv_target.py --all
"""

import argparse
import json
import logging
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from regime import DB_PATH, MODEL_DIR, load_all_tickers
from regime_causal import TRAIN_END_DATE, train_mask

logger = logging.getLogger(__name__)

# ── Label definition ───────────────────────────────────────────────────────
# Forward horizon in TRADING days (rows), not calendar days. The frame is one
# row per trading session, so t+1..t+5 is the next five sessions.
HORIZON = 5

# Daily volatility -> annualised, the market convention. 252 trading days/year.
ANNUALISATION = np.sqrt(252.0)

# Sample standard deviation. ddof=1 because the 5 forward returns are a SAMPLE
# drawn from the period's return distribution, not the whole population of it;
# ddof=0 would bias the estimate low, and biased low by a factor that depends on
# the window length would make the label incomparable across any future change
# of HORIZON.
DDOF = 1

# Terciles. Stated to the precision the task fixed them at, not as 100/3.
LOWER_PERCENTILE = 33.33
UPPER_PERCENTILE = 66.67

# Fixed, ticker-independent name -> id map, mirroring regime_causal's
# REGIME_NAME_TO_ID. vol_target_id = 2 means "turbulent" on every ticker.
VOL_TARGET_NAMES = {0: "calm", 1: "normal", 2: "turbulent"}
VOL_NAME_TO_ID = {name: vid for vid, name in VOL_TARGET_NAMES.items()}

THRESHOLDS_PATH = MODEL_DIR / "rv_thresholds.json"


def forward_realized_vol(close, horizon: int = HORIZON) -> np.ndarray:
    """Annualised std-dev of the log returns over rows t+1 .. t+horizon.

    STRICTLY FORWARD. Row t's window starts at t+1, so row t's OWN return
    log(close[t] / close[t-1]) is never in it. This is not a stylistic choice:
    if the label included the return realised on the feature row's own date, the
    label would be partly readable off the feature row itself, and any model
    would learn that shortcut rather than the forecasting problem.

    A subtlety worth stating because it looks like a violation and is not:
    log_ret[t+1] = log(close[t+1] / close[t]) does reference close[t]. Row t's
    own CLOSE is therefore an input to the first forward return. Its own RETURN
    is not. Knowing close[t] alone reveals nothing about the label, because
    every forward return also needs a price that has not happened yet.

    Index algebra, since it is the part that is easy to get wrong:
        log_ret[i]        = log(close[i] / close[i-1]);  log_ret[0] is undefined
        r = log_ret[1:]   so r[j] = log_ret[j+1]
        row t wants        log_ret[t+1 .. t+h]  ==  r[t .. t+h-1]
    which is exactly the t-th window of a length-h sliding view over r.

    Each window is reduced INDEPENDENTLY, deliberately, rather than through
    pandas' rolling().std(). Rolling accumulates a running variance left to
    right, so its output at one index carries float error from every earlier
    row: perturbing close[t-1] shifted fwd_rv[t] by ~1e-13 even though every
    return in row t's window was unchanged. That is past->future rounding noise,
    not lookahead — rolling accumulates forward, so a LATER mutation can never
    move an EARLIER output, and the thresholds were bit-identical either way.
    But the claim this label has to support is that fwd_rv[t] is a function of
    close[t .. t+h] and of nothing else, and "nothing else" should mean exactly
    that. Independent windows make it exact and bit-reproducible.

    Rows with fewer than `horizon` complete forward returns are NaN. Never 0.0,
    never a default, never forward-filled. This is the U3 lesson
    (docs/STATE_REPORT.md:577): ingest.py's directional target defaults the
    final row per ticker to 1 (Flat) because a bare Series carried the NaN and
    dropna() could not see it, fabricating six labels at the newest dates — the
    rows live inference touches first. A NaN that cannot be mistaken for a class
    is the whole point.
    """
    close = np.asarray(close, dtype=float)
    n = close.size

    out = np.full(n, np.nan, dtype=float)
    if n < horizon + 1:
        # Not one complete forward window anywhere. All NaN, and no exception:
        # too short is an absence of labels, not an error.
        return out

    r = np.log(close[1:]) - np.log(close[:-1])          # r[j] = log_ret[j+1]
    windows = np.lib.stride_tricks.sliding_window_view(r, horizon)

    # windows[t] == r[t .. t+horizon-1] == log_ret[t+1 .. t+horizon]
    out[: windows.shape[0]] = windows.std(axis=1, ddof=DDOF) * ANNUALISATION
    return out


def add_forward_rv(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Attach fwd_rv_5d to one ticker's frame, which must be date-ascending."""
    dates = pd.to_datetime(df["date"])
    if not dates.is_monotonic_increasing:
        raise ValueError(
            "rows are not in ascending date order; a forward window computed "
            "over unsorted rows would silently look sideways, not forward."
        )

    out = df.copy()
    out["fwd_rv_5d"] = forward_realized_vol(out["close"].values, horizon)
    return out


def threshold_fit_mask(dates, horizon: int = HORIZON) -> np.ndarray:
    """Rows whose ENTIRE forward window lies on or before TRAIN_END_DATE.

    Stricter than train_mask() by exactly `horizon` rows per ticker. See the
    module docstring: this is what stops validation-period closes from moving a
    fitted parameter. Because dates ascend, train_mask is a prefix of Trues, so
    "row t+horizon is still a training row" is the complete condition.
    """
    in_train = np.asarray(train_mask(dates), dtype=bool)
    shifted = np.zeros_like(in_train)
    if horizon < len(in_train):
        shifted[: len(in_train) - horizon] = in_train[horizon:]
    return in_train & shifted


def fit_thresholds(ticker_dfs: dict, horizon: int = HORIZON) -> dict:
    """Fit pooled tercile thresholds on threshold-eligible training rows only.

    ticker_dfs: {ticker -> frame carrying date and fwd_rv_5d, date-ascending}.
    """
    pooled, per_ticker = [], {}

    for tkr in sorted(ticker_dfs):
        df = ticker_dfs[tkr]
        mask = threshold_fit_mask(df["date"], horizon)
        values = df.loc[mask, "fwd_rv_5d"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        pooled.append(values)
        per_ticker[tkr] = int(values.size)

    pooled = np.concatenate(pooled) if pooled else np.array([], dtype=float)
    if pooled.size == 0:
        raise ValueError(
            "no threshold-eligible training rows; refusing to fit terciles."
        )

    # numpy's default 'linear' interpolation between order statistics. Named
    # here because the threshold values are persisted and must be reproducible.
    lower, upper = np.percentile(pooled, [LOWER_PERCENTILE, UPPER_PERCENTILE])
    if not lower < upper:
        raise ValueError(f"degenerate terciles: lower={lower} >= upper={upper}")

    return {
        "lower": float(lower),
        "upper": float(upper),
        "lower_percentile": LOWER_PERCENTILE,
        "upper_percentile": UPPER_PERCENTILE,
        "train_end_date": TRAIN_END_DATE,
        "pooled_train_rows": int(pooled.size),
        "per_ticker_rows": per_ticker,
        "horizon": horizon,
        "ddof": DDOF,
        "annualisation_factor": float(ANNUALISATION),
        "annualisation_trading_days": 252,
        "boundary_convention": "calm: rv < lower; normal: lower <= rv < upper; "
                               "turbulent: rv >= upper",
        "name_to_id": VOL_NAME_TO_ID,
        "pooled_across_tickers": True,
        "fitted_at": datetime.now(timezone.utc).isoformat(),
    }


def save_thresholds(thresholds: dict, path=THRESHOLDS_PATH) -> None:
    """Freeze the thresholds. Serving READS this; it must never refit."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2, sort_keys=True)
    logger.info(f"  Thresholds → {path}")


def load_thresholds(path=THRESHOLDS_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def assign_labels(fwd_rv, lower: float, upper: float):
    """Map forward RV to (name, id), NaN -> (None, None).

    BOUNDARY CONVENTION — lower-inclusive on the upper side of each cut:
        rv <  lower   -> calm       (0)
        lower <= rv <  upper   -> normal     (1)
        rv >= upper   -> turbulent  (2)
    A value exactly equal to `lower` is normal; exactly equal to `upper` is
    turbulent. Stated because a tercile cut sits on an observed order statistic,
    so exact equality is reachable, not hypothetical.

    NaN yields NULL, never a default class — the U3 lesson again.
    """
    rv = np.asarray(fwd_rv, dtype=float)
    names, ids = [], []

    for value in rv:
        if not np.isfinite(value):
            names.append(None)
            ids.append(None)
        elif value < lower:
            names.append("calm")
            ids.append(VOL_NAME_TO_ID["calm"])
        elif value < upper:
            names.append("normal")
            ids.append(VOL_NAME_TO_ID["normal"])
        else:
            names.append("turbulent")
            ids.append(VOL_NAME_TO_ID["turbulent"])

    return names, ids


def add_labels(df: pd.DataFrame, lower: float, upper: float) -> pd.DataFrame:
    """Attach vol_target / vol_target_id from an existing fwd_rv_5d column."""
    out = df.copy()
    names, ids = assign_labels(out["fwd_rv_5d"].to_numpy(dtype=float), lower, upper)
    out["vol_target"] = names
    out["vol_target_id"] = pd.array(ids, dtype="Int64")  # nullable; NULL != 0
    return out


def save_vol_target_to_db(df: pd.DataFrame, db_path: str, ticker: str) -> int:
    """Write ONLY fwd_rv_5d / vol_target / vol_target_id. Nothing else."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    for ddl in (
        "ALTER TABLE market_data ADD COLUMN fwd_rv_5d REAL",
        "ALTER TABLE market_data ADD COLUMN vol_target TEXT",
        "ALTER TABLE market_data ADD COLUMN vol_target_id INTEGER",
    ):
        try:
            cur.execute(ddl)
        except sqlite3.OperationalError as exc:
            if "duplicate column" not in str(exc).lower():
                raise
            logger.info(f"  Column already present — {ddl.split()[-2]}")
    conn.commit()

    # Exact-stored-string date resolution, the regime.py:489-492 pattern: dates
    # are stored as 26-char TEXT with a microsecond suffix, so comparing against
    # a naive strftime form matches ZERO rows and leaves the columns NULL with
    # no error. Resolve every date through a normalised-Timestamp lookup.
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

        rv = row["fwd_rv_5d"]
        rv = None if not np.isfinite(rv) else float(rv)
        name = row["vol_target"]
        name = None if pd.isna(name) else str(name)
        vid = row["vol_target_id"]
        vid = None if pd.isna(vid) else int(vid)

        params.append((rv, name, vid, stored_date, ticker))

    before = conn.total_changes
    cur.executemany(
        "UPDATE market_data SET fwd_rv_5d=?, vol_target=?, vol_target_id=? "
        "WHERE date=? AND ticker=?",
        params,
    )
    conn.commit()
    updated = conn.total_changes - before
    conn.close()

    if updated == 0:
        raise RuntimeError(
            f"0 rows updated for {ticker} — refusing to report success. "
            f"{missing} of {len(df)} dates did not resolve to a stored date string."
        )
    logger.info(f"  ✅ {ticker}: {updated} rows updated ({missing} dates unmatched)")
    return updated


def run_rv_target_pipeline(db_path: str = DB_PATH, save_to_db_flag: bool = True) -> dict:
    """Compute forward RV, fit pooled train-only terciles, label, persist."""
    ticker_dfs = {
        tkr: add_forward_rv(df) for tkr, df in load_all_tickers(db_path).items()
    }

    thresholds = fit_thresholds(ticker_dfs)
    logger.info(
        f"  Pooled terciles on {thresholds['pooled_train_rows']} eligible train "
        f"rows (<= {TRAIN_END_DATE}, full forward window inside): "
        f"lower={thresholds['lower']!r} upper={thresholds['upper']!r}"
    )
    save_thresholds(thresholds)

    total_updated, per_ticker = 0, {}
    for tkr in sorted(ticker_dfs):
        labelled = add_labels(ticker_dfs[tkr], thresholds["lower"], thresholds["upper"])
        ticker_dfs[tkr] = labelled

        if save_to_db_flag:
            total_updated += save_vol_target_to_db(labelled, db_path, tkr)

        per_ticker[tkr] = {
            "rows": len(labelled),
            "null_labels": int(labelled["vol_target"].isna().sum()),
        }

    logger.info(f"\n✅ Volatility target complete — {total_updated} rows updated")
    return {
        "thresholds": thresholds,
        "per_ticker": per_ticker,
        "updated": total_updated,
        "frames": ticker_dfs,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    parser = argparse.ArgumentParser(description="MMIS T3 — forward volatility target")
    parser.add_argument("--db", type=str, default=DB_PATH, help="Path to mmis.db")
    parser.add_argument("--all", action="store_true", help="Process all tickers")
    parser.add_argument("--no-save", action="store_true", help="Do not write to the DB")
    args = parser.parse_args()

    if args.all:
        run_rv_target_pipeline(db_path=args.db, save_to_db_flag=not args.no_save)
    else:
        print("⚠️  Specify --all")
