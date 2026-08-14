"""CORRECTNESS tests for the classical volatility baselines (T4).

Like tests/test_rv_target.py and unlike most of this suite, these are not
characterization tests: they say what har_baseline.py SHOULD do and are expected
to stay green.

Two properties matter most.

The first is that the fitted OLS coefficients cannot be influenced by anything
dated after the training boundary. That is verified EMPIRICALLY by mutating
post-boundary prices and refitting — a fit that quietly admitted a
boundary-straddling row would still look correct on the page.

The second is that the baseline is FAIR. A handicapped baseline flatters
whatever is compared against it later, which would silently invalidate the
comparison this module exists to enable. So the predictor alignment is pinned by
an exact identity (backward-5 at t+5 == forward-5 at t), the lookback lengths are
pinned, and the metrics are checked against hand-worked arithmetic.

Everything synthetic here needs no database, no artifacts and no network. The
few tests that read data/mmis.db SKIP when it is absent (it is gitignored).
"""

import numpy as np
import pandas as pd
import pytest

import rv_target
from conftest import require
from har_baseline import (
    CLASS_IDS,
    HAR_WINDOWS,
    MAX_LOOKBACK,
    add_har_predictors,
    backward_realized_vol,
    bucket,
    build_design,
    complete_predictors,
    design_matrix,
    fit_har,
    fit_ols,
    predictor_columns,
    qlike,
    regression_metrics,
)
from regime_causal import TRAIN_END_DATE

TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "SPY", "TSLA"]


def synthetic_ticker(seed: int = 0, n: int = 900, name: str = "SYN") -> pd.DataFrame:
    """Business-day close series straddling TRAIN_END_DATE. Deterministic."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-06-01", periods=n)

    vol = np.full(n, 0.008)
    vol[(dates >= pd.Timestamp("2023-09-01")) & (dates < pd.Timestamp("2024-02-01"))] = 0.022
    vol[dates >= pd.Timestamp("2025-08-01")] = 0.040

    close = 100 * np.exp(np.cumsum(rng.standard_normal(n) * vol))
    frame = pd.DataFrame({"date": dates, "ticker": name, "close": close})
    frame["fwd_rv_5d"] = rv_target.forward_realized_vol(close)
    names, ids = rv_target.assign_labels(
        frame["fwd_rv_5d"].to_numpy(float), 0.18390286612477044, 0.32590977697139434
    )
    frame["vol_target"] = names
    frame["vol_target_id"] = pd.array(ids, dtype="Int64")
    return frame


def synthetic_panel() -> dict:
    return {t: synthetic_ticker(seed=i, name=t) for i, t in enumerate(TICKERS)}


def mutate_after_boundary(frames: dict, mode: str) -> dict:
    """Rewrite every close strictly AFTER TRAIN_END_DATE, leaving train rows alone.

    The target is recomputed from the mutated closes, exactly as it would be if
    the post-boundary world had genuinely unfolded differently.
    """
    rng = np.random.default_rng(4321)
    out = {}
    for tkr, df in frames.items():
        d = df.copy()
        post = pd.to_datetime(d["date"]) > pd.Timestamp(TRAIN_END_DATE)
        idx = d.index[post]
        if mode == "randomwalk":
            d.loc[idx, "close"] = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.05, len(idx))))
        elif mode == "x10":
            d.loc[idx, "close"] = d.loc[idx, "close"] * 10.0
        else:
            raise ValueError(mode)
        d["fwd_rv_5d"] = rv_target.forward_realized_vol(d["close"].to_numpy(float))
        out[tkr] = d
    return out


# ══════════════════════════════════════════════════════════════════
#  D1 — causality and leakage
# ══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("mode", ["randomwalk", "x10"])
def test_coefficients_are_bit_identical_under_post_boundary_mutation(mode):
    """THE CENTRAL TEST. If future prices move the coefficients, the fit leaks.

    `randomwalk` is the PRIMARY mutation and the one that carries the argument.
    T3's verifier established that a uniform rescale is a weak probe, because
    realized volatility is scale-invariant — x10 barely perturbs the
    post-boundary predictors at all. It is kept as a second case only because it
    bites hard at the seam, on the boundary-straddling rows the embargo excludes.

    Asserted as exact equality AND as identical hex representations, so a
    difference below printing precision cannot pass.
    """
    base = synthetic_panel()
    baseline = fit_har(build_design(base))
    mutated = fit_har(build_design(mutate_after_boundary(base, mode)))

    for variant in ("level", "log"):
        for i, (a, b) in enumerate(
            zip(baseline[variant]["coefficients"], mutated[variant]["coefficients"])
        ):
            assert a == b, f"{mode}: {variant} coefficient {i} moved {a!r} -> {b!r}"
            assert a.hex() == b.hex()

    assert mutated["train_rows"] == baseline["train_rows"]
    assert mutated["train_rows_per_ticker"] == baseline["train_rows_per_ticker"]
    assert mutated["train_mean_fwd_rv"] == baseline["train_mean_fwd_rv"]
    assert mutated["train_median_fwd_rv"] == baseline["train_median_fwd_rv"]


def test_mutation_actually_changed_the_post_boundary_data():
    """Anti-vacuity guard. A no-op mutation would make the test above pass for
    the wrong reason, so pin that the post-boundary predictors AND targets really
    move under the random-walk replacement."""
    base = synthetic_panel()
    mutated = mutate_after_boundary(base, "randomwalk")

    for tkr in TICKERS:
        b = add_har_predictors(base[tkr])
        m = add_har_predictors(mutated[tkr])
        post = pd.to_datetime(b["date"]) > pd.Timestamp(TRAIN_END_DATE)

        assert not np.allclose(b.loc[post, "close"], m.loc[post, "close"])

        for col in predictor_columns():
            x, y = b.loc[post, col].to_numpy(), m.loc[post, col].to_numpy()
            fin = np.isfinite(x) & np.isfinite(y)
            assert not np.allclose(x[fin], y[fin]), f"{tkr}: {col} did not move"

        x = b.loc[post, "fwd_rv_5d"].to_numpy()
        y = m.loc[post, "fwd_rv_5d"].to_numpy()
        fin = np.isfinite(x) & np.isfinite(y)
        assert not np.allclose(x[fin], y[fin])


def test_embargo_rule_is_imported_from_rv_target_not_reimplemented():
    """The rule must be the SAME object, not a copy that can drift.

    T3 measured that a naive `date <= TRAIN_END_DATE` pool leaks. Restating that
    rule here would let the two definitions diverge silently, so identity is
    asserted directly rather than behaviour being spot-checked.
    """
    import har_baseline

    assert har_baseline.threshold_fit_mask is rv_target.threshold_fit_mask


def test_training_row_count_is_the_embargo_pool_minus_the_lookback(db):
    """The arithmetic, pinned against the real panel: T3's 7,500 embargo-eligible
    rows, less a 66-day lookback for each of the 6 tickers, is 7,104."""
    from conftest import DB_FILE
    from har_baseline import load_panel

    require(DB_FILE, "data/mmis.db")
    panel = build_design(load_panel(str(DB_FILE)))

    assert int(panel["embargo_eligible"].sum()) == 7500
    assert int((panel["embargo_eligible"] & ~panel["has_predictors"]).sum()) == \
        MAX_LOOKBACK * len(TICKERS) == 396
    assert int(panel["is_train"].sum()) == 7500 - 396 == 7104

    per_ticker = panel[panel["is_train"]].groupby("ticker").size()
    assert set(per_ticker) == {1184}, "every ticker must contribute equally"


def test_predictors_use_no_close_after_their_own_row():
    """Truncation proof: a predictor computed on a short series must equal the
    one computed on the long series, for every row the truncation retains."""
    df = synthetic_ticker()
    cut = 700

    full = add_har_predictors(df)
    truncated = add_har_predictors(df.iloc[:cut].copy())

    for col in predictor_columns():
        np.testing.assert_array_equal(
            truncated[col].to_numpy(), full[col].to_numpy()[:cut]
        )


def test_backward_window_aligns_exactly_with_the_forward_target():
    """THE ALIGNMENT IDENTITY, and the strongest fairness check available.

    backward-5 RV at row t+5 and forward-5 RV at row t are the same quantity —
    both are stdev(log_ret[t+1 .. t+5], ddof=1) * sqrt(252). They must be
    bit-equal. An off-by-one in EITHER direction breaks this, and an off-by-one
    in the predictors would silently weaken the baseline.
    """
    df = synthetic_ticker()
    close = df["close"].to_numpy(float)

    back5 = backward_realized_vol(close, 5)
    fwd5 = rv_target.forward_realized_vol(close)

    t = np.arange(len(close) - 5)
    a, b = fwd5[t], back5[t + 5]
    fin = np.isfinite(a) & np.isfinite(b)

    assert fin.sum() > 800
    np.testing.assert_array_equal(a[fin], b[fin])


def test_rows_without_a_complete_lookback_are_nan_and_excluded():
    """Each window leaves exactly `window` leading NaNs, never 0.0, and those
    rows are excluded from the fit rather than imputed."""
    df = synthetic_ticker()
    out = add_har_predictors(df)

    for w in HAR_WINDOWS:
        col = out[f"har_rv_{w}"].to_numpy(float)
        assert np.isnan(col[:w]).all(), f"window {w}: leading rows are not NaN"
        assert np.isfinite(col[w:]).all(), f"window {w}: interior gap"
        assert int(np.isnan(col).sum()) == w
        assert not (col[:w] == 0).any()

    complete = complete_predictors(out)
    assert not complete[:MAX_LOOKBACK].any()
    assert complete[MAX_LOOKBACK:].all()


def test_fit_excludes_incomplete_lookback_rows_from_the_design():
    """A NaN predictor reaching lstsq would poison every coefficient."""
    panel = build_design(synthetic_panel())
    train = panel[panel["is_train"]]

    x = design_matrix(train)
    assert np.isfinite(x).all()
    assert np.isfinite(train["fwd_rv_5d"].to_numpy(float)).all()

    model = fit_har(panel)
    assert all(np.isfinite(model["level"]["coefficients"]))
    assert model["level"]["rank"] == 4


def test_series_shorter_than_the_window_is_all_nan_and_does_not_raise():
    for w in HAR_WINDOWS:
        for n in range(0, w + 1):
            close = 100 * np.exp(np.cumsum(np.full(n, 0.01)))
            out = backward_realized_vol(close, w)
            assert len(out) == n
            assert np.isnan(out).all()


# ══════════════════════════════════════════════════════════════════
#  D2 — metric correctness against hand-computed values
# ══════════════════════════════════════════════════════════════════

def test_regression_metrics_match_hand_computed_arithmetic():
    """Hand-worked example, arithmetic shown.

        actual    = [0.20, 0.30]
        predicted = [0.25, 0.25]
        errors    = [-0.05, +0.05]

        SSE  = 0.05^2 + 0.05^2 = 0.0025 + 0.0025 = 0.0050
        MSE  = 0.0050 / 2                        = 0.0025
        RMSE = sqrt(0.0025)                      = 0.05
        MAE  = (0.05 + 0.05) / 2                 = 0.05

    R^2 against a TRAINING mean of 0.20 (not the sample mean, which would be
    0.25 and would make this degenerate):
        SST = (0.20-0.20)^2 + (0.30-0.20)^2 = 0 + 0.01 = 0.01
        R2  = 1 - 0.0050 / 0.01             = 1 - 0.5  = 0.5
    """
    actual = np.array([0.20, 0.30])
    predicted = np.array([0.25, 0.25])

    m = regression_metrics(actual, predicted, train_mean=0.20)

    assert m["mse"] == pytest.approx(0.0025, rel=1e-12)
    assert m["rmse"] == pytest.approx(0.05, rel=1e-12)
    assert m["mae"] == pytest.approx(0.05, rel=1e-12)
    assert m["r2"] == pytest.approx(0.5, rel=1e-12)
    assert m["n"] == 2


def test_qlike_matches_hand_computed_arithmetic():
    """QLIKE is computed in VARIANCE space, so square the volatilities first.

        actual    = [0.20, 0.30]  ->  s2    = [0.04, 0.09]
        predicted = [0.25, 0.25]  ->  s2hat = [0.0625, 0.0625]

        ratio = s2 / s2hat = [0.64, 1.44]

        QLIKE_i = ratio - ln(ratio) - 1
          i=0: 0.64 - ln(0.64) - 1 = 0.64 + 0.4462871026284195 - 1 = 0.0862871026284195
          i=1: 1.44 - ln(1.44) - 1 = 1.44 - 0.3646431135879093 - 1 = 0.0753568864120907

        mean = (0.0862871026284195 + 0.0753568864120907) / 2
             = 0.1616439890405102 / 2
             = 0.0808219945202551
    """
    actual = np.array([0.20, 0.30])
    predicted = np.array([0.25, 0.25])

    out = qlike(actual, predicted, floor=1e-6)

    assert out["qlike"] == pytest.approx(0.0808219945202551, rel=1e-12)
    assert out["n_floored"] == 0


def test_qlike_is_zero_at_equality_and_strictly_positive_otherwise():
    """x - log(x) - 1 >= 0, with equality only at x = 1."""
    actual = np.array([0.1, 0.2, 0.35, 0.8])

    assert qlike(actual, actual, floor=1e-6)["qlike"] == pytest.approx(0.0, abs=1e-15)

    for scale in (0.5, 0.9, 1.1, 2.0, 5.0):
        assert qlike(actual, actual * scale, floor=1e-6)["qlike"] > 0.0

    # Asymmetric: under-predicting variance is penalised harder than over.
    under = qlike(actual, actual * 0.5, floor=1e-6)["qlike"]
    over = qlike(actual, actual * 2.0, floor=1e-6)["qlike"]
    assert under > over


def test_qlike_reports_floored_predictions_rather_than_hiding_them():
    """Non-positive predictions are undefined under a log and must be counted."""
    actual = np.array([0.2, 0.2, 0.2])
    predicted = np.array([0.2, 0.0, -0.5])

    out = qlike(actual, predicted, floor=0.01)

    assert out["n_floored"] == 2
    assert out["floor_used"] == 0.01
    assert np.isfinite(out["qlike"])


def test_bucketing_uses_the_frozen_thresholds_at_exact_boundaries():
    """Convention identical to T3: each cut inclusive on its UPPER side, so a
    value exactly at `lower` is normal and exactly at `upper` is turbulent."""
    lower = 0.18390286612477044
    upper = 0.32590977697139434
    eps = 1e-15

    values = np.array([
        lower - eps, lower, lower + eps,
        upper - eps, upper, upper + eps,
    ])
    assert list(bucket(values, lower, upper)) == [0, 1, 1, 1, 2, 2]

    assert list(bucket(np.array([lower]), lower, upper)) == [1]
    assert list(bucket(np.array([upper]), lower, upper)) == [2]
    assert set(bucket(np.array([0.0, 1.0]), lower, upper)) <= set(CLASS_IDS)


def test_bucketing_against_the_frozen_json_reproduces_the_stored_labels(db, q):
    """The buckets applied to the TARGET itself must reproduce vol_target_id
    exactly — proof the baseline is scored through the same cuts T3 froze."""
    from conftest import DB_FILE
    from rv_target import load_thresholds

    require(DB_FILE, "data/mmis.db")
    thresholds = load_thresholds()

    rows = q(
        "SELECT fwd_rv_5d, vol_target_id FROM market_data WHERE fwd_rv_5d IS NOT NULL"
    )
    rv = np.array([r[0] for r in rows], dtype=float)
    stored = np.array([r[1] for r in rows], dtype=int)

    np.testing.assert_array_equal(
        bucket(rv, thresholds["lower"], thresholds["upper"]), stored
    )


def test_ols_recovers_known_coefficients():
    """A fit that silently failed would invalidate the whole benchmark."""
    rng = np.random.default_rng(7)
    n = 500
    x = rng.normal(size=(n, 3))
    X = np.column_stack([np.ones(n), x])
    truth = np.array([0.5, 1.5, -2.0, 0.25])
    y = X @ truth

    out = fit_ols(X, y)

    np.testing.assert_allclose(out["coefficients"], truth, rtol=1e-10)
    assert out["rank"] == 4
    assert out["n"] == n
    assert out["dof"] == n - 4


def test_ols_rejects_a_rank_deficient_design():
    """A duplicated column must raise, not return arbitrary coefficients."""
    n = 50
    col = np.linspace(0, 1, n)
    X = np.column_stack([np.ones(n), col, col * 2.0, col])
    y = np.linspace(1, 2, n)

    with pytest.raises(ValueError, match="rank-deficient"):
        fit_ols(X, y)


def test_unsorted_dates_are_rejected():
    df = synthetic_ticker(n=200)
    with pytest.raises(ValueError, match="ascending date order"):
        add_har_predictors(df.iloc[::-1].reset_index(drop=True))
