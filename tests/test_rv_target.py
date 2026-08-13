"""CORRECTNESS tests for the forward volatility target (T3).

Like tests/test_regime_causality.py and unlike most of this suite, these are not
characterization tests: they say what rv_target.py SHOULD do and are expected to
stay green.

The central property is that the THRESHOLDS — the only fitted parameter in the
label — cannot be influenced by anything dated after the training boundary. That
is verified EMPIRICALLY, by mutating post-boundary prices and refitting, not by
reading the code. A pool that quietly admitted a boundary-straddling forward
window would still look correct on the page; it is caught here by mutation and
nowhere else.

The synthetic tests need no database, no artifacts and no network. The handful
that read data/mmis.db SKIP when it is absent (it is gitignored).
"""

import numpy as np
import pandas as pd
import pytest

from conftest import require
from regime_causal import TRAIN_END_DATE
from rv_target import (
    ANNUALISATION,
    HORIZON,
    VOL_NAME_TO_ID,
    VOL_TARGET_NAMES,
    add_forward_rv,
    add_labels,
    assign_labels,
    fit_thresholds,
    forward_realized_vol,
    threshold_fit_mask,
)

TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "SPY", "TSLA"]

# Well after TRAIN_END_DATE, so a leak would have to reach backwards across the
# boundary to become visible.
MUTATION_START = "2025-08-01"


def synthetic_ticker(seed: int = 0, n: int = 900, name: str = "SYN") -> pd.DataFrame:
    """Business-day close series straddling TRAIN_END_DATE. Deterministic."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-06-01", periods=n)

    vol = np.full(n, 0.008)
    vol[(dates >= pd.Timestamp("2023-09-01")) & (dates < pd.Timestamp("2024-02-01"))] = 0.022
    vol[dates >= pd.Timestamp(MUTATION_START)] = 0.040

    close = 100 * np.exp(np.cumsum(rng.standard_normal(n) * vol))
    return pd.DataFrame({"date": dates, "ticker": name, "close": close})


def synthetic_panel() -> dict:
    """Six tickers, as the real pooled fit sees them."""
    return {
        t: add_forward_rv(synthetic_ticker(seed=i, name=t))
        for i, t in enumerate(TICKERS)
    }


def mutate_after_boundary(frames: dict, mode: str) -> dict:
    """Rewrite every close strictly AFTER TRAIN_END_DATE. Leaves train rows alone."""
    rng = np.random.default_rng(1234)
    out = {}
    for tkr, df in frames.items():
        d = df.copy()
        post = pd.to_datetime(d["date"]) > pd.Timestamp(TRAIN_END_DATE)
        idx = d.index[post]
        if mode == "x2":
            d.loc[idx, "close"] = d.loc[idx, "close"] * 2.0
        elif mode == "x10":
            d.loc[idx, "close"] = d.loc[idx, "close"] * 10.0
        elif mode == "randomwalk":
            d.loc[idx, "close"] = 100.0 * np.exp(
                np.cumsum(rng.normal(0, 0.05, len(idx)))
            )
        else:
            raise ValueError(mode)
        out[tkr] = add_forward_rv(d)
    return out


# ══════════════════════════════════════════════════════════════════
#  D1 — the leakage proof
# ══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("mode", ["x2", "x10", "randomwalk"])
def test_thresholds_are_bit_identical_under_post_boundary_mutation(mode):
    """THE CENTRAL TEST. If future prices can move the thresholds, the label leaks.

    Every close strictly after TRAIN_END_DATE is replaced — doubled, multiplied
    by ten, and finally overwritten with an unrelated random walk. The thresholds
    are a fitted parameter applied to the validation period, so any movement at
    all means validation data reached them.

    Weight the three cases correctly: realized volatility is SCALE-INVARIANT, so
    x2 and x10 barely perturb the post-boundary labels at all (~1e-14, pure
    rounding) once the rescaling is uniform over a contiguous block. They are
    weak probes on their own. `randomwalk` is the mutation that carries the
    argument — it destroys the post-boundary return series outright — and
    test_mutation_actually_changed_the_post_boundary_data below pins that it
    really does move those labels, so this cannot pass vacuously.

    Where x2/x10 DO bite is the seam: they move the boundary-straddling rows
    enormously, which is exactly the leak vector that threshold_fit_mask
    excludes. See test_threshold_fit_pool_excludes_the_boundary_straddling_rows.

    Asserted as exact float equality AND as identical hex representations, so a
    difference below printing precision cannot pass.
    """
    base = synthetic_panel()
    baseline = fit_thresholds(base)
    mutated = fit_thresholds(mutate_after_boundary(base, mode))

    assert mutated["lower"] == baseline["lower"], f"{mode} moved the lower threshold"
    assert mutated["upper"] == baseline["upper"], f"{mode} moved the upper threshold"
    assert mutated["lower"].hex() == baseline["lower"].hex()
    assert mutated["upper"].hex() == baseline["upper"].hex()
    # The pool itself must be unchanged too, not merely coincidentally equal.
    assert mutated["pooled_train_rows"] == baseline["pooled_train_rows"]
    assert mutated["per_ticker_rows"] == baseline["per_ticker_rows"]


def test_mutation_actually_changed_the_post_boundary_data():
    """Guards the test above: a no-op mutation would make it vacuously pass."""
    base = synthetic_panel()
    mutated = mutate_after_boundary(base, "randomwalk")

    for tkr in TICKERS:
        post = pd.to_datetime(base[tkr]["date"]) > pd.Timestamp(TRAIN_END_DATE)
        assert not np.allclose(
            base[tkr].loc[post, "close"], mutated[tkr].loc[post, "close"]
        )
        # And the post-boundary LABELS must move, or nothing was really perturbed.
        a = base[tkr].loc[post, "fwd_rv_5d"].to_numpy()
        b = mutated[tkr].loc[post, "fwd_rv_5d"].to_numpy()
        finite = np.isfinite(a) & np.isfinite(b)
        assert not np.allclose(a[finite], b[finite])


def test_threshold_fit_pool_excludes_the_boundary_straddling_rows():
    """The mechanism that makes the test above pass, pinned directly.

    A row dated <= TRAIN_END_DATE whose forward window runs past it would drag
    post-boundary prices into the fit. Eligibility is therefore exactly
    `HORIZON` rows tighter than the plain training mask.
    """
    from regime_causal import train_mask

    df = synthetic_ticker()
    in_train = np.asarray(train_mask(df["date"]), dtype=bool)
    eligible = threshold_fit_mask(df["date"])

    assert eligible.sum() == in_train.sum() - HORIZON
    # Eligible is a strict subset, and the dropped rows are the last training ones.
    assert not (eligible & ~in_train).any()
    dropped = np.where(in_train & ~eligible)[0]
    assert list(dropped) == list(range(in_train.sum() - HORIZON, in_train.sum()))


def test_truncating_the_series_does_not_change_completed_labels():
    """Truncation proof: a row whose forward window fits inside the truncation
    must get the same label from the short series as from the long one.

    This is the property that no row is labelled using data beyond t+HORIZON.
    """
    df = synthetic_ticker()
    cut = 700

    full = add_forward_rv(df)["fwd_rv_5d"].to_numpy()
    truncated = add_forward_rv(df.iloc[:cut].copy())["fwd_rv_5d"].to_numpy()

    # Rows 0 .. cut-HORIZON-1 have a complete forward window inside the truncation.
    complete = cut - HORIZON
    np.testing.assert_array_equal(truncated[:complete], full[:complete])
    # And everything after that in the truncated series is NULL, not guessed.
    assert np.isnan(truncated[complete:]).all()


def test_label_does_not_depend_on_the_feature_rows_own_return():
    """Row t's label must not be readable from row t's own return.

    A note on the mutation, because the obvious one is arithmetically impossible:
    changing close[t] ALONE cannot leave returns t+1..t+5 fixed, since
    log_ret[t+1] = log(close[t+1] / close[t]) is defined against close[t]. The
    property that actually matters is that fwd_rv[t] is a function of
    close[t .. t+HORIZON] and of nothing earlier. So we perturb close[t-1],
    which changes row t's own return log(close[t]/close[t-1]) — and row t-1's —
    while leaving every forward return of row t untouched.
    """
    df = synthetic_ticker()
    t = 500

    before = add_forward_rv(df)["fwd_rv_5d"].to_numpy()

    mutated = df.copy()
    mutated.loc[t - 1, "close"] = mutated.loc[t - 1, "close"] * 1.5
    after = add_forward_rv(mutated)["fwd_rv_5d"].to_numpy()

    # Row t's own return genuinely moved.
    own_before = np.log(df.loc[t, "close"] / df.loc[t - 1, "close"])
    own_after = np.log(mutated.loc[t, "close"] / mutated.loc[t - 1, "close"])
    assert not np.isclose(own_before, own_after)

    # Its label did not.
    assert after[t] == before[t]
    # Nor did any later row's, since all of them look forward of t-1.
    np.testing.assert_array_equal(after[t:-HORIZON], before[t:-HORIZON])


def test_last_five_rows_per_ticker_are_null_not_defaulted():
    """The U3 lesson. No forward window means NO LABEL — never a default class.

    ingest.py:135-140 defaults its final row per ticker to 1 (Flat) because a
    bare Series carried the NaN where dropna() could not see it. Here the
    absence must be representable and must be NULL.
    """
    panel = synthetic_panel()
    thresholds = fit_thresholds(panel)

    for tkr, df in panel.items():
        labelled = add_labels(df, thresholds["lower"], thresholds["upper"])

        assert labelled["fwd_rv_5d"].isna().sum() == HORIZON
        assert labelled["vol_target"].isna().sum() == HORIZON
        assert labelled["vol_target_id"].isna().sum() == HORIZON

        tail = labelled.tail(HORIZON)
        assert tail["vol_target"].isna().all(), f"{tkr}: tail is not NULL"
        assert tail["vol_target_id"].isna().all()
        # Nothing was quietly zero-filled: NULL must not be class 0 ("calm").
        assert not (tail["vol_target"] == "calm").any()
        assert not (tail["vol_target_id"] == 0).any()

        # And every earlier row IS labelled.
        assert labelled.head(len(labelled) - HORIZON)["vol_target"].notna().all()


def test_name_to_id_map_is_identical_across_all_six_tickers():
    """One fixed map everywhere, mirroring T2's REGIME_NAME_TO_ID. This is the
    property regime_id never had (U1: raw per-ticker HMM state index)."""
    panel = synthetic_panel()
    thresholds = fit_thresholds(panel)

    observed = []
    for tkr in TICKERS:
        labelled = add_labels(panel[tkr], thresholds["lower"], thresholds["upper"])
        pairs = labelled[["vol_target", "vol_target_id"]].dropna().drop_duplicates()
        observed.append({r.vol_target: int(r.vol_target_id) for r in pairs.itertuples()})

    for tkr, mapping in zip(TICKERS, observed):
        for name, vid in mapping.items():
            assert VOL_NAME_TO_ID[name] == vid, f"{tkr}: {name} -> {vid}"

    # Every ticker agrees with every other on the pairs they share.
    for other in observed[1:]:
        shared = set(observed[0]) & set(other)
        assert {k: observed[0][k] for k in shared} == {k: other[k] for k in shared}

    assert VOL_NAME_TO_ID == {"calm": 0, "normal": 1, "turbulent": 2}
    assert VOL_TARGET_NAMES == {0: "calm", 1: "normal", 2: "turbulent"}


# ══════════════════════════════════════════════════════════════════
#  D2 — label correctness against hand-computed values
# ══════════════════════════════════════════════════════════════════

def test_forward_rv_matches_hand_computed_arithmetic():
    """Hand-worked example, arithmetic shown.

    Build closes so the log returns are exactly 0.01, 0.02, 0.03, 0.04, 0.05:
        close = [100, 100e^.01, 100e^.03, 100e^.06, 100e^.10, 100e^.15]
        log_ret = [nan, 0.01, 0.02, 0.03, 0.04, 0.05]

    Row 0's forward window is log_ret[1..5] = 0.01 .. 0.05.
        mean      = (0.01+0.02+0.03+0.04+0.05) / 5 = 0.03
        deviations= -0.02, -0.01, 0.00, +0.01, +0.02
        squares   = 4e-4, 1e-4, 0, 1e-4, 4e-4      sum = 1.0e-3
        var(ddof=1) = 1.0e-3 / (5-1)               = 2.5e-4
        sd          = sqrt(2.5e-4)                 = 0.0158113883008418966
        annualised  = sd * sqrt(252) = sqrt(2.5e-4 * 252) = sqrt(0.063)
                    = 0.25099800796022265
    """
    cum = np.array([0.0, 0.01, 0.03, 0.06, 0.10, 0.15])
    close = 100 * np.exp(cum)

    rv = forward_realized_vol(close)

    expected = 0.25099800796022265
    assert rv[0] == pytest.approx(expected, rel=1e-12)
    assert rv[0] == pytest.approx(np.sqrt(0.063), rel=1e-12)

    # Only row 0 has a complete forward window in a 6-row series.
    assert np.isnan(rv[1:]).all()
    assert ANNUALISATION == pytest.approx(np.sqrt(252.0))


def test_forward_window_excludes_the_rows_own_return_by_construction():
    """A second hand-check that pins WHICH five returns are used.

    log_ret = [nan, 0.10, 0.01, 0.01, 0.01, 0.01, 0.01]. Row 0's window is
    log_ret[1..5] = (0.10, 0.01, 0.01, 0.01, 0.01) and includes the 0.10 spike;
    row 1's window is log_ret[2..6] = (0.01,)*5, which has ZERO dispersion.
    If row 1's window wrongly started at its own return it would be non-zero.
    """
    steps = [0.0, 0.10, 0.01, 0.01, 0.01, 0.01, 0.01]
    close = 100 * np.exp(np.cumsum(steps))

    rv = forward_realized_vol(close)

    assert rv[0] > 0.5              # the spike is inside row 0's window
    assert rv[1] == pytest.approx(0.0, abs=1e-12)   # row 1 sees five equal returns
    assert np.isnan(rv[2:]).all()


def test_boundary_convention_at_exactly_the_threshold_values():
    """Each cut is inclusive on its UPPER side: == lower is normal, == upper is
    turbulent. A tercile lands on an observed order statistic, so exact equality
    is reachable and must be pinned rather than left to chance."""
    lower, upper = 0.2, 0.4
    eps = 1e-15

    values = [
        lower - eps, lower, lower + eps,
        upper - eps, upper, upper + eps,
    ]
    names, ids = assign_labels(values, lower, upper)

    assert names == ["calm", "normal", "normal", "normal", "turbulent", "turbulent"]
    assert ids == [0, 1, 1, 1, 2, 2]

    # Exactly at a threshold is never the class below it.
    assert assign_labels([lower], lower, upper)[0] == ["normal"]
    assert assign_labels([upper], lower, upper)[0] == ["turbulent"]


def test_series_shorter_than_the_window_is_all_null_and_does_not_raise():
    """Fewer than HORIZON+1 rows cannot produce a single label. That must be an
    all-NULL column, not an exception and not a fabricated value."""
    for n in range(0, HORIZON + 1):
        close = 100 * np.exp(np.cumsum(np.full(n, 0.01)))
        rv = forward_realized_vol(close)

        assert len(rv) == n
        assert np.isnan(rv).all(), f"n={n} produced a non-NULL label"

        names, ids = assign_labels(rv, 0.2, 0.4)
        assert names == [None] * n
        assert ids == [None] * n


def test_nan_forward_rv_never_becomes_a_class():
    names, ids = assign_labels([np.nan, 0.1, np.inf, -np.inf, 0.5], 0.2, 0.4)
    assert names == [None, "calm", None, None, "turbulent"]
    assert ids == [None, 0, None, None, 2]


def test_unsorted_dates_are_rejected_rather_than_silently_mislabelled():
    """A forward window over unsorted rows looks sideways, not forward."""
    df = synthetic_ticker(n=50)
    shuffled = df.iloc[::-1].reset_index(drop=True)

    with pytest.raises(ValueError, match="ascending date order"):
        add_forward_rv(shuffled)


# ══════════════════════════════════════════════════════════════════
#  Against the database (skips on a fresh clone)
# ══════════════════════════════════════════════════════════════════

def test_db_new_columns_exist_with_declared_types(q):
    from conftest import DB_FILE
    require(DB_FILE, "data/mmis.db")

    declared = {r[1]: r[2] for r in q("PRAGMA table_info(market_data)")}
    assert declared["fwd_rv_5d"] == "REAL"
    assert declared["vol_target"] == "TEXT"
    assert declared["vol_target_id"] == "INTEGER"


def test_db_null_pattern_is_exactly_the_last_five_rows_per_ticker(q, scalar):
    """30 NULLs, all at the end of a ticker's series, and no others."""
    from conftest import DB_FILE
    require(DB_FILE, "data/mmis.db")

    for col in ("fwd_rv_5d", "vol_target", "vol_target_id"):
        assert scalar(f"SELECT COUNT(*) FROM market_data WHERE {col} IS NULL") == 30

    # The three columns must be NULL on exactly the SAME rows.
    assert scalar(
        "SELECT COUNT(*) FROM market_data WHERE "
        "(fwd_rv_5d IS NULL) <> (vol_target IS NULL) "
        "OR (vol_target IS NULL) <> (vol_target_id IS NULL)"
    ) == 0

    for (ticker,) in q("SELECT DISTINCT ticker FROM market_data ORDER BY ticker"):
        nulls = [r[0] for r in q(
            "SELECT date FROM market_data WHERE ticker=? AND vol_target IS NULL "
            "ORDER BY date", (ticker,)
        )]
        tail = [r[0] for r in q(
            "SELECT date FROM market_data WHERE ticker=? ORDER BY date DESC LIMIT 5",
            (ticker,)
        )]
        assert len(nulls) == 5
        assert nulls == sorted(tail), f"{ticker}: NULLs are not the final five rows"


def test_db_vol_target_and_id_agree_on_the_fixed_map(q, scalar):
    from conftest import DB_FILE
    require(DB_FILE, "data/mmis.db")

    for name, vid in VOL_NAME_TO_ID.items():
        mismatched = scalar(
            "SELECT COUNT(*) FROM market_data WHERE vol_target = ? "
            "AND vol_target_id <> ?", (name, vid)
        )
        assert mismatched == 0, f"{name} does not map to {vid} on every row"

    names = {r[0] for r in q("SELECT DISTINCT vol_target FROM market_data")}
    assert names == set(VOL_NAME_TO_ID) | {None}

    # Every ticker uses the same map — the U1 property, in the new columns.
    pairs = q(
        "SELECT DISTINCT ticker, vol_target, vol_target_id FROM market_data "
        "WHERE vol_target IS NOT NULL ORDER BY ticker"
    )
    for _, name, vid in pairs:
        assert VOL_NAME_TO_ID[name] == vid


def test_db_labels_agree_with_the_persisted_thresholds(q):
    """The stored label must be reproducible from fwd_rv_5d and the frozen
    thresholds — serving reads the JSON, it never refits."""
    from pathlib import Path

    from conftest import DB_FILE, PROJECT_ROOT
    from rv_target import load_thresholds

    require(DB_FILE, "data/mmis.db")
    path = Path(PROJECT_ROOT) / "models" / "rv_thresholds.json"
    require(path, "models/rv_thresholds.json")

    thresholds = load_thresholds(path)
    assert thresholds["train_end_date"] == TRAIN_END_DATE
    assert thresholds["pooled_train_rows"] == 7500
    assert thresholds["lower"] < thresholds["upper"]

    rows = q(
        "SELECT fwd_rv_5d, vol_target, vol_target_id FROM market_data "
        "WHERE fwd_rv_5d IS NOT NULL"
    )
    rv = [r[0] for r in rows]
    names, ids = assign_labels(rv, thresholds["lower"], thresholds["upper"])

    assert names == [r[1] for r in rows]
    assert ids == [r[2] for r in rows]
