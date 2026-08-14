"""CORRECTNESS tests for fusion.py's date-pinned train/validation split.

Like tests/test_rv_target.py and unlike most of this suite, these are not
characterization tests: they say what temporal_split SHOULD do and are expected
to stay green.

The central property is ROW-COUNT INVARIANCE. The split this replaced was
`int(len(df) * 0.8)` over the market-visual aligned frame, which moved whenever
visual_features gained or lost rows — deleting rows from the VALIDATION period
silently relocated the boundary and handed previously-held-out days to training.
test_boundary_is_invariant_to_row_count pins the fix, and the test immediately
after it demonstrates that the old rule genuinely failed the same check, so the
invariance assertion cannot pass vacuously.

IMPORT COST, measured: importing fusion.py takes ~12 s, because it pulls in
torch, mlflow, and now hmmlearn via `from regime_causal import TRAIN_END_DATE`.
That is paid ONCE at collection, not per test, and it is nowhere near a timeout.
Nothing here executes fusion.py's training path or its __main__ block; only
module-level definitions are touched.
"""

import inspect
import re
import sqlite3

import numpy as np
import pandas as pd
import pytest

import fusion
import regime_causal
from conftest import DB_FILE, require
from fusion import EMBARGO_DAYS, temporal_split
from regime_causal import TRAIN_END_DATE

TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "SPY", "TSLA"]
BOUNDARY = pd.Timestamp(TRAIN_END_DATE)


@pytest.fixture(scope="module")
def aligned():
    """fusion.load_aligned_data's frame, rebuilt READ-ONLY.

    Deliberately NOT calling load_aligned_data itself, which opens a read-write
    SQLAlchemy engine against the only copy of this database. The join mirrors
    fusion.py:241-260; an inner join on (date, ticker) against visual_features
    is exactly equivalent to fusion's left-merge followed by
    dropna(subset=["feature_vector"]), since that dropna removes precisely the
    market rows with no visual match. Only the columns the split needs are
    pulled.
    """
    require(DB_FILE, "data/mmis.db")
    con = sqlite3.connect(f"{DB_FILE.as_uri()}?mode=ro", uri=True)
    try:
        market = pd.read_sql("SELECT date, ticker, target FROM market_data", con)
        visual = pd.read_sql("SELECT date, ticker FROM visual_features", con)
    finally:
        con.close()

    for frame in (market, visual):
        frame["date"] = pd.to_datetime(frame["date"]).dt.date

    df = market.merge(visual, on=["date", "ticker"], how="inner")
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


def synthetic_frame(start="2024-01-01", n_dates=700, tickers=TICKERS):
    """A (date, ticker) panel straddling TRAIN_END_DATE, six rows per date.

    700 business days from 2024-01-01 runs well past the boundary, so the
    validation period is large enough to delete hundreds of rows from without
    exhausting it — which the invariance test needs.
    """
    dates = pd.bdate_range(start, periods=n_dates)
    rows = [
        {"date": d.date(), "ticker": t, "target": (i + j) % 3}
        for i, d in enumerate(dates)
        for j, t in enumerate(tickers)
    ]
    frame = pd.DataFrame(rows)
    assert frame["date"].min() < BOUNDARY.date() < frame["date"].max(), \
        "synthetic frame must straddle the boundary or it proves nothing"
    return frame.sort_values(["date", "ticker"]).reset_index(drop=True)


def dates_of(frame):
    return pd.to_datetime(frame["date"]).dt.normalize()


# ══════════════════════════════════════════════════════════════════
#  The split on the real aligned frame
# ══════════════════════════════════════════════════════════════════

def test_train_dates_all_precede_validation_dates(aligned):
    """The property a row-count cut could not guarantee."""
    train, val = temporal_split(aligned)

    assert dates_of(train).max() < dates_of(val).min()
    assert dates_of(train).max() <= BOUNDARY
    assert dates_of(val).min() > BOUNDARY


def test_no_date_appears_in_both_splits(aligned):
    train, val = temporal_split(aligned)
    assert not (set(dates_of(train).unique()) & set(dates_of(val).unique()))


def test_all_six_tickers_for_a_date_land_on_the_same_side(aligned):
    """The old cut fell INSIDE 2025-03-17: AAPL/AMZN/GOOGL/MSFT/SPY went to
    train and TSLA to validation, decided purely by alphabetical order within
    the day. No date may be divided again."""
    train, val = temporal_split(aligned)

    train_dates = set(dates_of(train).unique())
    val_dates = set(dates_of(val).unique())

    # Every date is wholly on one side, so per-date ticker counts must match
    # the full frame's count for that date.
    per_date = aligned.groupby(dates_of(aligned)).size()
    for frame, owned in ((train, train_dates), (val, val_dates)):
        counts = frame.groupby(dates_of(frame)).size()
        for date, n in counts.items():
            assert n == per_date[date], f"{date} is split across the boundary"

    # And specifically: the date the old code straddled is now undivided.
    straddled = pd.Timestamp("2025-03-17")
    assert not (straddled in train_dates and straddled in val_dates)


def test_validation_is_large_enough_to_be_meaningful(aligned):
    """A boundary that leaves a token validation set is a wrong boundary."""
    _, val = temporal_split(aligned)
    assert len(val) >= 500


def test_split_partitions_the_frame_without_duplicating_rows(aligned):
    """Train and validation must be disjoint, and together account for every
    row except the embargoed dates."""
    train, val = temporal_split(aligned, embargo_days=0)
    assert len(train) + len(val) == len(aligned)
    assert not (set(train.index) & set(val.index))

    train1, val1 = temporal_split(aligned, embargo_days=1)
    embargoed = len(aligned) - len(train1) - len(val1)
    assert embargoed == 6, "one embargoed date should remove exactly six rows"


# ══════════════════════════════════════════════════════════════════
#  The boundary is imported, not restated
# ══════════════════════════════════════════════════════════════════

def test_boundary_is_imported_from_regime_causal_not_a_duplicate_literal():
    """A second copy of the date is exactly the drift this change removes."""
    assert fusion.TRAIN_END_DATE is regime_causal.TRAIN_END_DATE
    assert inspect.signature(temporal_split).parameters[
        "train_end_date"].default is regime_causal.TRAIN_END_DATE

    # No date literal in the split path itself. Strip comments, then strip the
    # docstring (which legitimately names 2025-03-17 when describing the old
    # defect), leaving only executable source.
    source = inspect.getsource(temporal_split)
    body = "\n".join(
        line for line in source.splitlines()
        if not line.strip().startswith("#")
    )
    parts = body.split('"""')
    body = parts[0] + "".join(parts[2:])
    assert not re.search(r"\d{4}-\d{2}-\d{2}", body), \
        f"date literal found in temporal_split's executable body: {body}"


# ══════════════════════════════════════════════════════════════════
#  THE CENTRAL PROPERTY — invariance to row count
# ══════════════════════════════════════════════════════════════════

def test_boundary_is_invariant_to_row_count():
    """Dropping rows from the VALIDATION period must not move the boundary.

    This is the entire point of the task. The old rule computed the cut as a
    fraction of len(df), so deleting validation rows dragged the boundary
    backwards into the training period and silently reclassified days that had
    been held out.
    """
    frame = synthetic_frame()
    train, val = temporal_split(frame)
    baseline_last_train = dates_of(train).max()
    baseline_first_val = dates_of(val).min()
    baseline_train_rows = len(train)

    rng = np.random.default_rng(0)
    val_positions = frame.index[dates_of(frame) > BOUNDARY]

    for n_drop in (1, 50, 200, 600):
        shrunk = frame.drop(index=rng.choice(val_positions, n_drop, replace=False))
        shrunk = shrunk.reset_index(drop=True)
        tr, va = temporal_split(shrunk)

        assert dates_of(tr).max() == baseline_last_train, \
            f"dropping {n_drop} validation rows moved the last training date"
        assert dates_of(va).min() == baseline_first_val
        assert len(tr) == baseline_train_rows, \
            f"dropping {n_drop} validation rows changed the training set size"


def test_the_old_row_count_rule_was_not_invariant():
    """Anti-vacuity companion: prove the check above discriminates.

    If `int(len(df) * 0.8)` also passed the invariance test, that test would be
    proving nothing. It does not pass — the boundary moves substantially.
    """
    frame = synthetic_frame()

    def old_rule(df):
        cut = int(len(df) * 0.8)
        return pd.Timestamp(df.iloc[cut]["date"])

    baseline = old_rule(frame)

    rng = np.random.default_rng(0)
    val_positions = frame.index[dates_of(frame) > BOUNDARY]
    shrunk = frame.drop(index=rng.choice(val_positions, 600, replace=False))
    shrunk = shrunk.reset_index(drop=True)

    assert old_rule(shrunk) != baseline, \
        "the old rule did not move; the invariance test would be vacuous"
    # And it moves backwards, into what used to be training data.
    assert old_rule(shrunk) < baseline


def test_boundary_is_invariant_to_added_rows():
    """The mirror case: extending the series must not move the boundary either.

    A re-ingest appends new bars, which under the old rule pushed the cut
    forward and pulled fresh dates into training.
    """
    short = synthetic_frame(n_dates=350)
    long = synthetic_frame(n_dates=500)

    tr_short, _ = temporal_split(short)
    tr_long, _ = temporal_split(long)

    assert dates_of(tr_short).max() == dates_of(tr_long).max()
    assert len(tr_short) == len(tr_long)


# ══════════════════════════════════════════════════════════════════
#  The embargo
# ══════════════════════════════════════════════════════════════════

def test_default_embargo_matches_the_current_next_day_target():
    """fusion.py trains on market_data.target, a NEXT-DAY label, so 1 is
    correct today. T5 must raise it to 5 with the vol_target switch."""
    assert EMBARGO_DAYS == 1


@pytest.mark.parametrize("embargo", [0, 1, 5])
def test_embargo_drops_whole_dates_not_rows(embargo):
    """Six tickers per date must stay aligned: an embargo that removed rows
    rather than dates would leave a partial day in training."""
    frame = synthetic_frame()
    train, _ = temporal_split(frame, embargo_days=embargo)

    eligible = sorted(set(dates_of(frame)[dates_of(frame) <= BOUNDARY].unique()))
    expected_dates = eligible[: len(eligible) - embargo] if embargo else eligible

    assert sorted(set(dates_of(train).unique())) == expected_dates

    # Every retained date keeps all six tickers.
    counts = train.groupby(dates_of(train)).size()
    assert set(counts.unique()) == {len(TICKERS)}


@pytest.mark.parametrize("embargo,expected_dates_dropped", [(0, 0), (1, 1), (5, 5)])
def test_embargo_removes_exactly_the_trailing_training_dates(embargo, expected_dates_dropped):
    frame = synthetic_frame()
    no_embargo, _ = temporal_split(frame, embargo_days=0)
    train, _ = temporal_split(frame, embargo_days=embargo)

    all_train_dates = sorted(set(dates_of(no_embargo).unique()))
    kept = sorted(set(dates_of(train).unique()))
    dropped = [d for d in all_train_dates if d not in kept]

    assert len(dropped) == expected_dates_dropped
    # The dropped dates are the LAST ones, not an arbitrary subset.
    assert dropped == all_train_dates[len(all_train_dates) - expected_dates_dropped:] \
        if expected_dates_dropped else dropped == []
    # Six rows per dropped date.
    assert len(no_embargo) - len(train) == expected_dates_dropped * len(TICKERS)


def test_embargo_never_moves_rows_into_validation(aligned):
    """Embargoed rows are dropped from BOTH sides, never reclassified."""
    _, val0 = temporal_split(aligned, embargo_days=0)
    _, val5 = temporal_split(aligned, embargo_days=5)
    assert len(val0) == len(val5)
    assert dates_of(val0).min() == dates_of(val5).min()


# ══════════════════════════════════════════════════════════════════
#  The guards actually fire
# ══════════════════════════════════════════════════════════════════

def test_empty_side_guard_fires_when_the_boundary_excludes_a_whole_side():
    """A boundary outside the data must fail loudly, not return an empty frame.

    Named for what it actually exercises. The OTHER two guards in
    temporal_split — the max(train) >= min(val) check and the shared-date set
    intersection — are unreachable by construction, because train_mask and
    val_mask are a strict partition of one normalised date series. They are
    defence-in-depth against a future edit breaking that partition, and this
    test deliberately does not claim to cover them.
    """
    frame = synthetic_frame(n_dates=60, start="2025-02-03")

    # Boundary before all data -> empty train side.
    with pytest.raises(ValueError, match="empty side"):
        temporal_split(frame, train_end_date="2019-01-01")

    # Boundary after all data -> empty validation side.
    with pytest.raises(ValueError, match="empty side"):
        temporal_split(frame, train_end_date="2030-01-01")


def test_embargo_consuming_the_whole_training_side_raises():
    """An embargo longer than the training window must fail loudly, not
    silently return an empty training set."""
    frame = synthetic_frame(n_dates=60, start="2025-02-03")
    n_train_dates = len(set(dates_of(frame)[dates_of(frame) <= BOUNDARY].unique()))

    with pytest.raises(ValueError, match="empty side"):
        temporal_split(frame, embargo_days=n_train_dates + 5)


def test_split_is_deterministic_and_preserves_columns(aligned):
    a_train, a_val = temporal_split(aligned)
    b_train, b_val = temporal_split(aligned)

    pd.testing.assert_frame_equal(a_train, b_train)
    pd.testing.assert_frame_equal(a_val, b_val)
    assert list(a_train.columns) == list(aligned.columns)
    assert list(a_val.columns) == list(aligned.columns)
