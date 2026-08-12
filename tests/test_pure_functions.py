"""Characterization tests for the pure maths in evaluation.py.

No database, no artifacts, no I/O. Every expected value below is hand-computed
in the comment above its assertion, so the test documents the arithmetic rather
than restating whatever the code happens to return.

Two tests carry defect IDs and deliberately assert WRONG behaviour. See
tests/README.md and docs/STATE_REPORT.md §5.
"""

import numpy as np
import pandas as pd
import pytest

from evaluation import (
    calmar_ratio,
    expected_calibration_error,
    hit_rate,
    max_drawdown,
    precision_at_up,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    walk_forward_split,
)

pytestmark = pytest.mark.characterization


# ══════════════════════════════════════════════════════════════════
#  Risk / return metrics
# ══════════════════════════════════════════════════════════════════

def test_sharpe_ratio_matches_hand_computation():
    """sqrt(252) * mean / std, with numpy's population std (ddof=0)."""
    r = np.array([0.01, -0.01, 0.02, 0.00, 0.03])
    # mean  = (0.01 - 0.01 + 0.02 + 0.00 + 0.03) / 5 = 0.05 / 5 = 0.01
    # devs  = [0.00, -0.02, 0.01, -0.01, 0.02]
    # sq    = [0, 4e-4, 1e-4, 1e-4, 4e-4]; sum = 1.0e-3
    # var   = 1.0e-3 / 5 = 2.0e-4          (ddof=0 — np.std default)
    # std   = sqrt(2.0e-4)               = 0.014142135623730951
    # sharpe= sqrt(252) * 0.01 / 0.0141421356
    #       = 15.874507866387544 * 0.7071067811865476
    #       = 11.224972160321824
    assert sharpe_ratio(r) == pytest.approx(11.224972160321824, rel=1e-12)


def test_sharpe_ratio_returns_zero_when_std_is_zero():
    """Constant returns have zero dispersion; the guard branch returns 0.0, not inf/nan."""
    assert sharpe_ratio(np.array([0.01, 0.01, 0.01, 0.01])) == 0.0


def test_sharpe_ratio_risk_free_is_divided_by_trading_days():
    """risk_free is an ANNUAL figure, deducted per-period as risk_free / 252."""
    r = np.array([0.01, -0.01, 0.02, 0.00, 0.03])
    # excess = r - 0.0252/252 = r - 0.0001
    # mean(excess) = 0.01 - 0.0001 = 0.0099 ; std is unchanged by a constant shift
    # sharpe = 15.874507866387544 * 0.0099 / 0.014142135623730951 = 11.112722438718604
    assert sharpe_ratio(r, risk_free=0.0252) == pytest.approx(11.112722438718604, rel=1e-12)


def test_max_drawdown_matches_hand_computation():
    """Largest peak-to-trough fall, returned as a positive magnitude."""
    c = np.array([1.0, 1.2, 0.9, 1.5, 1.2])
    # running peak = [1.0, 1.2, 1.2, 1.5, 1.5]
    # drawdown     = [0.0, 0.0, (0.9-1.2)/1.2, 0.0, (1.2-1.5)/1.5]
    #              = [0.0, 0.0, -0.25,          0.0, -0.20]
    # min = -0.25 -> abs -> 0.25
    assert max_drawdown(c) == pytest.approx(0.25, rel=1e-12)


def test_max_drawdown_is_zero_for_a_monotonic_rise():
    """A series that never falls below its running peak has no drawdown."""
    assert max_drawdown(np.array([1.0, 1.1, 1.2, 1.3])) == pytest.approx(0.0)


def test_calmar_ratio_matches_hand_computation():
    """Calmar = annualised return / max drawdown = 0.30 / 0.25 = 1.2."""
    assert calmar_ratio(0.30, 0.25) == pytest.approx(1.2, rel=1e-12)


def test_calmar_ratio_returns_zero_when_drawdown_is_zero():
    """The zero-drawdown guard returns 0.0 rather than dividing by zero."""
    assert calmar_ratio(0.30, 0.0) == 0.0


def test_sortino_ratio_matches_hand_computation():
    """Same numerator as Sharpe; denominator is the std of the NEGATIVE returns only."""
    r = np.array([0.02, -0.01, 0.03, -0.03, 0.04])
    # mean(excess)   = (0.02 - 0.01 + 0.03 - 0.03 + 0.04) / 5 = 0.05 / 5 = 0.01
    # downside       = [-0.01, -0.03]
    # mean(downside) = -0.02 ; devs = [0.01, -0.01] ; var = 1e-4 / 1 ... ddof=0 -> 1e-4
    # std(downside)  = 0.01
    # sortino = sqrt(252) * 0.01 / 0.01 = sqrt(252) = 15.874507866387544
    assert sortino_ratio(r) == pytest.approx(15.874507866387544, rel=1e-12)


def test_sortino_ratio_returns_zero_when_there_is_no_downside():
    """No negative return -> len(downside) == 0 -> the guard returns 0.0."""
    assert sortino_ratio(np.array([0.01, 0.02, 0.03])) == 0.0


def test_sortino_ratio_returns_zero_for_a_single_negative_return():
    """One negative value has population std 0.0, so the same guard fires.

    Not a defect, but a sharp edge worth pinning: a strategy with exactly one
    losing day scores 0.0, not infinity.
    """
    assert sortino_ratio(np.array([0.02, -0.01, 0.03])) == 0.0


def test_profit_factor_matches_hand_computation():
    """Gross profit / gross loss."""
    r = np.array([0.02, -0.01, 0.03, -0.03, 0.04])
    # gains  = 0.02 + 0.03 + 0.04 = 0.09
    # losses = |-0.01 - 0.03|     = 0.04
    # pf     = 0.09 / 0.04        = 2.25
    assert profit_factor(r) == pytest.approx(2.25, rel=1e-12)


def test_profit_factor_is_infinite_when_there_are_no_losses():
    """Zero gross loss returns float('inf') rather than raising."""
    assert profit_factor(np.array([0.01, 0.02])) == float("inf")


def test_hit_rate_matches_hand_computation():
    """Fraction of days where pred_direction equals (actual_return > 0)."""
    pred = np.array([1, 0, 1, 1, 0])
    act = np.array([0.01, -0.01, -0.02, 0.03, 0.005])
    # actual_direction = [1, 0, 0, 1, 1]
    # matches          = [T, T, F, T, F]  -> 3 of 5
    assert hit_rate(pred, act) == pytest.approx(0.6, rel=1e-12)


def test_precision_at_up_matches_hand_computation():
    """Of the days predicted UP, the fraction that actually rose."""
    pred = np.array([1, 0, 1, 1, 0])
    act = np.array([0.01, -0.01, -0.02, 0.03, 0.005])
    # up_mask     = [T, F, T, T, F] -> actual[up] = [0.01, -0.02, 0.03]
    # actually up = [T, F, T] -> 2 of 3
    assert precision_at_up(pred, act) == pytest.approx(2.0 / 3.0, rel=1e-12)


def test_precision_at_up_returns_zero_when_nothing_is_predicted_up():
    """No UP predictions -> the guard returns 0.0 rather than nan."""
    assert precision_at_up(np.array([0, 0, 0]), np.array([0.01, 0.02, -0.01])) == 0.0


def test_hit_rate_scores_a_flat_day_as_down():
    """actual_return == 0.0 is classed as NOT up, so a 0-prediction scores a hit.

    Pins the boundary convention: `> 0`, not `>= 0`. Load-bearing because the
    training label uses a 0.5% threshold instead (see U6).
    """
    assert hit_rate(np.array([0]), np.array([0.0])) == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════
#  Expected Calibration Error
# ══════════════════════════════════════════════════════════════════

def test_expected_calibration_error_matches_hand_computation():
    """Weighted mean gap between confidence and accuracy, over 10 equal bins."""
    proba = np.array([0.15, 0.25])
    actual = np.array([0, 1])
    # bins = linspace(0, 1, 11); n = 2
    #   bin [0.1, 0.2): {0.15}, conf 0.15, acc 0 -> (1/2) * |0.15 - 0| = 0.075
    #   bin [0.2, 0.3): {0.25}, conf 0.25, acc 1 -> (1/2) * |0.25 - 1| = 0.375
    # ECE = 0.075 + 0.375 = 0.45
    ece, bin_data = expected_calibration_error(proba, actual)
    assert ece == pytest.approx(0.45, rel=1e-12)
    assert bin_data["count"] == [1, 1]
    assert sum(bin_data["count"]) == len(proba)


@pytest.mark.defect
def test_expected_calibration_error_drops_proba_of_exactly_one_U7():
    """Pins U7 (docs/STATE_REPORT.md §5): a prediction of exactly 1.0 is binned
    out of existence but still counted in the ECE denominator, biasing ECE LOW
    on precisely the most overconfident predictions.

    evaluation.py:111 masks with `(pred_proba >= lo) & (pred_proba < hi)`. The
    final bin's `hi` is exactly 1.0, so `pred_proba == 1.0` matches no bin, while
    `n = len(pred_proba)` at :107 still includes it. inference.py:215 builds an
    IsotonicRegression with y_max=1.0, which emits exactly-1.0 routinely (2 such
    rows in the current final_predictions.csv), so this is live, not theoretical.

    EXPECTED TO GO RED when U7 is fixed. That red is the intended signal.
    """
    proba = np.array([1.0, 0.05])
    actual = np.array([0, 0])  # the 1.0 prediction is confidently WRONG
    ece, bin_data = expected_calibration_error(proba, actual)

    #   bin [0.0, 0.1): {0.05}, conf 0.05, acc 0 -> (1/2) * 0.05 = 0.025
    #   bin [0.9, 1.0): 1.0 fails `< 1.0` -> EMPTY, contributes nothing
    # ECE (as coded)  = 0.025
    # ECE (if fixed)  = 0.025 + (1/2) * |1.0 - 0| = 0.525, i.e. 21x larger
    assert ece == pytest.approx(0.025, rel=1e-12)

    # The dropped row is invisible in the bins but present in the denominator.
    assert sum(bin_data["count"]) == 1
    assert len(proba) == 2

    # The top bin [0.9, 1.0) is not even emitted — it matched nothing.
    assert 0.9 not in bin_data["bin_lower"]


@pytest.mark.defect
def test_expected_calibration_error_is_biased_low_by_the_dropped_bin_U7():
    """Pins U7 quantitatively: the reported ECE is strictly below the honest one.

    Same construction as the test above, but asserts the RELATIONSHIP rather
    than the literal, so it stays meaningful if the bin edges ever change.

    EXPECTED TO GO RED when U7 is fixed.
    """
    proba = np.array([1.0, 0.05])
    actual = np.array([0, 0])
    reported, _ = expected_calibration_error(proba, actual)

    # Honest ECE, computed here with an inclusive final bin.
    honest = 0.5 * abs(0.05 - 0.0) + 0.5 * abs(1.0 - 0.0)
    assert honest == pytest.approx(0.525, rel=1e-12)
    assert reported < honest
    assert reported == pytest.approx(0.025, rel=1e-12)


# ══════════════════════════════════════════════════════════════════
#  Walk-forward split
# ══════════════════════════════════════════════════════════════════

def _wf_frame():
    """Two years of deterministic business-day rows. No randomness anywhere."""
    dates = pd.date_range("2021-01-01", periods=500, freq="B")
    idx = np.arange(len(dates))
    return pd.DataFrame({
        "date": dates,
        "ticker": "TEST",
        # Deterministic alternating +/-1% returns.
        "actual_return": np.where(idx % 2 == 0, 0.01, -0.01),
        # Deterministic alternating direction, out of phase with the returns.
        "pred_direction": (idx % 3 == 0).astype(int),
        "pred_proba": 0.5 + (idx % 5) / 100.0,
    })


@pytest.mark.defect
def test_walk_forward_folds_are_identical_D7():
    """Pins D7 (docs/STATE_REPORT.md §5): the walk-forward loop never advances
    `train_start`, so every fold is byte-identical and the function is not a
    walk-forward at all.

    evaluation.py:333 assigns `train_start = min_date` once and never reassigns
    it. train_end/test_start/test_end are all recomputed FROM it at :336-338 on
    every pass, so the `test_start = test_end` at :364 is overwritten before use
    — dead code. The loop terminates only on the `fold_idx > 24` safety cap at
    :367, emitting 24 copies of the same fold, all testing the same month.

    EXPECTED TO GO RED when D7 is fixed. That red is the intended signal.
    """
    folds = walk_forward_split(_wf_frame())

    # Terminated by the safety cap, not by exhausting the data.
    assert len(folds) == 24

    # Every fold is the same fold, differing only in its index.
    assert len({f.train_start for f in folds}) == 1
    assert len({f.train_end for f in folds}) == 1
    assert len({f.test_start for f in folds}) == 1
    assert len({f.test_end for f in folds}) == 1
    assert len({f.sharpe for f in folds}) == 1
    assert len({f.precision_at_up for f in folds}) == 1
    assert len({f.n_test_days for f in folds}) == 1

    # The fold index is the ONLY thing that moves.
    assert [f.fold for f in folds] == list(range(1, 25))


@pytest.mark.defect
def test_walk_forward_train_start_never_leaves_the_first_date_D7():
    """Pins D7: `train_start` is pinned to the dataset minimum on every fold.

    A correct expanding-window implementation keeps train_start fixed but must
    still ADVANCE the test window; here the test window is frozen too, so all
    24 folds score the same month — months 6->7 after min_date, deep in-sample.

    EXPECTED TO GO RED when D7 is fixed.
    """
    df = _wf_frame()
    folds = walk_forward_split(df)
    min_date = pd.to_datetime(df["date"]).min()

    assert all(f.train_start == str(min_date.date()) for f in folds)
    # 6-month train -> test window opens exactly 6 months after the first date
    # and closes 1 month later, for every single fold.
    assert all(f.test_start == "2021-07-01" for f in folds)
    assert all(f.test_end == "2021-08-01" for f in folds)


def test_walk_forward_returns_no_folds_when_the_span_is_too_short():
    """Under 7 months of data, `test_end > max_date` on the first pass -> no folds.

    Pins the termination branch that DOES work, so a future D7 fix is not free
    to change it silently.
    """
    df = _wf_frame().head(60)  # ~3 months of business days
    assert walk_forward_split(df) == []
