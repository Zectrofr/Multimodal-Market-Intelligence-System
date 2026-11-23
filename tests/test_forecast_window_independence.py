
"""Independent temporal integrity tests for MMIS volatility targets.

Uses synthetic data only. No database access, model training,
network requests, or pipeline execution.
"""

import numpy as np
import pandas as pd

from rv_target import (
    HORIZON,
    add_forward_rv,
    fit_thresholds,
    forward_realized_vol,
    threshold_fit_mask,
)
from regime_causal import TRAIN_END_DATE


def synthetic_prices(n=120, seed=123):
    """Deterministic positive prices with nonconstant volatility."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.015, n - 1)
    return 100.0 * np.exp(
        np.r_[0.0, np.cumsum(returns)]
    )


def test_forward_target_has_missing_tail():
    prices = synthetic_prices()
    result = forward_realized_vol(prices)

    assert len(result) == len(prices)
    assert np.isfinite(result[:-HORIZON]).all()
    assert np.isnan(result[-HORIZON:]).all()


def test_future_mutation_cannot_change_earlier_windows():
    prices = synthetic_prices()
    original = forward_realized_vol(prices)

    mutated = prices.copy()
    mutation_index = 80
    mutated[mutation_index:] *= 1.5

    changed = forward_realized_vol(mutated)

    # All target windows ending before the mutation are identical.
    np.testing.assert_array_equal(
        original[:mutation_index - HORIZON],
        changed[:mutation_index - HORIZON],
    )

    # Anti-vacuity: the mutation must affect at least one
    # target window that actually contains the changed return.
    assert not np.array_equal(
        original[mutation_index - HORIZON:mutation_index],
        changed[mutation_index - HORIZON:mutation_index],
    )


def test_past_mutation_does_not_change_later_windows():
    prices = synthetic_prices()
    original = forward_realized_vol(prices)

    mutated = prices.copy()
    mutated[15] *= 1.2

    changed = forward_realized_vol(mutated)

    np.testing.assert_array_equal(
        original[16: -HORIZON],
        changed[16: -HORIZON],
    )

    assert not np.array_equal(
        original[10:16],
        changed[10:16],
    )


def test_short_series_produces_no_fabricated_labels():
    for length in range(HORIZON + 1):
        prices = synthetic_prices(n=max(length, 2))[:length]
        result = forward_realized_vol(prices)

        assert len(result) == length
        assert np.isnan(result).all()


def test_unsorted_market_rows_are_rejected():
    dates = pd.bdate_range("2025-01-01", periods=20)
    frame = pd.DataFrame({
        "date": dates,
        "close": synthetic_prices(20),
    })

    frame = frame.iloc[::-1].reset_index(drop=True)

    try:
        add_forward_rv(frame)
    except ValueError as exc:
        assert "date" in str(exc).lower()
    else:
        raise AssertionError(
            "Unsorted market data was silently accepted"
        )


def test_threshold_mask_excludes_boundary_crossing_windows():
    boundary = pd.Timestamp(TRAIN_END_DATE)

    dates = pd.bdate_range(
        boundary - pd.Timedelta(days=35),
        periods=50,
    )

    mask = threshold_fit_mask(dates, horizon=HORIZON)
    train_rows = dates <= boundary
    train_indices = np.flatnonzero(train_rows)

    assert train_indices.size > HORIZON
    assert mask.sum() == train_indices.size - HORIZON

    for index in np.flatnonzero(mask):
        assert dates[index + HORIZON] <= boundary

    assert not mask[train_indices[-HORIZON:]].any()


def build_synthetic_ticker_frame(seed):
    dates = pd.bdate_range("2024-06-03", periods=260)
    prices = synthetic_prices(len(dates), seed=seed)

    frame = pd.DataFrame({
        "date": dates,
        "close": prices,
    })

    return add_forward_rv(frame)


def test_validation_prices_cannot_change_fitted_thresholds():
    original = {
        "AAPL": build_synthetic_ticker_frame(101),
        "MSFT": build_synthetic_ticker_frame(202),
    }

    before = fit_thresholds(original)

    modified = {
        ticker: frame.copy(deep=True)
        for ticker, frame in original.items()
    }

    boundary = pd.Timestamp(TRAIN_END_DATE)

    for ticker, frame in modified.items():
        future = frame["date"] > boundary

        # Replace validation prices using a different deterministic
        # random walk, rather than a scale-only transformation.
        count = int(future.sum())
        rng = np.random.default_rng(
            501 if ticker == "AAPL" else 502
        )
        future_prices = 100.0 * np.exp(
            np.cumsum(rng.normal(0, 0.08, count))
        )

        frame.loc[future, "close"] = future_prices
        modified[ticker] = add_forward_rv(
            frame[["date", "close"]]
        )

    after = fit_thresholds(modified)

    assert before["lower"].hex() == after["lower"].hex()
    assert before["upper"].hex() == after["upper"].hex()
    assert before["pooled_train_rows"] == after["pooled_train_rows"]

    # Anti-vacuity: confirm validation-period volatility changed.
    for ticker in original:
        old = original[ticker]["fwd_rv_5d"].to_numpy()
        new = modified[ticker]["fwd_rv_5d"].to_numpy()
        future = original[ticker]["date"] > boundary

        assert not np.array_equal(
            old[future.to_numpy()],
            new[future.to_numpy()],
            equal_nan=True,
        )
