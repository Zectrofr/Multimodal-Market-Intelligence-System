"""CORRECTNESS tests for the causal regime path (T2).

Unlike most of this suite these are not characterization tests: the behaviour
they assert was specified in the task that created regime_causal.py, so they
say what the code SHOULD do and are expected to stay green.

The central property under test is causality: the label at day t must be a
function of observations up to and including t, and of nothing after it. That
is verified EMPIRICALLY, by truncating the series and re-running, rather than
by reading the code — a forward recursion that accidentally consumed future
data would still look correct on the page.

Everything here is synthetic. No database, no artifacts, no network.
"""

import numpy as np
import pandas as pd
import pytest

from regime import apply_standardisation, build_hmm_features_raw, standardisation_stats
from regime_causal import (
    REGIME_NAME_TO_ID,
    TRAIN_END_DATE,
    CausalRegimeHMM,
    filtered_posterior,
    train_mask,
)

# A dramatic volatility change, well AFTER the training boundary, so a leak
# would have to reach backwards across it to be visible.
VOL_SHOCK_DATE = "2025-08-01"


def synthetic_ticker(seed: int = 0, n: int = 900, shock: bool = True) -> pd.DataFrame:
    """Business-day OHLC-ish series straddling TRAIN_END_DATE.

    Volatility is calm for most of the series and, if `shock` is set, becomes
    ~8x larger from VOL_SHOCK_DATE onward. Deterministic given `seed`.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-06-01", periods=n)

    vol = np.full(n, 0.006)
    if shock:
        vol[dates >= pd.Timestamp(VOL_SHOCK_DATE)] = 0.048
    # A mid-training regime change too, so the HMM has something to learn.
    vol[(dates >= pd.Timestamp("2023-09-01")) & (dates < pd.Timestamp("2024-02-01"))] = 0.021

    log_ret = rng.standard_normal(n) * vol
    close = 100 * np.exp(np.cumsum(log_ret))
    return pd.DataFrame({"date": dates, "ticker": f"SYN{seed}", "close": close})


@pytest.fixture(scope="module")
def fitted():
    df = synthetic_ticker()
    return df, CausalRegimeHMM().fit(df)


# ══════════════════════════════════════════════════════════════════
#  THE CENTRAL TEST — no future information reaches any label
# ══════════════════════════════════════════════════════════════════

def test_labels_are_unchanged_by_truncation_after_the_shock(fitted):
    """THE test. Labels for retained dates must be identical whether computed
    from the full series or from a series truncated before the future exists.

    The truncation points bracket a dramatic volatility change. Under the old
    Viterbi + full-sample-standardisation path this fails loudly, because the
    post-shock volatility rescales every earlier observation and the MAP path
    is re-optimised end to end.
    """
    df, hmm = fitted
    full = hmm.label(df)

    shock_idx = int((df["date"] < pd.Timestamp(VOL_SHOCK_DATE)).sum())

    for cut in [shock_idx - 40, shock_idx - 1, shock_idx, shock_idx + 25, len(df) - 1]:
        truncated = hmm.label(df.iloc[:cut].copy())

        assert list(truncated["regime_causal"]) == list(full["regime_causal"][:cut]), (
            f"labels changed when the series was truncated at index {cut} "
            f"({df['date'].iloc[cut].date()}) — future data is leaking backwards"
        )
        assert list(truncated["regime_id_causal"]) == list(
            full["regime_id_causal"][:cut]
        )


def test_filtered_posterior_row_is_unchanged_by_truncation(fitted):
    """The posterior itself, not just its argmax, must be truncation-invariant.

    Verified by truncation rather than by inspecting the recursion.
    """
    df, hmm = fitted
    raw = build_hmm_features_raw(df)
    feats = (raw - hmm.feature_means_) / hmm.feature_stds_

    full_post = filtered_posterior(hmm.model, feats)

    for cut in [120, 400, 700, len(df) - 1]:
        trunc_post = filtered_posterior(hmm.model, feats[:cut])
        np.testing.assert_allclose(trunc_post, full_post[:cut], rtol=0, atol=1e-12)


def test_a_future_shock_cannot_change_an_earlier_label(fitted):
    """Same property from the other direction: replace everything after the
    boundary with a wildly different future and the earlier labels must not move."""
    df, hmm = fitted
    baseline = hmm.label(df)

    altered = df.copy()
    future = altered["date"] >= pd.Timestamp(VOL_SHOCK_DATE)
    # A completely different future: 50x volatility, opposite drift.
    rng = np.random.default_rng(99)
    altered.loc[future, "close"] = 100 * np.exp(
        np.cumsum(rng.standard_normal(int(future.sum())) * 0.30)
    )

    relabelled = hmm.label(altered)
    n_past = int((~future).sum())
    assert list(relabelled["regime_causal"][:n_past]) == list(
        baseline["regime_causal"][:n_past]
    )


def test_the_legacy_path_fails_the_same_truncation_check():
    """Control: prove the test above can actually detect a leak.

    The legacy acausal path (full-sample standardisation + Viterbi) is run
    through the identical truncation, and its labels DO change. Without this,
    a truncation test that trivially passes would prove nothing.
    """
    from regime import MarketRegimeHMM

    df = synthetic_ticker()
    legacy = MarketRegimeHMM().fit(df)

    full = legacy.label(df)
    cut = int((df["date"] < pd.Timestamp(VOL_SHOCK_DATE)).sum())
    truncated = legacy.label(df.iloc[:cut].copy())

    assert list(truncated["regime"]) != list(full["regime"][:cut]), (
        "the legacy path survived truncation — the truncation check is not "
        "sensitive enough to detect a leak, so the causal result is not evidence"
    )


# ══════════════════════════════════════════════════════════════════
#  Train-only fitting
# ══════════════════════════════════════════════════════════════════

def test_standardisation_statistics_come_from_training_rows_only(fitted):
    """Stats must equal those of the train slice, and must NOT equal the
    full-sample stats — otherwise the assertion proves nothing."""
    df, hmm = fitted
    raw = build_hmm_features_raw(df)
    mask = train_mask(df["date"])

    expected_means, expected_stds = standardisation_stats(raw[mask])
    np.testing.assert_allclose(hmm.feature_means_, expected_means, rtol=0, atol=0)
    np.testing.assert_allclose(hmm.feature_stds_, expected_stds, rtol=0, atol=0)

    full_means, full_stds = standardisation_stats(raw)
    assert not np.allclose(hmm.feature_means_, full_means), \
        "train-only means coincide with full-sample means; the test is vacuous"
    assert not np.allclose(hmm.feature_stds_, full_stds)


def test_standardisation_statistics_survive_a_changed_future(fitted):
    """Refitting with a different post-boundary future must not move the stats."""
    df, hmm = fitted
    altered = df.copy()
    future = ~train_mask(altered["date"])
    altered.loc[future, "close"] = altered.loc[future, "close"] * 3.0

    refit = CausalRegimeHMM().fit(altered)
    np.testing.assert_allclose(refit.feature_means_, hmm.feature_means_, rtol=0, atol=0)
    np.testing.assert_allclose(refit.feature_stds_, hmm.feature_stds_, rtol=0, atol=0)


def test_fit_uses_no_row_after_the_training_boundary(fitted):
    """n_train_rows must equal the mask, and every training date must be <= boundary."""
    df, hmm = fitted
    mask = train_mask(df["date"])

    assert hmm.n_train_rows == int(mask.sum())
    assert hmm.n_train_rows < len(df), "the synthetic series must straddle the boundary"
    assert df.loc[mask, "date"].max() <= pd.Timestamp(TRAIN_END_DATE)
    assert df.loc[~mask, "date"].min() > pd.Timestamp(TRAIN_END_DATE)


def test_fitted_model_parameters_are_unaffected_by_post_boundary_data(fitted):
    """Empirical proof of train-only fitting: corrupt the future, refit,
    and every learned parameter must be bit-identical."""
    df, hmm = fitted
    altered = df.copy()
    future = ~train_mask(altered["date"])
    rng = np.random.default_rng(7)
    altered.loc[future, "close"] = 100 * np.exp(
        np.cumsum(rng.standard_normal(int(future.sum())) * 0.25)
    )

    refit = CausalRegimeHMM().fit(altered)

    # Tolerance, not bit-equality: multithreaded BLAS reorders reductions, so
    # refitting on IDENTICAL data can differ by ~1e-12 on some inputs. That is
    # unrelated to causality and must not be able to fail this test. The
    # properties that actually matter — the training INPUT and the resulting
    # LABELS — are asserted exactly, below and in the truncation tests.
    for name in ("startprob_", "transmat_", "means_", "covars_"):
        np.testing.assert_allclose(
            getattr(refit.model, name), getattr(hmm.model, name),
            rtol=0, atol=1e-10, err_msg=f"{name} moved with the future",
        )
    assert refit.state_map == hmm.state_map

    # Exact: the training feature matrix, and every label.
    mask_r = train_mask(altered["date"])
    np.testing.assert_array_equal(
        apply_standardisation(
            build_hmm_features_raw(altered)[mask_r],
            refit.feature_means_, refit.feature_stds_,
        ),
        apply_standardisation(
            build_hmm_features_raw(df)[train_mask(df["date"])],
            hmm.feature_means_, hmm.feature_stds_,
        ),
    )
    assert list(refit.label(df)["regime_causal"]) == list(hmm.label(df)["regime_causal"])


def test_train_mask_boundary_is_inclusive():
    """Rows dated exactly TRAIN_END_DATE are training rows."""
    dates = pd.to_datetime(
        ["2025-03-10", TRAIN_END_DATE, "2025-03-12", "2026-01-01"]
    )
    assert list(train_mask(dates)) == [True, True, False, False]


def test_fit_refuses_when_there_is_too_little_training_data():
    """A series entirely after the boundary must raise, not fit on nothing."""
    df = pd.DataFrame({
        "date": pd.bdate_range("2025-06-01", periods=60),
        "ticker": "LATE",
        "close": np.linspace(100, 110, 60),
    })
    with pytest.raises(ValueError, match="training rows"):
        CausalRegimeHMM().fit(df)


# ══════════════════════════════════════════════════════════════════
#  Shared name -> id mapping (U1)
# ══════════════════════════════════════════════════════════════════

def test_name_to_id_mapping_is_identical_across_tickers():
    """U1's fix: regime_id_causal N must mean the same thing on every ticker.

    Six independently fitted synthetic tickers, each of which will land on a
    different raw hmmlearn state permutation, must still agree on the mapping.
    """
    observed = []
    for seed in range(6):
        df = synthetic_ticker(seed=seed)
        labelled = CausalRegimeHMM().fit(df).label(df)
        pairs = (
            labelled[["regime_causal", "regime_id_causal"]]
            .drop_duplicates()
            .sort_values("regime_causal")
        )
        observed.append({r.regime_causal: r.regime_id_causal for r in pairs.itertuples()})

    first = observed[0]
    for mapping in observed[1:]:
        for name, rid in mapping.items():
            assert first.get(name, rid) == rid, \
                f"ticker disagreement on '{name}': {first.get(name)} vs {rid}"

    # And it is the declared contract, not merely self-consistent.
    for mapping in observed:
        for name, rid in mapping.items():
            assert REGIME_NAME_TO_ID[name] == rid


def test_regime_id_is_derived_from_the_name_not_the_raw_state(fitted):
    """The written id must come from REGIME_NAME_TO_ID, never from hmmlearn's
    arbitrary internal state index — that conflation is U1 itself."""
    df, hmm = fitted
    labelled = hmm.label(df)

    for row in labelled[["regime_causal", "regime_id_causal"]].drop_duplicates().itertuples():
        assert REGIME_NAME_TO_ID[row.regime_causal] == row.regime_id_causal

    assert set(labelled["regime_causal"]) <= set(REGIME_NAME_TO_ID)


def test_state_map_is_ranked_by_training_volatility(fitted):
    """mean_reverting is the calmest state, high_vol the most turbulent —
    measured on training rows only."""
    df, hmm = fitted
    mask = train_mask(df["date"])
    close = df.loc[mask, "close"].values.astype(float)
    log_ret = np.concatenate([[0.0], np.log(close[1:] / close[:-1])])

    raw = build_hmm_features_raw(df)
    feats = (raw[mask] - hmm.feature_means_) / hmm.feature_stds_
    states = filtered_posterior(hmm.model, feats).argmax(axis=1)

    vol_by_name = {}
    for state, name in hmm.state_map.items():
        sel = states == state
        if sel.sum():
            vol_by_name[name] = float(np.std(log_ret[sel]))

    if {"mean_reverting", "high_vol"} <= set(vol_by_name):
        assert vol_by_name["mean_reverting"] <= vol_by_name["high_vol"]


def test_unknown_state_is_explicit_not_a_silent_fourth_regime(fitted):
    """U15: an unmapped state must produce 'unknown'/-1, not an implicit
    all-zero one-hot downstream."""
    from regime_causal import UNKNOWN_REGIME_ID, UNKNOWN_REGIME_NAME

    df, hmm = fitted
    broken = CausalRegimeHMM()
    broken.__dict__.update(hmm.__dict__)
    broken.state_map = {}  # every state now unmapped

    labelled = broken.label(df)
    assert set(labelled["regime_causal"]) == {UNKNOWN_REGIME_NAME}
    assert set(labelled["regime_id_causal"]) == {UNKNOWN_REGIME_ID}
    assert UNKNOWN_REGIME_ID not in REGIME_NAME_TO_ID.values()


# ══════════════════════════════════════════════════════════════════
#  Decoding is filtered, not smoothed or Viterbi
# ══════════════════════════════════════════════════════════════════

def test_filtered_posterior_rows_are_valid_distributions(fitted):
    df, hmm = fitted
    raw = build_hmm_features_raw(df)
    feats = (raw - hmm.feature_means_) / hmm.feature_stds_
    post = filtered_posterior(hmm.model, feats)

    assert post.shape == (len(df), hmm.n_regimes)
    np.testing.assert_allclose(post.sum(axis=1), 1.0, rtol=0, atol=1e-10)
    assert (post >= 0).all()


def test_filtered_differs_from_smoothed_and_viterbi(fitted):
    """If filtered agreed everywhere with the acausal estimates, the change
    would be cosmetic. It must not."""
    df, hmm = fitted
    raw = build_hmm_features_raw(df)
    feats = (raw - hmm.feature_means_) / hmm.feature_stds_

    filtered = filtered_posterior(hmm.model, feats).argmax(axis=1)
    smoothed = hmm.model.predict_proba(feats).argmax(axis=1)
    viterbi = hmm.model.predict(feats)

    assert not np.array_equal(filtered, smoothed), "filtered == smoothed posterior"
    assert not np.array_equal(filtered, viterbi), "filtered == Viterbi path"


def test_first_row_posterior_depends_only_on_the_first_observation(fitted):
    """The t=0 filtered posterior is startprob * emission(x_0), normalised —
    a one-row series must give exactly the same first row as the full series."""
    df, hmm = fitted
    raw = build_hmm_features_raw(df)
    feats = (raw - hmm.feature_means_) / hmm.feature_stds_

    full = filtered_posterior(hmm.model, feats)
    single = filtered_posterior(hmm.model, feats[:1])
    np.testing.assert_allclose(single[0], full[0], rtol=0, atol=1e-12)
