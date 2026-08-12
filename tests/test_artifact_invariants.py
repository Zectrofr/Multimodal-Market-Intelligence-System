"""Characterization tests for results/ and models/ — read-only.

Everything here is gitignored, so a fresh clone SKIPS this file rather than
failing it. On the machine that produced the artifacts, these pin the numbers
recorded in docs/STATE_REPORT.md §6.

Nothing here loads or unpickles a model. models/*.pkl and models/*.pt are
checked for existence and size ONLY — unpickling executes arbitrary code, and a
test suite has no business doing that.
"""

import pytest

from conftest import PROJECT_ROOT, require

pytestmark = pytest.mark.characterization

PREDICTION_COLUMNS = [
    "date", "ticker", "close", "regime", "pred_proba", "pred_direction",
    "uncertainty", "actual_return", "high_uncertainty", "pred_proba_uncalibrated",
]

# docs/STATE_REPORT.md §6 — the mutually consistent seed-42 serving triple.
SERVING_TRIPLE = {
    "models/best_fusion_model.pt": 4_396_394,
    "models/feature_scalers.pkl": 32_116,
    "models/calibrator.pkl": 656,
}


# ══════════════════════════════════════════════════════════════════
#  results/final_predictions.csv
# ══════════════════════════════════════════════════════════════════

def test_final_predictions_shape_and_columns(predictions):
    assert len(predictions) == 9286
    assert list(predictions.columns) == PREDICTION_COLUMNS


def test_final_predictions_has_no_nulls_and_no_duplicates(predictions):
    assert predictions.isnull().sum().sum() == 0
    assert not predictions.duplicated(subset=["date", "ticker"]).any()


def test_final_predictions_dates_are_bare_iso_unlike_the_database(predictions):
    """§6: the CSV stores 'YYYY-MM-DD' with NO time component, while the database
    stores a 26-char microsecond form (§3.4). Any code joining the two on a raw
    string match gets zero rows. Pinned because it is a live trap."""
    assert predictions["date"].map(len).unique().tolist() == [10]
    assert predictions["date"].min() == "2020-04-13"
    assert predictions["date"].max() == "2026-06-09"


def test_final_predictions_covers_six_tickers(predictions):
    counts = predictions["ticker"].value_counts().to_dict()
    assert sorted(counts) == ["AAPL", "AMZN", "GOOGL", "MSFT", "SPY", "TSLA"]
    assert counts == {
        "AAPL": 1547, "GOOGL": 1547,
        "AMZN": 1548, "MSFT": 1548, "SPY": 1548, "TSLA": 1548,
    }


def test_pred_direction_is_binary_despite_a_three_class_head(predictions):
    """§6: class 2 is never emitted; pred_direction only ever takes 0 or 1."""
    assert sorted(predictions["pred_direction"].unique()) == [0, 1]
    assert predictions["pred_direction"].value_counts().to_dict() == {0: 5847, 1: 3439}


def test_pred_proba_is_an_isotonic_step_function(predictions):
    """§6: the isotonic calibrator collapses 9,286 rows onto 15 distinct values,
    including exactly 0.0 and exactly 1.0 — the latter is what U7 discards."""
    assert predictions["pred_proba"].nunique() == 15
    assert predictions["pred_proba"].min() == 0.0
    assert predictions["pred_proba"].max() == 1.0
    assert (predictions["pred_proba"] == 1.0).sum() == 2


def test_uncertainty_never_reaches_the_default_filter_threshold(predictions):
    """Pins the fact behind U5: MC-Dropout uncertainty maxes around 0.00375, so
    the default `uncertainty <= 0.02` filter passes every single row."""
    assert predictions["uncertainty"].max() < 0.02
    assert predictions["uncertainty"].max() == pytest.approx(0.00375, abs=1e-4)
    assert predictions["uncertainty"].min() == pytest.approx(0.000484, abs=1e-5)


def test_pred_proba_uncalibrated_ceiling(predictions):
    """The raw head essentially never says 'up' on a >0.5 basis.

    NOTE — divergence from docs/STATE_REPORT.md §6, which states the max is
    0.5228. The true value is 0.5228198, i.e. the audit figure is a 4-dp
    rounding and the raw maximum strictly EXCEEDS it. This test pins the actual
    value, per the audit's own instruction to prefer measured reality.
    """
    observed = predictions["pred_proba_uncalibrated"].max()
    assert observed == pytest.approx(0.5228198, abs=1e-7)
    assert observed > 0.5228          # the audit's rounded figure is exceeded
    assert observed < 0.5229          # but only in the 5th decimal place
    assert (predictions["pred_proba_uncalibrated"] > 0.5).mean() < 0.05


@pytest.mark.defect
def test_pred_direction_does_not_follow_pred_proba_U11(predictions):
    """Pins U11 (docs/STATE_REPORT.md §5): pred_direction is the 3-class argmax
    while pred_proba is the calibrated binary P(up), so they agree on only ~37%
    of rows.

    This matters because every RETURN metric is driven by pred_direction, while
    the advertised ECE is measured on pred_proba — the calibration number
    describes a probability that does not drive the trades.

    EXPECTED TO GO RED when U11 is fixed. That red is the intended signal.
    """
    implied = (predictions["pred_proba"] > 0.5).astype(int)
    agreement = 100.0 * (predictions["pred_direction"] == implied).mean()

    # Derived, not hardcoded — the literal is only the tolerance anchor.
    assert agreement == pytest.approx(37.06, abs=0.01), \
        f"agreement is {agreement:.4f}%"

    # A column derivable from the other would agree on 100% of rows.
    assert agreement < 50.0

    # The two columns even disagree on their marginal UP rate.
    assert predictions["pred_direction"].mean() == pytest.approx(0.3703, abs=0.001)
    assert implied.mean() != pytest.approx(predictions["pred_direction"].mean(), abs=0.01)


# ══════════════════════════════════════════════════════════════════
#  models/ — existence and size only. Nothing is loaded or unpickled.
# ══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("rel,size", sorted(SERVING_TRIPLE.items()))
def test_serving_triple_exists_with_expected_size(rel, size):
    path = require(PROJECT_ROOT / rel, rel)
    assert path.is_file()
    assert path.stat().st_size == size


def test_serving_triple_is_complete():
    """All three must be present together — inference.py needs every one of them."""
    missing = [rel for rel in SERVING_TRIPLE if not (PROJECT_ROOT / rel).exists()]
    if missing:
        pytest.skip(f"serving artifacts absent (gitignored): {missing}")
    assert len(SERVING_TRIPLE) == 3


def test_six_hmm_pickles_exist():
    """§6: six per-ticker HMMs. Per U12 these cannot be unpickled under uvicorn,
    which is why this test checks existence only and never loads them."""
    models = PROJECT_ROOT / "models"
    require(models, "models/")
    found = sorted(p.name for p in models.glob("hmm_*.pkl"))
    assert found == [
        "hmm_AAPL.pkl", "hmm_AMZN.pkl", "hmm_GOOGL.pkl",
        "hmm_MSFT.pkl", "hmm_SPY.pkl", "hmm_TSLA.pkl",
    ]


def test_synthetic_predictions_are_quarantined():
    """The dummy_model output (D3) must not sit loose in results/ where a glob
    or a careless --csv could pick it up alongside the real file."""
    results = PROJECT_ROOT / "results"
    require(results, "results/")
    assert not (results / "regime_tagged_predictions.csv").exists(), \
        "fabricated D3 output is loose in results/"
