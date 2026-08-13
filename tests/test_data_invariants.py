"""Characterization tests for data/mmis.db — read-only.

These pin the schema and shape facts recorded in docs/STATE_REPORT.md §3. The
database is gitignored and is the only copy of several irreplaceable tables, so
any silent change to it should surface here as a visible red rather than as a
quietly different number in a report months later.

Two tests carry defect IDs and deliberately assert WRONG behaviour. See
tests/README.md.
"""

import re

import pytest

pytestmark = pytest.mark.characterization

# docs/STATE_REPORT.md §3.1 — the exact 44 columns, in ordinal order.
MARKET_DATA_COLUMNS = [
    "date", "ticker", "open", "high", "low", "close", "volume",
    "rsi_14", "macd", "macd_signal", "macd_diff",
    "ema_20", "ema_50",
    "bb_upper", "bb_lower", "bb_mid", "bb_bandwidth", "bb_position",
    "atr_14",
    "returns_1d", "returns_5d", "returns_10d", "returns_20d",
    "volume_ma_20", "volume_ratio", "volume_change_1d",
    "body_size", "upper_shadow", "lower_shadow", "high_low_range",
    "day_of_week", "month", "day_of_year",
    "day_of_year_sin", "day_of_year_cos", "month_sin", "month_cos",
    "time_idx", "group_id", "target", "split", "created_at",
    "regime", "regime_id",
    # T2 shadow columns — causal regime labels, added by regime_causal.py.
    # The two above are deliberately left untouched for comparison.
    "regime_causal", "regime_id_causal",
    # T3 shadow columns — forward volatility target, added by rv_target.py.
    # `target` above is likewise left untouched for comparison.
    "fwd_rv_5d", "vol_target", "vol_target_id",
]

# The only columns in this database permitted to hold NULL, and the reason they
# are: a 5-day FORWARD label does not exist for the last 5 rows of a ticker.
# Representing that absence as NULL is the entire point (U3 — ingest.py:135-140
# instead defaults its final row to 1/Flat, fabricating six labels). Their NULL
# count is not merely allowed but pinned exactly, in
# test_t3_columns_hold_exactly_thirty_nulls_on_the_final_rows below.
T3_NULLABLE_COLUMNS = ["fwd_rv_5d", "vol_target", "vol_target_id"]

# regime.py:69-73 inverted — the fixed name->id map every ticker must honour.
CAUSAL_NAME_TO_ID = {"mean_reverting": 0, "trending": 1, "high_vol": 2}

EXPECTED_ROWS = {
    "market_data": 9412,
    "visual_features": 9292,
    "sentiment_data": 124,
    "chart_images": 0,
}

# regime.py:69-73 — the contract regime_id is SUPPOSED to satisfy.
REGIME_NAMES = {0: "mean_reverting", 1: "trending", 2: "high_vol"}

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}$")

POPULATED_TABLES = ["market_data", "visual_features", "sentiment_data"]


def columns_of(q, table):
    return [r[1] for r in q(f"PRAGMA table_info({table})")]


# ══════════════════════════════════════════════════════════════════
#  Tables and shapes
# ══════════════════════════════════════════════════════════════════

def test_database_has_exactly_four_tables(q):
    names = [r[0] for r in q(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )]
    assert names == ["chart_images", "market_data", "sentiment_data", "visual_features"]


@pytest.mark.parametrize("table,expected", sorted(EXPECTED_ROWS.items()))
def test_table_row_counts(scalar, table, expected):
    assert scalar(f"SELECT COUNT(*) FROM {table}") == expected


def test_market_data_has_49_columns_with_exact_names(q):
    """Was 44 before T2, 46 after it; T3's C1 appended three more.

    Renamed from test_market_data_has_46_columns_with_exact_names. The exact
    name list is still asserted in full — this pins the new shape, it does not
    loosen the check.
    """
    cols = columns_of(q, "market_data")
    assert len(cols) == 49
    assert cols == MARKET_DATA_COLUMNS
    # The originals keep their ordinal positions; the new ones are appended.
    assert cols[42:44] == ["regime", "regime_id"]
    assert cols[44:46] == ["regime_causal", "regime_id_causal"]
    assert cols[46:] == ["fwd_rv_5d", "vol_target", "vol_target_id"]


def test_chart_images_exists_but_is_dead(scalar, q):
    """§3.9: created by ingest.py:226-234 with a foreign key, never written to."""
    assert scalar("SELECT COUNT(*) FROM chart_images") == 0
    assert columns_of(q, "chart_images") == ["date", "ticker", "image_path", "window_size"]


def test_sentiment_and_visual_column_names(q):
    assert columns_of(q, "sentiment_data") == [
        "date", "ticker", "sentiment_neg", "sentiment_neu",
        "sentiment_pos", "sentiment_score", "headline_count",
    ]
    assert columns_of(q, "visual_features") == [
        "date", "ticker", "image_path", "feature_vector",
    ]


# ══════════════════════════════════════════════════════════════════
#  Date storage format — load-bearing for any date-matching code
# ══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("table", POPULATED_TABLES)
def test_dates_are_text_of_length_26(q, scalar, table):
    """§3.4: every date is TEXT, uniformly 26 chars, with a microsecond suffix.

    This is why regime.py must resolve dates through a normalised-Timestamp
    lookup: comparing against strftime('%Y-%m-%d') matches ZERO rows on '='.
    """
    assert [r[0] for r in q(f"SELECT DISTINCT typeof(date) FROM {table}")] == ["text"]
    assert [r[0] for r in q(f"SELECT DISTINCT LENGTH(date) FROM {table}")] == [26]
    assert scalar(f"SELECT COUNT(*) FROM {table} WHERE date LIKE '%.000000'") == \
        scalar(f"SELECT COUNT(*) FROM {table}")


@pytest.mark.parametrize("table", POPULATED_TABLES)
def test_every_date_matches_the_microsecond_format(q, table):
    """Regex-check every distinct value, not just the length."""
    bad = [d for (d,) in q(f"SELECT DISTINCT date FROM {table}") if not DATE_RE.match(d)]
    assert bad == []


def test_a_bare_iso_date_matches_no_row(scalar):
    """The trap, made explicit: '2020-03-13' finds nothing; the 26-char form does."""
    assert scalar("SELECT COUNT(*) FROM market_data WHERE date = '2020-03-13'") == 0
    assert scalar(
        "SELECT COUNT(*) FROM market_data WHERE date = '2020-03-13 00:00:00.000000'"
    ) == 6


# ══════════════════════════════════════════════════════════════════
#  Coverage and spans
# ══════════════════════════════════════════════════════════════════

def test_six_tickers_with_the_expected_span(q, scalar):
    assert scalar("SELECT COUNT(DISTINCT ticker) FROM market_data") == 6
    assert [r[0] for r in q("SELECT DISTINCT ticker FROM market_data ORDER BY ticker")] == \
        ["AAPL", "AMZN", "GOOGL", "MSFT", "SPY", "TSLA"]
    assert scalar("SELECT MIN(date) FROM market_data") == "2020-03-13 00:00:00.000000"
    assert scalar("SELECT MAX(date) FROM market_data") == "2026-06-10 00:00:00.000000"


def test_no_duplicate_date_ticker_pairs(scalar):
    """§3.9: neither sentiment_data nor visual_features has a PK, yet neither
    currently holds a duplicate. Pinned because to_sql(if_exists='replace')
    destroys the declared PK, so nothing enforces this."""
    for table in POPULATED_TABLES:
        total = scalar(f"SELECT COUNT(*) FROM {table}")
        distinct = scalar(f"SELECT COUNT(*) FROM (SELECT DISTINCT date, ticker FROM {table})")
        assert total == distinct, f"{table} has duplicate (date, ticker) rows"


@pytest.mark.parametrize("table", POPULATED_TABLES)
def test_zero_nulls_in_every_column(q, scalar, table):
    """§3.5: 0 NULLs in all 46 + 7 + 4 columns, plus T3's three by exception.

    Was "all 44 + 7 + 4". The three T3 forward-label columns are excluded here
    and asserted SEPARATELY and more strictly below — exact count, exact rows —
    because for them NULL is the correct value on 30 rows rather than a defect.
    Every other column, including all 46 that existed before T3, is still held
    to zero NULLs by the loop below, so this is narrower in scope and not weaker
    in force.
    """
    for col in columns_of(q, table):
        if table == "market_data" and col in T3_NULLABLE_COLUMNS:
            continue
        assert scalar(f'SELECT COUNT(*) FROM {table} WHERE "{col}" IS NULL') == 0, \
            f"{table}.{col} contains NULLs"


def test_t3_columns_hold_exactly_thirty_nulls_on_the_final_rows(q, scalar):
    """The exception carved out above, pinned exactly rather than waived.

    A 5-day forward label cannot exist for the last 5 rows of a ticker. Those
    rows — 5 x 6 tickers = 30 — must be NULL, on the same rows in all three
    columns, and nowhere else. Anything else means either a fabricated label
    (U3's failure mode) or a gap in the middle of the series.
    """
    for col in T3_NULLABLE_COLUMNS:
        assert scalar(f"SELECT COUNT(*) FROM market_data WHERE {col} IS NULL") == 30, \
            f"market_data.{col} does not hold exactly 30 NULLs"

    # All three NULL on identical rows — never a label without its RV, or vice versa.
    assert scalar(
        "SELECT COUNT(*) FROM market_data WHERE "
        "(fwd_rv_5d IS NULL) <> (vol_target IS NULL) "
        "OR (vol_target IS NULL) <> (vol_target_id IS NULL)"
    ) == 0

    # And they are the final five rows of each ticker, not an interior gap.
    for (ticker,) in q("SELECT DISTINCT ticker FROM market_data ORDER BY ticker"):
        nulls = [r[0] for r in q(
            "SELECT date FROM market_data WHERE ticker=? AND vol_target IS NULL "
            "ORDER BY date", (ticker,)
        )]
        tail = [r[0] for r in q(
            "SELECT date FROM market_data WHERE ticker=? ORDER BY date DESC LIMIT 5",
            (ticker,)
        )]
        assert nulls == sorted(tail), f"{ticker}: NULLs are not the final five rows"


def test_regime_column_is_fully_populated_with_three_labels(q, scalar):
    assert scalar("SELECT COUNT(*) FROM market_data WHERE regime IS NULL") == 0
    counts = dict(q("SELECT regime, COUNT(*) FROM market_data GROUP BY regime"))
    assert counts == {"mean_reverting": 3834, "trending": 3683, "high_vol": 1895}


# ══════════════════════════════════════════════════════════════════
#  Vision features
# ══════════════════════════════════════════════════════════════════

def test_feature_vector_blobs_are_uniformly_5120_bytes(q):
    """§3.7: 5,120 bytes = 1,280 float32. Nothing in the DB records the dtype,
    so readers must hardcode np.frombuffer(..., dtype=np.float32)."""
    lengths = [r[0] for r in q("SELECT DISTINCT LENGTH(feature_vector) FROM visual_features")]
    assert lengths == [5120]
    assert 5120 // 4 == 1280


def test_feature_vector_is_stored_as_blob_despite_being_declared_text(q):
    """§3.1: declared types in this database are unreliable."""
    assert [r[0] for r in q("SELECT DISTINCT typeof(feature_vector) FROM visual_features")] \
        == ["blob"]
    declared = {r[1]: r[2] for r in q("PRAGMA table_info(visual_features)")}
    assert declared["feature_vector"] == "TEXT"


def test_visual_features_covers_9292_of_9412_market_rows(scalar):
    """The 120-row gap is the first 20 bars per ticker (vision.py loop start)."""
    assert scalar("SELECT COUNT(*) FROM visual_features") == 9292
    assert scalar("SELECT COUNT(*) FROM market_data") - 9292 == 120


# ══════════════════════════════════════════════════════════════════
#  Pinned defects
# ══════════════════════════════════════════════════════════════════

@pytest.mark.defect
def test_regime_and_regime_id_are_mutually_inconsistent_U1(q, scalar):
    """Pins U1 (docs/STATE_REPORT.md §5): `regime` and `regime_id` are written
    by one UPDATE but come from different value spaces, so they disagree.

    `regime_id` is hmmlearn's RAW state index, arbitrary per EM initialisation.
    `regime` is that index remapped through a state_map built by sorting states
    on realised volatility (regime.py:188-194). A separate HMM is fit per ticker
    (regime.py:595-601), so the raw->name permutation is redrawn six times.
    REGIME_NAMES (regime.py:69-73) declares a contract that nothing enforces.

    Any consumer that groups or joins on regime_id silently merges three
    different semantic regimes.

    EXPECTED TO GO RED when U1 is fixed. That red is the intended signal.
    """
    total = scalar("SELECT COUNT(*) FROM market_data")

    # Derive the honouring count from REGIME_NAMES rather than hardcoding it.
    clauses = " OR ".join(
        f"(regime_id = {rid} AND regime = '{name}')" for rid, name in REGIME_NAMES.items()
    )
    honouring = scalar(f"SELECT COUNT(*) FROM market_data WHERE {clauses}")

    pct = 100.0 * honouring / total
    assert honouring == 6666
    assert total == 9412
    assert pct == pytest.approx(70.82, abs=0.01), f"contract-honouring share is {pct:.4f}%"

    # A contract that held would be 100%. It is not.
    assert honouring < total

    # The mapping is 1:1 WITHIN a ticker but permuted ACROSS tickers.
    per_ticker = {}
    for ticker, regime, regime_id in q(
        "SELECT DISTINCT ticker, regime, regime_id FROM market_data ORDER BY ticker, regime"
    ):
        per_ticker.setdefault(ticker, {})[regime] = regime_id

    assert len(per_ticker) == 6
    for ticker, mapping in per_ticker.items():
        assert len(mapping) == 3, f"{ticker} has a non-1:1 regime->regime_id map"
        assert len(set(mapping.values())) == 3, f"{ticker} reuses a regime_id"

    distinct_mappings = {tuple(sorted(m.items())) for m in per_ticker.values()}
    assert len(distinct_mappings) > 1, "regime->regime_id agrees across tickers"

    # Concretely: GOOGL and SPY disagree with the other four.
    assert per_ticker["AAPL"]["mean_reverting"] == 0
    assert per_ticker["GOOGL"]["mean_reverting"] == 1
    assert per_ticker["SPY"]["mean_reverting"] == 2


@pytest.mark.defect
def test_regime_id_marginals_do_not_reconcile_with_regime_U1(q):
    """Pins U1 from the other side: the two columns' marginal distributions
    cannot be reconciled, which is the cheapest way to detect the inconsistency.

    EXPECTED TO GO RED when U1 is fixed.
    """
    by_name = dict(q("SELECT regime, COUNT(*) FROM market_data GROUP BY regime"))
    by_id = dict(q("SELECT regime_id, COUNT(*) FROM market_data GROUP BY regime_id"))

    assert by_name == {"mean_reverting": 3834, "trending": 3683, "high_vol": 1895}
    assert by_id == {0: 3754, 1: 3149, 2: 2509}

    # If the contract held, these two would be the same multiset of counts.
    assert sorted(by_name.values()) != sorted(by_id.values())


# ══════════════════════════════════════════════════════════════════
#  T2 shadow columns — causal regime labels
# ══════════════════════════════════════════════════════════════════

def test_causal_regime_columns_are_fully_populated(scalar, q):
    assert scalar("SELECT COUNT(*) FROM market_data WHERE regime_causal IS NULL") == 0
    assert scalar("SELECT COUNT(*) FROM market_data WHERE regime_id_causal IS NULL") == 0
    names = {r[0] for r in q("SELECT DISTINCT regime_causal FROM market_data")}
    assert names == set(CAUSAL_NAME_TO_ID)
    ids = {r[0] for r in q("SELECT DISTINCT regime_id_causal FROM market_data")}
    assert ids == set(CAUSAL_NAME_TO_ID.values())


def test_causal_regime_id_honours_the_contract_on_every_row(scalar):
    """U1 is FIXED in the shadow columns: 100%, not 70.82%.

    Derived from CAUSAL_NAME_TO_ID rather than hardcoded, so the test tracks the
    contract rather than a snapshot of it.
    """
    clauses = " OR ".join(
        f"(regime_id_causal = {rid} AND regime_causal = '{name}')"
        for name, rid in CAUSAL_NAME_TO_ID.items()
    )
    total = scalar("SELECT COUNT(*) FROM market_data")
    honouring = scalar(f"SELECT COUNT(*) FROM market_data WHERE {clauses}")
    assert honouring == total == 9412
    assert 100.0 * honouring / total == 100.0


def test_causal_name_to_id_mapping_is_identical_across_tickers(q):
    """The whole point of the U1 fix: regime_id_causal N means the same regime
    on every ticker, unlike regime_id."""
    per_ticker = {}
    for ticker, name, rid in q(
        "SELECT DISTINCT ticker, regime_causal, regime_id_causal FROM market_data"
    ):
        per_ticker.setdefault(ticker, {})[name] = rid

    assert len(per_ticker) == 6
    for ticker, mapping in per_ticker.items():
        assert mapping == CAUSAL_NAME_TO_ID, f"{ticker} disagrees: {mapping}"

    # Exactly one distinct mapping across all six — contrast with the old
    # columns, which have three.
    assert len({tuple(sorted(m.items())) for m in per_ticker.values()}) == 1


def test_causal_marginals_agree_between_name_and_id(q):
    """A consequence of the fixed map: the two columns' marginal distributions
    are the same multiset. The legacy pair fails this (see the U1 tests)."""
    by_name = dict(q("SELECT regime_causal, COUNT(*) FROM market_data GROUP BY 1"))
    by_id = dict(q("SELECT regime_id_causal, COUNT(*) FROM market_data GROUP BY 1"))
    assert sorted(by_name.values()) == sorted(by_id.values())
    for name, rid in CAUSAL_NAME_TO_ID.items():
        assert by_name[name] == by_id[rid]


def test_causal_labels_differ_substantially_from_the_legacy_labels(scalar):
    """If the causal path had reproduced the acausal one, the leak would not
    have been removed. Agreement is ~73%, well under the 95% suspicion line
    agreed for T2."""
    total = scalar("SELECT COUNT(*) FROM market_data")
    agreeing = scalar(
        "SELECT COUNT(*) FROM market_data WHERE regime_causal = regime"
    )
    agreement = 100.0 * agreeing / total
    assert agreeing == 6887
    assert agreement == pytest.approx(73.1725, abs=0.01)
    assert agreement < 95.0, "suspiciously high agreement — is the leak still there?"


def test_legacy_regime_columns_were_not_modified_by_T2(q, scalar):
    """The shadow-column design only works if the originals are untouched.

    Pins the exact legacy distributions measured before T2 ran.
    """
    assert dict(q("SELECT regime, COUNT(*) FROM market_data GROUP BY regime")) == {
        "mean_reverting": 3834, "trending": 3683, "high_vol": 1895,
    }
    assert dict(q("SELECT regime_id, COUNT(*) FROM market_data GROUP BY regime_id")) == {
        0: 3754, 1: 3149, 2: 2509,
    }
    assert scalar("SELECT COUNT(*) FROM market_data WHERE regime IS NULL") == 0


@pytest.mark.defect
def test_sentiment_covers_1_3_percent_in_a_30_day_window_D8(q, scalar):
    """Pins D8 (docs/STATE_REPORT.md §5): NewsAPI's free tier serves ~30 days,
    so sentiment covers 1.3% of a market table spanning 2020-2026, and
    to_sql(if_exists='replace') at sentiment.py means re-runs cannot accumulate
    coverage — each run's window overwrites the last entirely.

    EXPECTED TO GO RED when D8 is fixed (by a paid tier, an accumulating write,
    or a backfill). That red is the intended signal.
    """
    sentiment_rows = scalar("SELECT COUNT(*) FROM sentiment_data")
    market_rows = scalar("SELECT COUNT(*) FROM market_data")

    coverage = 100.0 * sentiment_rows / market_rows
    assert sentiment_rows == 124
    assert coverage == pytest.approx(1.3175, abs=0.001), f"coverage is {coverage:.4f}%"

    # Every row joins market_data exactly — the dates match, there just aren't many.
    assert scalar(
        "SELECT COUNT(*) FROM sentiment_data s "
        "JOIN market_data m ON s.date = m.date AND s.ticker = m.ticker"
    ) == 124

    # All 124 rows sit inside one ~30-day window in 2026.
    assert scalar("SELECT MIN(date) FROM sentiment_data") == "2026-05-12 00:00:00.000000"
    assert scalar("SELECT MAX(date) FROM sentiment_data") == "2026-06-10 00:00:00.000000"
    assert scalar("SELECT COUNT(*) FROM sentiment_data WHERE date < '2026-01-01'") == 0

    # 20-21 rows per ticker, against 1,568-1,569 market rows each.
    per_ticker = dict(q("SELECT ticker, COUNT(*) FROM sentiment_data GROUP BY ticker"))
    assert len(per_ticker) == 6
    assert all(20 <= n <= 21 for n in per_ticker.values())


@pytest.mark.defect
def test_sentiment_placeholder_rows_are_only_identifiable_by_headline_count_D8(q, scalar):
    """Pins D8's second half: 24 of the 124 rows are the no-news placeholder
    (0.0, 1.0, 0.0), indistinguishable from a genuine perfectly-neutral reading
    except via headline_count == 0. Over-quota NewsAPI failures are swallowed
    into the same vector (sentiment.py:93-95).

    EXPECTED TO GO RED when D8 is fixed.
    """
    placeholders = scalar(
        "SELECT COUNT(*) FROM sentiment_data "
        "WHERE sentiment_neg = 0.0 AND sentiment_neu = 1.0 "
        "AND sentiment_pos = 0.0 AND sentiment_score = 0.0"
    )
    assert placeholders == 24
    assert scalar("SELECT COUNT(*) FROM sentiment_data WHERE headline_count = 0") == 24

    # The two sets coincide exactly.
    assert scalar(
        "SELECT COUNT(*) FROM sentiment_data "
        "WHERE headline_count = 0 AND NOT (sentiment_neu = 1.0)"
    ) == 0

    # Spread evenly: exactly 4 per ticker.
    per_ticker = dict(q(
        "SELECT ticker, COUNT(*) FROM sentiment_data WHERE headline_count = 0 GROUP BY ticker"
    ))
    assert set(per_ticker.values()) == {4}
