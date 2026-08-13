"""
regime_causal.py — T2: causal regime labelling.

Fixes U4 (regime look-ahead bias) and U1 (regime/regime_id incoherence) from
docs/STATE_REPORT.md §5, writing to SHADOW columns so nothing that currently
depends on market_data.regime / regime_id changes.

The four leaks in the legacy path (regime.py) and what replaces each:

  U4(1) full-sample standardisation      -> statistics fitted on TRAINING ROWS
        (regime.py build_hmm_features)      ONLY, persisted with the model and
                                            applied unchanged to every later date.
  U4(2) HMM fit over the entire history  -> fit restricted to rows on or before
        (regime.py:601-602)                 TRAIN_END_DATE.
  U4(3) Viterbi / smoothed decoding      -> FILTERED posterior via an explicit
        (regime.py:205 .predict())          forward recursion in log space:
                                            P(state_t | obs_1..t), never t+1..T.
  U4(4) full-sample volatility ranking   -> states ranked on TRAINING ROWS ONLY.
        (regime.py:185)

  U1    per-ticker raw hmmlearn state    -> name assigned per ticker, then a
        index written as regime_id           FIXED name->id map shared by every
                                             ticker, so regime_id_causal N means
                                             the same thing everywhere.

Why a separate module: the legacy acausal path must keep producing exactly what
it produced before (the existing columns and the demo CSVs were generated with
it), so regime.py is left almost untouched and everything new lives here.

Usage:
    python regime.py --causal          (preferred entry point)
    python regime_causal.py --all
"""

import argparse
import logging
import pickle
import sqlite3

import numpy as np
import pandas as pd
from hmmlearn import _hmmc
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp

from regime import (
    DB_PATH,
    HMM_ITERATIONS,
    MODEL_DIR,
    N_REGIMES,
    REGIME_NAMES,
    apply_standardisation,
    build_hmm_features_raw,
    load_all_tickers,
    standardisation_stats,
)

logger = logging.getLogger(__name__)

# ── Training boundary ──────────────────────────────────────────────────────
# Provenance: market_data.split, written by ingest.py:243-245 as a PER-TICKER
# 80/20 row-position split. The last date carrying split='train' anywhere in
# the table is 2025-03-11 (see docs/STATE_REPORT.md §3.9). Rows on or before
# this date train the HMM; everything after is labelled forward-only.
#
# NOTE — 2025-03-11 is not a clean global cut: it carries 4 train rows
# (AMZN, MSFT, SPY, TSLA) and 2 validation rows (AAPL, GOOGL), because the
# split was cut per ticker by row position rather than by calendar date. Using
# <= 2025-03-11 therefore admits at most 2 rows that ingest.py called
# validation. That is a 2-row, single-day imprecision inherited from the data.
#
# THIS BOUNDARY MUST MATCH THE SPLIT fusion.py USES. It currently does NOT --
# fusion.py:419 takes int(len(df) * 0.8) over the market-visual aligned frame,
# a row-count fraction landing on 2025-03-17. assert_boundary_matches_fusion()
# below measures the disagreement. It REPORTS; it never reconciles.
TRAIN_END_DATE = "2025-03-11"

# Fixed, ticker-independent name -> id map. Inverse of regime.py's REGIME_NAMES,
# derived rather than restated so the two cannot drift apart.
REGIME_NAME_TO_ID = {name: rid for rid, name in REGIME_NAMES.items()}

# U15: an unmapped state must be explicit, never a silent implicit 4th regime.
UNKNOWN_REGIME_NAME = "unknown"
UNKNOWN_REGIME_ID = -1


def train_mask(dates) -> np.ndarray:
    """Boolean mask of rows on or before TRAIN_END_DATE.

    market_data.date is 26-char TEXT ('YYYY-MM-DD HH:MM:SS.ffffff'), so the
    comparison is done on normalised Timestamps rather than on raw strings.
    """
    ts = pd.to_datetime(pd.Series(list(dates))).dt.normalize()
    return (ts <= pd.Timestamp(TRAIN_END_DATE)).values


def filtered_posterior(model: GaussianHMM, x: np.ndarray) -> np.ndarray:
    """P(state_t | obs_1..t) for every t — the CAUSAL forward (alpha) recursion.

    Deliberately NOT model.predict() (Viterbi over the whole sequence) and NOT
    model.predict_proba() (smoothed posterior, conditioned on obs_1..T). Both
    would let observations after t determine the label at t.

    hmmlearn 0.3.3 exposes the recursion as _hmmc.forward_log, which returns the
    log alpha lattice; normalising each row gives the filtered distribution.
    Kept in log space until the final exponentiation for numerical stability.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    framelogprob = model._compute_log_likelihood(x)
    framelogprob = np.ascontiguousarray(framelogprob, dtype=np.float64)

    _, fwdlattice = _hmmc.forward_log(
        np.ascontiguousarray(model.startprob_, dtype=np.float64),
        np.ascontiguousarray(model.transmat_, dtype=np.float64),
        framelogprob,
    )
    fwdlattice = np.asarray(fwdlattice, dtype=np.float64)
    return np.exp(fwdlattice - logsumexp(fwdlattice, axis=1, keepdims=True))


class CausalRegimeHMM:
    """Per-ticker HMM fit on the training period only and decoded causally."""

    def __init__(self, n_regimes: int = N_REGIMES, random_state: int = 42):
        if n_regimes != len(REGIME_NAMES):
            raise ValueError(
                f"n_regimes={n_regimes} but REGIME_NAMES declares "
                f"{len(REGIME_NAMES)} names; the state map would be ambiguous."
            )
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.model = None
        self.state_map = {}          # raw hmmlearn state index -> regime name
        self.feature_means_ = None   # train-only standardisation statistics,
        self.feature_stds_ = None    # persisted so serving reuses them exactly
        self.train_end_date = TRAIN_END_DATE
        self.n_train_rows = 0
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "CausalRegimeHMM":
        """Fit on rows dated on or before TRAIN_END_DATE. Nothing later is seen."""
        mask = train_mask(df["date"])
        if mask.sum() < self.n_regimes * 10:
            raise ValueError(
                f"only {int(mask.sum())} training rows on or before "
                f"{TRAIN_END_DATE}; refusing to fit a {self.n_regimes}-state HMM."
            )

        raw = build_hmm_features_raw(df)

        # U4(1): statistics from training rows ONLY, then frozen.
        self.feature_means_, self.feature_stds_ = standardisation_stats(raw[mask])
        train_features = apply_standardisation(
            raw[mask], self.feature_means_, self.feature_stds_
        )

        # U4(2): the fit never sees a post-boundary row.
        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="diag",
            n_iter=HMM_ITERATIONS,
            random_state=self.random_state,
            tol=1e-4,
        )
        self.model.fit(train_features, [len(train_features)])
        self.n_train_rows = int(mask.sum())
        logger.info(
            f"  Causal HMM fit on {self.n_train_rows} train rows "
            f"(<= {TRAIN_END_DATE}); converged={self.model.monitor_.converged}"
        )

        # U4(4): name the states from training data only, using the same
        # ascending-realised-volatility rule the legacy path uses.
        self._build_state_map(df.loc[mask, "close"].values, train_features)
        self._fitted = True
        return self

    def _build_state_map(self, train_close: np.ndarray, train_features: np.ndarray):
        """Rank states by realised volatility over TRAINING rows only.

        Uses the filtered state assignment, not Viterbi, so the map itself is
        causal within the training window too.
        """
        states = filtered_posterior(self.model, train_features).argmax(axis=1)
        close = np.asarray(train_close, dtype=float)
        log_ret = np.concatenate([[0.0], np.log(close[1:] / close[:-1])])

        vol_per_state = {}
        for s in range(self.n_regimes):
            sel = states == s
            vol_per_state[s] = float(np.std(log_ret[sel])) if sel.sum() else 0.0

        # REGIME_NAMES is ordered calm -> turbulent by id, matching the
        # ascending volatility sort. Names are preserved, not renamed.
        names_by_rank = [REGIME_NAMES[i] for i in sorted(REGIME_NAMES)]
        ranked = sorted(vol_per_state.items(), key=lambda kv: kv[1])
        self.state_map = {state: name for (state, _), name in zip(ranked, names_by_rank)}
        logger.info(f"  Causal state map (train-only vol ranking): {self.state_map}")

    def label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add regime_causal / regime_id_causal using filtered decoding.

        Precisely what is guaranteed:
          * No label uses any information dated after TRAIN_END_DATE via the
            model, because the fit and the scaler saw only training rows.
          * No label at row t uses any observation after t via the decoder,
            because the forward recursion is one-directional.

        What is NOT claimed: labels on TRAINING rows are in-sample. Row t inside
        the training window was influenced by the other training rows through
        the fit itself, so perturbing a late training row moves early training
        labels. That is ordinary in-sample fitting, not a validation leak — no
        row after the boundary informs anything. Verified empirically by
        truncation in tests/test_regime_causality.py.
        """
        assert self._fitted, "Call fit() before label()"
        out = df.copy()

        raw = build_hmm_features_raw(df)
        features = apply_standardisation(raw, self.feature_means_, self.feature_stds_)
        states = filtered_posterior(self.model, features).argmax(axis=1)

        names = [self.state_map.get(int(s), UNKNOWN_REGIME_NAME) for s in states]
        out["regime_causal"] = names
        # U1: one fixed name -> id map for every ticker.
        out["regime_id_causal"] = [
            REGIME_NAME_TO_ID.get(n, UNKNOWN_REGIME_ID) for n in names
        ]

        n_unknown = int((out["regime_causal"] == UNKNOWN_REGIME_NAME).sum())
        if n_unknown:
            # U15: surface it loudly rather than emitting a silent 4th regime.
            logger.error(
                f"  {n_unknown} rows decoded to a state absent from the state map; "
                f"labelled '{UNKNOWN_REGIME_NAME}' / {UNKNOWN_REGIME_ID}."
            )
        return out

    def save(self, path: str):
        """Persist model, state map AND the train-only standardisation stats.

        The legacy MarketRegimeHMM pickles no scaler, so a reloaded model
        silently re-standardises against whatever data it is handed. Storing
        them here is what makes serving reproduce training exactly.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"  Causal HMM saved → {path}")

    @staticmethod
    def load(path: str) -> "CausalRegimeHMM":
        with open(path, "rb") as f:
            return pickle.load(f)


def assert_boundary_matches_fusion(db_path: str = DB_PATH) -> dict:
    """Measure fusion.py's effective split boundary and compare to TRAIN_END_DATE.

    Replays fusion.py's logic read-only: load_aligned_data left-joins market_data
    to visual_features and drops rows with no feature_vector, sorts by
    (date, ticker), then cuts at int(len(df) * 0.8).

    Returns a dict describing the comparison. Reports only — never reconciles.
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT m.date, m.ticker FROM market_data m "
            "JOIN visual_features v ON m.date = v.date AND m.ticker = v.ticker "
            "ORDER BY m.date, m.ticker"
        ).fetchall()
    finally:
        con.close()

    cut = int(len(rows) * 0.8)
    fusion_boundary = pd.Timestamp(rows[cut - 1][0]).normalize()
    ours = pd.Timestamp(TRAIN_END_DATE).normalize()

    result = {
        "aligned_rows": len(rows),
        "fusion_split_index": cut,
        "fusion_train_end": str(fusion_boundary.date()),
        "regime_causal_train_end": str(ours.date()),
        "matches": fusion_boundary == ours,
        "gap_days": int((fusion_boundary - ours).days),
    }
    if not result["matches"]:
        logger.error(
            "  ⚠️  SPLIT BOUNDARY MISMATCH — fusion.py trains through "
            f"{result['fusion_train_end']} (row-count fraction "
            f"{cut}/{len(rows)}), regime_causal trains through "
            f"{result['regime_causal_train_end']} (market_data.split). "
            f"Gap: {result['gap_days']} calendar days. Reported, NOT reconciled."
        )
    return result


def save_causal_regimes_to_db(df: pd.DataFrame, db_path: str, ticker: str) -> int:
    """Write ONLY regime_causal / regime_id_causal. Never touches regime/regime_id."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    for ddl in (
        "ALTER TABLE market_data ADD COLUMN regime_causal TEXT",
        "ALTER TABLE market_data ADD COLUMN regime_id_causal INTEGER",
    ):
        try:
            cur.execute(ddl)
        except sqlite3.OperationalError as exc:
            if "duplicate column" not in str(exc).lower():
                raise
            logger.info(f"  Column already present — {ddl.split()[-2]}")
    conn.commit()

    # Same exact-stored-string date resolution the legacy writer uses
    # (regime.py:489-492): the stored form carries a microsecond suffix, so a
    # naive strftime comparison would match zero rows.
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
        params.append(
            (row["regime_causal"], int(row["regime_id_causal"]), stored_date, ticker)
        )

    before = conn.total_changes
    cur.executemany(
        "UPDATE market_data SET regime_causal=?, regime_id_causal=? "
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


def run_causal_regime_pipeline(db_path: str = DB_PATH, save_to_db_flag: bool = True) -> dict:
    """Fit, label and (optionally) persist causal regimes for every ticker."""
    boundary = assert_boundary_matches_fusion(db_path)

    ticker_dfs = load_all_tickers(db_path)
    total_updated, per_ticker = 0, {}

    for tkr, df in ticker_dfs.items():
        logger.info(f"\n── {tkr} ──")
        hmm = CausalRegimeHMM(n_regimes=N_REGIMES)
        hmm.fit(df)
        labelled = hmm.label(df)

        if save_to_db_flag:
            total_updated += save_causal_regimes_to_db(labelled, db_path, tkr)

        hmm.save(str(MODEL_DIR / f"hmm_causal_{tkr}.pkl"))
        per_ticker[tkr] = {
            "state_map": dict(hmm.state_map),
            "n_train_rows": hmm.n_train_rows,
            "n_rows": len(labelled),
        }

    logger.info(f"\n✅ Causal regime labelling complete — {total_updated} rows updated")
    return {"boundary": boundary, "per_ticker": per_ticker, "updated": total_updated}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    parser = argparse.ArgumentParser(description="MMIS T2 — causal regime labelling")
    parser.add_argument("--db", type=str, default=DB_PATH, help="Path to mmis.db")
    parser.add_argument("--all", action="store_true", help="Process all tickers")
    parser.add_argument("--no-save", action="store_true", help="Do not write to the DB")
    parser.add_argument(
        "--check-boundary",
        action="store_true",
        help="Only compare the training boundary against fusion.py's split, then exit",
    )
    args = parser.parse_args()

    if args.check_boundary:
        print(assert_boundary_matches_fusion(args.db))
    elif args.all:
        run_causal_regime_pipeline(db_path=args.db, save_to_db_flag=not args.no_save)
    else:
        print("⚠️  Specify --all or --check-boundary")
