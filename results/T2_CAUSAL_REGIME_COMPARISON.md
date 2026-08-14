# T2 — Causal regime labelling: old vs new

Measured 2026-08-13 against `data\mmis.db` (9,412 rows, 6 tickers, 2020-03-13 → 2026-06-10).

The legacy columns `regime` / `regime_id` are **unchanged**. The causal labels were written to
two new shadow columns, `regime_causal` / `regime_id_causal`, so nothing that currently reads
the old columns is affected. Verified: the SHA-256 of `(ticker, date, regime, regime_id)` over
all 9,412 rows is `d7776f9a…48cffc6` both before and after the write.

**This document contains no performance claim.** It measures how much the labels moved, not
whether anything got better. See "What this does and does not mean".

---

## 1. What changed in the labelling

| Leak (docs/STATE_REPORT.md §5, U4) | Legacy path | Causal path |
|---|---|---|
| U4(1) standardisation | full-sample mean/std, recomputed per call, never stored (`regime.py:105-108`) | statistics fitted on training rows only, frozen, persisted with the model |
| U4(2) HMM fit | entire per-ticker history (`regime.py:601-602`) | rows dated ≤ `TRAIN_END_DATE` only |
| U4(3) decoding | Viterbi over the whole sequence (`regime.py:205`, `.predict()`) | filtered forward recursion, `P(state_t \| obs_1..t)` |
| U4(4) state naming | realised volatility over the full sample (`regime.py:185`) | realised volatility over training rows only |
| U1 id/name coherence | `regime_id` = raw hmmlearn state index, re-permuted per ticker | fixed `REGIME_NAME_TO_ID`, identical on every ticker |

`TRAIN_END_DATE = 2025-03-11`, taken from `market_data.split`. **1,255 training rows for every
one of the six tickers** — the boundary is applied as a date, so AAPL and GOOGL get 1,255 too
even though only 1,254 of their rows carry `split='train'` (see Findings for T3, item 2, for
why those two differ).

### Scope of the causality guarantee

Two things are guaranteed: no label uses information dated after `TRAIN_END_DATE`, because the
fit and the scaler saw only training rows; and no label at row *t* uses an observation after
*t*, because the forward recursion is one-directional.

One thing is **not** claimed: labels on training rows are in-sample. A training row was
influenced by the other training rows through the fit, so perturbing a late training row does
move earlier training labels. That is ordinary in-sample fitting, not a validation leak — the
property that matters, that nothing after the boundary informs anything, holds absolutely.
Independent verification measured exactly this: mutating training rows 1200–1255 moved 685/1000
early AAPL labels, while the same mutation applied *after* the boundary moved 0/1000.

### The seam with fusion.py, and why it is safe

`fusion.py:419` splits at `int(len(df) * 0.8)` over the market⋈visual aligned frame, which
lands on **2025-03-17** — six calendar days after the HMM's training boundary of **2025-03-11**.
The two do not match, and this was reported rather than reconciled.

**This is not leakage, and the direction is what makes it safe.** The HMM stops learning at
2025-03-11 while fusion keeps training to 2025-03-17. The rows in between, 2025-03-12 to
2025-03-17, sit in fusion's *training* set carrying regime labels produced by a model that
never saw those dates — a forward prediction. No information from those rows, or from any
later row, reached the model that labelled them.

The leaky configuration is the mirror image: an HMM fit *past* fusion's boundary, which would
push future knowledge into rows fusion holds out for validation. That is what the legacy path
did, fitting through 2026-06-10 — the entire series — and it is precisely what this change
removes. Choosing 2025-03-11 is the strictly safe side of the seam, not a tolerated compromise.

---

## 2. Distributions — before and after

### Overall

| regime | before (count) | before % | after (count) | after % |
|---|---|---|---|---|
| mean_reverting | 3,834 | 40.7352 | 4,390 | 46.6426 |
| trending | 3,683 | 39.1309 | 3,203 | 34.0310 |
| high_vol | 1,895 | 20.1339 | 1,819 | 19.3264 |

| regime_id | before | before % | after | after % |
|---|---|---|---|---|
| 0 | 3,754 | 39.8853 | 4,390 | 46.6426 |
| 1 | 3,149 | 33.4573 | 3,203 | 34.0310 |
| 2 | 2,509 | 26.6575 | 1,819 | 19.3264 |

After the change the `regime_id_causal` marginals are identical to the `regime_causal`
marginals by construction. Before, they were not — that discrepancy *was* U1.

### Per ticker — `regime` (before)

| ticker | mean_reverting | trending | high_vol |
|---|---|---|---|
| AAPL | 583 | 574 | 411 |
| AMZN | 561 | 587 | 421 |
| GOOGL | 583 | 594 | 391 |
| MSFT | 747 | 613 | 209 |
| SPY | 758 | 667 | 144 |
| TSLA | 602 | 648 | 319 |

### Per ticker — `regime_causal` (after)

| ticker | mean_reverting | trending | high_vol |
|---|---|---|---|
| AAPL | 544 | 535 | 489 |
| AMZN | 756 | 425 | 388 |
| GOOGL | 743 | 525 | 300 |
| MSFT | 859 | 554 | 156 |
| SPY | 777 | 579 | 213 |
| TSLA | 711 | 585 | 273 |

---

## 3. id ↔ name coherence (U1)

### Before — the permutation differed per ticker

| ticker | mean_reverting → | trending → | high_vol → | vs contract |
|---|---|---|---|---|
| AAPL | 0 | 1 | 2 | identity |
| AMZN | 0 | 1 | 2 | identity |
| GOOGL | **1** | **0** | 2 | 0↔1 swap |
| MSFT | 0 | 1 | 2 | identity |
| SPY | **2** | **0** | **1** | 3-cycle |
| TSLA | 0 | 1 | 2 | identity |

Contract compliance: **6,666 / 9,412 = 70.8245 %**. All 2,746 violations came from GOOGL
(1,177) and SPY (1,569 — SPY honoured none).

### After — one mapping everywhere

| ticker | mean_reverting → | trending → | high_vol → |
|---|---|---|---|
| AAPL, AMZN, GOOGL, MSFT, SPY, TSLA | 0 | 1 | 2 |

Contract compliance: **9,412 / 9,412 = 100.0000 %**.

The underlying raw hmmlearn state indices still differ per ticker. Read directly from the
persisted `models\hmm_causal_*.pkl`:

| raw state map | tickers |
|---|---|
| `{2: mean_reverting, 0: trending, 1: high_vol}` | AMZN, GOOGL, MSFT |
| `{0: mean_reverting, 1: trending, 2: high_vol}` | AAPL, SPY, TSLA |

That permutation is unavoidable — it is set by EM initialisation. What changed is that the raw
index is no longer written to the database. The name is resolved per ticker, then mapped
through one fixed `REGIME_NAME_TO_ID`, so `regime_id_causal = 2` means `high_vol` everywhere.

---

## 4. Row-level agreement: `regime_causal` == `regime`

| Scope | Agreeing | Total | Agreement |
|---|---|---|---|
| **Overall** | 6,887 | 9,412 | **73.1725 %** |
| Train period (date ≤ 2025-03-11) | 5,491 | 7,530 | 72.9216 % |
| Validation period (date > 2025-03-11) | 1,396 | 1,882 | 74.1764 % |

Using `market_data.split`'s own labels instead of the date boundary gives 72.9277 % (train,
5,490/7,528) and 74.1507 % (validation, 1,397/1,884) — a two-row difference, from the six rows
dated exactly 2025-03-11.

Per ticker:

| ticker | overall | train | validation |
|---|---|---|---|
| AAPL | 77.9337 % | 77.3705 % | 80.1917 % |
| AMZN | 39.4519 % | 39.8406 % | 37.8981 % |
| GOOGL | 56.6964 % | 57.2112 % | 54.6326 % |
| MSFT | 87.6354 % | 86.3745 % | 92.6752 % |
| SPY | 88.7827 % | 88.7649 % | 88.8535 % |
| TSLA | 88.5277 % | 87.9681 % | 90.7643 % |

**Suspicion check: not triggered.** The threshold agreed in advance was that overall agreement
above 95 % would be treated as suspicious — it would suggest the causal path had quietly
reproduced the acausal one and that the leak had not actually been removed. Observed agreement
is 73.17 %, so roughly one row in four carries a different regime label than before. That is
the artefact being removed, and it is large.

Per-ticker agreement varies widely (AMZN 39 %, SPY 89 %). Nothing here explains that spread,
and no explanation should be inferred from these numbers alone.

---

## 5. Label stability (day-over-day switching)

| ticker | before | after | comparable pairs |
|---|---|---|---|
| AAPL | 75.4308 % | 67.1985 % | 1,567 |
| AMZN | 10.9056 % | 9.8852 % | 1,568 |
| GOOGL | 77.6005 % | 15.7626 % | 1,567 |
| MSFT | 10.2041 % | 11.4158 % | 1,568 |
| SPY | 4.4643 % | 4.4643 % | 1,568 |
| TSLA | 11.0969 % | 12.8189 % | 1,568 |
| **Pooled** | **31.6075 %** (2,973) | **20.2530 %** (1,905) | 9,406 |

**This contradicts the prior expectation and is reported as measured.** A filtered estimate
lacks the smoothing that a Viterbi path applies, so more switching was expected after the
change, not less. Pooled switching *fell* by 11.35 points.

The direction is not uniform. MSFT (+1.21) and TSLA (+1.72) moved as predicted; SPY is
unchanged to four decimal places; AAPL (−8.23) and especially GOOGL (−61.84) dominate the
pooled figure. The two tickers that fell hardest are the two whose legacy labels switched on
roughly three days in four — a rate that is difficult to interpret as regime structure at all.

What produced that is not established here. Both the fit window and the standardisation changed
alongside the decoder, so the pooled movement cannot be attributed to filtering alone, and this
document does not attempt to. It is recorded as an open observation, not a result.

---

## 6. What this change means, and what it does not

It means the regime label attached to any given day is now computed from that day and the days
before it, and from a model fitted only on data through 2025-03-11. Previously every label —
including labels on rows used to validate the model — was computed with knowledge of the full
series through 2026-06-10, via four independent mechanisms. Separately, `regime_id_causal` now
denotes the same regime on every ticker, which `regime_id` did not.

It does **not** mean the model is better, that any metric improved, or that any earlier number
is now correct. No model was retrained, no prediction was regenerated, and no evaluation was
run. Nothing in this repository consumes the shadow columns yet.

What it does mean for earlier work is narrower and worth stating plainly: any prior finding
conditioned on regime — per-regime performance breakdowns, regime-conditioned analysis, the
regime one-hot occupying 3 of 34 market input dimensions in `fusion.py` — rested on labels that
encoded their own future. Those findings were not measuring what they appeared to measure. This
change removes that artefact from the shadow columns. It does not retroactively repair any
number that was computed from the old ones, and every published figure in this repository was
computed from the old ones.

---

## Findings for T3

Documentation only. None of these was fixed in T2.

**1. `fusion.py:419`'s train/validation boundary is a row-count artefact, not a date.**
The split is `int(len(df) * 0.8)` over the market⋈visual aligned frame, made temporal only by
the `sort_values(["date","ticker"])` at `fusion.py:260`. It currently lands on **2025-03-17**
(index 7,433 of 9,292 aligned rows). Because the row count is the join of `market_data` with
`visual_features`, the boundary **moves whenever `visual_features` gains or loses rows** — a
re-run of `vision.py`, or a change in chart coverage, silently shifts what "training data"
means, with no error and no log line. This is not recorded in `docs/STATE_REPORT.md`. It must
be pinned to a stored date before any retrain, otherwise two runs of `fusion.py` against
different vision coverage are not comparable and neither is reproducible.

**2. `market_data.split` is a per-ticker 80/20 label, not a global date cut.**
`ingest.py:243-245` computes it per ticker by row position, so the boundary differs by ticker:

| ticker | last `train` date | first `validation` date |
|---|---|---|
| AAPL | 2025-03-10 | 2025-03-11 |
| GOOGL | 2025-03-10 | 2025-03-11 |
| AMZN | 2025-03-11 | 2025-03-12 |
| MSFT | 2025-03-11 | 2025-03-12 |
| SPY | 2025-03-11 | 2025-03-12 |
| TSLA | 2025-03-11 | 2025-03-12 |

2025-03-11 therefore carries 4 `train` rows (AMZN, MSFT, SPY, TSLA) and 2 `validation` rows
(AAPL, GOOGL). T2's `TRAIN_END_DATE = 2025-03-11` is inclusive, so it admits those 2 AAPL/GOOGL
rows that `ingest.py` labelled validation — a 2-row, single-day imprecision inherited from the
data, noted for completeness.

**3. `regime.py:161` and the class docstring at `:131-132` are false.**
Both claim HMM states are named "via emission means". They are not: `_build_state_map` ranks
states by `np.std` of realised log returns per state at `regime.py:185`. `self.model.means_`
and `.covars_` are never consulted anywhere in the file. Additionally `ret_per_state`, computed
at `:178` and `:186`, is never read — dead code. The comment misleads anyone reasoning about
why a state got its name.

**4. The 34 `RuntimeWarning: invalid value encountered in divide` are pre-existing.**
Source is the `vol_ratio` line in `build_hmm_features_raw` (formerly `build_hmm_features`):
`np.where(vol_20 > 1e-8, vol_5 / vol_20, 1.0)`. NumPy evaluates both branches before selecting,
so the `vol_5 / vol_20` division is computed for every row including row 0, where
`vol_20[0] == 0.0` (a rolling std with `min_periods=1` over a single sample is NaN, zeroed by
`.fillna(0)`). The division yields `0/0 = nan` and emits the warning; `np.where` then discards
it and **`vol_ratio[0]` takes the value `1.0`**.

Measured on a 6-row probe: row 0 raw features are `[0.0, 0.0, 0.0, 1.0]`.

**Row 0 does reach the HMM** — `build_hmm_features_raw` returns exactly `len(df)` rows and
nothing downstream drops the first one. So every ticker's first observation is the synthetic
tuple `(log_ret=0, vol_5=0, vol_20=0, vol_ratio=1)` rather than a real measurement.

**Its influence is not small, and an earlier draft of this document wrongly said it was.**
Measured: refitting AAPL with row 0 dropped changes **649 / 1,567 labels (41.4 %)**. One row in
1,568 is not one part in 1,568 of the outcome — that row participates in the `startprob_`
estimate and in the standardisation statistics, so its leverage far exceeds its count. The
warning is a true signal about a fabricated first observation, not noise to be silenced. T2 did
not introduce it and did not change it.

**5. `converged=True` in the fit log is not evidence of convergence.**
`regime_causal.py` logs `model.monitor_.converged`, mirroring `regime.py:159`. hmmlearn reports
`converged` as True when the iteration cap is reached, not only when the tolerance is met. **AAPL
hits exactly that**: `iter = 200 / 200` against `n_iter=200`, yet the log says converged. The
other five genuinely converged (AMZN 75, GOOGL 64, MSFT 80, SPY 106, TSLA 112 iterations). The
log line is therefore reassuring in a case where it should not be. Pre-existing hmmlearn
behaviour, newly surfaced by T2's log line; not fixed here.

## Deferred: Option B � expanding-window HMM refit

T2 shipped Option A: one HMM per ticker fit on the training period only, applied forward with filtered decoding. Option B � periodic refit (quarterly) on all data up to each point, still filtered � was considered and deliberately deferred, not rejected.

Why A now: it isolates the causality fix so any change in results is attributable to removing leakage rather than to a new fitting regime. B also requires a stable state-matching procedure across every refit, since hmmlearn's raw state indices are re-permuted by EM initialisation � getting that wrong reintroduces U1 in a form that varies across time rather than across tickers, which is harder to detect.

Why B eventually: a regime model fit only on 2020-2025 and applied through 2026+ never sees post-boundary market structure. B mirrors how the system would actually run live. Both should be compared in the ablation harness � 'we tested both and refitting did not change the finding' is a stronger claim than silently choosing one.
