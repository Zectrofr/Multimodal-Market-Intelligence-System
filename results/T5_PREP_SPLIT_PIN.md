# T5-PREP — Pinning the fusion train/validation split to a date

Measured 2026-08-14 against `data\mmis.db` (9,412 rows, 6 tickers, 2020-03-13 → 2026-06-10;
9,292 rows after the market⋈visual alignment).

**No model was retrained.** `fusion.py`'s training loop was never executed. Nothing in this
document is a performance result — there is not a single accuracy, F1, loss, or return figure
here, because none was produced. This task changes **where the split falls** and nothing else.

The database was opened read-only throughout and is byte-identical: Length 55,455,744,
LastWriteTime `2026-08-13T16:11:07.2512829Z`, SHA-256 `214CC00F…F638C468`.

---

## 1. The defect

`fusion.py:419-420` computed its train/validation boundary as a **fraction of the row count**:

```python
split = int(len(df) * 0.8)
train_df, val_df = df.iloc[:split], df.iloc[split:]
```

over the market⋈visual aligned frame, made temporal only by the `sort_values(["date","ticker"])`
at `fusion.py:260`. **There is no date literal anywhere in that path.** Three consequences, all
measured:

### 1.1 The boundary cut *inside* a single date

`int(9292 * 0.8) = 7433`, and row 7,433 is **2025-03-17, ticker TSLA**. That date carries six
rows, and the cut fell in the middle of them:

| 2025-03-17 | tickers |
|---|---|
| landed in **train** | AAPL, AMZN, GOOGL, MSFT, SPY |
| landed in **validation** | TSLA |

Which side a ticker fell on was decided by **alphabetical order within the day** — an artefact of
the secondary sort key, nothing more. Five tickers' same-day information sat in training while the
sixth was held out.

### 1.2 The boundary moved whenever the row count moved

This is the more dangerous property, and it is easy to under-rate. The frame's length is the
market⋈visual join, so it changes whenever `visual_features` gains or loses rows — a re-run of
`vision.py`, a change in chart coverage, a new ticker, or fresh bars from `ingest.py`.

Measured directly: dropping 400 rows from the validation period (leaving the training period
completely untouched) moves the row-count boundary from **2025-03-17 to 2024-12-26** — nearly
three months earlier — and hands roughly 400 previously-training days to validation. Under the
pinned split the same deletion moves the boundary **not at all**: train stays at 7,404 rows ending
2025-03-10.

A split that relocates by three months in response to a change in *validation-period* data is not
a temporal split; it is a row counter that happens to be sorted by date.

### 1.3 It disagreed with every other boundary in the repository

`regime_causal.TRAIN_END_DATE = "2025-03-11"` governs the causal regime labels (T2), the frozen
volatility thresholds (T3), and the HAR-RV baselines (T4). Fusion's cut sat **6 calendar days
later**, at 2025-03-17, a gap of 23 aligned rows. `regime_causal.py:256-295`
(`assert_boundary_matches_fusion`) already measured this disagreement and logged it as an error,
but deliberately never reconciled it.

This defect is **not** recorded in `docs\STATE_REPORT.md`. It was found during T2 and documented
in the T2 and T3 reports.

---

## 2. The fix

`fusion.py` now calls a module-level `temporal_split(df, train_end_date, embargo_days)`:

```
train      = date <= TRAIN_END_DATE      (minus the embargo, below)
validation = date >  TRAIN_END_DATE
```

The rule operates on the **date**, so all six tickers for any given date land on the same side and
no date can straddle the boundary. `TRAIN_END_DATE` is **imported** from `regime_causal`, not
restated — there is no second date literal to drift. The sort at `fusion.py:260` is unchanged.

The function raises rather than proceeding if `max(train date) >= min(validation date)`, if any
date appears in both frames, or if either side is empty. Those are the two properties the
row-count cut could not guarantee, so they are now enforced at the point of the split and logged
with the boundary date and both row counts.

**One transitive consequence, stated because it is not free:** importing `regime_causal` into
`fusion.py` also pulls in `regime.py` (which runs `MODEL_DIR.mkdir(exist_ok=True)` at import time)
and makes `hmmlearn` a hard dependency of importing `fusion.py`. Both were already installed and
already imported by the test suite. The alternative — hardcoding a duplicate `"2025-03-11"` — is
exactly the drift this change exists to prevent.

---

## 3. Before / after

| | old (row-count) | new (date-pinned, embargo 1d) |
|---|---|---|
| rule | `int(len(df) * 0.8)` | `date <= TRAIN_END_DATE` |
| boundary | index 7,433 → **2025-03-17** | **2025-03-11** (imported) |
| last train date | 2025-03-17 | 2025-03-10 |
| first validation date | 2025-03-17 *(same date!)* | 2025-03-12 |
| train rows | 7,433 | **7,404** |
| validation rows | 1,859 | **1,882** |
| rows on the boundary date | 6, **split 5 / 1 across the cut** | 6, all on one side |
| dates in both splits | 1 | **0** |
| moves if row count changes | **yes** | no |

Without the embargo the date cut alone gives train 7,410 / validation 1,882. Rows changing side
between the old and new schemes: **23**, all in one direction (old-train → new-validation), on
2025-03-12/13/14/17 — AAPL 4, AMZN 4, GOOGL 4, MSFT 4, SPY 4, TSLA 3. No row moves the other way;
the new boundary is strictly earlier.

Validation is **1,882 rows**, comfortably above the 500-row floor this task was required to hold.

## 4. Class balance

`market_data.target` (0 = Down, 1 = Flat, 2 = Up). fusion.py computes an inverse-frequency
class-weighted loss from the **training** split (`fusion.py:466-467`), so a moved boundary moves
the loss surface. The question is whether it moves it materially.

| split | n | 0 (Down) | 1 (Flat) | 2 (Up) |
|---|---|---|---|---|
| OLD train | 7,433 | 2,514 (33.8221%) | 2,001 (26.9205%) | 2,918 (39.2574%) |
| **NEW train** | 7,404 | 2,499 (33.7520%) | 1,999 (26.9989%) | 2,906 (39.2491%) |
| OLD validation | 1,859 | 587 (31.5761%) | 587 (31.5761%) | 685 (36.8478%) |
| **NEW validation** | 1,882 | 601 (31.9341%) | 589 (31.2965%) | 692 (36.7694%) |

**It does not move materially.** The largest training-class shift is 0.08 percentage points, and
the resulting class weights move by under 0.3% relative:

| | Down | Flat | Up |
|---|---|---|---|
| OLD weights | 0.985548 | 1.238214 | 0.849098 |
| NEW weights | 0.987595 | 1.234617 | 0.849277 |

So the pinned boundary buys reproducibility and removes the straddle at essentially no cost in
class balance. Stated because the opposite result would have been a reason to reconsider before
T5 retrains, and it needed checking rather than assuming.

---

## 5. The embargo

`EMBARGO_DAYS` is an explicit, named, configurable module constant. It drops the last N training
**dates** (not rows, so the six tickers stay aligned), because those rows' labels are drawn from
dates that sit in validation.

**The correct value is the forward horizon of the target**, and the target is about to change:

| target | forward horizon | correct EMBARGO_DAYS | train rows | cost |
|---|---|---|---|---|
| `market_data.target` — next-day direction (**current**) | 1 day | **1** ← shipped default | 7,404 | 6 rows |
| `market_data.vol_target` — 5-day forward RV (**T5**) | 5 days | **5** | 7,380 | 30 rows |
| *(no embargo, for reference)* | — | 0 | 7,410 | 0 rows |

The default ships at **1**, which is correct for the target fusion.py trains on *today*. It
closes the one-day overlap recorded in `docs\STATE_REPORT.md` as D1: the last training row's
next-day label is drawn from the first validation date.

**T5 must raise this to 5 when it switches the target to `vol_target`.** Leaving it at 1 would
leak four days of validation outcomes into the training labels. Applying 5 now instead would drop
24 rows for no reason — the current label simply does not reach that far forward — which is why
the parameter is target-dependent rather than fixed at the larger value.

At the shipped default the six rows dated 2025-03-11 are in **neither** split: the embargo bars
them from training and they are not after the boundary, so they cannot be validation. That is the
same treatment T3 and T4 gave their embargoed rows, and it is correct rather than an oversight.

---

## 6. What was deliberately not changed

- The **target** is still `market_data.target`. Switching to `vol_target` is T5's job.
- The **regime column** is still `regime`, not `regime_causal`. That swap is also T5's.
- The model class, training loop, scaler handling, temperature fitting, MLflow logging, column
  lists, and hyperparameters are untouched — verified byte-identical by an independent verifier
  rather than asserted here.
- The sort at `fusion.py:260` is unchanged.

Two pre-existing observations surfaced while mapping the split path, recorded but **not** acted
on because they are out of scope:

1. `MARKET_DIM = 37` (`fusion.py:58`) is stale and unused — the real market width is 34 (31
   numeric columns + 3 regime one-hot).
2. The scaler pickle is written twice (`fusion.py` ~435 and ~545), and only the second write
   carries the `temperature` key. A run that dies mid-training leaves a `feature_scalers.pkl`
   without it.

Three other modules still derive the boundary their own way and now disagree with fusion.py:
`ingest.py:243-245` writes `market_data.split` as a **per-ticker** row-position cut;
`regime_causal.py:275-276` replays `int(len(rows) * 0.8)` to measure fusion's old boundary; and
`inference.py:212` / `evaluation.py:504` approximate it with `df["date"].quantile(0.80)`. Four
definitions of "validation starts here" existed before this change and three still do.

---

## 7. For T5

What remains, in the order it matters:

1. **Switch the target** from `market_data.target` to `market_data.vol_target` (3-class calm /
   normal / turbulent, ids 0/1/2) or to `fwd_rv_5d` for a regression head.
2. **Raise `EMBARGO_DAYS` to 5** in the same commit. These two changes must not be separated —
   the embargo is only correct relative to the target's horizon.
3. **Switch the regime column** from `regime` to `regime_causal`, and `regime_id` to
   `regime_id_causal`. The legacy columns carry the look-ahead bias T2 removed (U4), so any
   regime-conditioned result computed from them is measuring its own future.
4. **Score against T4's baselines on identical rows**, with the frozen thresholds from
   `models\rv_thresholds.json` — never refit to predictions.

On the bar to beat: `results\T4_CLASSICAL_BASELINES.md` headlines HAR-RV at out-of-sample QLIKE
**0.683165**, but its §11 fairness audit found an **unfitted EWMA(λ=0.94) at QLIKE 0.651918** that
beats it. **0.651918 is the real bar**, not 0.683165. T4 §11.7 also notes that per-ticker HAR
(0.658630) and ticker-fixed-effects HAR (0.662241) both beat the pooled fit, so if T5 trains per
ticker, the pooled baseline is the wrong comparator.

Note also that T4's baselines were scored on **1,852** out-of-sample rows defined by a 66-day
lookback plus a 5-day embargo, while this split yields **1,882** validation rows. Those row sets
are not identical and a like-for-like comparison must reconcile them explicitly before any claim
that one model beat another.

---

## 8. Independent verification

Two read-only verifiers that did not write the change checked it in parallel. Both passed. Neither
reported a failure of the split itself; the findings below are about *consumers* that were coupled
to the old rule, which this task is forbidden to modify.

**Correctness verifier.** Rebuilt the aligned frame independently (pandas replay *and* a pure-SQL
cross-check that never imports `fusion`): 9,292 rows both ways. Confirmed the new counts at every
embargo setting â€” train 7,410 / 7,404 / 7,380 at embargo 0 / 1 / 5, validation 1,882 throughout â€”
matching not only row counts but row *indices*. Confirmed 0 dates straddle the boundary (train
1,234 dates, validation 314, intersection empty) and that the old rule genuinely did straddle
2025-03-17 five rows to one. Confirmed the embargo drops whole dates only, that validation is
byte-identical at embargo 0/1/5, and that embargoed rows land in neither split.

On row-count invariance it ran eight perturbations. The new boundary did not move once:

| perturbation | new boundary | old rule's boundary |
|---|---|---|
| drop 1 validation row | 2025-03-10 | 2025-03-17 |
| drop 100 | 2025-03-10 | **2025-02-26** |
| drop 400 | 2025-03-10 | **2024-12-26** |
| drop 800 | 2025-03-10 | **2024-10-10** |
| append 6 rows | 2025-03-10 | 2025-03-18 |
| append 120 | 2025-03-10 | **2025-04-08** |
| append 600 | 2025-03-10 | **2025-07-11** |
| append 1,800 | 2025-03-10 | **2026-03-02** |

The old rule slides up to five months backwards when validation rows are deleted, and up to twelve
months forwards when new bars are appended â€” the append case silently pulling future dates into
*training*. The training row count under the new rule was 7,404 in all eight cases.

One semantic clarification it raised: `embargo_days` counts **trading dates present in the data**,
not calendar days, so an embargo of 5 spans seven calendar days across a weekend. That is the
correct semantics for a horizon measured in trading days, which is what `vol_target` uses.

**Scope verifier.** Confirmed the database unchanged (Length, LastWriteTime and SHA-256 all match;
49 columns, 9,412 rows). Extracted every top-level definition from `HEAD:fusion.py` and the working
copy by AST and compared source text exactly: `CrossModalAttention`, `MultimodalDataset`,
`load_aligned_data` (including the sort), `prepare_features`, `train_epoch`, `eval_epoch`,
`collect_logits`, `fit_temperature`, `set_seed`, `MARKET_COLS`, all twelve `mlflow.` call sites and
every pre-existing module constant are **byte-identical**. Only `EMBARGO_DAYS` and `temporal_split`
were added, and `run_fusion_pipeline` changed by exactly the seven deleted lines and one added
call. `fusion.py` is the only modified tracked file; no pre-existing test was touched.

Its verdict on the challenge: **"the new split is safer overall"**, with class weights moving at
most 0.29% relative, validation at 20.254% of the frame across 314 trading dates, no circular
import risk (the module graph is a DAG), and the added import cost measured at **0.15 s** on top of
fusion's existing ~10 s of torch/mlflow loading.

### 8.1 Downstream consumers this change leaves unreconciled

The verifier's own framing: *"None of these are defects in the diff â€” the diff is clean and tightly
scoped. They are consumers that were coupled to the old rule and were left unreconciled, which is
exactly the class of problem the change was made to prevent."* All four are **out of scope here**
and are listed in Â§7 as T5 work.

1. **`regime_causal.assert_boundary_matches_fusion()` is now a false alarm.** It still replays
   `int(len(rows) * 0.8)` at `regime_causal.py:275`, so it reports `fusion_train_end = 2025-03-17`,
   `matches: False`, and logs a `SPLIT BOUNDARY MISMATCH` error on every main-path run â€” even though
   fusion now matches `TRAIN_END_DATE` exactly. The repo's own drift detector measures a rule
   nothing uses, and no test covers it. Highest-value follow-up, and a one-function change.
2. **`inference.py:212` gains a small leak in the direction this repo cares about.** Its isotonic
   calibrator fits on `date < quantile(0.80)` = `< 2025-03-17`. Under the old split those were all
   training rows; under the new split **18 rows (2025-03-12 â€¦ 03-14) are now fusion's validation
   data**, so validation ECE becomes slightly optimistic. Small â€” 18 of 7,428 fit rows, 0.24% â€” but
   structurally wrong, and it is a *new* consequence of moving the boundary.
3. **`evaluation.py:503-504`** derives `val_start` from `quantile(0.80)` = 2025-03-17 and its
   docstring at line 421 claims this "matches fusion.py's temporal split". That claim is now false.
   The direction is conservative rather than leaky (it excludes 18 genuinely held-out rows and no
   longer includes 5 rows fusion trained on), but it should be pinned to the same constant.
4. **Importing `fusion` now creates `models/` as a side effect**, inherited from `regime.py:67`'s
   module-scope `MODEL_DIR.mkdir(exist_ok=True)`. At HEAD that ran only inside
   `run_fusion_pipeline`. Harmless in the project root, but `inference.py`, `live_inference.py` and
   pytest collection now all pay it, and it creates a stray CWD-relative directory elsewhere. The
   clean fix is to move `TRAIN_END_DATE` into a side-effect-free constants module â€” not to restate
   the literal, which the tests now forbid.

Also noted and left alone: the comment at `fusion.py:270-271` still explains the sort as existing
"so the downstream 80/20 split is a TRUE temporal split". The sort is still required â€” for
`temporal_split`'s ordering contract and for deterministic batching â€” but that rationale is stale.
Rewording it would have meant touching a byte-identical protected region for prose alone.
