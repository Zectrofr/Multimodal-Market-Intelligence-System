# MMIS test suite — characterization, not correctness

## What these tests are

This is a **characterization** suite. Its job is to describe what this codebase
does *today*, exactly as it is, so that any future change produces a visible,
reviewable diff in the test results. It is not a specification of what the code
*should* do, and most of it is not a quality bar.

That distinction matters because MMIS had zero tests before this suite existed.
Every fix in its history — the temporal split, the scaler leak, the regime
`UPDATE` that silently touched zero rows — was made without a safety net, and
the only way to check any of them was to re-run a pipeline that is itself
destructive. `docs/STATE_REPORT.md` §9 lists that as a top risk. These tests are
the net.

The consequence, and it is deliberate: **a number of these tests assert
behaviour that is wrong.** They pin bugs. A test named
`test_walk_forward_folds_are_identical_D7` asserts that the walk-forward
validator emits 24 byte-identical folds, which is a defect (D7), not a feature.
It passes today because that is what the code does.

The alternative — writing `test_walk_forward_advances_train_start`, asserting
the correct behaviour — would produce a suite that is red on its first run and
stays red. A permanently-red suite pins nothing, signals nothing, and is
ignored within a week. A suite that goes red the *moment something changes* is
the only kind that carries information.

## Tests named with a defect ID are expected to go red

Any test whose name ends in a defect ID (`_D7`, `_D8`, `_U1`, `_U7`, `_U11`)
carries a docstring citing that defect in `docs/STATE_REPORT.md` §5, and is
marked `@pytest.mark.defect`. Each one states in its docstring that it is
expected to fail when the defect is fixed.

**When one of these goes red, that is the intended signal, not a regression.**
The correct response is:

1. Confirm the red test is the one you meant to change.
2. Read its docstring to see exactly which broken behaviour it was holding.
3. Update the test to pin the *new* behaviour, and drop the `defect` marker and
   the ID from its name.
4. Update `docs/STATE_REPORT.md` §5 to record the defect as fixed.

What you must never do is weaken or delete a defect test to get back to green
without step 4. The ID in the name exists so that the day it goes red, whoever
sees it can find out in thirty seconds what it was protecting.

Currently pinned defects:

| Defect | Pinned by | What it holds |
|---|---|---|
| D7 | `test_walk_forward_folds_are_identical_D7`, `test_walk_forward_train_start_never_leaves_the_first_date_D7` | The walk-forward loop never advances; all 24 folds are identical |
| D8 | `test_sentiment_covers_1_3_percent_in_a_30_day_window_D8`, `test_sentiment_placeholder_rows_are_only_identifiable_by_headline_count_D8` | Sentiment covers 1.32% of market rows, all in one ~30-day window |
| U1 | `test_regime_and_regime_id_are_mutually_inconsistent_U1`, `test_regime_id_marginals_do_not_reconcile_with_regime_U1` | `regime` and `regime_id` disagree; only 70.82% honour `REGIME_NAMES` |
| U7 | `test_expected_calibration_error_drops_proba_of_exactly_one_U7`, `test_expected_calibration_error_is_biased_low_by_the_dropped_bin_U7` | ECE bins exclude `pred_proba == 1.0` but count it in the denominator |
| U11 | `test_pred_direction_does_not_follow_pred_proba_U11` | `pred_direction` agrees with `pred_proba > 0.5` on only 37.06% of rows |

`tests/test_guards.py` is the exception to everything above. Those tests assert
what the guard *should* do, because `db_guard.py` was written to a specification
rather than inherited. They are correctness tests and are expected to stay green.

## Safety properties

- **Nothing writes to `data/mmis.db`.** The `db` fixture opens it through the
  read-only SQLite URI (`?mode=ro`), so writes are refused by SQLite itself
  rather than merely avoided by convention.
- **No pipeline module is executed.** `ingest`, `sentiment`, `vision`, `regime`,
  `fusion`, `inference` are never imported. Only `evaluation.py` and
  `db_guard.py` are imported; the guard tests inspect `sentiment.py` /
  `vision.py` / `regime.py` by parsing their text with `ast`, precisely so that
  importing `torch` and `transformers` is unnecessary.
- **One import-time side effect, known and unfixed.** `evaluation.py:27` runs
  `warnings.filterwarnings("ignore")` at module scope, so importing it in
  `test_pure_functions.py` silences warnings for the whole pytest session,
  including any `DeprecationWarning` raised in the other test files. Nothing in
  this suite depends on seeing a warning, but do not read a clean run as
  evidence that no warning was raised. Fixing it means editing `evaluation.py`,
  which is out of scope for the task that created this suite.
- **No network calls, no model loading, no unpickling.** `models/*.pkl` and
  `models/*.pt` are checked for existence and byte size only. Unpickling
  executes arbitrary code and a test suite has no business doing it.
- **Guard tests use throwaway databases** under pytest's `tmp_path`. The real
  database is never handed to `assert_safe_to_replace` in a test.
- **`MMIS_ALLOW_DESTRUCTIVE` is cleared** by an autouse fixture and restored by
  `monkeypatch`, so nothing leaks into the surrounding shell.

## Skipping, not failing

`data/`, `results/`, and `models/` are all gitignored, so a fresh clone has none
of them. Tests that need an artifact call `require()` from `conftest.py`, which
issues `pytest.skip` with the missing path. A clone with no artifacts runs the
pure-function and static-guard tests and skips the rest — it does not go red.
A missing baseline is an absent measurement, not a regression.

## Running

```powershell
python -m pytest tests\ -v                 # everything
python -m pytest tests\ -q                 # summary only
python -m pytest tests\ -m defect          # only the pinned-defect tests
python -m pytest tests\ -m "not defect"    # everything except pinned defects
python -m pytest tests\test_guards.py -v   # the guard specification
```

The related static check on the guards, which imports nothing at all:

```powershell
python scripts\verify_guards.py
```

## Layout

| File | Scope | Needs |
|---|---|---|
| `conftest.py` | read-only DB fixtures, artifact skipping, marker registration | — |
| `test_pure_functions.py` | deterministic maths in `evaluation.py`, hand-computed | nothing |
| `test_data_invariants.py` | `data/mmis.db` schema, shapes, spans, NULLs | the database |
| `test_artifact_invariants.py` | `results/final_predictions.csv`, `models/` sizes | the artifacts |
| `test_guards.py` | `db_guard.py` behaviour + static placement checks | nothing |
