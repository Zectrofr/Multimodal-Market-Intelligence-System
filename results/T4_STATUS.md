# T4 — STATUS: INCOMPLETE, UNVERIFIED, UNCOMMITTED

Written 2026-08-14. This is a resumption record, not a result.

**Phases A–D are complete. Phase E (independent verification) has NOT run. Phase F (commit and
push) has NOT run and is gated on Phase E.** Nothing has been staged, committed, or pushed. The
working tree carries five new untracked files and no modified files.

Phase E was attempted: three read-only verifier subagents were launched (numerical, fairness,
integrity). All three terminated on an API monthly-spend limit before producing findings. No
independent verification of any kind has occurred.

The implementer then began self-verifying and was stopped. That was the right call and the reason
is recorded here so a fresh session does not repeat the mistake: **Phase E cannot be
self-administered.** E6 in particular asks whether the implementer handicapped their own baseline,
which the implementer is the one party incapable of answering. Self-verification here would not be
a weaker Phase E, it would be a different and much less useful artefact wearing Phase E's name.

---

## What completed

| Phase | Action | Result |
|---|---|---|
| A1 | Target and threshold provenance | `rv_target.threshold_fit_mask` at `rv_target.py:173`; thresholds bit-exact against `models\rv_thresholds.json`; 7,500 embargo-eligible rows (1,250/ticker) |
| A2 | Library survey | numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, scipy 1.15.3, Python 3.10.11. `statsmodels` and `arch` confirmed ABSENT. `np.linalg.lstsq` selected |
| B1 | HAR predictors | Backward RV over 5/22/66 days ending at t inclusive, independent windows |
| B2 | Train-only OLS | Pooled fit, level and log variants, persisted |
| B3 | Naive persistence | Backward 5-day RV, no fitting |
| B4 | Metrics | QLIKE (variance space), MSE/RMSE/MAE, R² vs training mean, macro-F1, 3×3 confusion |
| C1–C3 | Evaluation and report | In-sample and OOS scored separately; report and per-row CSV written |
| D1–D3 | Tests | 19 new tests; full suite **161 passed**; `verify_guards.py` **66/66, exit 0** |
| **E** | **Independent verification** | **NOT RUN — see PENDING below** |
| **F** | **Commit and push** | **NOT RUN — gated on E** |

## Row accounting

| Quantity | Rows |
|---|---|
| `threshold_fit_mask` embargo-eligible | 7,500 |
| less incomplete 66-day lookback (66 × 6 tickers) | −396 |
| **Training rows** | **7,104** (1,184 per ticker, equal across all six) |
| **Out-of-sample rows** | **1,852** (308 AAPL, 308 GOOGL, 309 each AMZN/MSFT/SPY/TSLA) |
| In neither set | 456 = 396 (no lookback) + 30 (embargoed) + 30 (NULL target) |

## Fitted coefficients — HAR-RV (level), n = 7,104, dof = 7,100, rank 4

| term | coefficient | std. error | t | p |
|---|---|---|---|---|
| intercept | `0.028519999862358796` | 0.0041585567 | 6.8581 | 7.563e-12 |
| `har_rv_5` | `0.06213605643419412` | 0.0141675996 | 4.3858 | 1.172e-05 |
| `har_rv_22` | `0.17268320231298484` | 0.0277459276 | 6.2237 | 5.131e-10 |
| `har_rv_66` | `0.6003418637908713` | 0.0262711432 | 22.8518 | 1.377e-111 |

Residual σ² = `0.02517978641695161`. Betas sum to 0.8351611225380502 (< 1, mean-reverting).

## Fitted coefficients — HAR-RV (log)

| term | coefficient |
|---|---|
| intercept | `-0.2610225119817661` |
| `log har_rv_5` | `0.08395773416117025` |
| `log har_rv_22` | `0.14862999140553163` |
| `log har_rv_66` | `0.6894906638034249` |

Residual σ² = `0.23588493790907167`; smearing factor `exp(σ²/2)` = `1.1251793767593556`.

Training mean `fwd_rv_5d` = `0.29618705284341523`; training median = `0.23991842875127906`.

## Headline — out-of-sample (n = 1,852), UNVERIFIED

| model | QLIKE ↓ | MSE ↓ | RMSE ↓ | MAE ↓ | R² ↑ | macro-F1 ↑ |
|---|---|---|---|---|---|---|
| HAR-RV (level) | 0.683165 | 0.03183593 | 0.178426 | 0.120536 | 0.278510 | 0.507551 |
| HAR-RV (log) | 0.800440 | 0.03154396 | 0.177606 | 0.114892 | 0.285126 | 0.532119 |
| HAR-RV (log, smearing-corrected) | 0.684845 | — | 0.178033 | 0.119750 | 0.281683 | 0.509409 |
| naive persistence | 1.834292 | 0.04771337 | 0.218434 | 0.143499 | −0.081317 | 0.515680 |
| constant median | 1.267400 | 0.04498646 | 0.212100 | 0.137468 | −0.019518 | 0.173195 |

In-sample (n = 7,104): HAR level QLIKE 0.463237, RMSE 0.158637, R² 0.425956, macro-F1 0.561789.

Every figure above is an implementer-produced number that no second party has checked.

---

## IMPLEMENTER SELF-CHECKS — not verification

These were run by the same agent that wrote the code. They are recorded because they are useful
evidence, and labelled because they are **not** a substitute for Phase E.

**1. Predictor alignment identity — came out bit-exact.** Backward-5 RV at row *t+5* and forward-5
RV at row *t* are the same quantity (both `stdev(log_ret[t+1…t+5], ddof=1) × √252`), so they must
be equal. Checked against the **stored database target** across all six tickers, 9,382 comparable
rows: **max |difference| = 0.000e+00**. Same result against `rv_target.forward_realized_vol`
directly. This is the check that would catch an off-by-one in either direction, which is the
error that would most damage the baseline's fairness. It is still a self-check.

**2. Database unchanged.** Fingerprint captured before any T4 work and re-checked after:

```
data/mmis.db
  Length:        55455744
  LastWriteTime: 2026-08-13T16:11:07.2512829Z
  SHA256:        214CC00FA48C3322A2191B15D4CFD84D603486E075982C095308B356F638C468

models/rv_thresholds.json
  SHA256:        98E7CDDD372906362E9A0DC80AC7D812A457F40265E36923CBFDE39B0A4DFE8C
```

All three DB attributes and the thresholds hash were identical after the run. The database was
opened read-only by URI throughout (`har_baseline.load_panel`, deliberately not
`regime.load_all_tickers`, which opens read-write).

**3. Lookback structure.** Each window leaves exactly `window` leading NaNs per ticker (5, 22, 66),
no interior gaps, no zero-fill. Training row arithmetic 7,500 − 396 = 7,104 confirmed empirically.

**4. Suite.** `python -m pytest tests\ -q` → **161 passed** (142 pre-existing + 19 new), 37
pre-existing warnings. `python scripts\verify_guards.py` → **66/66, exit 0**. No existing test was
modified; no existing test failed.

**5. Non-positive predictions.** Zero across all four models and both periods; the QLIKE floor
(0.010616091969664648) was available but never applied. Minimum prediction anywhere 0.0685.

---

## PENDING — Phase E, none of which has run

Verbatim from the task specification. A fresh session must spawn read-only verifier subagents that
did NOT perform the edits.

> **E1.** CENTRAL CHECK: independently prove the fitted coefficients cannot be influenced by
> validation-period data. Mutate post-boundary closes by random walk and by rescale, refit, compare
> coefficients bitwise. Report method and result. If they move, the task has FAILED.
>
> **E2.** Independently recompute HAR predictors and the OLS fit on at least 3 randomly chosen rows
> without using har_baseline.py, and confirm agreement to floating-point tolerance.
>
> **E3.** Independently recompute QLIKE on a sample of out-of-sample rows and confirm the reported
> headline figure.
>
> **E4.** Confirm data\mmis.db is unchanged: Length, LastWriteTime, and SHA-256 against
> data\mmis.db.pre-T3 for the pre-T3 columns and against the current file otherwise. This task must
> have written nothing.
>
> **E5.** Confirm no test was weakened: search for pytest.skip, xfail, loosened tolerances, and
> <=/>= replacing ==.
>
> **E6.** Challenge the baseline's fairness. Specifically check whether HAR-RV has been handicapped
> in any way — wrong lookbacks, dropped rows that should be included, a fit that silently failed to
> converge, predictors misaligned by one row. A weak baseline flatters a future model and is the
> most damaging possible error in this task. Report your reasoning, not just a conclusion.
>
> **E7.** git diff --stat shows only intended files; report the full diff.
>
> **E8.** Re-run the full suite independently.

**E6 is the one that cannot be delegated back to the implementer.** Suggested attacks for whoever
runs it, since the fairness question is empirical and not answerable by reading the code: refit
alternative specifications (5/22 only; a 1-day `|log_ret|` proxy alongside 5/22; 2/5/22) and
compare out-of-sample QLIKE/RMSE/R² against the shipped 5/22/66; test whether a stronger trivial
forecast (backward-22 RV, or an EWMA/RiskMetrics λ=0.94 variance forecast) beats the shipped HAR;
check the condition number of X'X and VIFs across the three overlapping windows; and judge whether
the plain-`exp` back-transform chosen as the headline for the log variant handicaps it, given the
smearing-corrected QLIKE is 0.684845 against the plain 0.800440.

---

## Files written by T4

All five are **new and untracked**. No existing file was modified.

| Path | Status |
|---|---|
| `har_baseline.py` | new, untracked |
| `models\har_baseline.json` | new, untracked |
| `tests\test_har_baseline.py` | new, untracked |
| `results\T4_CLASSICAL_BASELINES.md` | new, untracked (under gitignored `results/`) |
| `results\T4_BASELINE_PREDICTIONS.csv` | new, untracked (8,956 rows = 7,104 train + 1,852 validation) |
| `results\T4_STATUS.md` | this file |

## Phase F, when it eventually runs

Three commits, staged by explicit path, only after Phase E passes:

1. `har_baseline.py`, `models\har_baseline.json` — "feat: HAR-RV and naive persistence baselines for forward volatility"
2. `tests\test_har_baseline.py` — "test: prove HAR coefficients cannot see validation data"
3. `results\T4_CLASSICAL_BASELINES.md`, `results\T4_BASELINE_PREDICTIONS.csv` (needs `git add -f`) — "docs: record classical baseline scores the multimodal model must beat"

Then `git push origin main`. HEAD is currently `b63b5fe`, level with `origin/main`.
