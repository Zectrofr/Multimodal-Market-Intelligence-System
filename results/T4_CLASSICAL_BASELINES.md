# T4 — Classical baselines for the forward volatility target

Measured 2026-08-14 against `data\mmis.db` (9,412 rows, 6 tickers, 2020-03-13 → 2026-06-10).

This task **wrote nothing to the database**. It was opened read-only by URI throughout. Verified:
`data\mmis.db` Length 55,455,744, LastWriteTime `2026-08-13T16:11:07.2512829Z`, SHA-256
`214CC00F…F638C468` — identical before and after. `models\rv_thresholds.json` likewise unchanged
(SHA-256 `98E7CDDD…0A4DFE8C`); the thresholds were **loaded, never recomputed**.

No model was trained other than the four baselines described here. The fusion model was not
touched, and nothing in this document makes any claim about it.

---

## 1. The HAR specification

Corsi (2009), *A Simple Approximate Long-Memory Model of Realized Volatility*, regresses future
realized volatility on realized volatility measured over several backward horizons, mimicking the
heterogeneous horizons over which different market participants operate. It is deliberately simple
and genuinely hard to beat.

### This is an adaptation, not the original specification

Corsi builds a **daily** RV from **intraday** returns — with 5-minute bars a single day holds ~78
returns, so one day's realized volatility is estimable on its own, and the components are 1, 5 and
22 days. **This dataset has daily bars only.** A one-day RV from daily data would be the standard
deviation of a single return: undefined at ddof=1, and identically zero at ddof=0. The daily
component cannot be reproduced here.

What is used instead: backward-looking realized volatility over **5, 22 and 66 trading days**,
each ending at *t* **inclusive** — Corsi's weekly/monthly structure shifted up one horizon to
weekly/monthly/quarterly. That preserves the multi-horizon long-memory approximation that carries
the model. What is lost is the shortest horizon, and with it some responsiveness to very recent
shocks.

Each component uses **exactly the target's formula** — standard deviation of log returns, ddof=1,
annualised by √252 — so predictors and target share a scale and the coefficients are directly
interpretable as volatility-on-volatility loadings.

```
log_ret[i]     = log(close[i] / close[i-1])
har_rv_W[t]    = stdev(log_ret[t-W+1 … t],  ddof=1) × √252      W ∈ {5, 22, 66}
fwd_rv_5d[t]   = stdev(log_ret[t+1  … t+5], ddof=1) × √252      (the T3 target)
```

Each window is reduced independently via a sliding-window view rather than `pandas.rolling()`,
for the bit-locality reason T3 recorded.

### Why predictors include row *t*'s own return and the target does not

The target is **strictly forward** and excludes `log_ret[t]`, because a label must not be readable
off its own feature row. Every HAR component ends at *t* **inclusive** and therefore *includes*
`log_ret[t]`. The asymmetry is correct and is the point of the setup: a predictor is information
available at decision time — standing at the close of day *t*, having observed `close[t]`, you
know `log_ret[t]`. The target is an outcome that has not happened yet. Excluding `log_ret[t]` from
the predictors would discard genuinely available information and handicap the baseline; including
it in the target would be leakage.

### Alignment, verified rather than asserted

There is an exact identity available: backward-5 RV at row *t+5* and forward-5 RV at row *t* are
both `stdev(log_ret[t+1…t+5])×√252`, so they must be equal. Checked against the **stored database
target** across all six tickers — 9,382 comparable rows, **max |difference| = 0.000e+00**, and the
same against `rv_target.forward_realized_vol` directly. Any off-by-one in either direction would
break this identity. It does not break.

---

## 2. The fit

### The embargo, imported not restated

T3 established that "training row" cannot mean `date <= TRAIN_END_DATE`: a row dated 2025-03-11
has a forward window running to 2025-03-18, which is validation data, and T3 measured that pooling
such rows moves a fitted parameter. The rule is therefore **imported** from
`rv_target.threshold_fit_mask`, not reimplemented, so the two cannot drift apart.

Training rows are the intersection of three conditions:

| Condition | Rows |
|---|---|
| `threshold_fit_mask` — entire 5-day forward window on or before 2025-03-11 | 7,500 |
| complete 66-day backward lookback | −396 (66 per ticker × 6) |
| non-NULL target | −0 (all remaining rows have one) |
| **Training rows** | **7,104** (1,184 per ticker, exactly equal across all six) |

**Out-of-sample rows: 1,852** — strictly after 2025-03-11, complete predictors, non-NULL target
(308 AAPL, 308 GOOGL, 309 each for AMZN/MSFT/SPY/TSLA).

**456 rows belong to neither set, and are accounted for rather than dropped silently:**

| Reason | Rows |
|---|---|
| Training-dated, but no complete 66-day lookback (the first 66 bars per ticker) | 396 |
| Training-dated, but embargoed — forward window crosses the boundary | 30 |
| After the boundary, but the target is NULL (final 5 rows per ticker, T3) | 30 |

The middle 30 are worth naming explicitly: they are neither fit nor scored. They cannot join the
fit (the embargo bars them) and they cannot be scored as held-out (they are training-dated). That
is the correct treatment, not an oversight.

### Pooled, not per-ticker

**One** model is fit across all six tickers, matching T3's pooled thresholds. The trade-off is
real: a pooled HAR forces one set of coefficients onto SPY and TSLA alike, whose volatility levels
differ enormously (T3: SPY 5.6% turbulent, TSLA 78.3%). Six per-ticker fits would very likely
score better. That would be a different experiment, and a per-ticker baseline is not the thing a
pooled model must beat. Recorded as an open item rather than quietly chosen.

### Fitted coefficients — HAR-RV (level), the headline

OLS on the level of `fwd_rv_5d`, n = 7,104, dof = 7,100, design rank 4 (full).

| term | coefficient | std. error | t | p |
|---|---|---|---|---|
| intercept | **0.028519999862358796** | 0.0041585567 | 6.8581 | 7.563e-12 |
| `har_rv_5` | **0.06213605643419412** | 0.0141675996 | 4.3858 | 1.172e-05 |
| `har_rv_22` | **0.17268320231298484** | 0.0277459276 | 6.2237 | 5.131e-10 |
| `har_rv_66` | **0.6003418637908713** | 0.0262711432 | 22.8518 | 1.377e-111 |

Residual σ² = 0.02517978641695161. Standard errors computed directly as
σ²·diag((X'X)⁻¹) with σ² = RSS/(n−k), since statsmodels is not installed.

All four terms are significant, all loadings are positive, and **the betas sum to 0.8352 < 1** —
mean reversion, as HAR should show. The quarterly term dominates (0.600), which is the
long-memory signature the model exists to capture.

### Fitted coefficients — HAR-RV (log), secondary

OLS on `log(fwd_rv_5d)` against log predictors.

| term | coefficient | std. error | t | p |
|---|---|---|---|---|
| intercept | **−0.2610225119817661** | 0.0161586467 | −16.1537 | 1.130e-57 |
| `log har_rv_5` | **0.08395773416117025** | 0.0135712284 | 6.1865 | 6.495e-10 |
| `log har_rv_22` | **0.14862999140553163** | 0.0295744165 | 5.0256 | 5.141e-07 |
| `log har_rv_66` | **0.6894906638034249** | 0.0294835856 | 23.3856 | 1.397e-116 |

Residual σ² = 0.23588493790907167.

**A retransformation caveat that matters.** Back-transforming with a plain `exp(X'b)` yields the
conditional **median** of a lognormal, not its mean, so the log model systematically
under-predicts on a squared-error scale. The headline log row below uses the plain `exp`, as
specified. The Duan/smearing correction factor `exp(σ²/2) = 1.1251793767593556` is reported
alongside because omitting it would silently handicap the log specification — and the effect is
large: out-of-sample QLIKE improves from 0.800440 to **0.684845** once corrected, essentially
matching the level fit. Both are given so neither variant is flattered.

Persisted to `models\har_baseline.json`. Serving reads it; it never refits.

---

## 3. Metrics

**QLIKE**, in **variance** space. The forecasts are annualised *volatility*, so variance is its
square: `s2 = actual²`, `s2hat = predicted²`.

```
QLIKE = s2/s2hat − log(s2/s2hat) − 1
```

Zero exactly when `s2hat == s2`, strictly positive otherwise (`x − log x − 1 ≥ 0`, equality only
at `x = 1`). Lower is better. Unlike MSE it is asymmetric, penalising under-prediction of variance
far more heavily, which is why it is the standard volatility loss.

**Non-positive predictions**: none. A floor of 0.010616091969664648 (the smallest training
`fwd_rv_5d`) was available for the `log` in QLIKE, and **it was never used — 0 substitutions
across every model and both periods**. The minimum prediction anywhere was 0.0685 (HAR log, OOS).

**MSE / RMSE / MAE** on the volatility level.

**R²** against the **training** mean (0.29618705284341523), never the evaluation-period mean:
`R2 = 1 − SSE / Σ(actual − train_mean)²`. Scoring out-of-sample against the validation mean would
let the held-out sample supply its own benchmark — a forecaster cannot know that mean in advance.
For in-sample rows the two coincide by construction.

**Macro-F1** and per-class precision/recall, obtained by bucketing each continuous prediction
through the **frozen** thresholds from `models\rv_thresholds.json` (lower 0.18390286612477044,
upper 0.32590977697139434), never refit to the predictions. Macro-F1 rather than accuracy because
the validation period is unbalanced (T3: 38.28 / 35.10 / 26.62), so accuracy rewards leaning on
the majority class while macro-F1 weights all three equally.

---

## 4. Headline — out-of-sample (n = 1,852)

**This column is the result.** In-sample is shown alongside for reference only.

| model | QLIKE ↓ | MSE ↓ | RMSE ↓ | MAE ↓ | R² ↑ | macro-F1 ↑ | accuracy |
|---|---|---|---|---|---|---|---|
| **HAR-RV (level)** | **0.683165** | **0.03183593** | 0.178426 | 0.120536 | 0.278510 | 0.507551 | 0.5038 |
| HAR-RV (log) | 0.800440 | 0.03154396 | **0.177606** | **0.114892** | **0.285126** | **0.532119** | 0.5324 |
| HAR-RV (log, smearing-corrected) | 0.684845 | — | 0.178033 | 0.119750 | 0.281683 | 0.509409 | — |
| naive persistence | 1.834292 | 0.04771337 | 0.218434 | 0.143499 | −0.081317 | 0.515680 | 0.5189 |
| constant median | 1.267400 | 0.04498646 | 0.212100 | 0.137468 | −0.019518 | 0.173195 | 0.3510 |

In-sample (n = 7,104), for reference:

| model | QLIKE | MSE | RMSE | MAE | R² | macro-F1 |
|---|---|---|---|---|---|---|
| HAR-RV (level) | 0.463237 | 0.02516561 | 0.158637 | 0.108646 | 0.425956 | 0.561789 |
| HAR-RV (log) | 0.551833 | 0.02629264 | 0.162150 | 0.106370 | 0.400248 | 0.589362 |
| naive persistence | 1.460276 | 0.04341292 | 0.208358 | 0.140049 | 0.009723 | 0.532634 |
| constant median | 1.290728 | 0.04700532 | 0.216807 | 0.143787 | −0.072222 | 0.168298 |

The in-sample → out-of-sample degradation for HAR (R² 0.426 → 0.279) is ordinary generalisation
loss on a 4-parameter model, not a sign of overfitting.

---

## 5. Per-ticker, out-of-sample

| ticker | n | HAR level QLIKE / RMSE / F1 | HAR log QLIKE / RMSE / F1 | naive QLIKE / RMSE / F1 | constant QLIKE / RMSE / F1 |
|---|---|---|---|---|---|
| AAPL | 308 | 0.8326 / 0.1891 / 0.3428 | 0.9379 / 0.1840 / 0.4215 | 3.0888 / 0.2043 / 0.4181 | 0.9987 / 0.1844 / 0.1789 |
| AMZN | 309 | 0.6639 / 0.1793 / 0.2715 | 0.7809 / 0.1787 / 0.2923 | 1.6386 / 0.2156 / 0.3539 | 0.8488 / 0.1785 / 0.2311 |
| GOOGL | 308 | 0.5382 / 0.1451 / 0.2634 | 0.6325 / 0.1462 / 0.2493 | 1.3167 / 0.1868 / 0.4055 | 0.6471 / 0.1475 / 0.2203 |
| MSFT | 309 | 0.7826 / 0.1506 / 0.3600 | 0.9100 / 0.1494 / 0.3125 | 2.0568 / 0.1785 / 0.4581 | 0.7796 / 0.1513 / 0.1593 |
| SPY | 309 | 0.8852 / 0.1291 / 0.3970 | 1.0593 / 0.1244 / 0.3477 | 1.9826 / 0.1325 / 0.5007 | 1.1320 / 0.1609 / 0.0831 |
| TSLA | 309 | 0.3965 / 0.2503 / 0.2975 | 0.4818 / 0.2535 / 0.3431 | 0.9246 / 0.3371 / 0.3402 | 3.1953 / 0.3652 / 0.1371 |

Per-ticker macro-F1 is computed on ~309 rows over three classes and is correspondingly noisy; the
pooled figure in §4 is the reliable one. The TSLA column shows the pooled-fit trade-off in the
clear: the constant forecast scores QLIKE 3.1953 there, because a single pooled median is badly
wrong for the highest-volatility name in the panel.

## 6. Per true class, out-of-sample

| true class | n | HAR level RMSE / MAE / QLIKE | HAR log | naive | constant |
|---|---|---|---|---|---|
| calm | 709 | 0.1297 / 0.1061 / 0.6820 | 0.1056 / 0.0842 / 0.5498 | 0.1294 / 0.0889 / 0.8710 | 0.1281 / 0.1212 / 0.8382 |
| normal | 650 | 0.1038 / 0.0767 / 0.2327 | 0.0888 / 0.0687 / 0.3164 | 0.1591 / 0.1154 / 2.2973 | 0.0431 / 0.0368 / 0.0622 |
| turbulent | 493 | 0.2850 / 0.1990 / 1.2788 | 0.3034 / 0.2199 / 1.7991 | 0.3490 / 0.2591 / 2.6093 | 0.3781 / 0.2937 / 3.4736 |

Every model is worst on `turbulent`. The constant forecast trivially wins the `normal` row — it
predicts the training median, which sits inside the normal band — and is worst everywhere else.

### Per-class precision / recall / F1, out-of-sample

| model | calm P/R/F1 | normal P/R/F1 | turbulent P/R/F1 |
|---|---|---|---|
| HAR level | 0.7541 / 0.3850 / 0.5098 | 0.4045 / 0.6031 / 0.4842 | 0.5144 / 0.5436 / 0.5286 |
| HAR log | 0.7106 / 0.4640 / 0.5614 | 0.4268 / 0.6723 / 0.5221 | 0.6027 / 0.4462 / 0.5128 |
| naive | 0.6068 / 0.6051 / 0.6059 | 0.4303 / 0.4277 / 0.4290 | 0.5090 / 0.5152 / 0.5121 |
| constant | 0.0000 / 0.0000 / 0.0000 | 0.3510 / 1.0000 / 0.5196 | 0.0000 / 0.0000 / 0.0000 |

## 7. Confusion matrices, out-of-sample

Rows are true classes, columns predicted, order calm / normal / turbulent.

**HAR-RV (level)** — macro-F1 0.5076
```
             calm  normal  turb
calm          273     366    70
normal         75     392   183
turbulent      14     211   268
```
**HAR-RV (log)** — macro-F1 0.5321
```
             calm  normal  turb
calm          329     341    39
normal        107     437   106
turbulent      27     246   220
```
**naive persistence** — macro-F1 0.5157
```
             calm  normal  turb
calm          429     202    78
normal        205     278   167
turbulent      73     166   254
```
**constant median** — macro-F1 0.1732
```
             calm  normal  turb
calm            0     709     0
normal          0     650     0
turbulent       0     493     0
```

Predicted class distribution against the truth (709 / 650 / 493):

| model | calm | normal | turbulent |
|---|---|---|---|
| **truth** | 709 | 650 | 493 |
| HAR level | 362 | **969** | 521 |
| HAR log | 463 | **1,024** | 365 |
| naive | 707 | 646 | 499 |
| constant | 0 | 1,852 | 0 |

---

## 8. What the numbers say

**HAR-RV wins decisively on every continuous loss.** Out-of-sample QLIKE 0.683 against 1.834 for
persistence and 1.267 for a constant; R² 0.279 against −0.081 and −0.020. It comfortably earns its
three parameters on the forecasting problem it was built for. Volatility is persistent, and a
model that reads three horizons of it beats one that reads none.

**Naive persistence is worse than a constant on squared error, and that is not a typo.** Its
out-of-sample R² is −0.081 against the constant's −0.020, and its RMSE is higher (0.2184 vs
0.2121). Backward 5-day RV is an unbiased-ish but very noisy estimate of forward 5-day RV; a
single noisy draw is a worse squared-error forecast than a stable central value, even though it
carries real information. Both are far behind HAR.

**But the ranking substantially reverses once predictions are bucketed into classes, and this is
the most important caveat in this document.** On macro-F1, naive persistence (0.5157) **beats**
HAR-RV level (0.5076), and HAR-RV log (0.5321) leads it only modestly. A model that is better on
every continuous metric by a wide margin is not better at the 3-class problem.

The mechanism is visible in the distribution table in §7. OLS shrinks predictions toward the
conditional mean, so both HAR variants pile predictions into the middle bucket — 969 and 1,024
predicted `normal` against 650 true — while under-predicting the tails. Naive persistence, being
on exactly the same scale as the target and un-shrunk, reproduces the true class distribution
almost precisely (707 / 646 / 499 against 709 / 650 / 493) and so wins back on the tail classes
what it loses on precision. The frozen thresholds were fit on the *target's* distribution, not on
any *forecast's* distribution, and a shrunk forecast passed through them is systematically pulled
to the centre.

This is a property of the evaluation, not a defect in it: a well-calibrated point forecast under
squared error is *supposed* to be shrunk. It does mean the two metric families answer different
questions, and that a model must be compared on both.

**The constant forecast is not competitive**, except in the narrow sense that it beats naive
persistence on MSE and RMSE, and trivially wins the `normal` row of §6 by construction. Its
macro-F1 of 0.173 is the floor: it predicts one class for every row.

---

## 9. What this benchmark does and does not establish

**It establishes** the number a multimodal model must beat: out-of-sample QLIKE **0.683165**, RMSE
**0.178426**, MAE **0.120536**, R² **0.278510**, macro-F1 **0.507551** for HAR-RV (level), on the
**same 1,852 rows**, with the **same frozen thresholds**, and the same imported embargo rule. The
per-row predictions are in `results\T4_BASELINE_PREDICTIONS.csv` so a later comparison can join
against them rather than re-derive them.

**It does not establish** that the target is forecastable in any economically useful sense. There
is **no trading here, no returns, no Sharpe, no drawdown, no transaction costs, and no position
sizing**. An R² of 0.279 on forward realized volatility is a statement about a statistical fit and
nothing more. Volatility is the most predictable object in this dataset — that is precisely why it
was chosen as a target in T3 — and a good QLIKE does not convert into a strategy.

Nor does it say anything about the multimodal model, which has not been trained against this
target. Whether it beats these numbers is unmeasured. A negative finding, rigorously obtained,
would be a legitimate result.

---

## 10. Open items

**1. No Diebold-Mariano test.** Comparing two forecasts' loss differentials for statistical
significance requires two forecasts to compare, and at present only baselines exist. The gaps in
§4 are point estimates with no confidence interval attached; whether HAR's QLIKE advantage over
persistence is statistically distinguishable is untested here. When a second forecaster exists,
DM (with a Newey-West correction for the overlapping 5-day windows, which induce serial
correlation in the loss differential) is the appropriate test.

**2. No GARCH(1,1).** The `arch` package is not installed and this task does not install
packages. GARCH is the other standard volatility baseline and its absence is a real gap in the
comparison set.

**3. Per-ticker HAR not fitted.** §2 explains why the pooled fit is the right benchmark for a
pooled model. Six per-ticker fits would probably score better and would answer a different, also
interesting, question.

**4. Overlapping windows.** Consecutive targets share 4 of their 5 returns, so neither the
residuals nor the loss differentials are independent across adjacent rows. This inflates the
effective sample size behind every standard error and p-value in §2. The coefficients are point
estimates; their inference is optimistic.

**5. The retransformation choice is not neutral.** The log variant is reported with plain `exp`
back-transformation as the headline and the smearing correction alongside, because the two differ
materially on QLIKE (0.800 vs 0.685). Any future comparison must state which it uses.

---

## 11. Fairness challenge

Everything above Â§11 was written by the agent that implemented the baseline. This section records
an **independent adversarial audit** by a verifier that did not write the code, run in parallel
with two other verifiers covering the numerical and integrity checks. Its brief was a single
question: *has this baseline been handicapped, deliberately or accidentally?* A weak baseline
flatters every future model and would silently invalidate the comparison this document exists to
enable â€” and it is the one question the implementer cannot answer about their own work.

**Verdict: FAIRLY IMPLEMENTED, not handicapped.** Quoted: *"I found no handicap â€” not in the
windows, not in the estimator, not in the metric implementation, not in the split."* The auditor
re-implemented predictors, masks, OLS and every metric from scratch without importing
`har_baseline.py`, and reproduced all four models' headline figures to the digit.

**But the audit found that the bar is lower than Â§8 implies, in three independent ways that all
tilt the same direction. Those findings are recorded here rather than absorbed into the headline.**

### 11.1 Every specification tried

Out-of-sample, n = 1,852, identical rows throughout, sorted by QLIKE. Shipped models in bold.

| # | specification | OOS QLIKE â†“ | OOS MSE â†“ | RÂ² â†‘ |
|---|---|---|---|---|
| 1 | \|r_t\| + 5/22 (more-rows, n_train 7,368) | **0.627928** | **0.03048100** | 0.309212 |
| 2 | \|r_t\| + 5/22 (same-rows) | 0.627989 | 0.03067531 | 0.304812 |
| 3 | 5/22 (more-rows, n_train 7,368) | 0.629332 | 0.03059900 | 0.306542 |
| 4 | 2/5/22 (more-rows) | 0.629448 | 0.03060300 | 0.306444 |
| 5 | 5/22 (same-rows) | 0.629499 | 0.03079075 | 0.302196 |
| 6 | 2/5/22 (same-rows) | 0.629579 | 0.03079400 | 0.302127 |
| 7 | LOG 5/22 + smearing | 0.633540 | **0.03041862** | **0.310630** |
| 8 | 22-day RV only (single regressor) | 0.636609 | 0.03117652 | 0.293454 |
| 9 | **EWMA Î»=0.94, unfitted** | **0.651918** | 0.03419930 | 0.224949 |
| 10 | HAR per-ticker 5/22/66 (6 fits) | 0.658630 | 0.03058000 | 0.306968 |
| 11 | HAR pooled slopes + ticker fixed effects | 0.662241 | 0.03039900 | 0.311080 |
| 12 | EWMA Î»=0.94 Ã— 0.9518 (train-calibrated) | 0.676600 | 0.03264705 | 0.260127 |
| 13 | 5/22/66 + \|r_t\| | 0.680340 | 0.03173300 | 0.280837 |
| 14 | *(leaky refit incl. 30 embargoed rows â€” quantified, not recommended)* | 0.681960 | 0.03187051 | â€” |
| 15 | **SHIPPED HAR-RV 5/22/66 (level) â€” the headline** | **0.683165** | **0.03183593** | **0.278510** |
| 16 | **SHIPPED HAR-RV log + smearing** | 0.684845 | 0.03169592 | 0.281683 |
| 17 | 5/22 + EWMA | 0.684959 | 0.03247425 | 0.264043 |
| 18 | LOG 5/10/22/66 + smearing | 0.685012 | 0.03161780 | 0.283453 |
| 19 | 5/10/22/66 | 0.686851 | 0.03181100 | 0.279064 |
| 20 | 5/22/66 + EWMA | 0.689602 | 0.03244557 | 0.264693 |
| 21 | naive backward-22 RV | 0.692203 | 0.03562699 | 0.192594 |
| 22 | naive backward-66 RV | 0.728455 | 0.03799175 | 0.139002 |
| 23 | LOG 5/22 raw (median back-transform) | 0.751235 | 0.03056175 | 0.307386 |
| 24 | 0.5 Ã— (rv_5 + rv_22) | 0.776125 | 0.03583503 | 0.187879 |
| 25 | **SHIPPED HAR-RV log, raw (median)** | 0.800440 | 0.03154396 | 0.285126 |
| 26 | LOG 5/10/22/66 raw | 0.800617 | 0.03147652 | 0.286655 |
| 27 | naive backward-10 RV | 0.870075 | 0.03991229 | 0.095477 |
| 28 | constant = train QLIKE-optimal (0.362720) | 0.924148 | 0.05127727 | âˆ’0.162085 |
| 29 | constant = train mean (0.296187) | 0.973594 | 0.04412523 | 0.000000 |
| 30 | **SHIPPED constant = train median (0.239918)** | 1.267400 | 0.04498646 | âˆ’0.019518 |
| 31 | **SHIPPED naive persistence = backward-5 RV** | 1.834292 | 0.04771337 | âˆ’0.081317 |

### 11.2 Finding 1 â€” an unfitted EWMA beats the fitted HAR on QLIKE

**A RiskMetrics EWMA with Î» = 0.94 and no fitting at all scores OOS QLIKE 0.651918 against the
shipped HAR's 0.683165 â€” 4.6% better** â€” and also beats the smearing-corrected log HAR (0.684845).
It loses clearly on MSE (0.03419930 vs 0.03183593), so this is a disagreement between loss
functions rather than a refutation of HAR.

It belongs here because Â§9 names QLIKE 0.683165 as "the number a multimodal model must beat", and
a two-line recursion with a hardcoded Î» already gets 0.652. **That framing is too generous and is
corrected by this section.** The auditor's words: *"'beat QLIKE 0.683165' is not the demanding
target the write-up implies."*

### 11.3 Finding 2 â€” both crude references are the weakest members of their families

`naive persistence` uses backward-5 RV, which is the **worst of the four persistence horizons**:
backward-22 scores 0.692203 against backward-5's 1.834292, a factor of 2.6. Backward-22 is
essentially tied with the fitted HAR (0.692203 vs 0.683165, a 1.3% gap).

`constant median` (0.239918) is likewise the weakest sensible constant: the train mean scores
0.973594 and the in-sample QLIKE-optimal constant `sqrt(mean(yÂ²))` = 0.362720 scores 0.924148,
against the median's 1.267400.

Both shipped choices are individually defensible â€” backward-5 matches the target's 5-day horizon,
and the median is the natural central value for a right-skewed variable. But Â§8's claim that HAR
"comfortably earns its three parameters" rests on a 2.7Ã— QLIKE ratio against persistence that
shrinks to 1.3% against a fairly chosen persistence horizon. **That claim is overstated and this
section supersedes it.**

Compounding: RÂ² is computed against the *training* mean rather than the OOS mean, which inflates
SST and yields 0.278510 instead of 0.271585 â€” **+0.0069 of free RÂ²**. The Â§3 justification (a
forecaster cannot know the validation mean in advance) is legitimate, but the convention is the
generous one. Three independent conventions all tilt the same way.

### 11.4 Finding 3 â€” shorter specifications beat 5/22/66 out-of-sample, but the shipped choice was correct ex ante

Four alternatives beat the shipped spec OOS, the best (`|r_t| + 5/22`) by 8.1% on QLIKE and 4.3%
on MSE. The common ingredient in every winner is **dropping the 66-day term**. Notably, adding a
1-day proxy buys almost nothing (0.0015 QLIKE), so the daily-component limitation that Â§1
discusses at length is **not** where the difference lies.

**This is not evidence of handicapping, and the audit is explicit about why.** An expanding-window
walk-forward cross-validation run *entirely inside the training period* â€” the model selection an
honest implementer could have performed ex ante â€” endorses the shipped specification:

| spec | mean CV QLIKE (train-only) | mean CV MSE |
|---|---|---|
| **5/22/66 (shipped)** | **0.47056** | **0.023181** |
| 5/22 | 0.52795 | 0.025569 |
| \|r_t\| + 5/22 | 0.52669 | 0.025502 |
| 5/22/66 + EWMA | 0.48328 | 0.023162 |

The shipped spec wins by 12% **in all five folds**. The 5/22 advantage exists only in the
out-of-sample window, is concentrated in the 2025 Q1â€“Q2 volatility spike, and **reverses** in
2025 Q3/Q4 (0.6465 â†’ 0.6648 and 0.5088 â†’ 0.5318). Adopting 5/22 on the strength of the OOS number
would be selecting a specification on the test set â€” precisely the error this project's embargo
machinery exists to prevent. **The baseline is therefore left as shipped**, and the challengers are
recorded as an open item below.

### 11.5 What the audit confirmed

- **Alignment is bit-exact and the check discriminates.** The identity (backward-5 at *t+5* equals
  stored `fwd_rv_5d` at *t*) holds at max |diff| = 0.000e+00 over 9,382 rows. Crucially the auditor
  also ran the off-by-one variants â€” *t+4*, *t+6*, and a window ending at *tâˆ’1* â€” and all three
  fail at max |diff| = 1.504. The passing check is not passing trivially.
- **No lookahead anywhere.** Perturbing `close[500]` moved `rv_5` only on rows 500â€“505, `rv_22` on
  500â€“522, `rv_66` on 500â€“566 â€” every affected row at index â‰¥ 500. No predictor at row *t* is a
  function of any close after *t*. The predictors-include-`log_ret[t]` asymmetry of Â§1 is correct;
  excluding it would have been a genuine handicap and was not done.
- **Numerically healthy.** cond(X) = 22.157, cond(X'X) = 490.95 (far below any degeneracy
  threshold), singular values [96.573, 21.749, 9.214, 4.359], rank 4. VIFs 2.471 / 6.689 / 5.288,
  all under the conventional 10. Collinearity between the overlapping windows is inherent to HAR
  and is **not** materially hurting the fit.
- **Row accounting exact**, including that the 30 embargoed rows are 2025-03-05/06/07/10/11 Ã— 6.
- **The embargo is nearly free.** Refitting with the 30 embargoed rows included (leakage,
  quantified not recommended) moves coefficients only in the 3rdâ€“4th decimal: QLIKE 0.681960 vs
  0.683165, MSE 0.03187051 vs 0.03183593. Correctness cost essentially nothing here.
- **QLIKE floor never used** â€” 0 substitutions across every model and every variant tried.
- **Pooling costs more than the missing daily component**, as Â§2 predicted: per-ticker HAR scores
  0.658630 and ticker-fixed-effects 0.662241, both beating the pooled 0.683165. Disclosed in
  advance at `har_baseline.py:80-92`, and a deliberate experimental choice rather than a handicap.

### 11.6 A documentation error found by the audit

`har_baseline.py:308-312` states that the plain-`exp` back-transform "systematically under-predicts
on a squared-error scale", implying the smearing correction should improve MSE. **Empirically the
opposite holds here**: smearing *raises* MSE (0.03154396 â†’ 0.03169592) while sharply improving
QLIKE (0.800440 â†’ 0.684845). The correction itself is right and the factor
`exp(ÏƒÂ²/2) = 1.1251793767593556` was verified exactly; the stated *reason* is inverted. The
docstring was left unmodified â€” Phase E authorises only this section â€” and the error is recorded
here instead.

On whether the level headline understates the baseline: the log variant wins MSE, RÂ² and MAE while
the level wins QLIKE by 0.0017. The gap is ~0.9% of MSE, which the audit judged **not material**,
and Â§4 already tabulates both. Train-only CV also rates them equivalent (0.46981 log+smear vs
0.47056 level), so headlining the level model cost nothing.

### 11.7 Open items added by this audit

**6. A trivial EWMA(0.94) beats the fitted HAR on QLIKE (0.651918 vs 0.683165).** It should be
added to the standing reference set, and any future model claiming to beat "the classical
baseline" must beat this too, not merely the HAR.

**7. The reference set should be strengthened.** Backward-22 persistence (0.692203) and a
mean-or-QLIKE-optimal constant (0.973594 / 0.924148) are the fair members of their families and
should replace or accompany the currently shipped backward-5 and median.

**8. Shorter HAR specifications beat 5/22/66 out-of-sample but lose on train-only walk-forward
CV.** Left unchanged deliberately, since re-specifying on OOS evidence is test-set selection.
Whether the 66-day term should be dropped is a question for a task that can pre-register the
choice, not for this one.

**9. Per-ticker and ticker-fixed-effect HAR both beat the pooled fit** (0.658630 and 0.662241 vs
0.683165). If a future multimodal model is fit per ticker, the pooled baseline is the wrong
comparator and one of these should be used instead.
