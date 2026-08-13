# T3 — Forward realized volatility target

Measured 2026-08-13 against `data\mmis.db` (9,412 rows, 6 tickers, 2020-03-13 → 2026-06-10).

`market_data.target` — the 3-class next-day direction label — is **unchanged**, as are `regime`,
`regime_id`, `regime_causal` and `regime_id_causal`. The volatility label was written to three new
shadow columns, `fwd_rv_5d` / `vol_target` / `vol_target_id`, so nothing that currently reads the
old columns is affected. Verified: the SHA-256 of each protected column over all 9,412 rows is
identical before and after the write.

The digest recipe, stated so the check is reproducible rather than merely asserted — rows selected
as `SELECT ticker, date, <col> FROM market_data ORDER BY ticker, date`, then for each row the
UTF-8 bytes of `f"{ticker}|{date}|{value!r}\n"` fed to one running SHA-256:

| column | SHA-256 |
|---|---|
| `target` | `1c14e236acc4b1faae73939a39638d1872736460f65d4fb389ed9a3dc4ef5d95` |
| `regime` | `08e3653587794210a6566271445316e36d0fd80c435cf8aa59258c59283c0578` |
| `regime_id` | `1501a248a1a67db1cbbb321e25e2ababee7f2957eb60cd15efba40f4d6ef19a6` |
| `regime_causal` | `98efd61d95a9edb91821f9def2f33cd2dd5dbacca371367b85a87387cd824e88` |
| `regime_id_causal` | `0057ee2f6f51777c34d42740cb334e0971f8496c1a8e3bed63f7aef61456c232` |

Independently re-verified under a different serialization by a verifier that did not perform the
write, hashing `data\mmis.db.pre-T3` against `data\mmis.db`: all five columns match, row counts
9,412 in both, and the pre-T3 snapshot lacks the three new columns.

**This document contains no performance claim.** It describes a label. No model was trained, no
prediction was regenerated, and no evaluation was run against it. Nothing in this repository
consumes these columns yet.

---

## 1. The label, in full

| Property | Value |
|---|---|
| Horizon | **5 trading days** (rows, not calendar days) |
| Quantity | Realized volatility of daily log returns over the forward window |
| Window | Rows **t+1 … t+5 inclusive** — strictly forward, excludes row *t*'s own return |
| Dispersion | Sample standard deviation, **ddof = 1** |
| Annualisation | × **√252** = 15.874507866387544 |
| Classes | 3 — `calm` / `normal` / `turbulent` |
| Cut | Terciles, 33.33rd and 66.67th percentile |
| Cut fitted on | Pooled across all six tickers, training rows only, then **frozen** |
| Persisted to | `models\rv_thresholds.json` |

Formally, with `close` ordered ascending by date within a ticker:

```
log_ret[i]   = log(close[i] / close[i-1])              # log_ret[0] is undefined
fwd_rv_5d[t] = stdev(log_ret[t+1 … t+5], ddof=1) × √252
```

**ddof = 1** because the five forward returns are a *sample* drawn from that period's return
distribution, not the population of it. ddof = 0 would bias the estimate low by a factor that
depends on window length, which would make the label incomparable against any future change of
horizon.

**Each window is reduced independently**, via a sliding-window view, rather than through
`pandas.rolling().std()`. This was changed during testing, on evidence. `rolling()` accumulates a
running variance left to right, so its output at one index carries float error from every earlier
row: a test that perturbed `close[t-1]` — which changes row *t*'s own return but none of its five
forward returns — moved `fwd_rv[t]` by ~2e-14. That is past→future rounding noise and **not**
lookahead (rolling accumulates forward, so a later mutation can never move an earlier output, and
the thresholds were bit-identical under either implementation). But the claim this label must
support is that `fwd_rv[t]` is a function of `close[t … t+5]` and of nothing else, and "nothing
else" should mean exactly that. Independent windows make it exact and bit-reproducible. The
switch moved the fitted thresholds in their last two significant digits — lower
`0.18390286612477003` → `0.18390286612477044`, upper `0.3259097769713943` → `0.32590977697139434`
— and changed **no label on any of the 9,382 labelled rows**. Measured directly by running both
implementations over the same closes: 9,030 of the 9,382 `fwd_rv_5d` values differ bitwise, with
a maximum absolute difference of 3.4e-14, and 0 rows change class.

### Threshold values, full precision

| Threshold | Value |
|---|---|
| lower (33.33rd pct) | **0.18390286612477044** |
| upper (66.67th pct) | **0.32590977697139434** |
| Pooled fit rows | **7,500** (1,250 per ticker × 6) |
| Training boundary | **2025-03-11**, imported from `regime_causal.TRAIN_END_DATE` |

Percentiles use numpy's default `linear` interpolation between order statistics. Named because
the values are persisted and must reproduce exactly.

### Boundary convention

```
rv <  0.18390286612477044                                  ->  calm       (0)
     0.18390286612477044 <= rv <  0.32590977697139434       ->  normal     (1)
                                  rv >= 0.32590977697139434 ->  turbulent  (2)
```

Each cut is **inclusive on its upper side**: a value exactly equal to `lower` is `normal`, and a
value exactly equal to `upper` is `turbulent`. Stated explicitly because a tercile sits on an
observed order statistic, so exact equality is reachable rather than hypothetical.

### Name → id map

`calm = 0`, `normal = 1`, `turbulent = 2` — one fixed map, identical on every ticker, mirroring
T2's `REGIME_NAME_TO_ID`. `vol_target_id = 2` means `turbulent` everywhere. This is the property
`regime_id` never had (U1: it stored the raw per-ticker HMM state index, honoured on only 70.82%
of rows).

### NULL rows are NULL, not defaulted

The last **5 rows per ticker — 30 in total** — have no complete forward window and carry
`fwd_rv_5d IS NULL`, `vol_target IS NULL`, `vol_target_id IS NULL`. Never 0.0, never a class,
never forward-filled.

This is the direct lesson of **U3** (`docs\STATE_REPORT.md:577`), re-verified for this task:
`ingest.py:135-140` computes `next_return` as a bare Series never assigned into `df`, so on the
final row `close.shift(-1)` is NaN, both `.loc` masks evaluate False, and `target` silently keeps
its `:138` default of `1` (Flat). `dropna()` at `:143` cannot remove the row because no *column*
is NaN there, and the comment at `:142` claiming it does is false. Confirmed against the database:
all six terminal rows carry `target = 1`, against a true Flat base rate of 27.60% — six fabricated
labels sitting at the newest dates, which are the rows live inference touches first.

Exact NULL rows, verified positionally as the final five of each ticker and no others:

| ticker | NULL dates |
|---|---|
| AAPL | 2026-06-03, 06-04, 06-05, 06-08, 06-09 |
| GOOGL | 2026-06-03, 06-04, 06-05, 06-08, 06-09 |
| AMZN, MSFT, SPY, TSLA | 2026-06-04, 06-05, 06-08, 06-09, 06-10 |

AAPL and GOOGL end one session earlier (1,568 rows vs 1,569), which is why their tails differ.

---

## 2. Why the thresholds are pooled, not per-ticker

The terciles are fit on **one pooled distribution across all six tickers**. A single global
definition of "turbulent" makes the class comparable *across* tickers, which is what the
cross-sectional regime analysis this project exists to do requires: "AAPL is turbulent while SPY
is calm" is only a meaningful sentence under a shared scale.

The trade-off is real and is **not** an artefact. Under a pooled scale the classes are wildly
unbalanced per ticker:

| ticker | calm | normal | turbulent |
|---|---|---|---|
| SPY | **74.1688 %** | 20.2046 % | 5.6266 % |
| MSFT | 39.1944 % | 37.9795 % | 22.8261 % |
| AAPL | 36.8522 % | 39.7953 % | 23.3525 % |
| GOOGL | 27.5112 % | 43.7620 % | 28.7268 % |
| AMZN | 22.7621 % | 43.6701 % | 33.5678 % |
| TSLA | 4.7315 % | 16.9437 % | **78.3248 %** |

SPY is turbulent on 5.6% of days and TSLA on 78.3%. That is a property of the market — SPY is a
diversified index, TSLA a single high-beta name — not a defect in the labelling. Per-ticker
terciles would force every ticker to be turbulent exactly one third of the time, destroying
precisely the cross-sectional information the pooled scale preserves.

---

## 3. Why the threshold fit excludes the last 5 training rows

Threshold eligibility is **stricter** than `date <= TRAIN_END_DATE`. A row contributes to the fit
only if its **entire forward window** also lies on or before `TRAIN_END_DATE`. That costs 5 rows
per ticker: 7,530 rows are dated within the training period, 7,500 are eligible to fit.

The reason is a genuine leak, measured rather than assumed. A training row dated 2025-03-11 has a
forward window running into 2025-03-12 … 2025-03-18, which is validation data. Pooling those rows
would let post-boundary prices move a fitted parameter that is then applied to the validation
period.

Counterfactual, computed both ways on identical inputs — every close strictly after 2025-03-11
was mutated (×2, ×10, and replaced outright with a random walk):

| Pool | rows | lower | upper | thresholds move under post-boundary mutation? |
|---|---|---|---|---|
| **Strict (shipped)** | 7,500 | 0.18390286612477044 | 0.32590977697139434 | **No** — bit-identical under all three mutations |
| Naive `date <= TED` (rejected) | 7,530 | 0.18417683753516298 | 0.32652656649114353 | **Yes** — upper moves to 0.327501607568005 under all three |

The naive pool leaks by 0.00097504 in the upper threshold. Small in magnitude, unbounded in
principle: the size of the move is a function of what the market did after the boundary, which is
exactly the quantity that must not be readable.

The excluded rows still **receive** a label. A forward label on a training row is *supposed* to
depend on the following five days — that is what "forward" means. What must not happen is those
days informing a parameter applied to held-out data.

---

## 4. Class distribution

### Overall (9,382 labelled rows; 30 NULL)

| class | count | % |
|---|---|---|
| calm | 3,209 | 34.2038 % |
| normal | 3,164 | 33.7242 % |
| turbulent | 3,009 | 32.0721 % |

### By period

Period is taken by date against `TRAIN_END_DATE = 2025-03-11`, the same boundary the thresholds
were fit on.

| class | train (n=7,530) | % | dev. from 33.3333 | validation (n=1,852) | % | dev. from 33.3333 |
|---|---|---|---|---|---|---|
| calm | 2,500 | 33.2005 % | −0.1328 pts | 709 | 38.2829 % | **+4.9496 pts** |
| normal | 2,514 | 33.3865 % | +0.0531 pts | 650 | 35.0972 % | +1.7639 pts |
| turbulent | 2,516 | 33.4130 % | +0.0797 pts | 493 | 26.6199 % | **−6.7135 pts** |

**Train is near-equal thirds by construction** — maximum deviation 0.13 percentage points. It is
not exactly a third because the thresholds are cut at the 33.33rd/66.67th percentiles of the
7,500-row *fit-eligible* pool, while the train column above counts all 7,530 rows dated in the
training period.

**Validation deviates: 38.28 % calm against 26.62 % turbulent.** This is a property of the market
in that period, not a modelling error. The label is a frozen absolute scale — a threshold fixed in
March 2025 asks "was the following week more volatile than a typical week of 2020-2025?", and the
answer for June 2025 → June 2026 was, on balance, no. A label that re-terciled the validation
period would force 33/33/33 by construction and would thereby be unable to express "this year was
calm", which is information a forecaster should have to earn. The validation forward-RV median is
0.2261 against 0.2438 in training — the whole distribution shifted down.

### Per ticker × period

| ticker | period | calm | normal | turbulent |
|---|---|---|---|---|
| AAPL | train | 429 | 509 | 317 |
| AAPL | validation | 147 | 113 | 48 |
| AMZN | train | 284 | 519 | 452 |
| AMZN | validation | 72 | 164 | 73 |
| GOOGL | train | 354 | 532 | 369 |
| GOOGL | validation | 76 | 152 | 80 |
| MSFT | train | 460 | 497 | 298 |
| MSFT | validation | 153 | 97 | 59 |
| SPY | train | 906 | 272 | 77 |
| SPY | validation | 254 | 44 | 11 |
| TSLA | train | 67 | 185 | 1,003 |
| TSLA | validation | 7 | 80 | 222 |

---

## 5. Forward RV distribution

| statistic | train (n=7,530) | validation (n=1,852) |
|---|---|---|
| min | 0.0106160920 | 0.0121831273 |
| max | 2.3858194104 | 1.9790699114 |
| mean | 0.3045389998 | 0.2757056427 |
| median | 0.2437875025 | 0.2261061090 |

Deciles:

| | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 |
|---|---|---|---|---|---|---|---|---|---|
| train | 0.104566 | 0.141238 | 0.174132 | 0.207296 | 0.243788 | 0.289250 | 0.346480 | 0.427605 | 0.582524 |
| validation | 0.090094 | 0.124622 | 0.159639 | 0.189531 | 0.226106 | 0.269345 | 0.310701 | 0.375152 | 0.516203 |

The validation distribution is lower at every decile. Units are annualised volatility, so a
median of 0.2438 is roughly 24 % annualised.

---

## 6. Class persistence — the sanity check

How often the label on day *t* equals the label on day *t+1*, within a ticker:

| label | agreeing pairs | % |
|---|---|---|
| **`vol_target` (new)** | 7,415 / 9,376 | **79.0849 %** |
| `target` (old directional) | 3,344 / 9,406 | 35.5518 % |
| chance baseline (Σ marginal²) | — | 33.3583 % |

**Volatility clustering is present and large.** The new label persists on 79.08 % of consecutive
day pairs against a 33.36 % chance baseline — 45.7 points above chance. The old directional target
persists on 35.55 %, i.e. 2.2 points above chance, which is approximately the "no structure"
result that motivated replacing it.

This was the designated STOP condition: a persistence figure near chance would have meant the
label was wrong, because volatility clustering is one of the most robust facts in finance. It is
not near chance. Per ticker:

| ticker | persistence |
|---|---|
| SPY | 88.6756 % |
| TSLA | 84.1971 % |
| MSFT | 77.0953 % |
| AAPL | 76.8886 % |
| AMZN | 75.4319 % |
| GOOGL | 72.2151 % |

Note that part of the raw persistence is mechanical: consecutive forward windows overlap in 4 of
their 5 returns, so adjacent labels are not independent draws. That overlap is a property of any
overlapping-window forward label and is not corrected for here. It is stated so the 79 % is not
read as 79 % of *independent* evidence for clustering.

---

## 7. `vol_target` × `regime_causal`

`regime_causal` is a **contemporaneous** HMM state (what the market looks like now, decoded from
observations up to *t*); `vol_target` is a **forward outcome** (what the next five days did).
Related, not identical.

| regime_causal ↓ / vol_target → | calm | normal | turbulent |
|---|---|---|---|
| mean_reverting | 1,837 | 1,494 | 1,050 |
| trending | 1,051 | 1,112 | 1,022 |
| high_vol | 321 | 558 | 937 |

Row percentages:

| regime_causal | calm | normal | turbulent |
|---|---|---|---|
| mean_reverting | 41.9311 % | 34.1018 % | 23.9671 % |
| trending | 32.9984 % | 34.9137 % | 32.0879 % |
| high_vol | 17.6762 % | 30.7269 % | **51.5969 %** |

χ² = 534.8116 on n = 9,382, **Cramér's V = 0.1688**. Taking the obvious 1-1 identification
(mean_reverting↔calm, trending↔normal, high_vol↔turbulent) they agree on 3,886 / 9,382 =
**41.4197 %** of rows.

The association runs in the expected direction and is clearly non-trivial — `high_vol` days are
followed by a turbulent week 51.60 % of the time against a 32.07 % base rate — but V = 0.169 is a
weak-to-moderate association, nowhere near identity. `trending` is nearly uninformative about the
forward week (33.0 / 34.9 / 32.1, almost the marginal distribution).

**High correlation here would be expected and would NOT indicate leakage.** Both quantities are
functions of the same volatility process, so an HMM that reads current volatility well should
correlate with next week's volatility. The causality argument does not rest on the correlation
being small: `regime_causal` is causal by construction (T2 — train-only fit, train-only
standardisation, filtered forward-recursion decoding), and `vol_target` is strictly forward by
construction, using only rows *t+1 … t+5*. Neither can see past its own boundary, whatever their
correlation turns out to be.

---

## 8. Open items for later tasks

None of these was fixed in T3.

**1. `fusion.py:419`'s train/validation boundary is a row-count artefact, and it disagrees with
this label's boundary.** The split is `int(len(df) * 0.8)` over the market⋈visual aligned frame,
made temporal only by the `sort_values(["date","ticker"])` at `fusion.py:260`. Independently
re-derived for this task: the aligned frame is **9,292 rows**, `int(len(df) * 0.8) = 7,433`, and
row 7,433 is **2025-03-17** (ticker TSLA). The cut lands *inside* the 2025-03-17 block, which
occupies indices 7,428–7,433, so five of six tickers for that date land in train and TSLA lands in
validation.

Against `TRAIN_END_DATE = 2025-03-11`: fusion's boundary is **6 calendar days later**, with
**18 aligned rows (3 trading sessions) strictly between** them, and 23 rows between the two cut
indices (7,410 vs 7,433). The direction is the safe one — fusion trains *past* the point where the
volatility thresholds stop learning, so the intervening rows sit in fusion's training set carrying
labels from a scale that never saw them. The hazard is not the current offset but that it
**moves**: the boundary is positional, so re-running `vision.py`, changing chart coverage, adding
a ticker, or ingesting new bars silently shifts what "training data" means, with no error and no
log line. It must be pinned to a stored date before any retrain. Not recorded in
`docs\STATE_REPORT.md`.

**2. Row-0 fabrication in the regime feature path.** `build_hmm_features_raw` emits a synthetic
first observation `(log_ret=0, vol_5=0, vol_20=0, vol_ratio=1)` for every ticker, because the
rolling windows have no history and `np.where` discards the `0/0 = nan` it computes. Measured in
T2: refitting AAPL with row 0 dropped changes **649 / 1,567 labels (41.4 %)** — that row feeds
`startprob_` and the standardisation statistics, so its leverage far exceeds its single-row count.
This lives in the regime path, not the target path, and was deliberately left alone here. The
volatility target has no equivalent defect: row 0 gets a valid forward RV (its window is
`log_ret[1…5]`, all well-defined), and the only NULLs are the 5 rows per ticker at the *end* of
the series.

**3. U3 — the fabricated last row of the directional target — is still live.** It is moot for
`vol_target`, which NULLs those rows explicitly, but `market_data.target` still carries six
default-Flat labels at the six newest dates, and `ingest.py:142`'s comment still claims they were
dropped. Anything that trains or evaluates on `target` continues to inherit them.

**4. Forward labels need an embargo at the training boundary.** The last 5 training rows per
ticker carry labels whose forward windows extend past `TRAIN_END_DATE`. They are excluded from the
*threshold fit* (§3), but they are not excluded from the *label column* — nor should they be. Any
model trained on `vol_target` should embargo the final 5 training rows, or it trains on rows whose
targets peek across the boundary. This module does not and cannot enforce that; it is a property
of the training task.

**5. The 30 NULL rows are the rows live inference cares about most.** They are the most recent
five sessions per ticker. A serving path that expects a non-NULL `vol_target` on the latest bar
will not find one — correctly, because that label does not exist yet. This is a fact about
forward-looking labels, not a defect, but it must be handled explicitly rather than defaulted.
