# DO NOT USE — FABRICATED DATA

**The two CSV files in this directory are not experimental results. They contain predictions
invented by a numpy placeholder. No trained model produced them.**

They were moved here from `results\` on 2026-08-12 because they are dangerously easy to mistake
for real output. Nothing has been deleted — both files are byte-identical to what was on disk,
and both are also preserved in the verified baseline backup under `backups\`.

## What is in here

| File | Bytes | Origin |
|---|---|---|
| `regime_tagged_predictions.csv` | 1,231,157 | `dummy_model` placeholder, `regime.py:612-626` |
| `demo_regime_output.csv` | 92,600 | synthetic demo path, `regime.py:783-784` |

## Why these are fake

`regime.py:612-626` defines a function called `dummy_model`. Its own docstring says so:

> "Placeholder model: replace with real `model.predict_proba()` in Phase 4+.
> For now simulates a slight directional edge using log returns."

It computes probabilities with a hand-written formula — `up = clip(0.35 + ret * 5, 0.05, 0.85)`
and its mirror for `down` — and nothing else. There is no neural network, no checkpoint, no
learned parameter of any kind in that path. Those numbers are then pushed through a numpy
imitation of MC-Dropout (`regime.py:392`, which applies `np.random.binomial` masks) to
manufacture an "uncertainty" column, and written to CSV at `regime.py:662-677`.

`demo_regime_output.csv` is the same machinery run over a synthetic ticker literally named
`DEMO` on fabricated dates (2021-01-01 to 2023-04-20).

## Why this was dangerous

`regime_tagged_predictions.csv` shares **8 of its 10 column names** with the genuine
`results\final_predictions.csv` — `date`, `ticker`, `close`, `regime`, `uncertainty`,
`pred_direction`, `high_uncertainty`, `actual_return`, `pred_proba`. It carried **no marker
of any kind** indicating it was synthetic. Loading the wrong one is a single-character mistake.

They are not even consistent with the real file. Merged on `(date, ticker)`, `close`,
`actual_return`, and `regime` agree 100%, but **`pred_direction` agrees on only 44.16% of rows**
— so the fabricated file disagrees with the real model on the majority of trades while looking
identical at a glance.

Worse, `regime.py:676-677` used to **print the exact command to evaluate this file**:

```
Plug this directly into evaluation.py:
python evaluation.py --csv results/regime_tagged_predictions.csv --uncertainty-filter
```

Following that instruction produces a complete, professional-looking performance report —
Sharpe ratio, precision, drawdown, calibration — describing a model that does not exist. Those
two logger lines have since been replaced with a warning, and the generating block is now gated
behind an explicit `--demo-dummy` flag.

## Rules

1. **Never** pass either file to `evaluation.py`.
2. **Never** quote, chart, or report any number derived from them.
3. **Do not delete them.** They are kept for provenance — so that if any past result traces back
   to this data, that link stays visible.
4. If you need real predictions, use `results\final_predictions.csv`, and read
   `results\README_PROVENANCE.md` first for its caveats.

## Source references

- `regime.py:612-626` — the `dummy_model` placeholder function
- `regime.py:392` — `mc_dropout_numpy`, the fake uncertainty engine
- `regime.py:662-677` — the CSV write and the former "plug this into evaluation.py" instruction
- `docs\STATE_REPORT.md` §5 (defect D3) — full audit finding
