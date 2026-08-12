# What is real in `results\`

Index of this directory's contents by trustworthiness. All facts below are taken from
`docs\STATE_REPORT.md` §6 — nothing here was re-derived or re-measured.

**Read this before quoting any number from this directory.**

## Load-bearing and current (4 files)

These four are newer than the code that produced them, and all came from the **seed-42 leg of
the robustness sweep** on 2026-06-15 20:13–20:16.

| File | Modified | Note |
|---|---|---|
| `final_predictions.csv` | 2026-06-15 20:16:49 | 9,286 rows. The real predictions. |
| `evaluation_report.json` | 2026-06-15 20:16:51 | See the inconsistency warning below. |
| `robustness_sweep.csv` | 2026-06-15 20:16:51 | 4 seeds. The most honest artifact here. |
| `eval_final.log` | 2026-06-15 20:11:39 | Newer than `evaluation.py`. |

## Quarantined — fabricated (2 files)

Moved to `results\QUARANTINE_SYNTHETIC\`. Output of a numpy placeholder, not a model. See
`QUARANTINE_SYNTHETIC\README_DO_NOT_USE.md`.

- `regime_tagged_predictions.csv`
- `demo_regime_output.csv`

## Stale (17 files)

Older than the code that produced them. **Their numbers do not describe the current model.**

- `regime_tagged_predictions.csv` — newer than `regime.py` (+62s) but older than `fusion.py`,
  `inference.py`, and `evaluation.py`. (Also fabricated; now quarantined.)
- `demo_regime_output.csv` — 3 days older than `regime.py`; abandoned. (Also quarantined.)
- `fusion_final.log`, `fusion_v3.log`, `fusion_retrain.log`, `fusion_retrain_v2.log`
- `inference_final.log`, `inference_fixed.log`, `inference_v3.log`, `inference_v2.log`,
  `inference_new.log`
- `eval_fixed.log`, `eval_v3.log`, `eval_new.log`
- and the remaining older logs

## The log naming carries NO ordering guarantee

The generations `*_new` → `*_v2` → `*_v3` → `*_fixed` → `*_final` → `sweep_*` look
chronological. **They are not.** Three files named `*_final.log` predate their own producing code:

- `fusion_final.log` (2026-06-14 21:08) predates `fusion.py` (2026-06-15 00:12)
- `inference_final.log` (2026-06-15 20:11:37) predates the run that actually wrote
  `final_predictions.csv` (20:16:49)
- `eval_fixed.log` (2026-06-15 15:51:58) predates `evaluation.py` (15:56:54)

**Sort by mtime. Never trust the filename.**

Their headline numbers diverge wildly and all remain on disk: `eval_new.log` total return
−99.35% / ECE 0.11746 · `eval_v3.log` −11.22% / ECE 0.18139 · `eval_fixed.log` +466.94% /
ECE 0.19512 · `eval_final.log` +466.94% / ECE 0.00403.

## Warning: `evaluation_report.json` is internally inconsistent

Its headline block mixes filtered and unfiltered quantities:

- line 21 `n_trades: 3283` — from the **80%**-coverage uncertainty filter
- line 18 `precision_at_up: 0.5787` — the **100%**-coverage value
- line 14 `sharpe: 1.896` — matches **neither** (80% → 1.9383, 100% → 1.9765)

Which coverage the advertised Sharpe belongs to could not be determined without a re-run.

## Warning: the README's headline numbers cannot be traced to one run

Seed 42 is the **best of the four swept seeds** on both `full_sharpe` and `oos_sharpe`, and it
is the seed the README headlines. `robustness_sweep.csv` itself reports OOS precision edge > 0
in **0/4 seeds** and OOS beating buy-and-hold Sharpe in **0/4 seeds**. Full-sample Sharpe ranges
1.312 → 1.896 across seeds (mean 1.611); the advertised "1.90" is the top of that range.

Separately, the 24-fold walk-forward block is **one in-sample month repeated 24 times**
(`evaluation.py:333` never advances `train_start`), so `mean_sharpe 2.9229` and
`mean_precision_at_up 0.6267` are not meaningful.

Also note `pred_direction` (which drives every return metric) agrees with `pred_proba > 0.5`
(which the ECE measures) on only **37.06%** of rows — the calibration number does not describe
the traded signal.

## Nothing here is version-controlled

`.gitignore` excludes `results/`. A verified, hash-checked copy of this entire directory as it
stood on 2026-08-12 lives under `backups\baseline-*\results\`. That backup is the only other copy.
