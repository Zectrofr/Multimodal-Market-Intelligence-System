"""
robustness_sweep.py — Is the result signal or noise?

Re-runs the full train -> infer -> eval pipeline across several random seeds and
reports the spread of the headline metrics. If walk-forward precision and the
per-regime Sharpe ranking are stable across seeds, they're real; if they swing,
they're noise and must not be over-claimed.

Seed 42 (the committed canonical) runs LAST so the repo artifacts are restored.
"""
import os, sys, json, subprocess
import pandas as pd

SEEDS = [0, 7, 123, 42]
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
PY = sys.executable


def run(cmd, log, env):
    with open(log, "w", encoding="utf-8") as f:
        subprocess.run(cmd, env=env, check=True, stdout=f, stderr=subprocess.STDOUT)


rows = []
for seed in SEEDS:
    print(f"--- seed {seed}: training ---", flush=True)
    env = dict(ENV, MMIS_SEED=str(seed))
    run([PY, "fusion.py"],    f"results/sweep_fusion_{seed}.log", env)
    run([PY, "inference.py"], f"results/sweep_infer_{seed}.log",  env)
    run([PY, "evaluation.py", "--csv", "results/final_predictions.csv",
         "--uncertainty-filter"], f"results/sweep_eval_{seed}.log", env)

    rep = json.load(open("results/evaluation_report.json"))
    oos = rep["out_of_sample"]
    rows.append({
        "seed":            seed,
        # OUT-OF-SAMPLE (the honest metrics)
        "oos_precUP":      oos["precision_at_up"],
        "oos_base_up":     oos["base_up_rate"],
        "oos_edge_pts":    oos["precision_edge_pts"],
        "oos_sharpe":      oos["sharpe"],
        "oos_bh_sharpe":   oos["buy_hold_sharpe"],
        "oos_ece":         oos["ece"],
        # full-sample (inflated, in-sample) for reference
        "full_precUP":     rep["model"]["precision_at_up"],
        "full_sharpe":     rep["model"]["sharpe"],
    })
    print(f"--- seed {seed} done ---", flush=True)

df = pd.DataFrame(rows)
df.to_csv("results/robustness_sweep.csv", index=False)

print("\n================ ROBUSTNESS SWEEP ================")
print(df.to_string(index=False))

print("\n=== across seeds: mean +/- std  [min, max] ===")
for c in ["oos_precUP", "oos_edge_pts", "oos_sharpe", "oos_bh_sharpe",
          "oos_ece", "full_precUP", "full_sharpe"]:
    print(f"  {c:13s}: {df[c].mean():+.3f} +/- {df[c].std():.3f}   "
          f"[{df[c].min():+.3f}, {df[c].max():+.3f}]")

print(f"\nOUT-OF-SAMPLE precision edge > 0 in {(df['oos_edge_pts']>0).sum()}/{len(df)} seeds")
print(f"OOS beats buy-and-hold Sharpe in {(df['oos_sharpe']>df['oos_bh_sharpe']).sum()}/{len(df)} seeds")
