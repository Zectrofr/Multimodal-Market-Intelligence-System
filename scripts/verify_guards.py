"""Static verification that the fail-closed guards are present and correctly placed.

Imports NOTHING from the pipeline modules and executes no pipeline code. Every check is
either a parse of the file text or an inspection of the current environment.

Usage (from project root):  python scripts\\verify_guards.py
Exit code 0 = all checks pass, 1 = at least one failure.
"""

import ast
import os
import re
import sys

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def line_of(text, pattern, flags=0):
    """1-indexed line number of the first regex match, or None."""
    m = re.search(pattern, text, flags)
    if not m:
        return None
    return text[: m.start()].count("\n") + 1


def verify_replace_guard(path, table):
    """A destructive to_sql(if_exists='replace') must be preceded by its guard."""
    text = read(path)
    label = os.path.basename(path)

    try:
        ast.parse(text)
        check("%s parses as valid Python" % label, True)
    except SyntaxError as e:
        check("%s parses as valid Python" % label, False, str(e))
        return

    write_ln = line_of(text, r'to_sql\(\s*["\']%s["\'].*?if_exists\s*=\s*["\']replace["\']' % table, re.S)
    check("%s: destructive to_sql('%s', if_exists='replace') located" % (label, table),
          write_ln is not None, "line %s" % write_ln)
    if write_ln is None:
        return

    count_ln = line_of(text, r'SELECT COUNT\(\*\) FROM %s' % table)
    ro_ln = line_of(text, r'file:data/mmis\.db\?mode=ro')
    env_ln = line_of(text, r'MMIS_ALLOW_DESTRUCTIVE')
    raise_ln = line_of(text, r'raise RuntimeError')

    check("%s: guard counts existing %s rows" % (label, table),
          count_ln is not None, "line %s" % count_ln)
    check("%s: guard opens the DB READ-ONLY (mode=ro)" % label,
          ro_ln is not None, "line %s" % ro_ln)
    check("%s: guard consults MMIS_ALLOW_DESTRUCTIVE" % label,
          env_ln is not None, "line %s" % env_ln)
    check("%s: guard raises RuntimeError" % label,
          raise_ln is not None, "line %s" % raise_ln)

    for nm, ln in (("row count", count_ln), ("read-only open", ro_ln),
                   ("env-var check", env_ln), ("RuntimeError", raise_ln)):
        if ln is not None:
            check("%s: %s PRECEDES the destructive write" % (label, nm),
                  ln < write_ln, "guard line %s < write line %s" % (ln, write_ln))

    # The override must be an exact "1", not merely "set".
    check("%s: override requires MMIS_ALLOW_DESTRUCTIVE == '1'" % label,
          re.search(r'MMIS_ALLOW_DESTRUCTIVE["\']\s*\)\s*!=\s*["\']1["\']', text) is not None)


# ── D1 / D2 ─────────────────────────────────────────────────────────
verify_replace_guard("sentiment.py", "sentiment_data")
verify_replace_guard("vision.py", "visual_features")

# ── D3: regime.py dummy-prediction defusal ──────────────────────────
rg = read("regime.py")
try:
    ast.parse(rg)
    check("regime.py parses as valid Python", True)
except SyntaxError as e:
    check("regime.py parses as valid Python", False, str(e))

check("regime.py: --demo-dummy flag registered in argparse",
      re.search(r'add_argument\(\s*["\']--demo-dummy["\']', rg) is not None)
check("regime.py: --demo-dummy defaults to False (store_true)",
      re.search(r'add_argument\(\s*["\']--demo-dummy["\'][^)]*store_true', rg, re.S) is not None)
check("regime.py: run_regime_pipeline accepts demo_dummy parameter",
      re.search(r'demo_dummy:\s*bool\s*=\s*False', rg) is not None)
check("regime.py: flag is plumbed through to the pipeline call",
      len(re.findall(r'demo_dummy\s*=\s*args\.demo_dummy', rg)) >= 1,
      "%d call site(s)" % len(re.findall(r'demo_dummy\s*=\s*args\.demo_dummy', rg)))

gate_ln = line_of(rg, r'^\s*if demo_dummy:\s*$', re.M)
dummy_ln = line_of(rg, r'def dummy_model')
check("regime.py: dummy_model is gated behind 'if demo_dummy:'",
      gate_ln is not None and dummy_ln is not None and gate_ln < dummy_ln,
      "gate line %s < dummy_model line %s" % (gate_ln, dummy_ln))

csv_gate_ln = line_of(rg, r'if demo_dummy and export_csv and all_tagged:')
csv_write_ln = line_of(rg, r'eval_df\.to_csv\(')
check("regime.py: CSV export is gated behind demo_dummy",
      csv_gate_ln is not None and csv_write_ln is not None and csv_gate_ln < csv_write_ln,
      "gate line %s < write line %s" % (csv_gate_ln, csv_write_ln))

synth_ln = line_of(rg, r'SYNTHETIC_DO_NOT_EVALUATE')
check("regime.py: output filename carries SYNTHETIC_DO_NOT_EVALUATE",
      synth_ln is not None and csv_write_ln is not None and synth_ln < csv_write_ln,
      "filename line %s precedes write line %s" % (synth_ln, csv_write_ln))
check("regime.py: IS_SYNTHETIC column inserted as first column",
      re.search(r'insert\(\s*0\s*,\s*["\']IS_SYNTHETIC["\']\s*,\s*1\s*\)', rg) is not None)

check("regime.py: old 'regime_tagged_predictions.csv' write path removed",
      "regime_tagged_predictions.csv" not in rg)
check("regime.py: 'Plug this directly into evaluation.py' instruction removed",
      "Plug this directly into evaluation.py" not in rg)
check("regime.py: no 'evaluation.py --csv' invocation advertised",
      not re.search(r'python evaluation\.py --csv', rg))
check("regime.py: replaced with an explicit synthetic warning",
      re.search(r'logger\.warning\(', rg) is not None
      and "NOT by any trained model" in rg)

# regions that must NOT have been touched
check("regime.py: HMM regime UPDATE still intact (untouched)",
      "UPDATE market_data SET regime=?, regime_id=? WHERE date=? AND ticker=?" in rg)
check("regime.py: _build_state_map still intact (untouched)",
      "def _build_state_map" in rg)

# ── Environment ─────────────────────────────────────────────────────
env_val = os.environ.get("MMIS_ALLOW_DESTRUCTIVE")
check("environment: MMIS_ALLOW_DESTRUCTIVE is NOT set (guards are armed)",
      env_val is None,
      "value=%r" % env_val if env_val is not None else "unset")

# ── Report ──────────────────────────────────────────────────────────
width = max(len(n) for n, _, _ in CHECKS)
failed = 0
print("=" * (width + 22))
print("MMIS fail-closed guard verification (static; no pipeline code executed)")
print("=" * (width + 22))
for name, ok, detail in CHECKS:
    status = "PASS" if ok else "FAIL"
    if not ok:
        failed += 1
    print("[%s] %-*s %s" % (status, width, name, detail))
print("-" * (width + 22))
print("%d checks, %d passed, %d failed" % (len(CHECKS), len(CHECKS) - failed, failed))

if failed:
    print("\nRESULT: FAIL")
    sys.exit(1)
print("\nRESULT: ALL GUARDS PRESENT AND ARMED")
sys.exit(0)
