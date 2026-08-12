"""Static verification that the fail-closed guards are present, armed, and EARLY.

Imports NOTHING from the pipeline modules and executes no pipeline code. Every check is
either a parse of the file text, an AST inspection, or a read of the current environment.

Usage (from project root):  python scripts\\verify_guards.py
Exit code 0 = all checks pass, 1 = at least one failure.
"""

import ast
import os
import re
import sys

CHECKS = []

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))


def read(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        return fh.read()


def line_of(text, pattern, flags=0):
    """1-indexed line number of the first regex match, or None."""
    m = re.search(pattern, text, flags)
    if not m:
        return None
    return text[: m.start()].count("\n") + 1


def func_node(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def call_lines(fnode, callee):
    """Line numbers of every Call to `callee` inside fnode (by name or attribute)."""
    out = []
    for node in ast.walk(fnode):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        nm = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
        if nm == callee:
            out.append(node.lineno)
    return sorted(out)


def first_call(fnode, callee):
    ls = call_lines(fnode, callee)
    return ls[0] if ls else None


# ── The shared helper itself ────────────────────────────────────────
dg = read("db_guard.py")
try:
    dg_tree = ast.parse(dg)
    check("db_guard.py parses as valid Python", True)
except SyntaxError as e:
    check("db_guard.py parses as valid Python", False, str(e))
    dg_tree = None

if dg_tree is not None:
    fn = func_node(dg_tree, "assert_safe_to_replace")
    check("db_guard.py: exposes assert_safe_to_replace", fn is not None)

    check("db_guard.py: PROJECT_ROOT derived from __file__, not the CWD",
          re.search(r'PROJECT_ROOT\s*=\s*Path\(__file__\)\.resolve\(\)', dg) is not None)
    check("db_guard.py: opens the DB READ-ONLY (mode=ro)",
          "?mode=ro" in dg)
    check("db_guard.py: override requires MMIS_ALLOW_DESTRUCTIVE == '1'",
          re.search(r'OVERRIDE_ENV\s*=\s*["\']MMIS_ALLOW_DESTRUCTIVE["\']', dg) is not None
          and re.search(r'OVERRIDE_VALUE\s*=\s*["\']1["\']', dg) is not None
          and re.search(r'os\.environ\.get\(OVERRIDE_ENV\)\s*==\s*OVERRIDE_VALUE', dg) is not None)

    if fn is not None:
        # THE central property: every exception handler must terminate in a raise.
        # An except clause that assigns a permissive default (the old `_existing = 0`)
        # is precisely the fail-OPEN bug this rewrite removes.
        handlers = [h for h in ast.walk(fn) if isinstance(h, ast.ExceptHandler)]
        check("db_guard.py: assert_safe_to_replace has at least one except handler",
              len(handlers) > 0, "%d handler(s)" % len(handlers))
        bad = [h.lineno for h in handlers
               if not any(isinstance(n, ast.Raise) for n in ast.walk(h))]
        check("db_guard.py: EVERY except handler raises (no fail-open default)",
              not bad, "non-raising handlers at lines %s" % bad if bad else "all handlers raise")

        assigns_in_handlers = [
            n.lineno for h in handlers for n in ast.walk(h) if isinstance(n, ast.Assign)
        ]
        check("db_guard.py: no except handler assigns a fallback row count",
              not assigns_in_handlers,
              "assignments at %s" % assigns_in_handlers if assigns_in_handlers else "none")

        raises = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Raise)]
        check("db_guard.py: raises on missing DB, on query failure, and on populated table",
              len(raises) >= 3, "%d raise statements" % len(raises))

        check("db_guard.py: a missing database file raises rather than counting as empty",
              re.search(r'if not resolved\.is_file\(\):\s*\n\s*raise RuntimeError', dg) is not None)


# ── Pipeline guards: present, wired to the shared helper, and EARLY ──
def verify_pipeline_guard(path, func, table, expensive):
    """The guard call must precede BOTH the expensive work and the destructive write."""
    text = read(path)
    label = os.path.basename(path)

    try:
        tree = ast.parse(text)
        check("%s parses as valid Python" % label, True)
    except SyntaxError as e:
        check("%s parses as valid Python" % label, False, str(e))
        return

    check("%s: imports assert_safe_to_replace from db_guard" % label,
          re.search(r'from db_guard import assert_safe_to_replace', text) is not None)

    check("%s: old inline fail-open guard removed" % label,
          "except _sqlite3.Error" not in text and "_existing = 0" not in text)

    fnode = func_node(tree, func)
    check("%s: %s() found" % (label, func), fnode is not None)
    if fnode is None:
        return

    guard_ln = first_call(fnode, "assert_safe_to_replace")
    check("%s: %s() calls assert_safe_to_replace" % (label, func),
          guard_ln is not None, "line %s" % guard_ln)

    tosql_ln = first_call(fnode, "to_sql")
    check("%s: destructive to_sql located" % label, tosql_ln is not None, "line %s" % tosql_ln)
    check("%s: to_sql targets '%s' with if_exists='replace'" % (label, table),
          re.search(r'to_sql\(\s*["\']%s["\'].*?if_exists\s*=\s*["\']replace["\']' % table,
                    text, re.S) is not None)

    if guard_ln is None:
        return

    if tosql_ln is not None:
        check("%s: guard PRECEDES the destructive write" % label,
              guard_ln < tosql_ln, "guard line %s < write line %s" % (guard_ln, tosql_ln))

    # The point of the move: refuse BEFORE the money is spent, not after.
    for callee in expensive:
        exp_ln = first_call(fnode, callee)
        check("%s: %s() is expensive work located in %s()" % (label, callee, func),
              exp_ln is not None, "line %s" % exp_ln)
        if exp_ln is not None:
            check("%s: guard PRECEDES expensive work %s()" % (label, callee),
                  guard_ln < exp_ln, "guard line %s < %s line %s" % (guard_ln, callee, exp_ln))

    # And it must be the very first thing the function does.
    body_lines = [n.lineno for n in fnode.body]
    check("%s: guard is the FIRST statement of %s()" % (label, func),
          guard_ln == min(body_lines),
          "guard line %s, first statement line %s" % (guard_ln, min(body_lines)))


verify_pipeline_guard("sentiment.py", "run_sentiment_pipeline", "sentiment_data",
                      ["load_finbert", "NewsApiClient", "fetch_headlines", "get_sentiment"])
verify_pipeline_guard("vision.py", "run_vision_pipeline", "visual_features",
                      ["load_efficientnet", "generate_chart", "extract_features"])


# ── regime.py: dummy_model defusal (D3) ─────────────────────────────
rg = read("regime.py")
try:
    rg_tree = ast.parse(rg)
    check("regime.py parses as valid Python", True)
except SyntaxError as e:
    check("regime.py parses as valid Python", False, str(e))
    rg_tree = None

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

synth_ln = line_of(rg, r'SYNTHETIC_DO_NOT_EVALUATE_regime_predictions')
check("regime.py: dummy_model filename carries SYNTHETIC_DO_NOT_EVALUATE",
      synth_ln is not None and csv_write_ln is not None and synth_ln < csv_write_ln,
      "filename line %s precedes write line %s" % (synth_ln, csv_write_ln))
check("regime.py: dummy_model export inserts IS_SYNTHETIC as first column",
      re.search(r'eval_df\.insert\(\s*0\s*,\s*["\']IS_SYNTHETIC["\']\s*,\s*1\s*\)', rg) is not None)

check("regime.py: old 'regime_tagged_predictions.csv' write path removed",
      "regime_tagged_predictions.csv" not in rg)
check("regime.py: 'Plug this directly into evaluation.py' instruction removed",
      "Plug this directly into evaluation.py" not in rg)
check("regime.py: no 'evaluation.py --csv' invocation advertised",
      not re.search(r'python evaluation\.py --csv', rg))


# ── regime.py: run_demo defusal (DEFERRED-3) ────────────────────────
if rg_tree is not None:
    demo_fn = func_node(rg_tree, "run_demo")
    check("regime.py: run_demo() found", demo_fn is not None)
    if demo_fn is not None:
        demo_write_ln = first_call(demo_fn, "to_csv")
        check("regime.py: run_demo() CSV write located", demo_write_ln is not None,
              "line %s" % demo_write_ln)

        demo_name_ln = line_of(rg, r'SYNTHETIC_DO_NOT_EVALUATE_demo_regime_output\.csv')
        check("regime.py: run_demo() output filename carries SYNTHETIC_DO_NOT_EVALUATE",
              demo_name_ln is not None and demo_write_ln is not None
              and demo_name_ln < demo_write_ln,
              "filename line %s precedes write line %s" % (demo_name_ln, demo_write_ln))

        insert_ln = line_of(rg, r'df_labelled\.insert\(\s*0\s*,\s*["\']IS_SYNTHETIC["\']\s*,\s*1\s*\)')
        check("regime.py: run_demo() inserts IS_SYNTHETIC as the FIRST column",
              insert_ln is not None)
        check("regime.py: IS_SYNTHETIC is inserted BEFORE the CSV is written",
              insert_ln is not None and demo_write_ln is not None and insert_ln < demo_write_ln,
              "insert line %s < write line %s" % (insert_ln, demo_write_ln))

        warn_lns = call_lines(demo_fn, "warning")
        check("regime.py: run_demo() announces its output via logger.warning",
              len(warn_lns) > 0, "line(s) %s" % warn_lns)
        check("regime.py: run_demo() states the output is fabricated and must not be evaluated",
              "FABRICATED" in rg and "Never pass it to evaluation.py" in rg)
        check("regime.py: old bare 'demo_regime_output.csv' write path removed",
              not re.search(r'["\']results/demo_regime_output\.csv["\']', rg))

check("regime.py: run_demo is reachable only via the --demo flag",
      re.search(r'add_argument\(\s*["\']--demo["\']\s*,\s*action\s*=\s*["\']store_true["\']', rg)
      is not None
      and re.search(r'if args\.demo:\s*\n\s*run_demo\(\)', rg) is not None)

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
print("\nRESULT: ALL GUARDS PRESENT, ARMED, AND EARLY")
sys.exit(0)
