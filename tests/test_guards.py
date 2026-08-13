"""Tests for the fail-closed destructive-write guard.

These are the one part of this suite that are NOT characterization tests: they
assert what assert_safe_to_replace SHOULD do, because that function was written
to a specification in this task rather than inherited.

Every dynamic test runs against a throwaway SQLite database under tmp_path. The
real data/mmis.db is never passed to the guard in these tests, and the static
tests parse file text without importing any pipeline module.
"""

import ast
import sqlite3

import pytest

from conftest import PROJECT_ROOT
from db_guard import OVERRIDE_ENV, PROJECT_ROOT as GUARD_ROOT, assert_safe_to_replace


@pytest.fixture(autouse=True)
def _guard_disarmed_by_default(monkeypatch):
    """Every test starts with the override UNSET, whatever the ambient env holds.

    monkeypatch restores the previous value automatically at teardown, so this
    cannot leak into the rest of the session or the developer's shell.
    """
    monkeypatch.delenv(OVERRIDE_ENV, raising=False)


def make_db(path, rows=0, table="t"):
    """Create a throwaway SQLite file with `rows` rows in `table`."""
    con = sqlite3.connect(str(path))
    try:
        con.execute(f"CREATE TABLE {table} (x INTEGER)")
        for i in range(rows):
            con.execute(f"INSERT INTO {table} VALUES (?)", (i,))
        con.commit()
    finally:
        con.close()
    return path


# ══════════════════════════════════════════════════════════════════
#  The four fail-closed paths
# ══════════════════════════════════════════════════════════════════

def test_guard_raises_when_table_is_populated_and_override_unset(tmp_path):
    db = make_db(tmp_path / "x.db", rows=7)
    with pytest.raises(RuntimeError) as err:
        assert_safe_to_replace("t", db_path=str(db), human_reason="irreplaceable")

    message = str(err.value)
    assert "REFUSING TO RUN" in message
    assert "7" in message                    # the row count that would be lost
    assert "'t'" in message                  # the table name
    assert "irreplaceable" in message        # the human_reason passed in
    assert OVERRIDE_ENV in message           # the exact override instruction


def test_guard_raises_when_database_file_does_not_exist(tmp_path):
    """FAIL CLOSED: a missing database is unverifiable, never 'zero rows'."""
    missing = tmp_path / "nope" / "absent.db"
    assert not missing.exists()
    with pytest.raises(RuntimeError) as err:
        assert_safe_to_replace("t", db_path=str(missing), human_reason="why")
    assert "does not exist" in str(err.value)
    assert "REFUSING TO RUN" in str(err.value)


def test_guard_returns_when_table_does_not_exist(tmp_path):
    """A table that was never created holds nothing to destroy, so a first run
    may proceed. Renamed from test_guard_raises_when_table_does_not_exist: T1
    made this raise, which meant sentiment.py and vision.py could never
    bootstrap on a fresh clone. T2 reverses it deliberately.

    This is not a fail-open path — absence is established by a SUCCESSFUL
    sqlite_master query, not by catching an error.
    """
    db = make_db(tmp_path / "x.db", rows=3, table="other")
    assert assert_safe_to_replace("t", db_path=str(db), human_reason="why") is None


def test_guard_still_protects_the_other_tables_when_one_is_missing(tmp_path):
    """The bootstrap allowance is per-table, not a blanket pass."""
    db = make_db(tmp_path / "x.db", rows=3, table="other")
    assert assert_safe_to_replace("t", db_path=str(db), human_reason="why") is None
    with pytest.raises(RuntimeError):
        assert_safe_to_replace("other", db_path=str(db), human_reason="why")


def test_missing_table_is_established_positively_not_by_catching_an_error(tmp_path):
    """A corrupt file must still raise, proving the missing-table return is not
    just a swallowed OperationalError under another name."""
    bogus = tmp_path / "corrupt2.db"
    bogus.write_bytes(b"not a database at all" * 40)
    with pytest.raises(RuntimeError) as err:
        assert_safe_to_replace("anything", db_path=str(bogus), human_reason="why")
    assert "row-count query failed" in str(err.value)


def test_guard_raises_when_the_file_is_not_a_database(tmp_path):
    """FAIL CLOSED on a corrupt/garbage file, which is the 'query raises for ANY
    reason' branch."""
    bogus = tmp_path / "corrupt.db"
    bogus.write_bytes(b"this is definitely not a sqlite file" * 10)
    with pytest.raises(RuntimeError) as err:
        assert_safe_to_replace("t", db_path=str(bogus), human_reason="why")
    assert "row-count query failed" in str(err.value)


def test_guard_returns_when_table_exists_and_is_empty(tmp_path):
    """The only non-override path that returns normally."""
    db = make_db(tmp_path / "x.db", rows=0)
    assert assert_safe_to_replace("t", db_path=str(db), human_reason="why") is None


def test_guard_returns_when_override_is_exactly_one(tmp_path, monkeypatch):
    db = make_db(tmp_path / "x.db", rows=99)
    monkeypatch.setenv(OVERRIDE_ENV, "1")
    assert assert_safe_to_replace("t", db_path=str(db), human_reason="why") is None


@pytest.mark.parametrize("value", ["0", "true", "TRUE", "yes", "", " 1", "1 ", "11"])
def test_guard_still_raises_for_any_override_value_other_than_one(tmp_path, monkeypatch, value):
    """The override is an exact '1'. Anything else leaves the guard armed."""
    db = make_db(tmp_path / "x.db", rows=5)
    monkeypatch.setenv(OVERRIDE_ENV, value)
    with pytest.raises(RuntimeError):
        assert_safe_to_replace("t", db_path=str(db), human_reason="why")


def test_override_does_not_bypass_a_missing_database(tmp_path, monkeypatch):
    """Even with the override set, a missing DB raises — the override authorises
    destroying KNOWN rows, not proceeding blind."""
    monkeypatch.setenv(OVERRIDE_ENV, "1")
    with pytest.raises(RuntimeError) as err:
        assert_safe_to_replace("t", db_path=str(tmp_path / "absent.db"), human_reason="why")
    assert "does not exist" in str(err.value)


def test_environment_is_restored_after_monkeypatch(tmp_path):
    """Sanity check that the previous test's setenv did not leak."""
    import os
    assert os.environ.get(OVERRIDE_ENV) is None


# ══════════════════════════════════════════════════════════════════
#  Path resolution — the CWD failure mode
# ══════════════════════════════════════════════════════════════════

def test_relative_db_path_resolves_against_project_root_not_cwd(tmp_path, monkeypatch):
    """Running from another directory must not silently point at a different DB.

    The guard is called with the default relative 'data/mmis.db' from a
    completely unrelated CWD; it must still resolve to the project root's copy.
    """
    from db_guard import resolve_db_path

    monkeypatch.chdir(tmp_path)
    resolved = resolve_db_path("data/mmis.db")
    assert resolved == GUARD_ROOT / "data" / "mmis.db"
    assert resolved.is_absolute()
    assert GUARD_ROOT == PROJECT_ROOT


def test_absolute_db_path_is_left_alone(tmp_path):
    from db_guard import resolve_db_path
    target = tmp_path / "somewhere.db"
    assert resolve_db_path(str(target)) == target


def test_guard_from_a_foreign_cwd_still_refuses(tmp_path, monkeypatch):
    """End-to-end version of the above: chdir away, and the guard still refuses
    on a populated table rather than finding nothing and proceeding."""
    db = make_db(tmp_path / "real.db", rows=4)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    with pytest.raises(RuntimeError) as err:
        assert_safe_to_replace("t", db_path=str(db), human_reason="why")
    assert "4" in str(err.value)


def test_guard_opens_the_database_read_only(tmp_path):
    """The guard must not be able to mutate the file it inspects.

    Verified by asserting the mtime and size are untouched after a refusal, and
    that db_guard builds a mode=ro URI.
    """
    db = make_db(tmp_path / "x.db", rows=2)
    before = (db.stat().st_size, db.stat().st_mtime_ns)
    with pytest.raises(RuntimeError):
        assert_safe_to_replace("t", db_path=str(db), human_reason="why")
    assert (db.stat().st_size, db.stat().st_mtime_ns) == before
    assert "?mode=ro" in (PROJECT_ROOT / "db_guard.py").read_text(encoding="utf-8")


# ══════════════════════════════════════════════════════════════════
#  Static checks — no pipeline module is imported
# ══════════════════════════════════════════════════════════════════

def _first_call_line(tree, func_name, callee):
    """Line of the first Call to `callee` inside function `func_name`."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    f = sub.func
                    name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
                    if name == callee:
                        return sub.lineno
    return None


def _parse(rel):
    text = (PROJECT_ROOT / rel).read_text(encoding="utf-8")
    return text, ast.parse(text)


@pytest.mark.parametrize("rel,func,expensive", [
    ("sentiment.py", "run_sentiment_pipeline", ["load_finbert", "NewsApiClient"]),
    ("vision.py", "run_vision_pipeline", ["load_efficientnet", "generate_chart"]),
])
def test_guard_call_precedes_the_expensive_work(rel, func, expensive):
    """DEFERRED-2: the guard must refuse BEFORE the quota and the compute are
    spent, not after. Parsed from file text — the modules are never imported,
    because importing vision.py would pull in torch and torchvision."""
    _, tree = _parse(rel)
    guard_line = _first_call_line(tree, func, "assert_safe_to_replace")
    assert guard_line is not None, f"{rel}:{func} does not call the guard"

    for callee in expensive:
        exp_line = _first_call_line(tree, func, callee)
        assert exp_line is not None, f"{rel}:{func} no longer calls {callee}"
        assert guard_line < exp_line, (
            f"{rel}: guard at line {guard_line} runs AFTER {callee} at line {exp_line}"
        )


@pytest.mark.parametrize("rel,func", [
    ("sentiment.py", "run_sentiment_pipeline"),
    ("vision.py", "run_vision_pipeline"),
])
def test_guard_is_the_first_statement_of_the_pipeline(rel, func):
    _, tree = _parse(rel)
    guard_line = _first_call_line(tree, func, "assert_safe_to_replace")
    fnode = next(n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == func)
    assert guard_line == min(stmt.lineno for stmt in fnode.body)


@pytest.mark.parametrize("rel,func", [
    ("sentiment.py", "run_sentiment_pipeline"),
    ("vision.py", "run_vision_pipeline"),
])
def test_guard_precedes_the_destructive_write(rel, func):
    _, tree = _parse(rel)
    guard_line = _first_call_line(tree, func, "assert_safe_to_replace")
    write_line = _first_call_line(tree, func, "to_sql")
    assert write_line is not None
    assert guard_line < write_line


@pytest.mark.parametrize("rel", ["sentiment.py", "vision.py"])
def test_old_fail_open_inline_guard_is_gone(rel):
    """DEFERRED-1: the `except _sqlite3.Error: _existing = 0` pattern silently
    turned every error into 'safe to proceed'. It must not come back."""
    text, _ = _parse(rel)
    assert "except _sqlite3.Error" not in text
    assert "_existing = 0" not in text
    assert "from db_guard import assert_safe_to_replace" in text


def test_db_guard_has_no_fail_open_exception_handler():
    """DEFERRED-1, at the source: every except handler in assert_safe_to_replace
    must terminate in a raise, and none may assign a fallback value."""
    _, tree = _parse("db_guard.py")
    fnode = next(n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == "assert_safe_to_replace")
    handlers = [h for h in ast.walk(fnode) if isinstance(h, ast.ExceptHandler)]
    assert handlers, "no exception handling at all"
    for h in handlers:
        assert any(isinstance(n, ast.Raise) for n in ast.walk(h)), \
            f"except handler at line {h.lineno} does not raise"
        assert not [n for n in ast.walk(h) if isinstance(n, ast.Assign)], \
            f"except handler at line {h.lineno} assigns a fallback value"


def test_run_demo_output_is_marked_synthetic():
    """DEFERRED-3: run_demo writes fabricated prices AND fabricated predictions.
    Its output must be named and columned so it cannot be mistaken for a result."""
    text, tree = _parse("regime.py")
    fnode = next(n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == "run_demo")

    write_line = _first_call_line(tree, "run_demo", "to_csv")
    assert write_line is not None

    assert "SYNTHETIC_DO_NOT_EVALUATE_demo_regime_output.csv" in text
    assert '"results/demo_regime_output.csv"' not in text
    assert "'results/demo_regime_output.csv'" not in text

    # IS_SYNTHETIC is inserted at position 0, before the write.
    insert_lines = [
        sub.lineno for sub in ast.walk(fnode)
        if isinstance(sub, ast.Call)
        and getattr(sub.func, "attr", None) == "insert"
        and len(sub.args) == 3
        and isinstance(sub.args[0], ast.Constant) and sub.args[0].value == 0
        and isinstance(sub.args[1], ast.Constant) and sub.args[1].value == "IS_SYNTHETIC"
        and isinstance(sub.args[2], ast.Constant) and sub.args[2].value == 1
    ]
    assert len(insert_lines) == 1
    assert insert_lines[0] < write_line

    # And it says so out loud, at warning level.
    assert _first_call_line(tree, "run_demo", "warning") is not None
    assert "FABRICATED" in text
    assert "Never pass it to evaluation.py" in text


def test_run_demo_is_reachable_only_behind_the_demo_flag():
    """DEFERRED-3(a): confirming what was ALREADY true before this task —
    run_demo() has exactly one call site, guarded by `if args.demo:`."""
    text, tree = _parse("regime.py")
    call_sites = [
        n.lineno for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "run_demo"
    ]
    assert len(call_sites) == 1
    assert "if args.demo:" in text
    assert 'add_argument("--demo",   action="store_true"' in text
