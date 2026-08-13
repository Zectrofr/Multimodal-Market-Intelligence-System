"""Shared fail-closed guard for destructive table replacements.

Placed at the project ROOT rather than under scripts\\ deliberately: the pipeline
modules that call it (sentiment.py, vision.py) live at the root and are executed
as top-level scripts, so `from db_guard import assert_safe_to_replace` resolves
through the same import mechanics they already rely on, with no sys.path
manipulation in production code. scripts\\ is not a package (no __init__.py) and
importing from it would require exactly that manipulation.

The contract is FAIL CLOSED: every path that cannot POSITIVELY establish "there
is nothing here to destroy" or "the operator explicitly authorised the loss"
raises. A missing database, a locked file, a corrupt page, a wrong working
directory — all of them raise. None of them is permitted to degrade into
"0 rows, safe to proceed", which is the failure mode this module exists to
eliminate.

A table that does not exist yet is the one absence established positively, by a
successful sqlite_master query rather than by catching an error, so a fresh
clone can bootstrap. If that query itself fails, the guard raises as usual.
"""

import os
import sqlite3
from pathlib import Path

# Project root derived from THIS file's location, never from the current working
# directory. This closes the CWD failure mode: the guard resolves the same
# database no matter where the interpreter was launched from.
PROJECT_ROOT = Path(__file__).resolve().parent

OVERRIDE_ENV = "MMIS_ALLOW_DESTRUCTIVE"
OVERRIDE_VALUE = "1"


def resolve_db_path(db_path: str) -> Path:
    """Resolve db_path against the PROJECT ROOT unless it is already absolute."""
    candidate = Path(db_path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


def assert_safe_to_replace(
    table_name: str,
    db_path: str = "data/mmis.db",
    human_reason: str = "",
) -> None:
    """Raise RuntimeError unless replacing `table_name` is demonstrably safe.

    Returns normally in exactly two cases, BOTH of which require that the
    database file exists, the table exists, and the row-count query succeeds:
      1. The table genuinely contains zero rows.
      2. MMIS_ALLOW_DESTRUCTIVE is set to exactly "1".

    Every other outcome raises, including every error condition. There is no
    exception path on which this function returns normally.

    Note that the override is NOT a blanket bypass: it authorises destroying a
    known number of known rows, so it is consulted only on the populated-table
    branch. A missing database or a missing table still raises even with the
    override set, because there is nothing to authorise the destruction of.

    A table that does not exist yet also returns normally: there is nothing to
    destroy, so a fresh clone can bootstrap. This is NOT a fail-open path — the
    absence is established positively by querying sqlite_master, and any error
    while doing so still raises. A missing DATABASE FILE remains fatal, because
    that means the process is looking in the wrong place, not that the data is
    genuinely absent.
    """
    override_active = os.environ.get(OVERRIDE_ENV) == OVERRIDE_VALUE
    resolved = resolve_db_path(db_path)

    # A missing database is NOT "zero rows, safe to proceed". It usually means
    # the process is running from the wrong directory, and proceeding would
    # create a fresh database while the real one sits untouched elsewhere.
    if not resolved.is_file():
        raise RuntimeError(
            f"REFUSING TO RUN: cannot verify it is safe to replace table "
            f"'{table_name}' because the database file does not exist at "
            f"{resolved}. Resolved from db_path={db_path!r} against project root "
            f"{PROJECT_ROOT}. This guard fails CLOSED: a missing database is "
            f"treated as unverifiable, never as empty. {human_reason}"
        )

    # Read-only URI form. Even the counting query cannot mutate the database.
    uri = f"{resolved.as_uri()}?mode=ro"
    try:
        con = sqlite3.connect(uri, uri=True)
        try:
            # Establish absence POSITIVELY. A table that has never been created
            # holds nothing to destroy, so a first run may proceed; but that
            # conclusion is reached by a successful query against sqlite_master,
            # never by catching an error.
            table_exists = con.execute(
                "SELECT COUNT(*) FROM sqlite_master "
                "WHERE type IN ('table', 'view') AND name = ?",
                (table_name,),
            ).fetchone()[0] > 0
            existing = (
                con.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
                if table_exists
                else 0
            )
        finally:
            con.close()
    except Exception as exc:
        # ANY failure — connection refused, database locked, file corrupt,
        # disk error — is fatal. Never swallowed into a default.
        raise RuntimeError(
            f"REFUSING TO RUN: cannot verify it is safe to replace table "
            f"'{table_name}' in {resolved}. The row-count query failed with "
            f"{type(exc).__name__}: {exc}. This guard fails CLOSED: an "
            f"unverifiable table is never assumed empty. {human_reason}"
        ) from exc

    if existing > 0 and not override_active:
        raise RuntimeError(
            f"REFUSING TO RUN: this would DROP table '{table_name}' in {resolved} "
            f"and destroy {existing} existing rows. {human_reason} "
            f"If you have read the above and still intend to destroy those "
            f"{existing} rows, set {OVERRIDE_ENV}={OVERRIDE_VALUE} in the "
            f"environment and re-run. In PowerShell: "
            f'$env:{OVERRIDE_ENV} = "{OVERRIDE_VALUE}"'
        )
