"""Shared fixtures for the MMIS characterization suite.

Every database fixture here opens data/mmis.db through the read-only SQLite URI
form. Nothing in this suite may write to the database, and the read-only handle
makes that a guarantee enforced by SQLite rather than a promise made in a
comment.

Artifacts (data/, results/, models/) are gitignored, so a fresh clone has none
of them. Tests that need one SKIP rather than FAIL — a missing artifact is an
absent baseline, not a regression.
"""

import sqlite3
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Make the root-level modules (evaluation.py, db_guard.py, ...) importable
# without installing the project.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DB_FILE = PROJECT_ROOT / "data" / "mmis.db"
PREDICTIONS_CSV = PROJECT_ROOT / "results" / "final_predictions.csv"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "characterization: pins CURRENT behaviour so future changes show a visible diff",
    )
    config.addinivalue_line(
        "markers",
        "defect: pins behaviour that is KNOWN-WRONG; expected to go red when the "
        "cited defect is fixed, and that red is the intended signal",
    )


def require(path: Path, what: str) -> Path:
    """Skip the calling test when a gitignored artifact is not on this machine."""
    if not path.exists():
        pytest.skip(f"{what} not present at {path} (gitignored; skipping, not failing)")
    return path


@pytest.fixture(scope="session")
def db():
    """Session-scoped READ-ONLY connection to data/mmis.db."""
    require(DB_FILE, "data/mmis.db")
    con = sqlite3.connect(f"{DB_FILE.as_uri()}?mode=ro", uri=True)
    try:
        yield con
    finally:
        con.close()


@pytest.fixture(scope="session")
def q(db):
    """Run a query against the read-only DB and return all rows."""
    def _q(sql, params=()):
        return db.execute(sql, params).fetchall()
    return _q


@pytest.fixture(scope="session")
def scalar(q):
    """Run a query expected to yield a single scalar."""
    def _scalar(sql, params=()):
        return q(sql, params)[0][0]
    return _scalar


@pytest.fixture(scope="session")
def predictions():
    """results/final_predictions.csv as a DataFrame, or skip if absent."""
    import pandas as pd
    require(PREDICTIONS_CSV, "results/final_predictions.csv")
    return pd.read_csv(PREDICTIONS_CSV)
