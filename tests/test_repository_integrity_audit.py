
"""Tests for the read-only MMIS repository integrity auditor."""

from pathlib import Path

import pytest

from scripts.check_repository_integrity import (
    REQUIRED_FILES,
    audit,
    get_tracked_files,
    is_protected,
)


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        "data/mmis.db",
        "data/mmis.db.backup",
        "data/images/chart.png",
        "models/best_fusion_model.pt",
        "models/calibrator.pkl",
        "mlflow.db",
        "mlruns/123/metrics/result",
        "backups/old_database.db",
        "results/final_predictions.csv",
    ],
)
def test_protected_artifacts_are_detected(path):
    assert is_protected(path)


@pytest.mark.parametrize(
    "path",
    [
        "fusion.py",
        "README.md",
        "models/rv_thresholds.json",
        "models/har_baseline.json",
        "results/T4_CLASSICAL_BASELINES.md",
        "scripts/verify_guards.py",
    ],
)
def test_legitimate_files_are_allowed(path):
    assert not is_protected(path)


def test_windows_paths_are_supported():
    assert is_protected(r"data\images\chart.png")
    assert is_protected(r"models\best_fusion_model.pt")


def test_missing_required_files_are_reported(tmp_path):
    findings = audit(tmp_path, [])

    assert len(findings) == len(REQUIRED_FILES)
    assert "MISSING REQUIRED FILE: fusion.py" in findings


def test_complete_repository_has_no_findings(tmp_path):
    for filename in REQUIRED_FILES:
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test fixture", encoding="utf-8")

    findings = audit(tmp_path, ["fusion.py", "README.md"])

    assert findings == []


def test_tracked_protected_artifact_is_reported(tmp_path):
    for filename in REQUIRED_FILES:
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test fixture", encoding="utf-8")

    findings = audit(
        tmp_path,
        ["fusion.py", "data/mmis.db"],
    )

    assert findings == [
        "PROTECTED ARTIFACT TRACKED: data/mmis.db"
    ]


def test_git_inspection_reports_tracked_repository_files():
    root = Path(__file__).resolve().parent.parent

    tracked = get_tracked_files(root)

    assert "CLAUDE.md" in tracked
    assert "fusion.py" in tracked
    assert "requirements.txt" in tracked
