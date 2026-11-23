# MMIS — Contributor Environment and Validation Guide

## Purpose

This guide describes how contributors can verify their local MMIS environment before submitting changes.

MMIS is a multimodal financial research platform focused on five-day forward realized volatility classification. It is not a live trading system.

## 1. Environment requirements

- Python 3.10, with Python 3.10.11 used in the recorded project environment.
- Python package versions pinned in `requirements.txt`.
- A local Git clone of the repository.
- Access to PowerShell on Windows.

Do not install or upgrade dependencies automatically during verification.

## 2. Check environment compatibility

Run from the repository root:

```powershell
python scripts\check_environment_compatibility.py
```

The checker reports matching, missing, and mismatched dependencies.

Exit codes:

- `0`: all checked requirements match.
- `1`: missing dependencies or version mismatches.
- `2`: the requirements file cannot be processed.

The checker is read-only. It does not modify the Python environment.

## 3. Verify repository integrity

```powershell
python scripts\check_repository_integrity.py
```

This checks required project files and detects protected artifacts that have accidentally entered Git's tracked file list.

It does not replace the project's database safety guards.

## 4. Run all automated tests

```powershell
python -m pytest tests\ -q
```

Always run the complete suite without selective filtering, skipped-test overrides, or weakened assertions.

A fresh clone may skip tests requiring Git-ignored research data and trained model artifacts. Record the number of passing, skipped, and failing tests.

## 5. Verify destructive-write protections

```powershell
python scripts\verify_guards.py
```

The historical baseline is 66 successful static checks. Any failure requires investigation.

## 6. Protect research artifacts

The following are locally generated or protected artifacts:

- `data/mmis.db`
- `data/images/`
- `models/*.pt`
- `models/*.pkl`
- `results/final_predictions.csv`
- `.env`
- `backups/`

Never assume that all files in `results/` are disposable. Some legitimate research records are tracked in the repository.

The original database must remain read-only unless a specific task authorizes modifications. Record SHA-256 checksums before and after authorized database inspection.

## 7. Temporal integrity

Forward volatility forecasting requires careful separation between training and validation periods.

Tests must verify that future observations cannot influence fitted training parameters.

Prefer independent mutation and truncation experiments, paired with anti-vacuity assertions proving that the test intervention actually changed its intended input.

Never weaken tests to obtain a passing result.

## 8. Integration readiness

Before requesting integration:

1. Confirm the working tree contains only intended changes.
2. Verify the full test suite.
3. Verify database safety guards.
4. Review environment compatibility findings.
5. Compare the contribution branch against the current upstream default branch.
6. Record any skipped tests and unresolved risks.
7. Use explicit-path Git staging only.
8. Preserve existing Git history and author identity.

No model-performance claims should be published without independently verifiable, reproducible evidence.

Local verification is not equivalent to full research-data integration validation.

## 9. Contribution boundaries

Contributors must preserve existing pipeline behavior unless an approved development task explicitly requires changes.

Local feature branches should be reviewed before integration.

Do not commit generated research databases, sensitive credentials, or unreviewed artifacts.

This guide supplements the repository's `CLAUDE.md`, which remains the authoritative source of repository operating rules. 