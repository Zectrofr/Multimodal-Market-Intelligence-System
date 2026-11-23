
"""Read-only repository integrity audit for MMIS.

Checks essential source files and detects accidentally tracked
sensitive or generated artifacts. Does not change repository state.
"""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent

REQUIRED_FILES = (
    "CLAUDE.md",
    "requirements.txt",
    "fusion.py",
    "inference.py",
    "regime_causal.py",
    "rv_target.py",
    "db_guard.py",
    "scripts/verify_guards.py",
    "docs/specs/T5A_WIRING_SPEC.py.txt",
)

PROTECTED_EXACT = {
    ".env",
    "data/mmis.db",
    "mlflow.db",
}

PROTECTED_PREFIXES = (
    "data/images/",
    "backups/",
    "mlruns/",
)

PROTECTED_SUFFIXES = (
    ".pt",
    ".pkl",
)


def is_protected(path):
    """Identify artifacts that should never be committed."""
    path = path.replace("\\", "/")
    if path in PROTECTED_EXACT:
        return True
    if path.startswith(PROTECTED_PREFIXES):
        return True
    if path.startswith("models/") and path.endswith(PROTECTED_SUFFIXES):
        return True
    if path.startswith("data/mmis.db."):
        return True
    if path.startswith("results/") and path.endswith(
        ("final_predictions.csv",)
    ):
        return True
    return False


def get_tracked_files(root):
    """Read Git's index without modifying it."""
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True,
        check=True,
    )
    return [
        path.decode("utf-8", errors="surrogateescape")
        for path in result.stdout.split(b"\0")
        if path
    ]


def audit(root, tracked_files):
    """Return findings without changing files or Git state."""
    findings = []

    for filename in REQUIRED_FILES:
        if not (root / filename).is_file():
            findings.append(f"MISSING REQUIRED FILE: {filename}")

    for filename in tracked_files:
        if is_protected(filename):
            findings.append(f"PROTECTED ARTIFACT TRACKED: {filename}")

    return sorted(findings)


def main():
    print("MMIS Repository Integrity Audit")
    print("=" * 40)

    try:
        tracked = get_tracked_files(ROOT)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: Git inspection failed: {exc}")
        return 2

    findings = audit(ROOT, tracked)

    print(f"Tracked files inspected: {len(tracked)}")

    if findings:
        for finding in findings:
            print(f"[FAIL] {finding}")
        print(f"\nAudit completed with {len(findings)} finding(s).")
        return 1

    print("[PASS] Required files are present.")
    print("[PASS] No protected artifacts are tracked.")
    print("[PASS] No repository modifications performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
