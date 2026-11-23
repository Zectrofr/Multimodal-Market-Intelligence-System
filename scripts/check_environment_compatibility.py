
"""Read-only Python environment compatibility checker for MMIS.

Compares installed distributions against requirements.txt.
Does not install, upgrade, or modify packages or project files.

Usage:
    python scripts/check_environment_compatibility.py
"""

import re
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = ROOT / "requirements.txt"
EXPECTED_PYTHON = (3, 10)

PIN_PATTERN = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9_.-]*)==([^\s;#]+)$"
)


def parse_requirements(path):
    """Read strict version pins; reject unsupported requirement syntax."""
    if not path.is_file():
        raise FileNotFoundError(f"Requirements missing: {path}")

    pins = {}
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue

        match = PIN_PATTERN.fullmatch(line)
        if match is None:
            raise ValueError(
                f"Unsupported requirement at line {line_number}: {line}"
            )

        name, version = match.groups()
        normalized = re.sub(r"[-_.]+", "-", name).lower()

        if normalized in pins:
            raise ValueError(f"Duplicate requirement: {name}")

        pins[normalized] = version

    if not pins:
        raise ValueError("No pinned dependencies were found")

    return pins


def inspect_dependencies(pins, version_lookup=None):
    """Return installed, missing and mismatched distributions."""
    if version_lookup is None:
        version_lookup = metadata.version

    matched = {}
    missing = {}
    mismatched = {}

    for name, expected in sorted(pins.items()):
        try:
            actual = version_lookup(name)
        except metadata.PackageNotFoundError:
            missing[name] = expected
            continue

        if actual == expected:
            matched[name] = actual
        else:
            mismatched[name] = {
                "expected": expected,
                "installed": actual,
            }

    return {
        "matched": matched,
        "missing": missing,
        "mismatched": mismatched,
    }


def render_report(report, python_version=None):
    """Produce a deterministic, human-readable report."""
    if python_version is None:
        python_version = sys.version_info[:3]

    lines = [
        "MMIS Environment Compatibility Report",
        "=" * 45,
        "Python: " + ".".join(map(str, python_version)),
        f"Dependencies matching: {len(report['matched'])}",
        f"Dependencies missing: {len(report['missing'])}",
        f"Version mismatches: {len(report['mismatched'])}",
    ]

    for name, expected in sorted(report["missing"].items()):
        lines.append(f"[MISSING] {name}: expected {expected}")

    for name, versions in sorted(report["mismatched"].items()):
        lines.append(
            f"[MISMATCH] {name}: "
            f"expected {versions['expected']}, "
            f"installed {versions['installed']}"
        )

    if tuple(python_version[:2]) != EXPECTED_PYTHON:
        lines.append(
            "[MISMATCH] Python minor version: expected 3.10"
        )

    if (
        not report["missing"]
        and not report["mismatched"]
        and tuple(python_version[:2]) == EXPECTED_PYTHON
    ):
        lines.append("[PASS] All pinned dependencies match")
    else:
        lines.append("[INFO] Environment differs from pinned requirements")

    lines.append("[INFO] No packages or project files were modified")
    return "\n".join(lines)


def main():
    try:
        pins = parse_requirements(REQUIREMENTS)
        report = inspect_dependencies(pins)
        print(render_report(report))
    except (OSError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    if (
        report["missing"]
        or report["mismatched"]
        or sys.version_info[:2] != EXPECTED_PYTHON
    ):
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
