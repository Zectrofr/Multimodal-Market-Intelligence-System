
"""Tests for the MMIS environment compatibility checker.

All tests use synthetic package inventories and temporary files.
No package installation, network access, or database writes.
"""

from importlib import metadata

import pytest

from scripts.check_environment_compatibility import (
    inspect_dependencies,
    parse_requirements,
    render_report,
)


def test_pinned_requirements_are_parsed(tmp_path):
    file = tmp_path / "requirements.txt"
    file.write_text(
        "numpy==2.2.6\n"
        "pandas==2.3.3  # data frames\n"
        "# comment\n",
        encoding="utf-8",
    )

    assert parse_requirements(file) == {
        "numpy": "2.2.6",
        "pandas": "2.3.3",
    }


def test_package_names_are_normalized(tmp_path):
    file = tmp_path / "requirements.txt"
    file.write_text(
        "Python_Dotenv==1.0.1\n",
        encoding="utf-8",
    )

    assert parse_requirements(file) == {
        "python-dotenv": "1.0.1"
    }


def test_unpinned_requirements_are_rejected(tmp_path):
    file = tmp_path / "requirements.txt"
    file.write_text("numpy>=2.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported"):
        parse_requirements(file)


def test_duplicate_requirements_are_rejected(tmp_path):
    file = tmp_path / "requirements.txt"
    file.write_text(
        "my_package==1.0\n"
        "my-package==2.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate"):
        parse_requirements(file)


def test_missing_requirements_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_requirements(tmp_path / "missing.txt")


def test_missing_packages_are_identified():
    def lookup(name):
        raise metadata.PackageNotFoundError(name)

    result = inspect_dependencies(
        {"example-package": "1.0"},
        version_lookup=lookup,
    )

    assert result["missing"] == {
        "example-package": "1.0"
    }


def test_version_mismatches_are_identified():
    result = inspect_dependencies(
        {"numpy": "2.2.6"},
        version_lookup=lambda name: "1.0.0",
    )

    assert result["mismatched"]["numpy"] == {
        "expected": "2.2.6",
        "installed": "1.0.0",
    }


def test_matching_versions_pass():
    result = inspect_dependencies(
        {"numpy": "2.2.6"},
        version_lookup=lambda name: "2.2.6",
    )

    assert result["matched"] == {"numpy": "2.2.6"}
    assert not result["missing"]
    assert not result["mismatched"]


def test_report_explains_version_mismatches():
    report = {
        "matched": {},
        "missing": {"hmmlearn": "0.3.3"},
        "mismatched": {
            "python-dotenv": {
                "expected": "1.0.1",
                "installed": "1.2.2",
            }
        },
    }

    output = render_report(
        report,
        python_version=(3, 10, 11),
    )

    assert "[MISSING] hmmlearn" in output
    assert "[MISMATCH] python-dotenv" in output
    assert "No packages or project files were modified" in output


def test_report_flags_incorrect_python_version():
    report = {
        "matched": {},
        "missing": {},
        "mismatched": {},
    }

    output = render_report(
        report,
        python_version=(3, 11, 0),
    )

    assert "expected 3.10" in output
