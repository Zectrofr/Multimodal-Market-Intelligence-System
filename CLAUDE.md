# CLAUDE.md — standing rules for this repository

These are the rules MMIS (Fincorp) already operates under. Nothing here is new
policy; this file exists because, until 2026-08-16, none of it was written down
anywhere an agent or a contributor would read it.

**Current authoritative state: [`results\T5_STATUS_AUDIT.md`](results/T5_STATUS_AUDIT.md)**
(written 2026-08-16, evidence-backed, independently verified). Read it before
trusting any other status document in this repository.

> **Warning — `results\T5A_REWIRE.md` and `results\T5A_REWIRE.UNIMPLEMENTED.md`
> describe work that was NEVER APPLIED.** Both carry an UNIMPLEMENTED banner.
> They are design documents for T5a, not records of completed work. Every
> statement of completion in them is false. The corresponding acceptance
> specification lives at `docs\specs\T5A_WIRING_SPEC.py.txt`.

---

## 1. Git history is not to be rewritten

**Forbidden, without exception:** `git filter-repo`, `git reset` (any mode),
`git rebase`, `git revert`, `git merge`, `git pull`, `git stash`,
`git restore`, `git checkout -- <path>`, `git checkout <branch>`, `git clean`,
`git commit --amend`, and any force-push.

This is not a stylistic preference. **This repository has already lost history
once.** A `git filter-repo` author-metadata rewrite on 2026-08-15 changed every
commit SHA in the history and left the reflog expired past that point. Commits
referenced by earlier documents (for example `25956c1`) no longer resolve and
cannot be recovered. Untracked and uncommitted work is not recoverable at all —
it exists in no object and no backup.

Read-only git commands are always fine: `status`, `log`, `show`, `diff`,
`ls-files`, `cat-file`, `rev-parse`, `hash-object`, `check-ignore`, `describe`,
`reflog`, `stash list`.

## 2. Explicit-path staging only

**Never** `git add -A`, `git add .`, or `git add -u`. Stage each path by name.
Most of this project's artifacts are gitignored (`data/mmis.db`, `data/mmis.db.*`,
`data/images/`, `results/`, `models/*.pt`, `models/*.pkl`, `mlruns/`, `mlflow.db`,
`.env`, `backups/`), and a blanket stage is how an unreviewed file enters the tree.

Note that `data/` as a *directory* is NOT ignored — only those three paths inside
it are, and `.gitignore` says so explicitly. A new file dropped into `data/` is
stageable, which is precisely why the rule above is absolute.

## 3. No attribution trailers in commit messages, ever

No `Co-Authored-By`, no "Generated with", no tool or model attribution of any
kind, in any commit message or PR body. The history is currently clean of these
across every commit; keep it that way.

## 4. The database is read-only unless a task explicitly authorises a write

`data\mmis.db` is the only copy of the ingested data and it is gitignored. Read
it through the read-only URI, always:

```python
sqlite3.connect("file:data/mmis.db?mode=ro", uri=True)
```

No `ALTER`, `UPDATE`, `INSERT`, `DELETE`, `DROP`, `VACUUM`, or writing PRAGMA
outside a task that names the write as its deliverable. Record the file's
SHA-256 before and after any task that touches it at all.

Two pipeline modules write with `to_sql(if_exists="replace")`, which DROPs the
target table: `sentiment.py:167` and `vision.py:189`. Both are protected by
fail-closed guards (`db_guard.assert_safe_to_replace`) that fire before any
expensive or irreversible work. The sentiment rows in particular came from a
NewsAPI free tier serving a ~30-day window and **can never be re-fetched**.
`scripts\verify_guards.py` verifies the guards statically; it must exit 0.

## 5. Shadow columns, never in-place replacement

Any new derived data is added as a **new column alongside** the existing one —
never by overwriting it. This is why the schema carries `regime` and
`regime_causal`, `regime_id` and `regime_id_causal`, `target` and `vol_target` /
`vol_target_id`. It keeps the old and new definitions comparable on identical
rows, and it means a mistake in the new column costs nothing.

## 6. Tests are not to be weakened

Do not weaken, skip, `xfail`, delete, or loosen the assertion of any test in
`tests\` to make a suite go green. If a test fails, the failure is the finding.

Run the full suite, unfiltered: `python -m pytest tests\ -q`. No `-x`, no
`--lf`, no `-k`, no `--maxfail` — anything that hides a failure defeats the
point. CI is equally unfiltered, though it reports differently:
`.github\workflows\tests.yml` runs `python -m pytest tests/ -v --tb=short`.

Two markers carry meaning: `characterization` pins current behaviour, and
`defect` pins behaviour that is KNOWN-WRONG. A `defect` test going red is the
intended signal that something was fixed, not a regression.

Relocating a test **out** of `tests\` because it specifies code that does not
exist is a re-scoping, not a weakening — but say so explicitly in the commit
message, and preserve the file byte-for-byte.

Note that a collection-time `ImportError` in any one test module is
**session-fatal**: pytest reports `Interrupted` and runs *zero* tests across
every other module. An untracked test file can therefore break the whole local
suite while CI stays green, because `actions/checkout@v4` only ever materialises
committed objects.

## 7. No performance number without a reproducible source

Do not publish, quote, or restate any metric — accuracy, precision, QLIKE,
Sharpe, ECE, returns — unless it can be traced to a reproducible run whose
artifact is on disk. Numbers that cannot be sourced are removed, not caveated.
Where a figure must be mentioned for context, attribute it explicitly
("claimed in `<document>`") rather than asserting it as current fact.

The README states this commitment directly and currently contains no
performance number.

## 8. Environment

Windows 11, **PowerShell 5.1**, Python 3.10.11 (system install, no venv).

- PowerShell-valid commands only. **No `&&` chaining** — use `;`, or `if ($?) { }`
  when the second command must depend on the first.
- Backslash paths (`tests\test_guards.py`).
- No `2>&1` redirection on native executables.
- Do not install packages as a side effect of a task.

Importing `regime.py` creates a `models\` directory as a module-scope side
effect (`regime.py:66-67`), and `fusion.py` pulls it in transitively via
`regime_causal`. Read these modules rather than importing them when you only
need to inspect them.

## 9. What this project is

A **research platform**. It is not a trading system, it is not financial
advice, and no capital is deployed against it. The project pivoted from
next-day price direction to 5-day forward realized volatility classification
(calm / normal / turbulent) with calibrated uncertainty; as of 2026-08-16 that
pivot exists in `rv_target.py`, `har_baseline.py` and the shadow DB columns
only, and the production chain still implements the direction target. Do not
describe the pivot as complete.
