"""Throwaway READ-ONLY inspection script for the MMIS state audit.

Opens data/mmis.db strictly via a read-only URI connection. Performs NO writes
of any kind to the database. Emits a plain-text dump to stdout.

Usage (from project root):  python scripts/audit_state.py
"""

import os
import sqlite3
import sys
from collections import Counter

DB_URI = "file:data/mmis.db?mode=ro"


def line(s=""):
    sys.stdout.write(str(s) + "\n")


def hr(title):
    line()
    line("=" * 78)
    line(title)
    line("=" * 78)


def q(cur, sql, params=()):
    cur.execute(sql, params)
    return cur.fetchall()


def main():
    if not os.path.exists("data/mmis.db"):
        line("FATAL: data/mmis.db does not exist relative to CWD " + os.getcwd())
        return 1

    con = sqlite3.connect(DB_URI, uri=True)
    cur = con.cursor()

    hr("0. SQLITE / FILE META")
    st = os.stat("data/mmis.db")
    line("path       : %s" % os.path.abspath("data/mmis.db"))
    line("size_bytes : %d" % st.st_size)
    line("sqlite_ver : %s" % sqlite3.sqlite_version)
    for pragma in ("page_size", "page_count", "journal_mode", "user_version"):
        try:
            line("%-12s: %s" % (pragma, q(cur, "PRAGMA %s" % pragma)[0][0]))
        except Exception as e:  # noqa: BLE001
            line("%-12s: ERR %s" % (pragma, e))

    hr("1. OBJECTS")
    objs = q(cur, "SELECT type, name, tbl_name FROM sqlite_master ORDER BY type, name")
    for t, n, tb in objs:
        line("%-8s %-32s (tbl=%s)" % (t, n, tb))

    tables = [n for (t, n, _tb) in objs if t == "table" and not n.startswith("sqlite_")]

    hr("2. SCHEMA + ROW COUNTS")
    meta = {}
    for tbl in tables:
        cols = q(cur, 'PRAGMA table_info("%s")' % tbl)
        n = q(cur, 'SELECT COUNT(*) FROM "%s"' % tbl)[0][0]
        meta[tbl] = {"cols": cols, "rows": n}
        line()
        line("TABLE %s  --  %d rows" % (tbl, n))
        line("  %-4s %-28s %-12s %-8s %-10s %s" % ("cid", "name", "decl_type", "notnull", "default", "pk"))
        for cid, name, ctype, notnull, dflt, pk in cols:
            line("  %-4s %-28s %-12s %-8s %-10s %s" % (cid, name, ctype, notnull, dflt, pk))
        idx = q(cur, 'PRAGMA index_list("%s")' % tbl)
        for row in idx:
            iname, uniq = row[1], row[2]
            icols = [r[2] for r in q(cur, 'PRAGMA index_info("%s")' % iname)]
            line("  INDEX %s unique=%s cols=%s" % (iname, uniq, icols))
        fks = q(cur, 'PRAGMA foreign_key_list("%s")' % tbl)
        for fk in fks:
            line("  FK %s" % (fk,))
        ddl = q(cur, "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (tbl,))
        if ddl and ddl[0][0]:
            line("  DDL: " + " ".join(ddl[0][0].split()))

    hr("3. NULL COUNTS PER COLUMN")
    for tbl in tables:
        n = meta[tbl]["rows"]
        if n == 0:
            line("TABLE %s : empty" % tbl)
            continue
        line()
        line("TABLE %s (%d rows)" % (tbl, n))
        line("  %-30s %-12s %s" % ("column", "nulls", "null_pct"))
        for cid, name, ctype, notnull, dflt, pk in meta[tbl]["cols"]:
            c = q(cur, 'SELECT COUNT(*) FROM "%s" WHERE "%s" IS NULL' % (tbl, name))[0][0]
            line("  %-30s %-12d %.2f%%" % (name, c, 100.0 * c / n))

    hr("4. DATE COLUMN STORAGE FORMAT")
    for tbl in tables:
        for cid, name, ctype, notnull, dflt, pk in meta[tbl]["cols"]:
            if "date" in name.lower() or name.lower() in ("ts", "timestamp", "time"):
                try:
                    rows = q(
                        cur,
                        'SELECT typeof("%s"), "%s" FROM "%s" WHERE "%s" IS NOT NULL LIMIT 3'
                        % (name, name, tbl, name),
                    )
                    types = q(
                        cur,
                        'SELECT typeof("%s"), COUNT(*) FROM "%s" GROUP BY 1' % (name, tbl),
                    )
                    line()
                    line("%s.%s  declared=%s" % (tbl, name, ctype))
                    line("  typeof histogram: %s" % (types,))
                    for tp, v in rows:
                        line("  example typeof=%-8s repr=%r" % (tp, v))
                except Exception as e:  # noqa: BLE001
                    line("  ERR %s.%s: %s" % (tbl, name, e))

    hr("5. PER-TICKER COVERAGE")
    for tbl in tables:
        names = [c[1] for c in meta[tbl]["cols"]]
        tick = next((c for c in names if c.lower() in ("ticker", "symbol")), None)
        dcol = next((c for c in names if c.lower() in ("date", "dt", "timestamp")), None)
        if not tick:
            continue
        line()
        line("TABLE %s  (ticker=%s date=%s)" % (tbl, tick, dcol))
        if dcol:
            rows = q(
                cur,
                'SELECT "%s", COUNT(*), MIN("%s"), MAX("%s") FROM "%s" GROUP BY 1 ORDER BY 1'
                % (tick, dcol, dcol, tbl),
            )
            line("  %-10s %-8s %-14s %s" % (tick, "rows", "min_date", "max_date"))
            for r in rows:
                line("  %-10s %-8s %-14s %s" % (r[0], r[1], r[2], r[3]))
        else:
            rows = q(cur, 'SELECT "%s", COUNT(*) FROM "%s" GROUP BY 1 ORDER BY 1' % (tick, tbl))
            for r in rows:
                line("  %-10s %s" % (r[0], r[1]))

    hr("6. REGIME COVERAGE")
    for tbl in tables:
        names = [c[1] for c in meta[tbl]["cols"]]
        for c in names:
            if "regime" in c.lower():
                n = meta[tbl]["rows"]
                nn = q(cur, 'SELECT COUNT(*) FROM "%s" WHERE "%s" IS NULL' % (tbl, c))[0][0]
                dist = q(
                    cur,
                    'SELECT "%s", COUNT(*) FROM "%s" GROUP BY 1 ORDER BY 2 DESC' % (c, tbl),
                )
                line()
                line("%s.%s : %d NULL of %d rows (%.2f%% NULL)" % (tbl, c, nn, n, 100.0 * nn / max(n, 1)))
                line("  value distribution: %s" % (dist,))

    hr("7. SENTIMENT COVERAGE")
    for tbl in tables:
        names = [c[1] for c in meta[tbl]["cols"]]
        scols = [c for c in names if any(k in c.lower() for k in ("sentiment", "positive", "negative", "neutral", "finbert"))]
        if not scols:
            continue
        n = meta[tbl]["rows"]
        line()
        line("TABLE %s  sentiment-like columns: %s  (%d rows)" % (tbl, scols, n))
        # modal tuple across all sentiment columns jointly
        try:
            sel = ", ".join('"%s"' % c for c in scols)
            rows = q(cur, "SELECT %s FROM \"%s\"" % (sel, tbl))
            cnt = Counter(rows)
            top = cnt.most_common(5)
            line("  distinct sentiment tuples: %d" % len(cnt))
            for val, k in top:
                line("    freq=%-8d pct=%6.2f%%  tuple=%s" % (k, 100.0 * k / max(n, 1), val))
            if top:
                modal, mfreq = top[0]
                nondefault = n - mfreq
                line("  MODAL(placeholder) rows = %d (%.4f%%)" % (mfreq, 100.0 * mfreq / max(n, 1)))
                line("  NON-DEFAULT rows        = %d (%.4f%%)" % (nondefault, 100.0 * nondefault / max(n, 1)))
        except Exception as e:  # noqa: BLE001
            line("  ERR %s" % e)
        # per-column distinct counts
        for c in scols:
            d = q(cur, 'SELECT COUNT(DISTINCT "%s") FROM "%s"' % (c, tbl))[0][0]
            line("  distinct(%s) = %d" % (c, d))

    hr("8. BLOB / VISION FEATURE COLUMNS")
    for tbl in tables:
        for cid, name, ctype, notnull, dflt, pk in meta[tbl]["cols"]:
            is_blobby = (ctype or "").upper().startswith("BLOB") or any(
                k in name.lower() for k in ("feature", "vector", "embedding", "blob")
            )
            if not is_blobby:
                continue
            try:
                nn = q(cur, 'SELECT COUNT(*) FROM "%s" WHERE "%s" IS NOT NULL' % (tbl, name))[0][0]
                lens = q(
                    cur,
                    'SELECT LENGTH("%s"), COUNT(*) FROM "%s" WHERE "%s" IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 5'
                    % (name, tbl, name),
                )
                tps = q(cur, 'SELECT typeof("%s"), COUNT(*) FROM "%s" GROUP BY 1' % (name, tbl))
                line()
                line("%s.%s declared=%s : %d non-NULL of %d" % (tbl, name, ctype, nn, meta[tbl]["rows"]))
                line("  typeof histogram : %s" % (tps,))
                line("  byte-length histo: %s" % (lens,))
            except Exception as e:  # noqa: BLE001
                line("  ERR %s.%s: %s" % (tbl, name, e))

    hr("9. LABEL / TARGET COLUMN DISTRIBUTIONS")
    for tbl in tables:
        for cid, name, ctype, notnull, dflt, pk in meta[tbl]["cols"]:
            if any(k in name.lower() for k in ("label", "target", "direction", "class", "y_")):
                try:
                    d = q(cur, 'SELECT "%s", COUNT(*) FROM "%s" GROUP BY 1 ORDER BY 2 DESC LIMIT 10' % (name, tbl))
                    line("%s.%s : %s" % (tbl, name, d))
                except Exception as e:  # noqa: BLE001
                    line("  ERR %s" % e)

    hr("10. SAMPLE ROW (non-blob columns only) PER TABLE")
    for tbl in tables:
        cols = [c for c in meta[tbl]["cols"] if not (c[2] or "").upper().startswith("BLOB")]
        names = [c[1] for c in cols]
        if not names or meta[tbl]["rows"] == 0:
            continue
        sel = ", ".join('"%s"' % c for c in names)
        try:
            r = q(cur, 'SELECT %s FROM "%s" LIMIT 1' % (sel, tbl))[0]
            line()
            line("TABLE %s sample row:" % tbl)
            for k, v in zip(names, r):
                sv = repr(v)
                if len(sv) > 90:
                    sv = sv[:90] + "...<truncated>"
                line("  %-28s = %s" % (k, sv))
        except Exception as e:  # noqa: BLE001
            line("  ERR %s" % e)

    con.close()
    hr("DONE (read-only, no writes performed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
