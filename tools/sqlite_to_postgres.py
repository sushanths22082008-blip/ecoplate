"""Copy an EcoPlate SQLite database into an EMPTY PostgreSQL database (for example Neon).

SAFE BY DEFAULT
* Without --execute this only READS the SQLite file and prints what it WOULD copy. It does not connect to PostgreSQL.
* The SQLite file is opened read-only. It is never changed or deleted.
* With --execute it refuses to run if the PostgreSQL database already contains any EcoPlate data. It never deletes,
  overwrites or merges anything.
* The whole copy is ONE transaction: either every row is copied and the counts match, or nothing is changed.
* It never prints DATABASE_URL, the password or any record contents (only table names and row counts).

Usage (PowerShell, inside the EcoPlate folder with the virtual environment active):
    python tools/sqlite_to_postgres.py --sqlite ecoplate.db                # dry run, reads only
    python tools/sqlite_to_postgres.py --sqlite ecoplate.db --execute      # real copy (needs DATABASE_URL)
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import ecoplate_dbapi as dbapi  # noqa: E402
import ecoplate_postgres as pgm  # noqa: E402

# Parents before children (surplus_events refers to surplus_items).
TABLE_ORDER = ("meal_records", "recipients", "surplus_items", "surplus_events", "day_context",
               "anomaly_reviews", "forecast_log")
BATCH = 1000


class ImportProblem(Exception):
    """A reason the import must not go ahead. The message never contains secrets or record contents."""


def open_sqlite_readonly(path):
    if not os.path.isfile(path):
        raise ImportProblem(f"SQLite file not found: {path}")
    uri = "file:" + os.path.abspath(path).replace("\\", "/").replace("?", "%3f").replace("#", "%23") + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    try:
        version = con.execute("PRAGMA user_version").fetchone()[0]
    except sqlite3.DatabaseError as exc:
        con.close()
        raise ImportProblem("That file is not a readable SQLite database") from exc
    if version > pgm.SCHEMA_VERSION:
        con.close()
        raise ImportProblem(f"SQLite schema version {version} is newer than this tool understands ({pgm.SCHEMA_VERSION})")
    return con


def table_columns(con, table):
    return [r[1] for r in con.execute(f"PRAGMA table_info({table})")]          # table names come from TABLE_ORDER only


def source_counts(src):
    """{table: rows} for the tables that exist in the SQLite file."""
    have = {r[0] for r in src.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    return {t: src.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLE_ORDER if t in have}


def check_source(src):
    """Problems that would make the copy fail or lose data. Returns a list of plain-English messages."""
    problems = []
    counts = source_counts(src)
    if "meal_records" not in counts:
        problems.append("There is no meal_records table in the SQLite file.")
    if "surplus_events" in counts and "surplus_items" in counts:
        orphans = src.execute("SELECT COUNT(*) FROM surplus_events e LEFT JOIN surplus_items i ON i.id = e.surplus_id "
                              "WHERE i.id IS NULL").fetchone()[0]
        if orphans:
            problems.append(f"{orphans} surplus event(s) refer to a surplus item that does not exist.")
    return problems


def run_copy(src, dst):
    """Copy every table from the open SQLite connection `src` into the Database `dst` (which must already have the EcoPlate
    schema). One transaction; verified; returns {table: rows_copied}. Raises ImportProblem and changes nothing on failure."""
    counts = source_counts(src)
    problems = check_source(src)
    if problems:
        raise ImportProblem(" ".join(problems))
    for t in TABLE_ORDER:                                                       # never touch a database that already has data
        existing = dst.scalar(f"SELECT COUNT(*) FROM {t}")
        if existing:
            raise ImportProblem(f"The target database already has {existing} row(s) in {t}. "
                                "This tool only fills an EMPTY database, so nothing was changed.")
    copied = {}
    dst.begin()
    try:
        for t in TABLE_ORDER:
            if t not in counts:
                continue
            cols = table_columns(src, t)
            marks = ",".join("?" for _ in cols)
            sql = f"INSERT INTO {t}({','.join(cols)}) VALUES({marks})"
            cur = src.execute(f"SELECT {','.join(cols)} FROM {t} ORDER BY id")
            n = 0
            while True:
                chunk = cur.fetchmany(BATCH)
                if not chunk:
                    break
                dst.executemany(sql, [tuple(r) for r in chunk])
                n += len(chunk)
            copied[t] = n
        if dst.is_postgres:
            pgm.sync_sequences(dst)                                             # next new row gets an id above the imported ones
        for t, n in copied.items():                                             # verify BEFORE committing
            got = dst.scalar(f"SELECT COUNT(*) FROM {t}")
            if got != n or got != counts[t]:
                raise ImportProblem(f"Row count mismatch in {t}: SQLite {counts[t]}, copied {n}, target {got}.")
        dst.commit()
    except ImportProblem:
        dst.rollback()
        raise
    except Exception as exc:
        try:
            dst.rollback()
        except Exception:                                                       # pragma: no cover
            pass
        raise ImportProblem(f"The copy failed and was rolled back; nothing was imported ({type(exc).__name__}).") from None
    return copied


def main(argv=None):
    ap = argparse.ArgumentParser(description="Copy an EcoPlate SQLite database into an empty PostgreSQL database.")
    ap.add_argument("--sqlite", required=True, help="path to the SQLite file, e.g. ecoplate.db")
    ap.add_argument("--execute", action="store_true", help="really copy (without this, only a dry run that reads the SQLite file)")
    args = ap.parse_args(argv)
    try:
        src = open_sqlite_readonly(args.sqlite)
        try:
            counts = source_counts(src)
            problems = check_source(src)
            print("SQLite file contains:")
            for t, n in counts.items():
                print(f"  {t:16s} {n:>8d} row(s)")
            for p in problems:
                print("PROBLEM:", p)
            if problems:
                return 2
            if not args.execute:
                print("\nDry run only. Nothing was copied and PostgreSQL was not contacted.")
                print("Add --execute to copy these rows into an EMPTY PostgreSQL database (DATABASE_URL).")
                return 0
            try:
                from dotenv import load_dotenv
                load_dotenv(os.path.join(ROOT, ".env"), override=False)
            except ImportError:
                pass
            url = (os.environ.get("DATABASE_URL") or "").strip()
            if not url:
                raise ImportProblem("DATABASE_URL is not set (put it in .env or set it in the terminal).")
            dst = dbapi.connect_postgres(url)
            try:
                pgm.migrate(dst)                                                # creates the EcoPlate tables if they are missing
                copied = run_copy(src, dst)
            finally:
                dst.close()
            print("\nCopied and verified:")
            for t, n in copied.items():
                print(f"  {t:16s} {n:>8d} row(s)")
            print("Done. The SQLite file was not changed.")
            return 0
        finally:
            src.close()
    except ImportProblem as exc:
        print("STOPPED:", dbapi.redact(exc))
        return 1
    except RuntimeError as exc:                                                 # connection errors (already redacted)
        print("STOPPED:", dbapi.redact(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
