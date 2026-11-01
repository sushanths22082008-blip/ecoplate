"""EcoPlate database migrations.

Rules this module follows:
* Migrations are ADDITIVE ONLY: they create new tables/indexes. They never alter, drop or rewrite `meal_records`.
* The schema version is stored in SQLite's `PRAGMA user_version`.
* If the database already holds meal records and a migration is about to run, a consistent copy of the
  database is written next to it first (`<name>.pre-migration-<timestamp>.bak`).
* Each migration runs inside one transaction. If anything fails, nothing is changed.

Version history
  0  original schema (meal_records only)
  1  surplus register on the server (surplus_items, surplus_events, recipients), calendar notes (day_context),
     anomaly review log (anomaly_reviews), and forecast log (forecast_log)
"""
from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone

SCHEMA_VERSION = 1
log = logging.getLogger("ecoplate.db")

MEAL_LIST = "('Breakfast','Lunch','Dinner')"

MIGRATION_1 = [
    f"""CREATE TABLE IF NOT EXISTS surplus_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        item TEXT NOT NULL CHECK (length(item) BETWEEN 1 AND 80),
        meal TEXT CHECK (meal IS NULL OR meal IN {MEAL_LIST}),
        portions INTEGER NOT NULL CHECK (portions BETWEEN 1 AND 10000),
        service_date TEXT NOT NULL,
        prepared_time TEXT,
        use_by_time TEXT,
        storage TEXT NOT NULL DEFAULT 'unknown'
            CHECK (storage IN ('hot_held','refrigerated','room_temp','frozen','unknown')),
        location TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT 'pending_review'
            CHECK (status IN ('pending_review','available','reserved','collected','withdrawn')),
        safety_confirmed_by TEXT,
        safety_confirmed_at TEXT,
        recipient_id INTEGER,
        legacy_id TEXT UNIQUE,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_surplus_status ON surplus_items(status)",
    "CREATE INDEX IF NOT EXISTS idx_surplus_dup ON surplus_items(service_date, meal)",
    """CREATE TABLE IF NOT EXISTS surplus_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        surplus_id INTEGER NOT NULL REFERENCES surplus_items(id),
        at TEXT NOT NULL,
        action TEXT NOT NULL,
        from_status TEXT,
        to_status TEXT,
        actor TEXT NOT NULL DEFAULT 'staff',
        note TEXT NOT NULL DEFAULT ''
    )""",
    "CREATE INDEX IF NOT EXISTS idx_surplus_events ON surplus_events(surplus_id)",
    """CREATE TABLE IF NOT EXISTS recipients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL COLLATE NOCASE CHECK (length(name) BETWEEN 1 AND 80) UNIQUE,
        contact TEXT NOT NULL DEFAULT '',
        requirements TEXT NOT NULL DEFAULT '',
        min_portions INTEGER CHECK (min_portions IS NULL OR min_portions >= 1),
        max_portions INTEGER CHECK (max_portions IS NULL OR max_portions >= 1),
        active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0,1)),
        created_at TEXT NOT NULL
    )""",
    f"""CREATE TABLE IF NOT EXISTS day_context (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        kind TEXT NOT NULL CHECK (kind IN ('holiday','exam','event','other')),
        meal TEXT CHECK (meal IS NULL OR meal IN {MEAL_LIST}),
        label TEXT NOT NULL DEFAULT '' CHECK (length(label) <= 80),
        created_at TEXT NOT NULL
    )""",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_context ON day_context(date, kind, COALESCE(meal,''), label)",
    "CREATE INDEX IF NOT EXISTS idx_context_date ON day_context(date)",
    f"""CREATE TABLE IF NOT EXISTS anomaly_reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        meal TEXT NOT NULL CHECK (meal IN {MEAL_LIST}),
        kind TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('reviewed_ok','confirmed_error','dismissed')),
        note TEXT NOT NULL DEFAULT '' CHECK (length(note) <= 300),
        snapshot_prepared INTEGER NOT NULL,
        snapshot_leftovers INTEGER NOT NULL,
        reviewed_at TEXT NOT NULL,
        UNIQUE (date, meal, kind)
    )""",
    f"""CREATE TABLE IF NOT EXISTS forecast_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        target_date TEXT NOT NULL,
        meal TEXT NOT NULL CHECK (meal IN {MEAL_LIST}),
        made_at TEXT NOT NULL,
        method TEXT NOT NULL,
        expected_demand REAL NOT NULL,
        recommended INTEGER NOT NULL,
        attendance_pct REAL NOT NULL,
        UNIQUE (target_date, meal)
    )""",
]

MIGRATIONS = {1: MIGRATION_1}


def _version(con):
    return con.execute("PRAGMA user_version").fetchone()[0]


def _has_records(con):
    try:
        return con.execute("SELECT COUNT(*) FROM meal_records").fetchone()[0] > 0
    except sqlite3.OperationalError:
        return False


def backup_database(con, db_path):
    """Write a consistent copy next to the database. Returns the path, or None for in-memory databases."""
    if not db_path or db_path == ":memory:":
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = f"{db_path}.pre-migration-{stamp}.bak"
    out = sqlite3.connect(dest)
    try:
        con.backup(out)
    finally:
        out.close()
    return dest


def migrate(con, db_path=None):
    """Bring the database up to SCHEMA_VERSION. `con` must use isolation_level=None (manual transactions).
    Returns {'from': v0, 'to': v1, 'backup': path-or-None}."""
    v0 = _version(con)
    if v0 > SCHEMA_VERSION:
        raise RuntimeError(f"Database schema version {v0} is newer than this EcoPlate ({SCHEMA_VERSION}). "
                           "Refusing to touch it; update the application instead.")
    result = {"from": v0, "to": v0, "backup": None}
    if v0 == SCHEMA_VERSION:
        return result
    if _has_records(con):
        result["backup"] = backup_database(con, db_path)
        if result["backup"]:
            log.info("Database backed up before migration: %s", os.path.basename(result["backup"]))
    for target in range(v0 + 1, SCHEMA_VERSION + 1):
        con.execute("BEGIN IMMEDIATE")
        try:
            for stmt in MIGRATIONS[target]:
                con.execute(stmt)
            con.execute(f"PRAGMA user_version = {int(target)}")
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        result["to"] = target
        log.info("Database migrated to schema version %d", target)
    return result
