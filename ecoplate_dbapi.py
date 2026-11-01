"""EcoPlate database layer: one small interface over SQLite and PostgreSQL (Psycopg 3).

The rest of the application talks to a `Database` object and never to `sqlite3` or `psycopg` directly.
SQL is written ONCE, with `?` placeholders. For PostgreSQL the placeholders are converted to `%s`
(by a converter that skips quoted text), and the few places where the two databases really differ are
handled by named methods:

    db.execute(sql, params)          run one statement; returns a cursor (rows support row["name"] AND row[0])
    db.executemany(sql, rows)        run one statement for many rows (Psycopg 3 needs a cursor for this)
    db.scalar(sql, params)           first column of the first row (for COUNT(*) and similar)
    db.insert_id(sql, params)        INSERT and return the new id (SQLite: lastrowid; PostgreSQL: RETURNING id)
    db.begin() / commit() / rollback()   explicit transactions (SQLite: BEGIN IMMEDIATE; PostgreSQL: BEGIN)
    db.execute(sql, params, for_update=True)   SELECT ... FOR UPDATE on PostgreSQL (row lock); ignored on SQLite,
                                         where BEGIN IMMEDIATE already locks the whole database for writing

Errors: catch `INTEGRITY_ERRORS` (unique / check / foreign-key violations) or `OPERATIONAL_ERRORS`; each is a
tuple that covers both drivers.

Nothing in this module prints or logs a connection string. `redact()` removes secrets from error text.
"""
from __future__ import annotations

import re
import sqlite3

try:                                    # PostgreSQL support is optional: SQLite-only installs still work
    import psycopg
    from psycopg.rows import dict_row
except ImportError:                     # pragma: no cover - exercised only when psycopg is not installed
    psycopg = None
    dict_row = None

SQLITE = "sqlite"
POSTGRES = "postgres"

INTEGRITY_ERRORS = (sqlite3.IntegrityError,) + ((psycopg.IntegrityError,) if psycopg else ())
OPERATIONAL_ERRORS = (sqlite3.OperationalError,) + ((psycopg.OperationalError,) if psycopg else ())


# ----------------------------------------------------------------------------- rows
class Row(dict):
    """A result row for PostgreSQL that behaves like sqlite3.Row: row["name"], row[0], dict(row), row.keys()."""
    __slots__ = ()

    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.values())[key]
        return dict.__getitem__(self, key)


# ----------------------------------------------------------------------------- placeholder conversion
def qmark_to_format(sql):
    """Convert SQLite `?` placeholders to Psycopg `%s`.

    * Text inside '...' or "..." is left alone (a `?` there is data, not a placeholder).
    * A literal `%` anywhere is doubled, because Psycopg treats `%` as special.
    """
    out, quote, i, n = [], None, 0, len(sql)
    while i < n:
        ch = sql[i]
        if quote:
            out.append("%%" if ch == "%" else ch)
            if ch == quote:
                if i + 1 < n and sql[i + 1] == quote:     # doubled quote = escaped quote, still inside the text
                    out.append(sql[i + 1])
                    i += 1
                else:
                    quote = None
        elif ch in ("'", '"'):
            quote = ch
            out.append(ch)
        elif ch == "?":
            out.append("%s")
        elif ch == "%":
            out.append("%%")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


# ----------------------------------------------------------------------------- secrets
def redact(text, secrets=()):
    """Remove connection strings and passwords from text that may be logged or shown."""
    text = str(text)
    for s in secrets:
        if s:
            text = text.replace(s, "[hidden]")
    text = re.sub(r"postgres(?:ql)?://\S+", "[hidden-connection-string]", text)
    text = re.sub(r"(?i)(password|passwd|pwd)\s*=\s*\S+", r"\1=[hidden]", text)
    return text


def url_secrets(url):
    """The parts of a connection URL that must never be shown (the whole URL and the password)."""
    secrets = [url]
    m = re.match(r"^[a-z][a-z0-9+.-]*://[^:/@]*:([^@]*)@", url or "", re.I)
    if m and m.group(1):
        secrets.append(m.group(1))
    return secrets


# ----------------------------------------------------------------------------- the interface
class Database:
    def __init__(self, raw, backend):
        self.raw = raw
        self.backend = backend

    @property
    def is_postgres(self):
        return self.backend == POSTGRES

    # ---- statements
    def execute(self, sql, params=(), for_update=False):
        if not self.is_postgres:
            return self.raw.execute(sql, params)
        text = qmark_to_format(sql)
        if for_update:
            text += " FOR UPDATE"
        # An empty tuple (not None) makes Psycopg process the "%%" escapes, which qmark_to_format added.
        return _PgCursor(self.raw.execute(text, () if params is None else params))

    def executemany(self, sql, rows):
        if not self.is_postgres:
            return self.raw.executemany(sql, rows)
        with self.raw.cursor() as cur:                        # Psycopg 3: executemany lives on the cursor
            cur.executemany(qmark_to_format(sql), rows)
        return None

    def scalar(self, sql, params=()):
        row = self.execute(sql, params).fetchone()
        return None if row is None else row[0]

    def insert_id(self, sql, params=()):
        """Run an INSERT into a table whose primary key is `id`, and return that id."""
        if not self.is_postgres:
            return self.raw.execute(sql, params).lastrowid
        row = self.raw.execute(qmark_to_format(sql) + " RETURNING id", () if params is None else params).fetchone()
        return row["id"]

    # ---- transactions (both connections run in autocommit mode, so a transaction exists only between these calls)
    def begin(self):
        self.raw.execute("BEGIN" if self.is_postgres else "BEGIN IMMEDIATE")

    def commit(self):
        self.raw.execute("COMMIT")

    def rollback(self):
        self.raw.execute("ROLLBACK")

    def close(self):
        self.raw.close()


class _PgCursor:
    """Wraps a Psycopg cursor so rows come back as `Row`, and exposes the `rowcount` the app relies on."""

    def __init__(self, cur):
        self._cur = cur

    @property
    def rowcount(self):
        return self._cur.rowcount

    def fetchone(self):
        r = self._cur.fetchone()
        return None if r is None else Row(r)

    def fetchall(self):
        return [Row(r) for r in self._cur.fetchall()]

    def __iter__(self):
        for r in self._cur:
            yield Row(r)


# ----------------------------------------------------------------------------- connecting
def connect_sqlite(path):
    raw = sqlite3.connect(path, isolation_level=None, timeout=15)
    raw.row_factory = sqlite3.Row
    return Database(raw, SQLITE)


def connect_postgres(url):
    """Open a PostgreSQL connection (autocommit, dict rows). Raises RuntimeError with a message that never
    contains the connection string or password."""
    if psycopg is None:
        raise RuntimeError("DATABASE_URL is set but the 'psycopg' package is not installed. "
                           "Run: pip install -r requirements.txt")
    try:
        raw = psycopg.connect(url, autocommit=True, row_factory=dict_row, connect_timeout=30,
                              prepare_threshold=None)       # no server-side prepared statements: safe behind poolers
    except Exception as exc:
        raise RuntimeError("Could not connect to PostgreSQL (" + type(exc).__name__ + "): "
                           + redact(exc, url_secrets(url))) from None
    return Database(raw, POSTGRES)
