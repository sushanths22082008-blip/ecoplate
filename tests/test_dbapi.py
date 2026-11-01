"""Tests for the SQLite/PostgreSQL database layer that need NO PostgreSQL server.

What these prove: the placeholder converter, row behaviour, the transaction/insert-id/row-lock logic (driven through a
recording fake of a Psycopg connection), backend selection, secret hiding, and that the PostgreSQL schema matches the
SQLite schema table-by-table and column-by-column.

What they CANNOT prove: that a real PostgreSQL server accepts the SQL. That is checked by tests/test_postgres.py,
which runs only when TEST_DATABASE_URL points at a throw-away database.
"""
import os
import re
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app as appmod  # noqa: E402
import ecoplate_db as sqlite_migrations  # noqa: E402
import ecoplate_dbapi as dbapi  # noqa: E402
import ecoplate_postgres as pgm  # noqa: E402

FAKE_URL = "postgresql://someuser:S3cretPassw0rd@ep-example.neon.tech/neondb?sslmode=require"


# ---------------------------------------------------------------- placeholder conversion
class PlaceholderConversion(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(dbapi.qmark_to_format("SELECT * FROM t WHERE a=? AND b=?"), "SELECT * FROM t WHERE a=%s AND b=%s")

    def test_question_mark_inside_text_is_data(self):
        self.assertEqual(dbapi.qmark_to_format("SELECT 'what?' , ?"), "SELECT 'what?' , %s")
        self.assertEqual(dbapi.qmark_to_format('SELECT "col?" FROM t WHERE x=?'), 'SELECT "col?" FROM t WHERE x=%s')

    def test_escaped_quote_does_not_end_the_text(self):
        self.assertEqual(dbapi.qmark_to_format("SELECT 'it''s ?' , ?"), "SELECT 'it''s ?' , %s")

    def test_percent_is_doubled(self):
        self.assertEqual(dbapi.qmark_to_format("SELECT * FROM t WHERE a LIKE 'x%' AND b=?"),
                         "SELECT * FROM t WHERE a LIKE 'x%%' AND b=%s")
        self.assertEqual(dbapi.qmark_to_format("SELECT 5 % 2, ?"), "SELECT 5 %% 2, %s")

    def test_real_statements_from_the_app(self):
        sql = "SELECT * FROM surplus_items WHERE service_date=? AND status!='withdrawn'"
        self.assertEqual(dbapi.qmark_to_format(sql), "SELECT * FROM surplus_items WHERE service_date=%s AND status!='withdrawn'")
        sql = "UPDATE surplus_items SET status=?, updated_at=? WHERE id=? AND status=?"
        self.assertEqual(dbapi.qmark_to_format(sql).count("%s"), 4)
        self.assertNotIn("?", dbapi.qmark_to_format(sql))


# ---------------------------------------------------------------- rows
class Rows(unittest.TestCase):
    def test_row_supports_name_index_dict_and_keys(self):
        r = dbapi.Row({"a": 1, "b": "x"})
        self.assertEqual((r["a"], r[0], r[1], r["b"]), (1, 1, "x", "x"))
        self.assertEqual(dict(r), {"a": 1, "b": "x"})
        self.assertEqual(list(r.keys()), ["a", "b"])
        with self.assertRaises(KeyError):
            r["missing"]

    def test_sqlite_rows_behave_the_same_way(self):
        db = dbapi.connect_sqlite(":memory:")
        row = db.execute("SELECT 1 AS a, 'x' AS b").fetchone()
        self.assertEqual((row["a"], row[0], row[1], row["b"]), (1, 1, "x", "x"))
        self.assertEqual(dict(row), {"a": 1, "b": "x"})
        db.close()


# ---------------------------------------------------------------- SQLite behaviour of the wrapper
class SqliteWrapper(unittest.TestCase):
    def setUp(self):
        self.db = dbapi.connect_sqlite(":memory:")
        self.db.raw.executescript(appmod.SCHEMA)

    def tearDown(self):
        self.db.close()

    ROW = ("2026-01-05", "Lunch", 100, 10, "manual", None, 0, "t")
    INS = "INSERT INTO meal_records(date,meal,prepared,leftovers,source,batch,synthetic,created_at) VALUES(?,?,?,?,?,?,?,?)"

    def test_insert_id_scalar_and_rowcount(self):
        self.assertEqual(self.db.insert_id(self.INS, self.ROW), 1)
        self.assertEqual(self.db.insert_id(self.INS, ("2026-01-05", "Dinner", 5, 1, "m", None, 0, "t")), 2)
        self.assertEqual(self.db.scalar("SELECT COUNT(*) FROM meal_records"), 2)
        cur = self.db.execute("DELETE FROM meal_records WHERE meal=?", ("Dinner",))
        self.assertEqual(cur.rowcount, 1)
        self.assertEqual(self.db.execute("DELETE FROM meal_records WHERE meal=?", ("Nope",)).rowcount, 0)

    def test_rollback_undoes_and_commit_keeps(self):
        self.db.begin()
        self.db.execute(self.INS, self.ROW)
        self.db.rollback()
        self.assertEqual(self.db.scalar("SELECT COUNT(*) FROM meal_records"), 0)
        self.db.begin()
        self.db.execute(self.INS, self.ROW)
        self.db.commit()
        self.assertEqual(self.db.scalar("SELECT COUNT(*) FROM meal_records"), 1)

    def test_executemany_is_atomic_inside_a_transaction(self):
        self.db.begin()
        with self.assertRaises(dbapi.INTEGRITY_ERRORS):
            self.db.executemany(self.INS, [self.ROW, self.ROW])
        self.db.rollback()
        self.assertEqual(self.db.scalar("SELECT COUNT(*) FROM meal_records"), 0)

    def test_for_update_is_ignored_on_sqlite(self):
        self.db.execute(self.INS, self.ROW)
        self.assertEqual(self.db.execute("SELECT * FROM meal_records WHERE id=?", (1,), for_update=True).fetchone()["meal"], "Lunch")

    def test_error_tuples_cover_sqlite(self):
        self.assertIn(sqlite3.IntegrityError, dbapi.INTEGRITY_ERRORS)
        self.assertIn(sqlite3.OperationalError, dbapi.OPERATIONAL_ERRORS)


# ---------------------------------------------------------------- PostgreSQL logic through a recording fake
class FakeCursor:
    def __init__(self, rows=None, rowcount=1):
        self.rows, self.rowcount = list(rows or []), rowcount

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)

    def __iter__(self):
        return iter(self.rows)


class FakeRaw:
    """Stands in for a psycopg connection: records every statement it is given."""
    def __init__(self, rows=None):
        self.calls, self.many, self.closed, self._rows = [], [], False, rows or []

    def execute(self, sql, params=()):
        self.calls.append((sql, params))
        return FakeCursor(self._rows)

    def cursor(self):
        outer = self

        class Ctx:
            def __enter__(s):
                return s

            def __exit__(s, *a):
                return False

            def executemany(s, sql, rows):
                outer.many.append((sql, list(rows)))
        return Ctx()

    def close(self):
        self.closed = True


class PostgresLogic(unittest.TestCase):
    def make(self, rows=None):
        raw = FakeRaw(rows)
        return dbapi.Database(raw, dbapi.POSTGRES), raw

    def test_placeholders_are_converted(self):
        db, raw = self.make()
        db.execute("SELECT * FROM meal_records WHERE date=? AND meal=?", ("2026-01-05", "Lunch"))
        self.assertEqual(raw.calls[0], ("SELECT * FROM meal_records WHERE date=%s AND meal=%s", ("2026-01-05", "Lunch")))

    def test_insert_id_uses_returning(self):
        db, raw = self.make([{"id": 42}])
        self.assertEqual(db.insert_id("INSERT INTO recipients(name) VALUES(?)", ("Hope",)), 42)
        self.assertEqual(raw.calls[0][0], "INSERT INTO recipients(name) VALUES(%s) RETURNING id")

    def test_for_update_appended_only_when_asked(self):
        db, raw = self.make()
        db.execute("SELECT * FROM surplus_items WHERE id=?", (1,), for_update=True)
        db.execute("SELECT * FROM surplus_items WHERE id=?", (1,))
        self.assertTrue(raw.calls[0][0].endswith("WHERE id=%s FOR UPDATE"))
        self.assertTrue(raw.calls[1][0].endswith("WHERE id=%s"))

    def test_transactions_use_plain_begin(self):
        db, raw = self.make()
        db.begin(); db.commit(); db.begin(); db.rollback()
        self.assertEqual([c[0] for c in raw.calls], ["BEGIN", "COMMIT", "BEGIN", "ROLLBACK"])

    def test_executemany_goes_through_a_cursor(self):
        db, raw = self.make()
        db.executemany("INSERT INTO t(a,b) VALUES(?,?)", [(1, 2), (3, 4)])
        self.assertEqual(raw.many, [("INSERT INTO t(a,b) VALUES(%s,%s)", [(1, 2), (3, 4)])])
        self.assertEqual(raw.calls, [])

    def test_rows_are_wrapped_and_scalar_works_with_dict_rows(self):
        db, _ = self.make([{"count": 7}])
        self.assertEqual(db.scalar("SELECT COUNT(*) FROM meal_records"), 7)
        row = db.execute("SELECT 1").fetchone()
        self.assertIsInstance(row, dbapi.Row)
        self.assertEqual((row["count"], row[0]), (7, 7))
        self.assertEqual(dict(db.execute("SELECT 1").fetchall()[0]), {"count": 7})
        self.assertEqual([r["count"] for r in db.execute("SELECT 1")], [7])

    def test_close_closes_the_connection(self):
        db, raw = self.make()
        db.close()
        self.assertTrue(raw.closed)


# ---------------------------------------------------------------- backend selection and secrets
class BackendSelection(unittest.TestCase):
    def env(self, **extra):
        base = {k: v for k, v in os.environ.items()
                if not k.startswith("ECOPLATE_") and k not in ("ANTHROPIC_API_KEY", "DATABASE_URL")}
        return mock.patch.dict(os.environ, {**base, "ECOPLATE_COOKIE_SECURE": "0", **extra}, clear=True)

    def test_no_database_url_means_sqlite(self):
        with tempfile.TemporaryDirectory() as t, self.env(), \
                mock.patch.object(dbapi, "connect_postgres", side_effect=AssertionError("must not be used")):
            app = appmod.create_app(os.path.join(t, "a.db"))
            self.assertEqual(app.config["DB_BACKEND"], "sqlite")
            self.assertEqual(app.test_client().get("/api/health").get_json()["database_backend"], "sqlite")

    def test_explicit_db_path_wins_over_database_url(self):
        with tempfile.TemporaryDirectory() as t, self.env(DATABASE_URL=FAKE_URL), \
                mock.patch.object(dbapi, "connect_postgres", side_effect=AssertionError("must not be used")):
            app = appmod.create_app(os.path.join(t, "a.db"))
            self.assertEqual(app.config["DB_BACKEND"], "sqlite")

    def test_database_url_without_path_selects_postgres_and_hides_the_url(self):
        seen = {}

        def fake_connect(url):
            seen["url"] = url
            raise RuntimeError("stop here")
        with self.env(DATABASE_URL=FAKE_URL), mock.patch.object(dbapi, "connect_postgres", side_effect=fake_connect):
            with self.assertRaises(RuntimeError):
                appmod.create_app()
        self.assertEqual(seen["url"], FAKE_URL)

    def test_postgres_app_starts_with_a_recording_fake_and_runs_the_migration(self):
        raw = FakeRaw([{"coalesce": 0, "max": 0}])
        fake_db = dbapi.Database(raw, dbapi.POSTGRES)
        with self.env(DATABASE_URL=FAKE_URL), mock.patch.object(dbapi, "connect_postgres", return_value=fake_db):
            app = appmod.create_app()
        self.assertEqual(app.config["DB_BACKEND"], "postgres")
        self.assertNotIn("DATABASE_URL", app.config)
        sql = [c[0] for c in raw.calls]
        self.assertEqual(sql[0], "BEGIN")
        self.assertTrue(any(s.startswith("SELECT pg_advisory_xact_lock") for s in sql))
        self.assertTrue(any("CREATE TABLE IF NOT EXISTS meal_records" in s for s in sql))
        self.assertEqual(sql[-1], "COMMIT")

    def test_redact_removes_url_and_password(self):
        msg = f'connection to "{FAKE_URL}" failed: password=S3cretPassw0rd'
        out = dbapi.redact(msg, dbapi.url_secrets(FAKE_URL))
        for secret in ("S3cretPassw0rd", "someuser", "neon.tech"):
            self.assertNotIn(secret, out)

    def test_connect_error_never_contains_the_password(self):
        class Boom(Exception):
            pass
        fake_psycopg = mock.Mock()
        fake_psycopg.connect.side_effect = Boom(f"could not connect using {FAKE_URL}")
        with mock.patch.object(dbapi, "psycopg", fake_psycopg):
            with self.assertRaises(RuntimeError) as cm:
                dbapi.connect_postgres(FAKE_URL)
        self.assertNotIn("S3cretPassw0rd", str(cm.exception))
        self.assertNotIn("neon.tech", str(cm.exception))

    def test_missing_driver_gives_a_clear_message(self):
        with mock.patch.object(dbapi, "psycopg", None):
            with self.assertRaises(RuntimeError) as cm:
                dbapi.connect_postgres(FAKE_URL)
        self.assertIn("psycopg", str(cm.exception))
        self.assertNotIn("S3cretPassw0rd", str(cm.exception))

    def test_dotenv_is_not_loaded_by_create_app(self):
        with tempfile.TemporaryDirectory() as t, self.env(), mock.patch.object(appmod, "_load_local_env") as loader:
            appmod.create_app(os.path.join(t, "a.db"))
        loader.assert_not_called()


# ---------------------------------------------------------------- PostgreSQL migration logic (fake connection)
class PostgresMigrationLogic(unittest.TestCase):
    def test_runs_in_one_transaction_with_a_lock_and_records_the_version(self):
        raw = FakeRaw([{"coalesce": 0}])
        pgm.migrate(dbapi.Database(raw, dbapi.POSTGRES))
        sql = [c[0] for c in raw.calls]
        self.assertEqual((sql[0], sql[-1]), ("BEGIN", "COMMIT"))
        self.assertIn("pg_advisory_xact_lock", sql[1])
        inserts = [c for c in raw.calls if c[0].startswith("INSERT INTO schema_migrations")]
        self.assertEqual(len(inserts), 1)
        self.assertEqual(inserts[0][1][0], pgm.SCHEMA_VERSION)

    def test_already_current_does_nothing_else(self):
        raw = FakeRaw([{"coalesce": pgm.SCHEMA_VERSION}])
        res = pgm.migrate(dbapi.Database(raw, dbapi.POSTGRES))
        self.assertEqual((res["from"], res["to"]), (pgm.SCHEMA_VERSION, pgm.SCHEMA_VERSION))
        self.assertFalse(any("CREATE TABLE IF NOT EXISTS meal_records" in c[0] for c in raw.calls))

    def test_newer_database_is_refused_and_rolled_back(self):
        raw = FakeRaw([{"coalesce": pgm.SCHEMA_VERSION + 1}])
        with self.assertRaises(RuntimeError):
            pgm.migrate(dbapi.Database(raw, dbapi.POSTGRES))
        self.assertEqual(raw.calls[-1][0], "ROLLBACK")

    def test_failure_rolls_back(self):
        class Failing(FakeRaw):
            def execute(s, sql, params=()):
                if "meal_records" in sql and sql.startswith("CREATE TABLE"):
                    raise RuntimeError("boom")
                return super().execute(sql, params)
        raw = Failing([{"coalesce": 0}])
        with self.assertRaises(RuntimeError):
            pgm.migrate(dbapi.Database(raw, dbapi.POSTGRES))
        self.assertEqual(raw.calls[-1][0], "ROLLBACK")


# ---------------------------------------------------------------- schema parity (SQLite <-> PostgreSQL)
def _pg_tables():
    """{table: [column names]} and {table: set of index names} parsed from the PostgreSQL DDL text."""
    cols, idx = {}, {}
    for stmt in pgm.MIGRATION_1:
        m = re.match(r"\s*CREATE TABLE IF NOT EXISTS (\w+) \((.*)\)\s*$", stmt, re.S)
        if m:
            names = []
            for line in re.split(r",\s*\n", m.group(2)):
                first = line.strip().split()[0]
                if first.upper() in ("CHECK", "UNIQUE", "PRIMARY", "FOREIGN", "CONSTRAINT"):
                    continue
                names.append(first)
            cols[m.group(1)] = names
        m = re.match(r"\s*CREATE (?:UNIQUE )?INDEX IF NOT EXISTS (\w+) ON (\w+)", stmt)
        if m:
            idx.setdefault(m.group(2), set()).add(m.group(1))
    return cols, idx


class SchemaParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        con = sqlite3.connect(os.path.join(cls.tmp.name, "p.db"), isolation_level=None)
        con.executescript(appmod.SCHEMA)
        sqlite_migrations.migrate(con)
        cls.sqlite_cols = {}
        cls.sqlite_idx = {}
        for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall():
            cls.sqlite_cols[t] = [r[1] for r in con.execute(f"PRAGMA table_info({t})")]
        for name, tbl in con.execute("SELECT name, tbl_name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"):
            cls.sqlite_idx.setdefault(tbl, set()).add(name)
        cls.sqlite_sql = {n: s for n, s in con.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")}
        con.close()
        cls.pg_cols, cls.pg_idx = _pg_tables()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_same_tables(self):
        self.assertEqual(set(self.sqlite_cols), set(self.pg_cols))

    def test_same_columns_in_the_same_order(self):
        for table, cols in self.sqlite_cols.items():
            self.assertEqual(cols, self.pg_cols[table], table)

    def test_same_named_indexes_except_the_deliberate_recipient_one(self):
        for table, names in self.sqlite_idx.items():
            pg = self.pg_idx.get(table, set())
            self.assertTrue(names <= pg, f"{table}: PostgreSQL is missing {names - pg}")
        self.assertIn("uq_recipients_name_ci", self.pg_idx["recipients"])

    def test_check_rules_are_all_present(self):
        pg_all = " ".join(pgm.MIGRATION_1)
        for table, sql in self.sqlite_sql.items():
            for rule in re.findall(r"CHECK \((?:[^()]|\([^()]*\))*\)", sql):
                norm = lambda x: re.sub(r"\s+", "", x)
                rule_cols = rule
                # the recipient name rule keeps its CHECK; SQLite writes COLLATE NOCASE in front of it
                self.assertIn(norm(rule_cols), norm(pg_all), f"{table}: {rule}")

    def test_unique_rules_are_present(self):
        pg_all = " ".join(pgm.MIGRATION_1)
        for needle in ("UNIQUE (date, meal)", "UNIQUE (date, meal, kind)", "UNIQUE (target_date, meal)", "legacy_id TEXT UNIQUE",
                       "REFERENCES surplus_items(id)"):
            self.assertIn(needle, pg_all)

    def test_every_table_with_an_id_is_in_the_sequence_list(self):
        with_id = {t for t, cols in self.pg_cols.items() if "id" in cols}
        self.assertEqual(with_id, set(pgm.TABLES_WITH_ID))


# ---------------------------------------------------------------- no SQLite-only code left in application modules
class NoSqliteOnlyCode(unittest.TestCase):
    FILES = ("app.py", "ecoplate_routes.py", "ecoplate_assistant.py", "ecoplate_insights.py", "ecoplate_surplus.py")

    def source(self, name):
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), name), encoding="utf-8") as f:
            return f.read()

    def test_no_driver_specific_calls_outside_the_database_layer(self):
        for name in self.FILES:
            src = self.source(name)
            for banned in ("lastrowid", "BEGIN IMMEDIATE", '"COMMIT"', '"ROLLBACK"', "sqlite3.IntegrityError",
                           "sqlite3.OperationalError", "import psycopg"):
                self.assertNotIn(banned, src, f"{name} still contains {banned}")

    def test_every_sql_string_uses_question_mark_placeholders_only(self):
        for name in ("app.py", "ecoplate_routes.py"):
            self.assertNotRegex(self.source(name), r"""execute(?:many)?\(\s*f?["'][^"']*%s""", name)


# ---------------------------------------------------------------- local .env loading
class LocalEnvFile(unittest.TestCase):
    def test_env_file_fills_missing_values_but_never_overrides_existing_ones(self):
        try:
            import dotenv  # noqa: F401
        except ImportError:
            self.skipTest("python-dotenv is not installed")
        with tempfile.TemporaryDirectory() as t:
            with open(os.path.join(t, ".env"), "w", encoding="utf-8") as f:
                f.write("ECOPLATE_TEST_FROM_FILE=file-value\nECOPLATE_TEST_ALREADY_SET=file-value\n")
            env = {**{k: v for k, v in os.environ.items()}, "ECOPLATE_TEST_ALREADY_SET": "real-environment-value"}
            with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(appmod, "BASE_DIR", t):
                appmod._load_local_env()
                self.assertEqual(os.environ["ECOPLATE_TEST_FROM_FILE"], "file-value")
                self.assertEqual(os.environ["ECOPLATE_TEST_ALREADY_SET"], "real-environment-value")

    def test_missing_env_file_or_package_is_harmless(self):
        with tempfile.TemporaryDirectory() as t, mock.patch.object(appmod, "BASE_DIR", t):
            appmod._load_local_env()                                    # no .env file: nothing happens
            with mock.patch.dict(sys.modules, {"dotenv": None}):        # package not installed: nothing happens
                appmod._load_local_env()


if __name__ == "__main__":
    unittest.main()
