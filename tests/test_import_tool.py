"""Tests for tools/sqlite_to_postgres.py. The copy logic is tested SQLite -> SQLite (no PostgreSQL needed);
the PostgreSQL-specific part (id sequences) is covered by tests/test_postgres.py when TEST_DATABASE_URL is set."""
import contextlib
import hashlib
import io
import os
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import app as appmod  # noqa: E402
import ecoplate_dbapi as dbapi  # noqa: E402
import sqlite_to_postgres as tool  # noqa: E402

ENV = {k: v for k, v in os.environ.items() if not k.startswith("ECOPLATE_") and k not in ("ANTHROPIC_API_KEY", "DATABASE_URL")}


def make_db(path):
    with mock.patch.dict(os.environ, {**ENV, "ECOPLATE_COOKIE_SECURE": "0"}, clear=True):
        return appmod.create_app(path)


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class ImportTool(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.src_path = os.path.join(self.tmp.name, "src.db")
        c = make_db(self.src_path).test_client()
        c.post("/api/records/bulk", json={"records": [
            {"date": "2026-01-05", "meal": "Lunch", "prepared": 100, "leftovers": 10},
            {"date": "2026-01-06", "meal": "Lunch", "prepared": 90, "leftovers": 5, "synthetic": True}]})
        c.post("/api/recipients", json={"name": "Hope Trust"})
        c.post("/api/context", json={"date": "2026-01-05", "kind": "event", "label": "Fest"})
        good = {"item": "Rice", "portions": 10, "storage": "hot_held", "prepared_time": "00:00", "use_by_time": "23:59"}
        sid = c.post("/api/surplus", json=good).get_json()["item"]["id"]
        c.post(f"/api/surplus/{sid}/action", json={"action": "confirm_safety", "attest": True})
        c.post("/api/records", json={"date": "2026-01-07", "meal": "Dinner", "prepared": 10, "leftovers": 1})
        c.post("/api/anomalies/review", json={"date": "2026-01-07", "meal": "Dinner", "kind": "high_leftovers", "status": "dismissed"})

    def empty_target(self):
        path = os.path.join(self.tmp.name, "dst.db")
        make_db(path)
        return dbapi.connect_sqlite(path)

    def rows(self, db, table):
        return [dict(r) for r in db.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()]

    def test_copy_is_exact_including_ids_and_leaves_the_source_alone(self):
        before = sha(self.src_path)
        src = tool.open_sqlite_readonly(self.src_path)
        dst = self.empty_target()
        try:
            copied = tool.run_copy(src, dst)
            self.assertEqual(copied["meal_records"], 3)
            self.assertEqual(copied["surplus_events"], 2)
            for t in tool.TABLE_ORDER:
                self.assertEqual(self.rows(dst, t), [dict(r) for r in src.execute(f"SELECT * FROM {t} ORDER BY id")], t)
        finally:
            src.close()
            dst.close()
        self.assertEqual(sha(self.src_path), before)

    def test_source_is_opened_read_only(self):
        src = tool.open_sqlite_readonly(self.src_path)
        try:
            with self.assertRaises(sqlite3.OperationalError):
                src.execute("DELETE FROM meal_records")
        finally:
            src.close()

    def test_refuses_a_target_that_already_has_data(self):
        src = tool.open_sqlite_readonly(self.src_path)
        dst = self.empty_target()
        try:
            dst.execute("INSERT INTO recipients(name,created_at) VALUES(?,?)", ("Existing", "t"))
            with self.assertRaises(tool.ImportProblem) as cm:
                tool.run_copy(src, dst)
            self.assertIn("EMPTY", str(cm.exception))
            self.assertEqual(dst.scalar("SELECT COUNT(*) FROM meal_records"), 0)
            self.assertEqual(dst.scalar("SELECT COUNT(*) FROM recipients"), 1)
        finally:
            src.close()
            dst.close()

    def test_failure_part_way_rolls_everything_back(self):
        src = tool.open_sqlite_readonly(self.src_path)
        dst = self.empty_target()
        real = dst.executemany
        calls = {"n": 0}

        def flaky(sql, rows):
            calls["n"] += 1
            if calls["n"] == 3:
                raise RuntimeError("simulated failure")
            return real(sql, rows)
        dst.executemany = flaky
        try:
            with self.assertRaises(tool.ImportProblem) as cm:
                tool.run_copy(src, dst)
            self.assertIn("rolled back", str(cm.exception))
            for t in tool.TABLE_ORDER:
                self.assertEqual(dst.scalar(f"SELECT COUNT(*) FROM {t}"), 0, t)
        finally:
            src.close()
            dst.close()

    def test_count_mismatch_is_caught_before_commit(self):
        src = tool.open_sqlite_readonly(self.src_path)
        dst = self.empty_target()
        real = dst.executemany

        def lossy(sql, rows):
            return real(sql, rows[:-1] if len(rows) > 1 else rows)         # silently drops a row
        dst.executemany = lossy
        try:
            with self.assertRaises(tool.ImportProblem):
                tool.run_copy(src, dst)
            self.assertEqual(dst.scalar("SELECT COUNT(*) FROM meal_records"), 0)
        finally:
            src.close()
            dst.close()

    def test_dry_run_reads_only_and_never_contacts_postgres(self):
        out = io.StringIO()
        with mock.patch.object(dbapi, "connect_postgres", side_effect=AssertionError("must not connect")), \
                mock.patch.dict(os.environ, {"DATABASE_URL": "postgresql://u:SECRETPW@host/db"}), contextlib.redirect_stdout(out):
            code = tool.main(["--sqlite", self.src_path])
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("meal_records", text)
        self.assertIn("Dry run only", text)
        self.assertNotIn("SECRETPW", text)

    def test_execute_without_database_url_stops_cleanly(self):
        out = io.StringIO()
        with mock.patch.dict(os.environ, {k: v for k, v in os.environ.items() if k != "DATABASE_URL"}, clear=True), \
                mock.patch.dict(sys.modules, {"dotenv": None}), contextlib.redirect_stdout(out):
            code = tool.main(["--sqlite", self.src_path, "--execute"])
        self.assertEqual(code, 1)
        self.assertIn("DATABASE_URL is not set", out.getvalue())

    def test_execute_connection_failure_never_prints_the_password(self):
        out = io.StringIO()
        url = "postgresql://u:SECRETPW@host.example/db"
        boom = RuntimeError("could not connect to " + url + " password=SECRETPW")
        with mock.patch.dict(os.environ, {"DATABASE_URL": url}), mock.patch.dict(sys.modules, {"dotenv": None}), \
                mock.patch.object(dbapi, "connect_postgres", side_effect=boom), contextlib.redirect_stdout(out):
            code = tool.main(["--sqlite", self.src_path, "--execute"])
        self.assertEqual(code, 1)
        self.assertNotIn("SECRETPW", out.getvalue())
        self.assertNotIn("host.example", out.getvalue())

    def test_missing_or_bad_file(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(tool.main(["--sqlite", os.path.join(self.tmp.name, "nope.db")]), 1)
            bad = os.path.join(self.tmp.name, "bad.db")
            with open(bad, "wb") as f:
                f.write(b"this is not a database" * 50)
            self.assertEqual(tool.main(["--sqlite", bad]), 1)

    def test_orphan_surplus_events_are_reported(self):
        raw = sqlite3.connect(self.src_path, isolation_level=None)
        raw.execute("PRAGMA foreign_keys=OFF")
        raw.execute("INSERT INTO surplus_events(surplus_id,at,action) VALUES(9999,'t','x')")
        raw.close()
        src = tool.open_sqlite_readonly(self.src_path)
        try:
            self.assertTrue(any("surplus event" in p for p in tool.check_source(src)))
        finally:
            src.close()


if __name__ == "__main__":
    unittest.main()
