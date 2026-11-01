"""The SAME application scenarios, run on both database backends.

* `SqliteScenarios` always runs.
* `PostgresScenarios` and `PostgresOnly` run only when the environment variable TEST_DATABASE_URL is set to a
  THROW-AWAY PostgreSQL database (for example a local PostgreSQL, or a separate Neon test branch). Otherwise they are
  reported as skipped, never as passed.

Safety rules built into the PostgreSQL tests:
* They refuse to run if TEST_DATABASE_URL is the same as DATABASE_URL.
* Every test works in its own temporary schema called ecoplate_test_<random> (created at the start, dropped with
  CASCADE at the end). Tables in other schemas, such as your real "public" data, are never touched.
* Use the DIRECT (non-pooled) connection string for TEST_DATABASE_URL, because the tests select the schema through a
  connection option that connection poolers do not accept.

How to run (PowerShell):
    $env:TEST_DATABASE_URL = "postgresql://user:password@host/dbname?sslmode=require"
    python -m unittest tests.test_postgres -v
"""
import os
import sys
import tempfile
import unittest
import uuid
from datetime import timedelta
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import app as appmod  # noqa: E402
import ecoplate_dbapi as dbapi  # noqa: E402
import ecoplate_postgres as pgm  # noqa: E402
import sqlite_to_postgres as importer  # noqa: E402
from ecoplate_util import local_today  # noqa: E402
from fixtures import gen  # noqa: E402

TEST_URL = (os.environ.get("TEST_DATABASE_URL") or "").strip()

INSERT_RECORD = ("INSERT INTO meal_records(date,meal,prepared,leftovers,source,batch,synthetic,created_at) "
                 "VALUES(?,?,?,?,?,?,?,?)")
ROW = ("2026-01-05", "Lunch", 100, 10, "manual", None, 0, "2026-01-05T00:00:00+00:00")


def clean_env():
    return {k: v for k, v in os.environ.items()
            if not k.startswith("ECOPLATE_") and k not in ("ANTHROPIC_API_KEY", "DATABASE_URL", "TEST_DATABASE_URL")}


# ================================================================ backend harnesses
class SqliteHarness:
    backend = "sqlite"

    def make_backend(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path = os.path.join(self.tmp.name, "t.db")
        with mock.patch.dict(os.environ, {**clean_env(), "ECOPLATE_COOKIE_SECURE": "0"}, clear=True):
            self.app = appmod.create_app(path)
        self.connect = lambda: dbapi.connect_sqlite(path)


class PostgresHarness:
    backend = "postgres"

    def make_backend(self):
        if not TEST_URL:
            self.skipTest("TEST_DATABASE_URL is not set (PostgreSQL tests are skipped, not passed)")
        if dbapi.psycopg is None:
            self.skipTest("psycopg is not installed")
        if TEST_URL == (os.environ.get("DATABASE_URL") or "").strip():
            self.skipTest("TEST_DATABASE_URL equals DATABASE_URL: refusing to test against the real database")
        from psycopg.conninfo import make_conninfo
        self.schema = "ecoplate_test_" + uuid.uuid4().hex[:12]
        admin = dbapi.psycopg.connect(TEST_URL, autocommit=True)
        self.addCleanup(admin.close)
        admin.execute(f"CREATE SCHEMA {self.schema}")
        self.addCleanup(lambda: admin.execute(f"DROP SCHEMA {self.schema} CASCADE"))
        self.url = make_conninfo(TEST_URL, options=f"-c search_path={self.schema}")
        with mock.patch.dict(os.environ, {**clean_env(), "ECOPLATE_COOKIE_SECURE": "0", "DATABASE_URL": self.url}, clear=True):
            self.app = appmod.create_app()
        self.connect = lambda: dbapi.connect_postgres(self.url)


# ================================================================ scenarios (identical on both backends)
class Scenarios:
    def setUp(self):
        self.make_backend()
        self.c = self.app.test_client()

    def count(self, table):
        con = self.connect()
        try:
            return con.scalar(f"SELECT COUNT(*) FROM {table}")
        finally:
            con.close()

    def rec(self, date="2026-01-05", meal="Lunch", prepared=100, leftovers=10, **kw):
        return {"date": date, "meal": meal, "prepared": prepared, "leftovers": leftovers, **kw}

    # ---- startup and health
    def test_startup_and_health(self):
        self.assertEqual(self.app.config["DB_BACKEND"], self.backend)
        h = self.c.get("/api/health").get_json()
        self.assertTrue(h["ok"])
        self.assertEqual(h["database_backend"], self.backend)
        self.assertEqual(h["records"], 0)
        self.assertEqual(h["schema_version"], 1)
        self.assertNotIn("postgresql://", str(h))

    def test_restart_keeps_data_and_does_not_remigrate_destructively(self):
        self.c.post("/api/records", json=self.rec())
        self.assertEqual(self.count("meal_records"), 1)
        # starting the application again on the same database must not lose or change anything
        if self.backend == "sqlite":
            app2 = appmod.create_app(os.path.join(self.tmp.name, "t.db"))
        else:
            with mock.patch.dict(os.environ, {**clean_env(), "ECOPLATE_COOKIE_SECURE": "0", "DATABASE_URL": self.url}, clear=True):
                app2 = appmod.create_app()
        self.assertEqual(app2.test_client().get("/api/records").get_json()["count"], 1)

    # ---- meal records: create / read / update-by-conflict / delete
    def test_record_crud(self):
        r = self.c.post("/api/records", json=self.rec())
        self.assertEqual(r.status_code, 201)
        self.assertEqual(r.get_json()["records"], 1)
        dup = self.c.post("/api/records", json=self.rec(prepared=999, leftovers=1))
        self.assertEqual(dup.status_code, 409)
        self.assertEqual(dup.get_json()["existing"], {"prepared": 100, "leftovers": 10})
        self.assertEqual(self.c.post("/api/records", json=self.rec(leftovers=500)).status_code, 400)
        listed = self.c.get("/api/records").get_json()
        self.assertEqual(listed["count"], 1)
        self.assertEqual(listed["records"][0]["synthetic"], False)
        self.assertEqual(self.c.delete("/api/records?date=2026-01-05&meal=Lunch").get_json()["deleted"], 1)
        self.assertEqual(self.c.delete("/api/records?date=2026-01-05&meal=Lunch").status_code, 404)
        self.assertEqual(self.count("meal_records"), 0)

    def test_records_are_listed_in_date_then_meal_order(self):
        for d, m in (("2026-01-06", "Dinner"), ("2026-01-05", "Dinner"), ("2026-01-05", "Breakfast"), ("2026-01-05", "Lunch")):
            self.c.post("/api/records", json=self.rec(date=d, meal=m))
        order = [(r["date"], r["meal"]) for r in self.c.get("/api/records").get_json()["records"]]
        self.assertEqual(order, [("2026-01-05", "Breakfast"), ("2026-01-05", "Lunch"), ("2026-01-05", "Dinner"), ("2026-01-06", "Dinner")])

    # ---- bulk operations
    def test_bulk_import_dry_run_duplicates_conflicts_and_invalid_rows(self):
        self.c.post("/api/records", json=self.rec())
        payload = {"records": [self.rec(),                                   # identical duplicate
                               self.rec(prepared=50, leftovers=5),           # conflict (kept existing)
                               self.rec(date="2026-01-06"),                  # new
                               self.rec(date="2026-01-07", meal="Dinner"),   # new
                               {"date": "nope"}]}                            # invalid
        dry = self.c.post("/api/records/bulk", json={**payload, "dry_run": True}).get_json()
        self.assertEqual((dry["would_add"], dry["added"]), (2, 0))
        self.assertEqual(self.count("meal_records"), 1)
        real = self.c.post("/api/records/bulk", json=payload).get_json()
        self.assertEqual((real["added"], real["duplicate_identical"], real["conflict_count"], real["invalid_count"]), (2, 1, 1, 1))
        self.assertEqual(real["records"], 3)
        kept = [r for r in self.c.get("/api/records").get_json()["records"] if r["date"] == "2026-01-05"][0]
        self.assertEqual((kept["prepared"], kept["leftovers"]), (100, 10))

    def test_bulk_batch_and_synthetic_deletes(self):
        self.c.post("/api/records/bulk", json={"records": [self.rec(date="2026-01-05"), self.rec(date="2026-01-06")],
                                               "batch": "b1", "synthetic": True})
        self.c.post("/api/records", json=self.rec(date="2026-02-01"))
        self.assertEqual(self.c.delete("/api/records/batch/b1").get_json()["deleted"], 2)
        self.c.post("/api/records/bulk", json={"records": [self.rec(date="2026-03-01")], "synthetic": True})
        self.assertEqual(self.c.delete("/api/records/synthetic").get_json()["deleted"], 1)
        self.assertEqual(self.count("meal_records"), 1)

    def test_replace_all_is_atomic(self):
        self.c.post("/api/records", json=self.rec())
        bad = self.c.post("/api/records/replace", json={"confirm": "REPLACE_ALL_RECORDS",
                                                        "records": [self.rec(date="2026-02-01"), self.rec(date="2026-02-01")]})
        self.assertEqual(bad.status_code, 400)
        self.assertEqual(self.count("meal_records"), 1)
        no_token = self.c.post("/api/records/replace", json={"records": [self.rec(date="2026-02-01")]})
        self.assertEqual(no_token.status_code, 400)
        ok = self.c.post("/api/records/replace", json={"confirm": "REPLACE_ALL_RECORDS",
                                                       "records": [self.rec(date="2026-02-01"), self.rec(date="2026-02-02")]})
        self.assertEqual((ok.get_json()["removed"], ok.get_json()["stored"]), (1, 2))
        self.assertEqual(self.count("meal_records"), 2)

    # ---- transactions at the database layer
    def test_transaction_rollback_leaves_no_trace(self):
        con = self.connect()
        try:
            con.begin()
            con.execute(INSERT_RECORD, ROW)
            con.rollback()
            self.assertEqual(con.scalar("SELECT COUNT(*) FROM meal_records"), 0)
            con.begin()                                                       # a failing batch must roll back completely
            with self.assertRaises(dbapi.INTEGRITY_ERRORS):
                con.executemany(INSERT_RECORD, [ROW, ROW])
            con.rollback()
            self.assertEqual(con.scalar("SELECT COUNT(*) FROM meal_records"), 0)
            con.begin()
            con.executemany(INSERT_RECORD, [ROW, ("2026-01-06",) + ROW[1:]])
            con.commit()
            self.assertEqual(con.scalar("SELECT COUNT(*) FROM meal_records"), 2)
        finally:
            con.close()

    def test_database_rules_reject_bad_rows(self):
        con = self.connect()
        try:
            bad_rows = [("2026-01-05", "Brunch", 100, 10, "m", None, 0, "t"),      # meal
                        ("2026-01-05", "Lunch", 0, 0, "m", None, 0, "t"),          # prepared >= 1
                        ("2026-01-05", "Lunch", 10, 11, "m", None, 0, "t"),        # leftovers <= prepared
                        ("2026-01-05", "Lunch", 10, 1, "m", None, 2, "t")]         # synthetic in (0,1)
            for row in bad_rows:
                with self.assertRaises(dbapi.INTEGRITY_ERRORS, msg=str(row)):
                    con.execute(INSERT_RECORD, row)
            self.assertEqual(con.scalar("SELECT COUNT(*) FROM meal_records"), 0)
            con.execute(INSERT_RECORD, ROW)                                       # the connection is still usable
            with self.assertRaises(dbapi.INTEGRITY_ERRORS):
                con.execute(INSERT_RECORD, ROW)                                   # UNIQUE (date, meal)
        finally:
            con.close()

    # ---- forecasting, model, history
    def test_forecast_history_is_logged_and_updated_in_place(self):
        self.assertEqual(self.c.post("/api/records/bulk", json={"records": gen(120)}).status_code, 200)
        target = (local_today() + timedelta(days=1)).isoformat()
        self.assertEqual(self.c.post("/api/context", json={"date": target, "kind": "exam", "label": "Test"}).status_code, 201)
        d = self.c.post("/api/forecast", json={"meal": "Lunch", "date": target}).get_json()
        self.assertTrue(d["ok"])
        self.assertEqual(d["calendar_context"][0]["kind"], "exam")
        self.assertEqual(self.count("forecast_log"), 1)
        self.c.post("/api/forecast", json={"meal": "Lunch", "date": target, "attendance_pct": 120})
        self.assertEqual(self.count("forecast_log"), 1)                          # ON CONFLICT ... DO UPDATE
        con = self.connect()
        try:
            row = con.execute("SELECT attendance_pct, expected_demand FROM forecast_log").fetchone()
            self.assertEqual(float(row["attendance_pct"]), 120.0)
            self.assertIsInstance(row["expected_demand"], float)
        finally:
            con.close()
        self.c.post("/api/forecast", json={"meal": "Lunch", "date": local_today().isoformat()})
        self.assertEqual(self.count("forecast_log"), 1)                          # same-day forecasts are not logged
        self.assertTrue(self.c.get("/api/model/report").get_json()["ok"])
        self.assertTrue(self.c.get("/api/insights").get_json()["ok"])

    # ---- anomaly reviews
    def test_anomaly_review_upsert_and_reopen(self):
        self.c.post("/api/records", json=self.rec(date="2026-01-13", prepared=100, leftovers=60))
        body = {"date": "2026-01-13", "meal": "Lunch", "kind": "high_leftovers", "status": "reviewed_ok", "note": "Fest day"}
        self.assertEqual(self.c.post("/api/anomalies/review", json=body).status_code, 200)
        self.assertEqual(self.c.post("/api/anomalies/review", json={**body, "status": "dismissed"}).status_code, 200)
        self.assertEqual(self.count("anomaly_reviews"), 1)
        self.assertEqual(self.c.post("/api/anomalies/review", json={**body, "date": "2026-02-02"}).status_code, 404)
        self.assertEqual(self.c.delete("/api/anomalies/review?date=2026-01-13&meal=Lunch&kind=high_leftovers").status_code, 200)
        self.assertEqual(self.c.delete("/api/anomalies/review?date=2026-01-13&meal=Lunch&kind=high_leftovers").status_code, 404)
        self.assertEqual(self.count("meal_records"), 1)                          # reviews never touch records

    # ---- calendar notes
    def test_day_context_uniqueness_including_blank_meal(self):
        note = {"date": "2026-01-05", "kind": "event", "label": "Fest"}
        self.assertEqual(self.c.post("/api/context", json=note).status_code, 201)
        self.assertEqual(self.c.post("/api/context", json=note).status_code, 409)             # meal NULL twice is still a duplicate
        self.assertEqual(self.c.post("/api/context", json={**note, "meal": "Lunch"}).status_code, 201)
        self.assertEqual(self.c.post("/api/context", json={**note, "meal": "Lunch"}).status_code, 409)
        notes = self.c.get("/api/context").get_json()["notes"]
        self.assertEqual(len(notes), 2)
        nid = notes[0]["id"]
        self.assertEqual(self.c.delete(f"/api/context/{nid}").get_json()["deleted"], 1)
        self.assertEqual(self.c.delete(f"/api/context/{nid}").status_code, 404)

    # ---- recipients
    def test_recipient_names_are_unique_ignoring_case_and_sorted_ignoring_case(self):
        self.assertEqual(self.c.post("/api/recipients", json={"name": "Hope Trust"}).status_code, 201)
        self.assertEqual(self.c.post("/api/recipients", json={"name": "hope trust"}).status_code, 409)
        self.assertEqual(self.c.post("/api/recipients", json={"name": "HOPE TRUST"}).status_code, 409)
        for n in ("beta home", "Alpha Care", "Charlie"):
            self.assertEqual(self.c.post("/api/recipients", json={"name": n}).status_code, 201)
        names = [r["name"] for r in self.c.get("/api/surplus").get_json()["recipients"]]
        self.assertEqual(names, ["Alpha Care", "beta home", "Charlie", "Hope Trust"])
        rid = self.c.get("/api/surplus").get_json()["recipients"][0]["id"]
        self.assertEqual(self.c.delete(f"/api/recipients/{rid}").status_code, 200)
        self.assertEqual(self.c.delete("/api/recipients/99999").status_code, 404)
        self.assertEqual(self.count("recipients"), 4)                                     # deactivated, never deleted

    # ---- surplus food: registration, transitions, events
    GOOD = {"item": "Veg biryani", "portions": 20, "meal": "Lunch", "storage": "hot_held",
            "prepared_time": "00:00", "use_by_time": "23:59", "location": "Back door"}

    def test_surplus_registration_transitions_and_events(self):
        r = self.c.post("/api/surplus", json=self.GOOD)
        self.assertEqual(r.status_code, 201)
        sid = r.get_json()["item"]["id"]
        self.assertEqual(r.get_json()["item"]["status"], "pending_review")
        self.assertEqual(self.c.post("/api/surplus", json=self.GOOD).status_code, 409)             # possible duplicate
        self.assertEqual(self.c.post("/api/surplus", json={**self.GOOD, "confirm_duplicate": True}).status_code, 201)
        act = lambda a, **kw: self.c.post(f"/api/surplus/{sid}/action", json={"action": a, **kw})
        self.assertEqual(act("confirm_safety").status_code, 422)                                   # needs the person's tick
        self.assertEqual(act("confirm_safety", attest=True).get_json()["item"]["status"], "available")
        self.assertEqual(act("reserve").status_code, 422)                                          # needs a recipient
        self.c.post("/api/recipients", json={"name": "Hope Trust", "min_portions": 5})
        rid = self.c.get("/api/surplus").get_json()["recipients"][0]["id"]
        self.assertEqual(act("reserve", recipient_id=rid).get_json()["item"]["status"], "reserved")
        self.assertEqual(act("release").get_json()["item"]["status"], "available")
        self.assertEqual(act("reserve", recipient_id=rid).status_code, 200)
        self.assertEqual(act("collect").get_json()["item"]["status"], "collected")
        self.assertEqual(act("withdraw").status_code, 422)                                         # collected is final
        self.assertEqual(self.c.post("/api/surplus/99999/action", json={"action": "withdraw"}).status_code, 404)
        events = [e["action"] for e in self.c.get(f"/api/surplus/{sid}/events").get_json()["events"]]
        self.assertEqual(events, ["registered", "confirm_safety", "reserve", "release", "reserve", "collect"])
        self.assertEqual(self.count("surplus_events"), 6 + 1)                                      # + the second item's "registered"

    def test_surplus_ids_and_registered_at_confirmation_are_recorded(self):
        r = self.c.post("/api/surplus", json={**self.GOOD, "safety_confirmed": True})
        item = r.get_json()["item"]
        self.assertEqual(item["status"], "available")
        events = self.c.get(f"/api/surplus/{item['id']}/events").get_json()["events"]
        self.assertEqual([e["action"] for e in events], ["registered", "confirm_safety"])
        second = self.c.post("/api/surplus", json={**self.GOOD, "item": "Dal", "safety_confirmed": True}).get_json()["item"]
        self.assertGreater(second["id"], item["id"])

    def test_legacy_browser_register_import_is_additive_and_idempotent(self):
        entries = [{"id": "s1", "item": "Rice", "portions": 10, "location": "Gate", "safeUntil": "18:30",
                    "created": "2026-05-01, 10:00:00", "status": "Available"},
                   {"id": "s2", "item": "Dal", "portions": 5, "status": "Collected"},
                   {"id": "s3", "item": "", "portions": 5}]
        r = self.c.post("/api/surplus/import", json={"entries": entries}).get_json()
        self.assertEqual((r["added"], len(r["invalid"])), (2, 1))
        r2 = self.c.post("/api/surplus/import", json={"entries": entries}).get_json()
        self.assertEqual((r2["added"], r2["skipped_already_imported"]), (0, 2))
        self.assertEqual(self.count("surplus_items"), 2)
        self.assertEqual(self.count("surplus_events"), 2)

    # ---- assistant
    def test_assistant_answers_and_cannot_change_data(self):
        self.c.post("/api/records/bulk", json={"records": gen(60)})
        before = [self.count(t) for t in ("meal_records", "surplus_items", "recipients", "day_context")]
        r = self.c.post("/api/assistant/ask", json={"question": "How much food was left over recently?"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("answer", r.get_json())
        self.assertEqual(before, [self.count(t) for t in ("meal_records", "surplus_items", "recipients", "day_context")])


class SqliteScenarios(Scenarios, SqliteHarness, unittest.TestCase):
    pass


class PostgresScenarios(Scenarios, PostgresHarness, unittest.TestCase):
    pass


# ================================================================ PostgreSQL-only checks
class PostgresOnly(PostgresHarness, unittest.TestCase):
    def setUp(self):
        self.make_backend()

    def test_migration_is_recorded_and_repeatable(self):
        con = self.connect()
        try:
            self.assertEqual(con.scalar("SELECT MAX(version) FROM schema_migrations"), pgm.SCHEMA_VERSION)
            again = pgm.migrate(con)
            self.assertEqual((again["from"], again["to"]), (pgm.SCHEMA_VERSION, pgm.SCHEMA_VERSION))
            self.assertEqual(con.scalar("SELECT COUNT(*) FROM schema_migrations"), 1)
        finally:
            con.close()

    def test_newer_schema_is_refused_without_changes(self):
        con = self.connect()
        try:
            con.execute("INSERT INTO schema_migrations(version,name,applied_at) VALUES(?,?,?)", (99, "future", "t"))
            with self.assertRaises(RuntimeError):
                pgm.migrate(con)
            self.assertEqual(con.scalar("SELECT COUNT(*) FROM schema_migrations"), 2)
        finally:
            con.close()

    def test_failed_migration_rolls_back_completely(self):
        con = self.connect()
        try:
            con.execute("DROP TABLE forecast_log")
            con.execute("DELETE FROM schema_migrations")
            broken = dict(pgm.MIGRATIONS)
            broken[1] = ("broken", list(pgm.MIGRATION_1) + ["CREATE TABLE this_is_not_valid ("])
            with mock.patch.object(pgm, "MIGRATIONS", broken):
                with self.assertRaises(Exception):
                    pgm.migrate(con)
            exists = con.scalar("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = current_schema() "
                                "AND table_name = 'forecast_log'")
            self.assertEqual(exists, 0)                                         # the failed attempt created nothing
            self.assertEqual(con.scalar("SELECT COUNT(*) FROM schema_migrations"), 0)
        finally:
            con.close()

    def test_sequences_are_synchronised_after_importing_explicit_ids(self):
        con = self.connect()
        try:
            con.execute("INSERT INTO recipients(id,name,created_at) VALUES(?,?,?)", (500, "Imported", "t"))
            con.execute("INSERT INTO meal_records(id,date,meal,prepared,leftovers,created_at) VALUES(?,?,?,?,?,?)",
                        (900, "2026-01-05", "Lunch", 10, 1, "t"))
            pgm.sync_sequences(con)
            self.assertEqual(con.insert_id("INSERT INTO recipients(name,created_at) VALUES(?,?)", ("Next", "t")), 501)
            self.assertEqual(con.insert_id(INSERT_RECORD, ("2026-01-06",) + ROW[1:]), 901)
            self.assertEqual(con.insert_id("INSERT INTO day_context(date,kind,created_at) VALUES(?,?,?)", ("2026-01-05", "event", "t")), 1)
        finally:
            con.close()

    def test_every_table_and_index_exists(self):
        con = self.connect()
        try:
            tables = {r["table_name"] for r in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema()")}
            for t in list(pgm.TABLES_WITH_ID) + ["schema_migrations"]:
                self.assertIn(t, tables)
            idx = {r["indexname"] for r in con.execute("SELECT indexname FROM pg_indexes WHERE schemaname = current_schema()")}
            for name in ("idx_records_batch", "idx_surplus_status", "idx_surplus_dup", "idx_surplus_events", "uq_context",
                         "idx_context_date", "uq_recipients_name_ci"):
                self.assertIn(name, idx)
        finally:
            con.close()

    def test_foreign_key_on_surplus_events_is_enforced(self):
        con = self.connect()
        try:
            with self.assertRaises(dbapi.INTEGRITY_ERRORS):
                con.execute("INSERT INTO surplus_events(surplus_id,at,action) VALUES(?,?,?)", (12345, "t", "x"))
        finally:
            con.close()

    def test_for_update_really_locks_the_row(self):
        a, b = self.connect(), self.connect()
        try:
            a.execute("INSERT INTO surplus_items(item,portions,service_date,created_at,updated_at) VALUES(?,?,?,?,?)",
                      ("Rice", 5, "2026-01-05", "t", "t"))
            a.begin()
            a.execute("SELECT * FROM surplus_items WHERE id=?", (1,), for_update=True).fetchone()
            b.execute("SET lock_timeout = '700ms'")
            b.begin()
            with self.assertRaises(dbapi.OPERATIONAL_ERRORS):                       # waits, then gives up: A holds the lock
                b.execute("SELECT * FROM surplus_items WHERE id=?", (1,), for_update=True)
            b.rollback()
            a.commit()
            b.begin()                                                                # lock released: B can take it now
            self.assertIsNotNone(b.execute("SELECT * FROM surplus_items WHERE id=?", (1,), for_update=True).fetchone())
            b.commit()
        finally:
            a.close()
            b.close()

    def test_rows_support_names_indexes_and_dict(self):
        con = self.connect()
        try:
            n = con.execute("SELECT COUNT(*) AS total FROM meal_records").fetchone()
            self.assertEqual((n["total"], n[0]), (0, 0))
            self.assertEqual(dict(n), {"total": 0})
        finally:
            con.close()

    def test_import_from_sqlite_keeps_ids_and_new_rows_continue_after_them(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "src.db")
        with mock.patch.dict(os.environ, {**clean_env(), "ECOPLATE_COOKIE_SECURE": "0"}, clear=True):
            c = appmod.create_app(path).test_client()
        c.post("/api/records/bulk", json={"records": [{"date": "2026-01-05", "meal": "Lunch", "prepared": 100, "leftovers": 10},
                                                      {"date": "2026-01-06", "meal": "Lunch", "prepared": 90, "leftovers": 5}]})
        c.delete("/api/records?date=2026-01-05&meal=Lunch")                      # leaves a gap: remaining id is 2
        c.post("/api/recipients", json={"name": "Hope Trust"})
        good = {"item": "Rice", "portions": 10, "storage": "hot_held", "prepared_time": "00:00", "use_by_time": "23:59"}
        sid = c.post("/api/surplus", json=good).get_json()["item"]["id"]
        c.post(f"/api/surplus/{sid}/action", json={"action": "confirm_safety", "attest": True})
        src = importer.open_sqlite_readonly(path)
        con = self.connect()
        try:
            copied = importer.run_copy(src, con)
            self.assertEqual((copied["meal_records"], copied["surplus_items"], copied["surplus_events"]), (1, 1, 2))
            self.assertEqual(con.scalar("SELECT id FROM meal_records"), 2)
            self.assertEqual(con.insert_id(INSERT_RECORD, ("2026-02-01",) + ROW[1:]), 3)          # continues after the imported id
            self.assertEqual(con.insert_id("INSERT INTO recipients(name,created_at) VALUES(?,?)", ("Next", "t")), 2)
            with self.assertRaises(importer.ImportProblem):                                     # second run refuses: not empty
                importer.run_copy(src, con)
        finally:
            src.close()
            con.close()

    def test_text_dates_keep_their_exact_representation(self):
        con = self.connect()
        try:
            con.execute(INSERT_RECORD, ("2026-01-05", "Lunch", 10, 1, "manual", None, 0, "2026-01-05T10:20:30+00:00"))
            row = con.execute("SELECT date, created_at FROM meal_records").fetchone()
            self.assertEqual((row["date"], row["created_at"]), ("2026-01-05", "2026-01-05T10:20:30+00:00"))
            self.assertIsInstance(row["date"], str)
        finally:
            con.close()


if __name__ == "__main__":
    unittest.main()
