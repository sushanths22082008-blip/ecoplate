import json
import math
import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fixtures import day_after, gen  # noqa: E402
import app as appmod  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def strict_json(resp):
    """Parse a response body refusing NaN/Infinity (which browsers cannot parse)."""
    def boom(c):
        raise AssertionError("non-finite number in JSON: " + c)
    return json.loads(resp.data, parse_constant=boom)


class ApiCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "t.db")
        self.app = appmod.create_app(self.db)
        self.c = self.app.test_client()

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, recs, **kw):
        body = {"records": [{k: v for k, v in r.items()} for r in recs], **kw}
        r = self.c.post("/api/records/bulk", json=body)
        self.assertEqual(r.status_code, 200, r.data)
        return r.get_json()

    def fc(self, meal, date, **kw):
        return self.c.post("/api/forecast", json={"meal": meal, "date": date, **kw})


class Basics(ApiCase):
    def test_health_and_frontend_served(self):
        h = self.c.get("/api/health").get_json()
        self.assertTrue(h["ok"])
        self.assertEqual(h["app"], "EcoPlate")
        self.assertEqual(h["records"], 0)
        with self.c.get("/") as page:
            self.assertEqual(page.status_code, 200)
            self.assertIn(b"EcoPlate", page.data)
            self.assertIn(b"/api/forecast", page.data)      # the page really calls the backend

    def test_unknown_api_path_is_json_404(self):
        r = self.c.get("/api/nope")
        self.assertEqual(r.status_code, 404)
        self.assertFalse(r.get_json()["ok"])

    def test_database_file_is_not_downloadable(self):
        self.assertEqual(self.c.get("/ecoplate.db").status_code, 404)
        self.assertEqual(self.c.get("/app.py").status_code, 404)


class SingleRecords(ApiCase):
    good = {"date": "2025-03-03", "meal": "Lunch", "prepared": 100, "leftovers": 10}

    def test_add_then_list_and_persist_across_restart(self):
        self.assertEqual(self.c.post("/api/records", json=self.good).status_code, 201)
        c2 = appmod.create_app(self.db).test_client()          # "restart" the server on the same file
        recs = c2.get("/api/records").get_json()["records"]
        self.assertEqual(len(recs), 1)
        self.assertEqual((recs[0]["date"], recs[0]["meal"], recs[0]["prepared"], recs[0]["leftovers"]),
                         ("2025-03-03", "Lunch", 100, 10))

    def test_duplicate_is_rejected_and_original_kept(self):
        self.c.post("/api/records", json=self.good)
        r = self.c.post("/api/records", json={**self.good, "prepared": 999, "leftovers": 1})
        self.assertEqual(r.status_code, 409)
        rec = self.c.get("/api/records").get_json()["records"]
        self.assertEqual(len(rec), 1)
        self.assertEqual(rec[0]["prepared"], 100)              # NOT overwritten

    def test_invalid_records_are_rejected(self):
        bad = [{"date": "2025-02-30"}, {"date": "03/03/2025"}, {"date": "2999-01-01"}, {"date": "1999-01-01"},
               {"meal": "lunch"}, {"meal": "Brunch"}, {"prepared": 0}, {"prepared": -5}, {"leftovers": -1},
               {"leftovers": 101}, {"prepared": 10.5}, {"prepared": "100"}, {"prepared": True},
               {"prepared": 10 ** 9}, {"source": "x" * 40}, {"synthetic": "yes"}]
        for patch in bad:
            r = self.c.post("/api/records", json={**self.good, **patch})
            self.assertEqual(r.status_code, 400, patch)
        self.assertEqual(self.c.get("/api/records").get_json()["count"], 0)

    def test_non_json_body(self):
        self.assertEqual(self.c.post("/api/records", data="nope").status_code, 400)
        self.assertEqual(self.c.post("/api/records", json=[1, 2]).status_code, 400)

    def test_delete_record(self):
        self.c.post("/api/records", json=self.good)
        self.assertEqual(self.c.delete("/api/records?date=2025-03-03&meal=Lunch").status_code, 200)
        self.assertEqual(self.c.delete("/api/records?date=2025-03-03&meal=Lunch").status_code, 404)
        self.assertEqual(self.c.delete("/api/records?date=bad&meal=Lunch").status_code, 400)


class Bulk(ApiCase):
    def test_mixed_batch_reports_everything_and_never_overwrites(self):
        self.c.post("/api/records", json={"date": "2025-03-03", "meal": "Lunch", "prepared": 100, "leftovers": 10})
        rows = [{"date": "2025-03-03", "meal": "Lunch", "prepared": 100, "leftovers": 10},     # identical duplicate
                {"date": "2025-03-03", "meal": "Lunch", "prepared": 200, "leftovers": 5},      # conflicting duplicate
                {"date": "2025-03-04", "meal": "Lunch", "prepared": 100, "leftovers": 10},     # new
                {"date": "2025-03-04", "meal": "Lunch", "prepared": 100, "leftovers": 10},     # repeated inside batch
                {"date": "2025-03-05", "meal": "Nope", "prepared": 100, "leftovers": 10}]      # invalid
        d = self.c.post("/api/records/bulk", json={"records": rows, "batch": "b1", "source": "import"}).get_json()
        self.assertEqual((d["added"], d["duplicate_identical"], d["conflict_count"], d["invalid_count"]), (1, 2, 1, 1))
        self.assertEqual(d["conflicts"][0]["kept"], {"prepared": 100, "leftovers": 10})
        self.assertEqual(d["records"], 2)
        stored = {(r["date"], r["meal"]): r for r in self.c.get("/api/records").get_json()["records"]}
        self.assertEqual(stored[("2025-03-03", "Lunch")]["prepared"], 100)

    def test_dry_run_saves_nothing(self):
        d = self.load(gen(10), dry_run=True)
        self.assertEqual(d["added"], 0)
        self.assertGreater(d["would_add"], 0)
        self.assertEqual(self.c.get("/api/records").get_json()["count"], 0)

    def test_undo_batch_and_remove_synthetic(self):
        self.load(gen(10, synthetic=False), batch="real1", source="import")
        self.load(gen(10, seed=2, synthetic=True)[:0] + [{"date": "2025-06-01", "meal": "Lunch", "prepared": 50, "leftovers": 5}],
                  synthetic=True, source="sample", batch="smp1")
        total = self.c.get("/api/records").get_json()["count"]
        self.assertEqual(self.c.delete("/api/records/batch/smp1").get_json()["deleted"], 1)
        self.load([{"date": "2025-06-01", "meal": "Lunch", "prepared": 50, "leftovers": 5}], synthetic=True)
        self.assertEqual(self.c.delete("/api/records/synthetic").get_json()["deleted"], 1)
        self.assertEqual(self.c.get("/api/records").get_json()["count"], total - 1)

    def test_old_browser_records_migration_payload_keeps_source_and_batch(self):
        d = self.load([{"date": "2025-03-03", "meal": "Dinner", "prepared": 80, "leftovers": 8, "source": "manual"},
                       {"date": "2025-03-04", "meal": "Dinner", "prepared": 80, "leftovers": 8, "source": "import", "batch": "bk3x"}],
                      source="legacy")
        self.assertEqual(d["added"], 2)
        r = {x["date"]: x for x in self.c.get("/api/records").get_json()["records"]}
        self.assertEqual((r["2025-03-03"]["source"], r["2025-03-04"]["source"], r["2025-03-04"]["batch"]), ("manual", "import", "bk3x"))
        again = self.load([{"date": "2025-03-03", "meal": "Dinner", "prepared": 80, "leftovers": 8}], source="legacy")
        self.assertEqual((again["added"], again["duplicate_identical"]), (0, 1))     # re-running migration is harmless


class Replace(ApiCase):
    def test_requires_confirmation_token(self):
        self.load(gen(5))
        n = self.c.get("/api/records").get_json()["count"]
        self.assertEqual(self.c.post("/api/records/replace", json={"records": []}).status_code, 400)
        self.assertEqual(self.c.get("/api/records").get_json()["count"], n)

    def test_is_all_or_nothing(self):
        self.load(gen(5))
        n = self.c.get("/api/records").get_json()["count"]
        r = self.c.post("/api/records/replace", json={"confirm": appmod.REPLACE_TOKEN, "records": [
            {"date": "2025-03-03", "meal": "Lunch", "prepared": 10, "leftovers": 1},
            {"date": "2025-03-03", "meal": "Lunch", "prepared": 10, "leftovers": 1}]})        # duplicate
        self.assertEqual(r.status_code, 400)
        r = self.c.post("/api/records/replace", json={"confirm": appmod.REPLACE_TOKEN, "records": [
            {"date": "2025-03-03", "meal": "Lunch", "prepared": 10, "leftovers": 99}]})        # invalid
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.c.get("/api/records").get_json()["count"], n)

    def test_replace_and_reset(self):
        self.load(gen(5))
        r = self.c.post("/api/records/replace", json={"confirm": appmod.REPLACE_TOKEN, "records": [
            {"date": "2025-03-03", "meal": "Lunch", "prepared": 10, "leftovers": 1, "synthetic": True}]})
        self.assertEqual(r.get_json()["stored"], 1)
        rec = self.c.get("/api/records").get_json()["records"]
        self.assertEqual(len(rec), 1)
        self.assertTrue(rec[0]["synthetic"])                      # the sample flag survives backup/restore
        self.c.post("/api/records/replace", json={"confirm": appmod.REPLACE_TOKEN, "records": []})
        self.assertEqual(self.c.get("/api/records").get_json()["count"], 0)


class Forecast(ApiCase):
    def test_input_validation(self):
        bad = [{"meal": "Brunch"}, {"date": "nope"}, {"date": "2999-01-01"}, {"attendance_pct": 5},
               {"attendance_pct": 301}, {"buffer_pct": -1}, {"buffer_pct": 51}, {"usable_leftovers": -1},
               {"usable_leftovers": 1.5}, {"attendance_pct": "100"}, {"attendance_pct": True}]
        for patch in bad:
            body = {"meal": "Lunch", "date": "2025-08-01", **patch}
            self.assertEqual(self.c.post("/api/forecast", json=body).status_code, 400, patch)

    def test_no_history_is_a_clear_422(self):
        r = self.fc("Lunch", "2025-08-01")
        self.assertEqual(r.status_code, 422)
        self.assertEqual(r.get_json()["code"], "no_history")

    def test_insufficient_data_is_labelled_fallback(self):
        recs = gen(20)
        self.load(recs)
        d = strict_json(self.fc("Lunch", day_after(recs).isoformat()))
        self.assertEqual(d["mode"], "insufficient_data")
        self.assertFalse(d["ml_used"])
        self.assertEqual(d["method"], "rule_based_fallback")
        self.assertIn("NOT machine learning", d["method_label"])
        self.assertIn("rule-based", d["fallback_reason"])

    def test_gradient_boosting_active_end_to_end(self):
        recs = gen(200, kind="weekday")
        self.load(recs, synthetic=True)
        d = strict_json(self.fc("Lunch", day_after(recs).isoformat(), attendance_pct=110, buffer_pct=5, usable_leftovers=3))
        self.assertEqual(d["mode"], "ml_active")
        self.assertTrue(d["ml_used"])
        self.assertEqual(d["method"], "ml_gradient_boosting")
        self.assertIsNone(d["fallback_reason"])
        self.assertEqual(d["recommended"], max(0, math.ceil(d["base_consumption"] * 1.10 * 1.05 - 3 - 1e-9)))
        rep = d["model_report"]
        self.assertTrue(rep["ml_active"])
        self.assertIn("SAMPLE", rep["data_warning"])               # synthetic flag reached the model
        for key in ("method", "method_label", "recommended", "expected_demand", "expected_surplus", "reliability",
                    "demand_range", "surplus_range", "context", "model_report"):
            self.assertIn(key, d)
        self.assertEqual(set(rep["metrics"]["gbm"]), {"mae", "rmse"})

    def test_trained_but_not_selected_end_to_end(self):
        recs = gen(200, kind="noise")
        self.load(recs, synthetic=True)
        d = strict_json(self.fc("Lunch", day_after(recs).isoformat()))
        self.assertEqual(d["mode"], "ml_not_selected")
        self.assertFalse(d["ml_used"])
        self.assertEqual(d["model_report"]["status"], "not_better_than_baseline")

    def test_meal_with_too_little_history(self):
        recs = gen(200, kind="weekday", meals=["Breakfast", "Lunch"]) + gen(200, kind="weekday", meals=["Dinner"])[:10]
        self.load(recs)
        d = strict_json(self.fc("Dinner", day_after(recs).isoformat()))
        self.assertEqual(d["mode"], "insufficient_meal_data")
        self.assertFalse(d["ml_used"])

    def test_retrains_when_a_record_is_added_or_deleted(self):
        recs = gen(120)
        self.load(recs)
        f1 = self.c.get("/api/model/report").get_json()["report"]["fingerprint"]
        self.assertTrue(self.c.get("/api/model/report").get_json()["cached"])      # unchanged data: no retrain
        nxt = day_after(recs).isoformat()
        self.assertEqual(self.c.post("/api/records", json={"date": nxt, "meal": "Lunch", "prepared": 300, "leftovers": 20}).status_code, 201)
        f2 = self.c.get("/api/model/report").get_json()
        self.assertNotEqual(f1, f2["report"]["fingerprint"])
        self.assertFalse(f2["cached"])
        self.c.delete(f"/api/records?date={nxt}&meal=Lunch")
        self.assertEqual(self.c.get("/api/model/report").get_json()["report"]["fingerprint"], f1)

    def test_forced_retrain_endpoint(self):
        self.load(gen(120))
        self.c.get("/api/model/report")
        r = self.c.post("/api/model/train").get_json()
        self.assertTrue(r["ok"])
        self.assertFalse(r["cached"])

    def test_report_on_empty_database(self):
        rep = strict_json(self.c.get("/api/model/report"))["report"]
        self.assertEqual(rep["status"], "insufficient_data")
        self.assertFalse(rep["ml_active"])


class FrontendStatic(unittest.TestCase):
    """Guards against features being lost from index.html. The ids below are every id in the ORIGINAL page."""
    ORIGINAL_IDS = """tabs notice tab-dashboard sRecords sPrepared sLeft sWaste sSurplus sRange chartMeal chartTrend tab-forecast
        fcForm fcMeal fcDate fcAtt fcLeft fcBuf fcResult tab-record recForm recDate recMeal recPrep recLeft recMsg tab-import
        btnTemplate btnSample btnBackup1 impFile impMsg impPreview impResult tab-history hMeal hSearch hCount hBody hPrev hPage
        hNext btnCsv2 hMsg tab-surplus sForm sItem sQty sLoc sUntil sSafe sMsg sBody tab-data btnCsv3 btnJson rsFile dataMsg
        storTxt storBar storWarn undoBox undoTxt btnUndo unmapBox unmapTxt btnUnmap btnReset""".split()

    def setUp(self):
        with open(os.path.join(ROOT, "index.html"), encoding="utf-8") as fh:
            self.html = fh.read()

    def test_all_original_elements_still_exist(self):
        ids = set(re.findall(r'\bid="([^"]+)"', self.html))
        self.assertEqual([i for i in self.ORIGINAL_IDS if i not in ids], [])

    def test_all_original_tabs_still_exist(self):
        for t in ("dashboard", "forecast", "record", "import", "history", "surplus", "data"):
            self.assertIn(f'data-tab="{t}"', self.html)
        self.assertIn('data-tab="model"', self.html)

    def test_original_rule_based_forecast_is_no_longer_computed_in_the_browser(self):
        self.assertNotIn("sample = same; method", self.html)       # the old JS averaging branch is gone
        self.assertIn("/api/forecast", self.html)

    def test_food_safety_confirmation_and_surplus_register_kept(self):
        self.assertIn("I confirm this food is freshly served", self.html)
        self.assertIn("Mark collected", self.html)


if __name__ == "__main__":
    unittest.main()
