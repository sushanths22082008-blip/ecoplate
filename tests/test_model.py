import math
import os
import sys
import unittest
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fixtures import TODAY, day_after, gen, ml  # noqa: E402


def trained(records):
    bundle, _ = ml.ModelStore().get(records, today=TODAY)
    return bundle


class CleanRecords(unittest.TestCase):
    def test_rejects_bad_rows_and_duplicates(self):
        good = {"date": "2025-03-03", "meal": "Lunch", "prepared": 100, "leftovers": 10}
        raw = [good, dict(good),                                           # duplicate
               {**good, "date": "2025-02-30"},                             # impossible date
               {**good, "meal": "lunch"},                                  # wrong case is not accepted by the model module
               {**good, "prepared": 0}, {**good, "leftovers": 101},        # prepared<1, leftovers>prepared
               {**good, "prepared": 10.5}, {**good, "leftovers": True},    # fraction, bool
               {**good, "date": "2030-01-01"}, {"meal": "Lunch"}]          # future, missing keys
        recs, sk = ml.clean_records(raw, today=TODAY)
        self.assertEqual(len(recs), 1)
        self.assertEqual(sk["duplicate"], 1)
        self.assertEqual(sk["future"], 1)
        self.assertEqual(sk["invalid"], 7)


class ThreeOutcomes(unittest.TestCase):
    def test_insufficient_data_falls_back_honestly(self):
        recs = gen(20)
        b = trained(recs)
        rep = b["report"]
        self.assertEqual(rep["status"], "insufficient_data")
        self.assertFalse(rep["ml_active"])
        self.assertIsNone(rep["metrics"])
        f = ml.forecast(b, "Lunch", day_after(recs))
        self.assertEqual(f["method"], "rule_based_fallback")
        self.assertIn("NOT machine learning", f["method_label"])

    def test_gradient_boosting_active_when_it_beats_baselines(self):
        recs = gen(200, seed=1, kind="weekday")
        b = trained(recs)
        rep = b["report"]
        self.assertEqual(rep["status"], "trained")
        self.assertTrue(rep["ml_active"])
        m = rep["metrics"]
        self.assertLess(m["gbm"]["mae"], m["baselines"][m["best_baseline"]]["mae"])
        for k in ("mae", "rmse"):
            self.assertTrue(math.isfinite(m["gbm"][k]))
        f = ml.forecast(b, "Lunch", day_after(recs))
        self.assertEqual(f["method"], "ml_gradient_boosting")
        self.assertIn("trained ML model", f["method_label"])

    def test_trained_but_not_selected_when_it_underperforms(self):
        recs = gen(200, seed=1, kind="noise")
        b = trained(recs)
        rep = b["report"]
        self.assertEqual(rep["status"], "not_better_than_baseline")
        self.assertFalse(rep["ml_active"])
        self.assertIsNotNone(b["model"])                         # it WAS trained...
        m = rep["metrics"]
        self.assertGreaterEqual(m["gbm"]["mae"], m["baselines"][m["best_baseline"]]["mae"])
        f = ml.forecast(b, "Lunch", day_after(recs))
        self.assertEqual(f["method"], "rule_based_fallback")      # ...but is not used
        self.assertIn("NOT", f["fallback_reason"])

    def test_meal_with_too_little_history_uses_fallback_even_if_model_is_active(self):
        recs = gen(200, seed=1, kind="weekday", meals=["Breakfast", "Lunch"])
        recs += [r for r in gen(200, seed=1, kind="weekday", meals=["Dinner"])][:10]
        b = trained(recs)
        self.assertTrue(b["report"]["ml_active"])
        f = ml.forecast(b, "Dinner", day_after(recs))
        self.assertEqual(f["method"], "rule_based_fallback")
        self.assertIn("not trusted for this meal", f["fallback_reason"])

    def test_no_history_for_meal(self):
        recs = gen(40, meals=["Lunch"])
        f = ml.forecast(trained(recs), "Dinner", day_after(recs))
        self.assertEqual(f["method"], "none")


class Safeguards(unittest.TestCase):
    def test_chronological_split_has_no_overlap(self):
        sp = trained(gen(200))["report"]["split"]
        self.assertLess(sp["train_to"], sp["validation_from"])
        self.assertEqual(sp["method"], "chronological (by date)")

    def test_features_use_only_earlier_records(self):
        recs, _ = ml.clean_records(gen(120), today=TODAY)
        before = ml._build_rows(recs, use_month=False)
        # change ONLY the last record of every meal: no earlier row's features may move
        changed = [dict(r) for r in recs]
        for m in ml.MEALS:
            last = max((i for i, r in enumerate(changed) if r["meal"] == m), key=lambda i: changed[i]["date"])
            changed[last]["prepared"] += 50
            changed[last]["leftovers"] = 0
        after = ml._build_rows(changed, use_month=False)
        self.assertEqual(len(before), len(after))
        diffs = sum(1 for a, b in zip(before, after) if a["x"] != b["x"] and
                    not all((x == y) or (math.isnan(x) and math.isnan(y)) for x, y in zip(a["x"], b["x"])))
        self.assertEqual(diffs, 0)

    def test_retrains_when_records_change_and_caches_when_not(self):
        store, recs = ml.ModelStore(), gen(120)
        b1, c1 = store.get(recs, today=TODAY)
        b2, c2 = store.get(recs, today=TODAY)
        self.assertFalse(c1)
        self.assertTrue(c2)
        recs2 = [dict(r) for r in recs]
        recs2[5]["prepared"] += 1
        b3, c3 = store.get(recs2, today=TODAY)
        self.assertFalse(c3)
        self.assertNotEqual(b1["report"]["fingerprint"], b3["report"]["fingerprint"])

    def test_synthetic_warning(self):
        self.assertIn("SAMPLE", trained(gen(200, synthetic=True))["report"]["data_warning"])
        self.assertIsNone(trained(gen(200, synthetic=False))["report"]["data_warning"])
        mixed = gen(100, synthetic=False)[:150] + gen(100, synthetic=True)[150:]
        self.assertIn("of", trained(mixed)["report"]["data_warning"])

    def test_recommendation_formula(self):
        recs = gen(200)
        b = trained(recs)
        f = ml.forecast(b, "Lunch", day_after(recs), attendance_pct=120, buffer_pct=15, usable_leftovers=7)
        expected = f["base_consumption"] * 1.2
        self.assertAlmostEqual(f["expected_demand"], expected)
        self.assertEqual(f["recommended"], max(0, math.ceil(expected * 1.15 - 7 - 1e-9)))
        self.assertEqual(f["available_total"], f["recommended"] + 7)    # leftovers subtracted exactly once


if __name__ == "__main__":
    unittest.main()
