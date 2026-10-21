"""EcoPlate demand model: Gradient Boosting on the canteen's own meal history.

Predicts meals CONSUMED (= prepared - leftovers) for a meal on a date, then turns
that into a preparation quantity. Everything here is leakage-safe: the features of
any row are computed only from records of the same meal with an EARLIER date.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
from collections import OrderedDict
from datetime import date, datetime, timezone

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

MODEL_VERSION = "1"
MEALS = ["Breakfast", "Lunch", "Dinner"]

# --- data-sufficiency thresholds (practical heuristics, not statistical guarantees) ---
MIN_PRIOR = 3                 # a row needs >= 3 earlier same-meal records to have meaningful history features
MIN_USABLE_ROWS = 60          # total training rows (after warm-up) before any model is trained
MIN_ROWS_PER_MEAL = 28        # ~4 weeks of daily records for the meal being forecast
VAL_FRACTION = 0.2            # most recent 20% of dates held out for validation
MIN_VAL_ROWS = 12
MIN_SPAN_FOR_MONTH_DAYS = 365  # month/season is only a feature once a full year has been seen

# Fixed, conservative hyper-parameters. Nothing is tuned on the validation set, so the
# validation score is an honest out-of-sample estimate. No internal early-stopping
# (sklearn's would use a RANDOM split of the training data).
HGB_PARAMS = dict(learning_rate=0.05, max_iter=100, max_depth=2, min_samples_leaf=20,
                  l2_regularization=1.0, early_stopping=False, random_state=0)


# ------------------------------------------------------------------ records
def clean_records(raw, today=None):
    """Validate raw records from the browser. Returns (records, skipped_counts)."""
    today = today or date.today()
    out, seen = [], set()
    skipped = {"invalid": 0, "duplicate": 0, "future": 0}
    for r in raw if isinstance(raw, list) else []:
        try:
            d = date.fromisoformat(str(r["date"])[:10])
            meal = r["meal"]
            p, l = r["prepared"], r["leftovers"]
            if isinstance(p, bool) or isinstance(l, bool):
                raise ValueError
            if float(p) != int(p) or float(l) != int(l):
                raise ValueError
            p, l = int(p), int(l)
            if meal not in MEALS or d.year < 2000 or p < 1 or l < 0 or l > p:
                raise ValueError
        except (KeyError, TypeError, ValueError, AttributeError):
            skipped["invalid"] += 1
            continue
        if d > today:
            skipped["future"] += 1      # a future date cannot be an observation
            continue
        if (d, meal) in seen:
            skipped["duplicate"] += 1
            continue
        seen.add((d, meal))
        out.append({"date": d, "meal": meal, "prepared": p, "leftovers": l,
                    "synthetic": bool(r.get("synthetic", False))})
    return out, skipped


def _cons(r):
    return r["prepared"] - r["leftovers"]


def _mean(xs):
    xs = list(xs)
    return float(sum(xs) / len(xs)) if xs else float("nan")


# ------------------------------------------------------------------ features
def feature_names(use_month):
    # All history features are RATIOS to the mean of the last 28 services (scale-free), because the model
    # predicts consumption relative to that mean. The mean itself is used only to scale the prediction back.
    return (["meal", "weekday"] + (["month"] if use_month else []) +
            ["last_service_vs_avg28", "avg7_vs_avg28", "same_weekday_avg4_vs_avg28",
             "prepared_avg7_vs_avg28", "leftover_rate_avg7"])


def _features(meal_idx, d, recs, i, use_month):
    """(feature vector, ref) for a service on date d, using ONLY recs[:i] (all earlier dates).
    ref = mean consumption of the last 28 services, used to scale the model's ratio prediction."""
    window = recs[max(0, i - 28):i]
    cons = [_cons(r) for r in window]
    last7 = window[-7:]
    dow = d.weekday()
    same = []
    for j in range(i - 1, max(-1, i - 71), -1):
        if recs[j]["date"].weekday() == dow:
            same.append(_cons(recs[j]))
            if len(same) == 4:
                break
    ref = _mean(cons)
    if not ref > 0:
        ref = float("nan")
    x = ([meal_idx, dow] + ([d.month] if use_month else []) +
         [cons[-1] / ref, _mean(cons[-7:]) / ref, _mean(same) / ref,
          _mean(r["prepared"] for r in last7) / ref,
          _mean(r["leftovers"] / r["prepared"] for r in last7)])
    return x, ref


def legacy_rule(recs, i, d):
    """The ORIGINAL EcoPlate rule, kept as fallback + baseline: mean consumption of the latest 8
    same-weekday records (if >= 4 exist), else of the latest 30 records of that meal."""
    dow = d.weekday()
    same = []
    for j in range(i - 1, -1, -1):
        if recs[j]["date"].weekday() == dow:
            same.append(recs[j])
            if len(same) == 8:
                break
    if len(same) >= 4:
        sample, kind = same, "weekday"
    else:
        sample, kind = recs[max(0, i - 30):i], "recent"
    return _mean(_cons(r) for r in sample), len(sample), kind


def _build_rows(records, use_month):
    rows = []
    for mi, m in enumerate(MEALS):
        recs = sorted((r for r in records if r["meal"] == m), key=lambda r: r["date"])
        for i in range(MIN_PRIOR, len(recs)):
            x, ref = _features(mi, recs[i]["date"], recs, i, use_month)
            if not ref > 0:
                continue                       # no demand history to scale by (all zeros)
            rows.append({"date": recs[i]["date"], "meal": m, "mi": mi, "x": x, "y": _cons(recs[i]),
                         "ref": ref, "recent7": _mean(_cons(r) for r in recs[max(0, i - 7):i]),
                         "rule": legacy_rule(recs, i, recs[i]["date"])[0]})
    rows.sort(key=lambda z: (z["date"], z["mi"]))
    return rows


# ------------------------------------------------------------------ model
def _new_model():
    return HistGradientBoostingRegressor(**HGB_PARAMS)


def _fit(X, y, ref, use_month):
    cat = np.zeros(X.shape[1], dtype=bool)
    cat[:2] = True                              # meal and weekday are categories, not magnitudes
    if use_month:
        cat[2] = True
    m = _new_model()
    m.set_params(categorical_features=cat)
    # Target is consumption relative to the recent 28-service mean: scale-free, so the trees
    # need not extrapolate when a canteen grows or shrinks.
    m.fit(X, y / ref)
    return m


def _predict(model, X, ref):
    return np.maximum(0.0, model.predict(X) * ref)


def _metrics(y, p):
    e = np.asarray(p, float) - np.asarray(y, float)
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2)))}


def _interval(y, p):
    """Empirical 80% band of relative errors (actual-pred)/pred on held-out data."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    ok = p > 0
    if ok.sum() < MIN_VAL_ROWS:
        return None
    rel = (y[ok] - p[ok]) / p[ok]
    return {"low": float(np.quantile(rel, 0.10)), "high": float(np.quantile(rel, 0.90)), "n": int(ok.sum())}


def _perm_importance(model, X, y, ref, names, repeats=10, seed=0):
    rng = np.random.default_rng(seed)
    base = _metrics(y, _predict(model, X, ref))["mae"]
    out = []
    for j, n in enumerate(names):
        d = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            d.append(_metrics(y, _predict(model, Xp, ref))["mae"] - base)
        out.append({"feature": n, "mae_increase": float(np.mean(d))})
    return sorted(out, key=lambda z: -z["mae_increase"])


def train(records, skipped=None, today=None):
    """Train + evaluate. Returns a bundle: {'report': json-safe dict, 'model', 'records', ...}."""
    today = today or date.today()
    skipped = skipped or {"invalid": 0, "duplicate": 0, "future": 0}
    d0 = min((r["date"] for r in records), default=None)
    d1 = max((r["date"] for r in records), default=None)
    span = (d1 - d0).days + 1 if d0 else 0
    use_month = span >= MIN_SPAN_FOR_MONTH_DAYS
    names = feature_names(use_month)
    rows = _build_rows(records, use_month)
    per_meal = {m: sum(1 for r in rows if r["meal"] == m) for m in MEALS}
    n_syn = sum(1 for r in records if r["synthetic"])

    report = {
        "model": "HistGradientBoostingRegressor (scikit-learn)", "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "insufficient_data", "ml_active": False, "reason": "",
        "data": {"valid_records": len(records), "skipped": skipped, "synthetic_records": n_syn,
                 "date_from": d0.isoformat() if d0 else None, "date_to": d1.isoformat() if d1 else None,
                 "span_days": span, "usable_rows": len(rows), "rows_per_meal": per_meal,
                 "zero_leftover_records": sum(1 for r in records if r["leftovers"] == 0)},
        "features": names, "month_feature": use_month,
        "month_note": None if use_month else "Month/season is not used: the history spans less than a year, "
                                             "so seasonal patterns cannot be learned.",
        "hyperparameters": {k: v for k, v in HGB_PARAMS.items() if k != "random_state"},
        "thresholds": {"min_usable_rows": MIN_USABLE_ROWS, "min_rows_per_meal": MIN_ROWS_PER_MEAL,
                       "min_validation_rows": MIN_VAL_ROWS},
        "data_warning": None, "split": None, "metrics": None, "importance": None, "interval": None,
    }
    if n_syn:
        report["data_warning"] = (
            f"{n_syn} of {len(records)} records are marked as SAMPLE (synthetic) data. Any score below describes "
            "how well the model fits that synthetic data. It is not real-world canteen accuracy."
            if n_syn < len(records) else
            "ALL records are marked as SAMPLE (synthetic) data. Any score below describes how well the model "
            "fits synthetic data. It is not real-world canteen accuracy.")

    bundle = {"report": report, "model": None, "records": records, "use_month": use_month,
              "rule_interval": None, "ml_interval": None}

    if len(rows) < MIN_USABLE_ROWS:
        report["reason"] = (f"Not enough history to train: {len(rows)} usable rows, at least {MIN_USABLE_ROWS} "
                            f"needed (each row needs {MIN_PRIOR}+ earlier records of the same meal). "
                            "The rule-based fallback is used.")
        return bundle

    # ---- chronological split by DATE: oldest 80% of dates train, newest 20% validate ----
    dates = sorted({r["date"] for r in rows})
    cut = dates[int(len(dates) * (1 - VAL_FRACTION))]
    tr = [r for r in rows if r["date"] < cut]
    va = [r for r in rows if r["date"] >= cut]
    if len(va) < MIN_VAL_ROWS or not tr:
        report["reason"] = (f"Not enough history to hold out a validation period ({len(va)} validation rows, "
                            f"need {MIN_VAL_ROWS}). The rule-based fallback is used.")
        return bundle

    def arr(rs, k):
        return np.array([r[k] for r in rs], dtype=float)
    Xtr, ytr, rtr = arr(tr, "x"), arr(tr, "y"), arr(tr, "ref")
    Xva, yva, rva = arr(va, "x"), arr(va, "y"), arr(va, "ref")

    m_val = _fit(Xtr, ytr, rtr, use_month)          # trained on train only -> honest validation scores
    p_gbm = _predict(m_val, Xva, rva)
    mean_by_meal = {m: float(np.mean([r["y"] for r in tr if r["meal"] == m] or [np.nan])) for m in MEALS}
    p_mean = np.array([mean_by_meal[r["meal"]] for r in va])
    p_mean = np.where(np.isnan(p_mean), float(np.mean(ytr)), p_mean)
    p_rule, p_r7 = arr(va, "rule"), arr(va, "recent7")
    base_pred = {"rule_based": p_rule, "recent7": p_r7, "train_mean": p_mean}

    met = {"gbm": _metrics(yva, p_gbm),
           "baselines": {"rule_based": _metrics(yva, p_rule), "recent7": _metrics(yva, p_r7),
                         "train_mean": _metrics(yva, p_mean)}}
    labels = {"rule_based": "previous rule-based method", "recent7": "average of last 7 services",
              "train_mean": "historical mean"}
    best = min(met["baselines"], key=lambda k: met["baselines"][k]["mae"])
    bm = met["baselines"][best]["mae"]
    met["best_baseline"], met["best_baseline_label"] = best, labels[best]
    met["improvement_pct"] = float((bm - met["gbm"]["mae"]) / bm * 100) if bm > 0 else 0.0
    met["mean_consumed_validation"] = float(np.mean(yva))
    met["per_meal"] = {}
    for m in MEALS:
        idx = [i for i, r in enumerate(va) if r["meal"] == m]
        if idx:
            met["per_meal"][m] = {"n": len(idx), "gbm_mae": _metrics(yva[idx], p_gbm[idx])["mae"],
                                  "rule_based_mae": _metrics(yva[idx], p_rule[idx])["mae"],
                                  "best_baseline_mae": _metrics(yva[idx], base_pred[best][idx])["mae"]}
    report["metrics"] = met
    report["split"] = {"method": "chronological (by date)", "train_rows": len(tr), "validation_rows": len(va),
                       "train_to": max(r["date"] for r in tr).isoformat(), "validation_from": cut.isoformat(),
                       "validation_to": max(r["date"] for r in va).isoformat()}
    report["importance"] = _perm_importance(m_val, Xva, yva, rva, names)
    bundle["ml_interval"], bundle["rule_interval"] = _interval(yva, p_gbm), _interval(yva, p_rule)

    # ---- production model: refit on ALL usable rows (same fixed settings) ----
    Xall, yall, rall = arr(rows, "x"), arr(rows, "y"), arr(rows, "ref")
    bundle["model"] = _fit(Xall, yall, rall, use_month)

    if met["gbm"]["mae"] < bm:
        report["status"], report["ml_active"] = "trained", True
        report["reason"] = (f"Gradient Boosting beat every baseline on the held-out recent period "
                            f"(MAE {met['gbm']['mae']:.1f} vs {bm:.1f} for the {labels[best]}).")
    else:
        report["status"] = "not_better_than_baseline"
        report["reason"] = (f"Gradient Boosting was trained but, on the held-out recent period, it was not more "
                            f"accurate (MAE {met['gbm']['mae']:.1f}) than the {labels[best]} (MAE {bm:.1f}). "
                            "It is therefore NOT used; the rule-based fallback is used.")
    return bundle


# ------------------------------------------------------------------ forecasting
def _reliability_ml(rep, days_ahead):
    m, low, med = rep["metrics"], False, False
    nv = rep["split"]["validation_rows"]
    reasons = [f"tested on {nv} unseen rows; error {m['improvement_pct']:.0f}% "
               f"{'lower' if m['improvement_pct'] >= 0 else 'higher'} than the best simple baseline"]
    if nv < 30:
        low = True; reasons.append("very few test rows")
    if m["improvement_pct"] < 5:
        low = True; reasons.append("gain over the baseline is too small to rely on")
    elif m["improvement_pct"] < 10:
        med = True
    if nv < 60:
        med = True
    if days_ahead > 14:
        low = True; reasons.append(f"forecast date is {days_ahead} days after the latest record, so recent-history inputs are stale")
    elif days_ahead > 7:
        med = True; reasons.append(f"forecast date is {days_ahead} days after the latest record")
    level = "Low" if low else "Medium" if med else "High"
    return {"level": level, "reasons": reasons}


def forecast(bundle, meal, target, attendance_pct=100.0, buffer_pct=10.0, usable_leftovers=0):
    rep, records = bundle["report"], bundle["records"]
    recs = sorted((r for r in records if r["meal"] == meal and r["date"] < target), key=lambda r: r["date"])
    n = len(recs)
    if not n:
        return {"method": "none", "error": f"There is no history for {meal} before {target.isoformat()}."}
    i = n
    rule_avg, rule_n, rule_kind = legacy_rule(recs, i, target)
    days_ahead = max(0, (target - recs[-1]["date"]).days - 1)

    use_ml, why = False, rep["reason"]
    if rep["ml_active"]:
        if rep["data"]["rows_per_meal"].get(meal, 0) < MIN_ROWS_PER_MEAL:
            why = (f"Only {rep['data']['rows_per_meal'].get(meal, 0)} usable {meal} rows "
                   f"(need {MIN_ROWS_PER_MEAL}); the model is not trusted for this meal.")
        elif n < MIN_PRIOR:
            why = f"Fewer than {MIN_PRIOR} earlier {meal} records."
        else:
            use_ml = True

    if use_ml:
        x, ref = _features(MEALS.index(meal), target, recs, i, bundle["use_month"])
        base = float(_predict(bundle["model"], np.array([x], float), np.array([ref]))[0]) if ref > 0 else rule_avg
        method, label, iv = "ml_gradient_boosting", "Gradient Boosting (trained ML model)", bundle["ml_interval"]
        rel = _reliability_ml(rep, days_ahead)
    else:
        base = rule_avg
        method, label, iv = "rule_based_fallback", "Rule-based fallback (NOT machine learning)", bundle["rule_interval"]
        lvl = "Low" if rule_n < 5 else "Medium" if rule_n < 15 else "High"
        rel = {"level": lvl, "reasons": [f"based on {rule_n} past record(s); no trained model"]}

    att, buf, lo = attendance_pct / 100.0, buffer_pct / 100.0, int(usable_leftovers)
    expected = base * att                         # expected meals eaten at this attendance
    target_qty = expected * (1 + buf)
    net = target_qty - lo                         # usable leftovers are subtracted exactly once
    rec = max(0, math.ceil(net - 1e-9))
    available = rec + lo
    out = {
        "method": method, "method_label": label, "fallback_reason": None if use_ml else why,
        "base_consumption": base, "expected_demand": expected, "target_quantity": target_qty,
        "net_need": net, "recommended": rec, "attendance_pct": attendance_pct, "buffer_pct": buffer_pct,
        "usable_leftovers": lo, "available_total": available,
        "expected_surplus": max(0.0, available - expected),
        "demand_range": None, "surplus_range": None, "shortfall_risk": None,
        "reliability": rel,
        "context": {"history_records": n, "latest_record": recs[-1]["date"].isoformat(), "days_after_latest": days_ahead,
                    "rule_based_reference": rule_avg, "rule_sample_size": rule_n, "rule_sample_kind": rule_kind,
                    "weekday_name": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][target.weekday()]},
        "model_report": rep,
    }
    if iv:
        lo_d, hi_d = expected * (1 + iv["low"]), expected * (1 + iv["high"])
        out["demand_range"] = {"low": lo_d, "high": hi_d, "basis_rows": iv["n"]}
        out["surplus_range"] = {"low": max(0.0, available - hi_d), "high": max(0.0, available - lo_d)}
        out["shortfall_risk"] = max(0.0, hi_d - available)
    return out


# ------------------------------------------------------------------ cache (retrain when data changes)
class ModelStore:
    """Trains on demand and caches by a fingerprint of the cleaned records, so any added, removed or
    changed record automatically triggers a retrain, and unchanged data is never retrained."""

    def __init__(self, max_items=4):
        self._c, self._max, self._lock = OrderedDict(), max_items, threading.Lock()

    def get(self, raw_records, today=None, force=False):
        today = today or date.today()
        records, skipped = clean_records(raw_records, today)
        key_src = json.dumps([MODEL_VERSION, today.isoformat(),
                              sorted((r["date"].isoformat(), r["meal"], r["prepared"], r["leftovers"], r["synthetic"]) for r in records)])
        key = hashlib.sha256(key_src.encode()).hexdigest()
        with self._lock:
            if key in self._c and not force:
                self._c.move_to_end(key)
                return self._c[key], True
            bundle = train(records, skipped, today)
            bundle["report"]["fingerprint"] = key[:12]
            self._c[key] = bundle
            while len(self._c) > self._max:
                self._c.popitem(last=False)
            return bundle, False
