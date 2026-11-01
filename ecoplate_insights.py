"""EcoPlate insights and anomaly detection.

Everything here is plain statistics on the stored meal records. There is no LLM and no randomness. Every
result carries the period, the number of records behind it and the thresholds that were applied, so a person can
check it. Observed facts, possible explanations and recommendations are kept in separate fields.

IMPORTANT reading rules (also shown in the UI):
* An insight describes what the RECORDS show. "Possible explanations" are hypotheses to check, not findings.
* An anomaly is a prompt to look at a record. It is not proof that the record is wrong or that something went wrong.
* Thresholds are practical heuristics for small canteen datasets, not guarantees; ordinary variation will sometimes
  trip them.
"""
from __future__ import annotations

import math
from datetime import date, timedelta
from statistics import mean, median, pstdev

import ecoplate_ml as ml

# ---- thresholds (all echoed in the output so nothing is hidden) ----
TREND_WINDOW = 5               # compare the latest 5 sessions of a meal with the 5 before them
TREND_MIN_PP = 2.0             # ...and require at least this many percentage points of change
TREND_MIN_Z = 2.0              # ...that is at least this many standard errors (indicative only: tiny samples)
MEAL_MIN_N = 10                # sessions per meal needed to compare meals
MEAL_GAP_PP = 3.0              # a meal is called out only if its rate differs from the overall rate by this much
WEEKDAY_MIN_N = 4              # sessions per weekday needed for a weekday pattern
WEEKDAY_GAP_PP = 2.0
PAIRS_MIN = 5                  # logged forecasts with outcomes needed to compare forecasts with actuals
PREP_REVIEW_MIN_N = 10         # sessions needed before suggesting a preparation review
PREP_REVIEW_MIN_RATE = 0.10    # ...and the average leftover rate must be at least 10%

A_REF_WINDOW = 28              # anomaly reference: up to the previous 28 sessions of the same meal
A_REF_MIN = 10                 # ...and at least 10 of them
A_Z = 3.5                      # robust z-score (median / MAD) threshold
A_MIN_RATE_GAP = 0.05          # leftover rate must also be >= 5 percentage points above the reference median
A_MIN_LEFTOVERS = 5            # ...and at least 5 portions, so tiny counts are not flagged
A_RATIO = 2.0                  # prepared quantity more than 2x / less than 0.5x the reference median
A_DEMAND_MIN_REL = 0.20        # eaten quantity must also differ by >= 20% from the reference median
A_SCALE_FLOOR = 0.01           # never divide by less than 1 percentage point of spread
A_WD_MIN = 4                   # quantity checks compare with the same weekday when >= 4 of the last 8 are available


def _rate(r):
    return r["leftovers"] / r["prepared"]


def _cons(r):
    return r["prepared"] - r["leftovers"]


def _pct(x):
    return round(100.0 * x, 1)


def _iso(d):
    return d.isoformat()


def select_records(records, exclude_synthetic=False, since=None):
    """records: output of ml.clean_records (dates are date objects)."""
    out = [r for r in records if not (exclude_synthetic and r["synthetic"])]
    if since:
        out = [r for r in out if r["date"] >= since]
    return sorted(out, key=lambda r: (r["date"], ml.MEALS.index(r["meal"])))


def _period(rs):
    return {"from": _iso(rs[0]["date"]), "to": _iso(rs[-1]["date"])} if rs else {"from": None, "to": None}


def _welch(a, b):
    """(mean_b - mean_a, standard error). SE is None if it cannot be computed."""
    if len(a) < 2 or len(b) < 2:
        return mean(b) - mean(a), None
    va = sum((x - mean(a)) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mean(b)) ** 2 for x in b) / (len(b) - 1)
    se = math.sqrt(va / len(a) + vb / len(b))
    return mean(b) - mean(a), (se if se > 0 else None)


# ------------------------------------------------------------------ insights
def _trend_insights(rs):
    found, short = [], []
    for meal in ml.MEALS:
        m = [r for r in rs if r["meal"] == meal]
        need = 2 * TREND_WINDOW
        if len(m) < need:
            if m:
                short.append({"topic": f"{meal} trend", "have": len(m), "needed": need,
                              "message": f"Only {len(m)} {meal} record(s); at least {need} are needed to compare the latest "
                                         f"{TREND_WINDOW} sessions with the {TREND_WINDOW} before them."})
            continue
        prev, cur = m[-need:-TREND_WINDOW], m[-TREND_WINDOW:]
        pr, cr = [_rate(r) for r in prev], [_rate(r) for r in cur]
        diff, se = _welch(pr, cr)
        diff_pp = diff * 100
        z = diff / se if se else None
        obs = {"meal": meal, "recent_period": _period(cur), "previous_period": _period(prev),
               "recent_sessions": len(cur), "previous_sessions": len(prev),
               "recent_leftover_rate_pct": _pct(mean(cr)), "previous_leftover_rate_pct": _pct(mean(pr)),
               "change_pp": round(diff_pp, 1), "z": None if z is None else round(z, 2),
               "recent_leftovers": sum(r["leftovers"] for r in cur), "recent_prepared": sum(r["prepared"] for r in cur),
               "previous_leftovers": sum(r["leftovers"] for r in prev), "previous_prepared": sum(r["prepared"] for r in prev)}
        thr = {"window": TREND_WINDOW, "min_change_pp": TREND_MIN_PP, "min_z": TREND_MIN_Z}
        clear = abs(diff_pp) >= TREND_MIN_PP and z is not None and abs(z) >= TREND_MIN_Z
        if clear:
            up = diff_pp > 0
            found.append({
                "id": f"trend-{meal.lower()}", "type": "trend", "severity": "attention" if up else "good",
                "title": f"{meal} leftover rate {'rose' if up else 'fell'} over the latest {TREND_WINDOW} sessions",
                "observed": obs, "thresholds": thr,
                "summary": (f"{meal} leftovers averaged {obs['recent_leftover_rate_pct']}% over the latest {TREND_WINDOW} sessions "
                            f"({obs['recent_period']['from']} to {obs['recent_period']['to']}), versus "
                            f"{obs['previous_leftover_rate_pct']}% over the {TREND_WINDOW} before "
                            f"({obs['previous_period']['from']} to {obs['previous_period']['to']})."),
                "possible_explanations": (["Attendance may have been lower than the quantity prepared for.",
                                           "Menu or portion changes may have reduced demand.",
                                           "Preparation quantities may have been raised without a matching rise in demand.",
                                           "Special events, exams or holidays in that period (not recorded here)."] if up else
                                          ["Preparation quantities may have been trimmed.",
                                           "Demand may have risen (check for days when food ran out)."]),
                "recommendation": ("Before changing tomorrow's plan, compare the prepared quantities and attendance for these "
                                   "sessions. This is a pattern in the records, not a confirmed cause." if up else
                                   "Check that the lower leftovers did not come with shortages before trimming further."),
                "data": {"records_used": len(prev) + len(cur), "enough": True},
                "caveat": "Small samples: the test is indicative, not conclusive."})
        else:
            found.append({
                "id": f"trend-{meal.lower()}", "type": "trend", "severity": "info",
                "title": f"No clear change in {meal} leftovers",
                "observed": obs, "thresholds": thr,
                "summary": (f"{meal} leftover rate: {obs['recent_leftover_rate_pct']}% over the latest {TREND_WINDOW} sessions "
                            f"vs {obs['previous_leftover_rate_pct']}% over the {TREND_WINDOW} before. The difference "
                            f"({obs['change_pp']:+.1f} points) is within normal day-to-day variation or below the "
                            f"{TREND_MIN_PP}-point threshold."),
                "possible_explanations": [], "recommendation": None,
                "data": {"records_used": len(prev) + len(cur), "enough": True}, "caveat": None})
    return found, short


def _meal_insight(rs):
    by = {m: [r for r in rs if r["meal"] == m] for m in ml.MEALS}
    ok = {m: v for m, v in by.items() if len(v) >= MEAL_MIN_N}
    if len(ok) < 2:
        have = {m: len(v) for m, v in by.items()}
        return None, {"topic": "Meal comparison", "have": have, "needed": MEAL_MIN_N,
                      "message": f"Need at least {MEAL_MIN_N} records for two or more meals to compare them "
                                 f"(have {', '.join(f'{m} {n}' for m, n in have.items())})."}
    stats = {}
    for m, v in ok.items():
        p, l = sum(r["prepared"] for r in v), sum(r["leftovers"] for r in v)
        stats[m] = {"sessions": len(v), "prepared": p, "leftovers": l, "leftover_rate_pct": _pct(l / p), "period": _period(v)}
    allp = sum(s["prepared"] for s in stats.values())
    alll = sum(s["leftovers"] for s in stats.values())
    overall = alll / allp
    hi = max(stats, key=lambda m: stats[m]["leftover_rate_pct"])
    lo = min(stats, key=lambda m: stats[m]["leftover_rate_pct"])
    gap = stats[hi]["leftover_rate_pct"] - overall * 100
    called = gap >= MEAL_GAP_PP
    return {
        "id": "meal-compare", "type": "meal_comparison", "severity": "attention" if called else "info",
        "title": (f"{hi} has the highest leftover rate" if called else "Meals have similar leftover rates"),
        "observed": {"overall_leftover_rate_pct": _pct(overall), "by_meal": stats, "highest": hi, "lowest": lo,
                     "highest_minus_overall_pp": round(gap, 1)},
        "thresholds": {"min_sessions_per_meal": MEAL_MIN_N, "call_out_gap_pp": MEAL_GAP_PP},
        "summary": ("; ".join(f"{m}: {s['leftover_rate_pct']}% ({s['leftovers']} of {s['prepared']} portions, {s['sessions']} sessions)"
                              for m, s in stats.items()) + f". Overall: {_pct(overall)}%."),
        "possible_explanations": (["Demand for this meal may be harder to predict.", "Portions or menu may suit this meal less.",
                                   "Timing (for example late-day service) may reduce uptake."] if called else []),
        "recommendation": (f"Review {hi} preparation quantities first; it accounts for the largest share of avoidable "
                           "leftovers in the records." if called else None),
        "data": {"records_used": sum(s["sessions"] for s in stats.values()), "enough": True},
        "caveat": "Differences between meals can also come from different menus and diner numbers."}, None


def _weekday_insight(rs):
    mean_by_meal = {}
    for m in ml.MEALS:
        v = [_rate(r) for r in rs if r["meal"] == m]
        if v:
            mean_by_meal[m] = mean(v)
    resid = {d: [] for d in range(7)}
    for r in rs:
        resid[r["date"].weekday()].append(_rate(r) - mean_by_meal[r["meal"]])
    ok = {d: v for d, v in resid.items() if len(v) >= WEEKDAY_MIN_N}
    if len(ok) < 3:
        return None, {"topic": "Weekday pattern", "have": {ml.WEEKDAYS[d]: len(v) for d, v in resid.items() if v},
                      "needed": WEEKDAY_MIN_N,
                      "message": f"Need at least {WEEKDAY_MIN_N} records on each of 3 or more weekdays to look for a weekday pattern."}
    avg = {d: mean(v) for d, v in ok.items()}
    worst = max(avg, key=avg.get)
    rest = [x for d, v in ok.items() if d != worst for x in v]
    diff, se = _welch(rest, ok[worst])
    z = diff / se if se else None
    clear = diff * 100 >= WEEKDAY_GAP_PP and z is not None and z >= 2.0
    table = {ml.WEEKDAYS[d]: {"sessions": len(ok[d]), "vs_meal_average_pp": round(avg[d] * 100, 1)} for d in sorted(ok)}
    return {
        "id": "weekday", "type": "weekday_pattern", "severity": "attention" if clear else "info",
        "title": (f"{ml.WEEKDAYS[worst]} sessions leave more food over than other days" if clear else "No clear weekday pattern"),
        "observed": {"by_weekday": table, "highest_weekday": ml.WEEKDAYS[worst],
                     "highest_vs_other_days_pp": round(diff * 100, 1), "z": None if z is None else round(z, 2),
                     "period": _period(rs)},
        "thresholds": {"min_sessions_per_weekday": WEEKDAY_MIN_N, "min_gap_pp": WEEKDAY_GAP_PP, "min_z": 2.0},
        "summary": ("Leftover rate relative to each meal's own average (percentage points): " +
                    ", ".join(f"{k} {v['vs_meal_average_pp']:+.1f}" for k, v in table.items()) + "."),
        "possible_explanations": (["Attendance may differ on this weekday (timetables, events, weekend travel).",
                                   "Preparation may follow a fixed weekly quantity that does not match demand."] if clear else []),
        "recommendation": (f"Check whether {ml.WEEKDAYS[worst]} quantities are set from a fixed habit rather than recent demand."
                           if clear else None),
        "data": {"records_used": sum(len(v) for v in ok.values()), "enough": True},
        "caveat": "A weekday association is not proof of cause."}, None


def _forecast_vs_actual(rs, pairs):
    """pairs: list of dicts {date, meal, expected_demand, recommended, made_at(date)} from the forecast log."""
    idx = {(r["date"], r["meal"]): r for r in rs}
    used = []
    for p in pairs:
        r = idx.get((p["date"], p["meal"]))
        if r is not None and p["made_at"] < p["date"]:           # forecast made BEFORE the day it predicts
            used.append((p, r))
    if len(used) < PAIRS_MIN:
        return None, {"topic": "Forecast vs actual", "have": len(used), "needed": PAIRS_MIN,
                      "message": f"{len(used)} saved forecast(s) have a recorded outcome; at least {PAIRS_MIN} are needed. "
                                 "Forecasts are saved automatically when you calculate them for a future date."}
    errs = [p["expected_demand"] - _cons(r) for p, r in used]
    over_prep = [r["prepared"] - p["recommended"] for p, r in used]
    mae = mean(abs(e) for e in errs)
    bias = mean(errs)
    return {
        "id": "forecast-vs-actual", "type": "forecast_vs_actual", "severity": "info",
        "title": "Saved forecasts compared with what happened",
        "observed": {"forecasts_compared": len(used), "period": {"from": _iso(min(r["date"] for _, r in used)),
                                                                  "to": _iso(max(r["date"] for _, r in used))},
                     "mean_absolute_error_portions": round(mae, 1), "mean_error_portions": round(bias, 1),
                     "forecast_too_high_count": sum(1 for e in errs if e > 0),
                     "forecast_too_low_count": sum(1 for e in errs if e < 0),
                     "avg_prepared_minus_recommended": round(mean(over_prep), 1)},
        "thresholds": {"min_pairs": PAIRS_MIN},
        "summary": (f"Across {len(used)} forecasts made before the day, expected demand was off by {mae:.1f} portions on "
                    f"average ({'higher' if bias > 0 else 'lower'} than actual by {abs(bias):.1f} on average). "
                    f"Kitchen prepared {mean(over_prep):+.1f} portions versus the recommendation on average."),
        "possible_explanations": [], "recommendation": None,
        "data": {"records_used": len(used), "enough": True},
        "caveat": "Only forecasts calculated before the service date are counted."}, None


def _prep_review(rs):
    found = []
    for meal in ml.MEALS:
        m = [r for r in rs if r["meal"] == meal][-PREP_REVIEW_MIN_N:]
        if len(m) < PREP_REVIEW_MIN_N:
            continue
        rate = mean(_rate(r) for r in m)
        least = min(r["leftovers"] for r in m)
        if rate >= PREP_REVIEW_MIN_RATE and least > 0:
            trial = least // 2
            found.append({
                "id": f"prep-{meal.lower()}", "type": "preparation_opportunity", "severity": "attention",
                "title": f"{meal}: food was left over at every one of the last {PREP_REVIEW_MIN_N} sessions",
                "observed": {"meal": meal, "period": _period(m), "sessions": len(m),
                             "average_leftover_rate_pct": _pct(rate), "smallest_leftover_portions": least,
                             "largest_leftover_portions": max(r["leftovers"] for r in m)},
                "thresholds": {"sessions": PREP_REVIEW_MIN_N, "min_avg_rate_pct": PREP_REVIEW_MIN_RATE * 100},
                "summary": (f"{meal}: at least {least} portions were left over in each of the last {PREP_REVIEW_MIN_N} sessions "
                            f"({m[0]['date']} to {m[-1]['date']}); the average leftover rate was {_pct(rate)}%."),
                "possible_explanations": ["A habitual buffer may be larger than current demand needs."],
                "recommendation": (f"A cautious trial could trim about {trial} portion(s) per session while watching for "
                                   "shortages (this is a rule of thumb: half the smallest recent leftover). "
                                   "Check the Forecast tab for the expected range first." if trial >= 1 else
                                   "Leftovers are small in absolute terms; no trimming is suggested."),
                "data": {"records_used": len(m), "enough": True},
                "caveat": "Leftovers do not show unmet demand. Make changes gradually and keep recording."})
    return found


def build_insights(records, forecast_pairs=(), exclude_synthetic=False, today=None):
    today = today or date.today()
    rs = select_records(records, exclude_synthetic=exclude_synthetic)
    n_syn = sum(1 for r in rs if r["synthetic"])
    out = {"generated_for": _iso(today), "records_used": len(rs), "period": _period(rs),
           "synthetic_records_included": n_syn, "excluded_synthetic": bool(exclude_synthetic),
           "insights": [], "insufficient_data": [], "warnings": []}
    if n_syn:
        out["warnings"].append(f"{n_syn} of {len(rs)} records are sample (synthetic) data. Insights built on them describe "
                               "invented data, not the real canteen.")
    if not rs:
        out["insufficient_data"].append({"topic": "All insights", "message": "No meal records yet."})
        return out
    ins, short = _trend_insights(rs)
    out["insights"] += ins
    out["insufficient_data"] += short
    for fn in (_meal_insight, _weekday_insight):
        i, s = fn(rs)
        if i:
            out["insights"].append(i)
        if s:
            out["insufficient_data"].append(s)
    i, s = _forecast_vs_actual(rs, forecast_pairs)
    if i:
        out["insights"].append(i)
    if s:
        out["insufficient_data"].append(s)
    out["insights"] += _prep_review(rs)
    order = {"attention": 0, "good": 1, "info": 2}
    out["insights"].sort(key=lambda x: order.get(x["severity"], 3))
    return out


# ------------------------------------------------------------------ anomalies
def _mad_scale(xs):
    med = median(xs)
    mad = median(abs(x - med) for x in xs)
    return med, max(1.4826 * mad, A_SCALE_FLOOR)


def detect_anomalies(records, exclude_synthetic=False, reviews=None, since=None):
    """Flags records that are unusual compared with the PREVIOUS up-to-28 sessions of the same meal (so the check
    never uses future data and a record is never compared with itself). Returns a dict with a list of flags.

    reviews: {(date_iso, meal, kind): {...}} from the review log; attached to each flag (marked stale if the record
    changed after it was reviewed)."""
    rs = select_records(records, exclude_synthetic=exclude_synthetic)
    reviews = reviews or {}
    flags = []
    for meal in ml.MEALS:
        m = [r for r in rs if r["meal"] == meal]
        for i, r in enumerate(m):
            if since and r["date"] < since:
                continue
            ref = m[max(0, i - A_REF_WINDOW):i]
            if len(ref) < A_REF_MIN:
                continue
            rr = [_rate(x) for x in ref]
            med_r, sc_r = _mad_scale(rr)
            # Quantities follow a weekly rhythm, so they are compared with the same weekday when enough exist.
            wd = [x for x in m[max(0, i - 70):i] if x["date"].weekday() == r["date"].weekday()][-8:]
            qref, qlabel = (wd, f"previous {len(wd)} {ml.WEEKDAYS[r['date'].weekday()]} {meal} sessions") if len(wd) >= A_WD_MIN \
                else (ref, f"previous {len(ref)} {meal} sessions")
            pp, cc = [x["prepared"] for x in qref], [_cons(x) for x in qref]
            med_p = median(pp)
            med_c, sc_c = _mad_scale(cc)
            refinfo = {"sessions": len(ref), "from": _iso(ref[0]["date"]), "to": _iso(ref[-1]["date"]),
                       "median_leftover_rate_pct": _pct(med_r), "median_prepared": med_p, "median_eaten": med_c,
                       "quantity_comparison": qlabel}
            rate = _rate(r)
            base = {"date": _iso(r["date"]), "meal": meal, "prepared": r["prepared"], "leftovers": r["leftovers"],
                    "leftover_rate_pct": _pct(rate), "synthetic": r["synthetic"], "reference": refinfo}

            def add(kind, title, reason, z=None, possible=()):
                flags.append({**base, "kind": kind, "title": title, "reason": reason,
                              "z": None if z is None else round(z, 1), "possible_explanations": list(possible)})

            z = (rate - med_r) / sc_r
            if z >= A_Z and rate - med_r >= A_MIN_RATE_GAP and r["leftovers"] >= A_MIN_LEFTOVERS:
                add("high_leftovers", "Leftovers much higher than usual",
                    f"{r['leftovers']} of {r['prepared']} portions were left over ({_pct(rate)}%). The previous "
                    f"{len(ref)} {meal} sessions ({refinfo['from']} to {refinfo['to']}) had a median of "
                    f"{_pct(med_r)}%. This is {z:.1f} robust standard deviations above that median (flag threshold {A_Z}).",
                    z, ["Lower attendance than planned.", "Over-preparation.", "A data-entry slip (for example an extra digit)."])
            if r["leftovers"] == r["prepared"]:
                add("all_left_over", "Every prepared portion was recorded as left over",
                    f"All {r['prepared']} portions are recorded as leftovers (0 eaten). That is possible if service was "
                    "cancelled, but it is also a common sign of swapped or mistyped columns.",
                    None, ["Service cancelled or held late.", "Prepared and leftover values swapped."])
            if med_p > 0 and (r["prepared"] >= A_RATIO * med_p or r["prepared"] <= med_p / A_RATIO):
                add("unusual_prepared", "Prepared quantity far from the usual",
                    f"{r['prepared']} portions were prepared; the {qlabel} had a median of "
                    f"{med_p:.0f} ({'at least 2x' if r['prepared'] >= med_p else 'at most half'} as much).",
                    None, ["A special event or a very small service.", "A data-entry slip."])
            eaten = _cons(r)
            zc = (eaten - med_c) / sc_c
            if abs(zc) >= A_Z and med_c > 0 and abs(eaten - med_c) / med_c >= A_DEMAND_MIN_REL:
                add("demand_discrepancy", "Eaten quantity far from the usual",
                    f"{eaten} portions were eaten (prepared minus leftovers); the {qlabel} had a "
                    f"median of {med_c:.0f}. That is {abs(zc):.1f} robust standard deviations "
                    f"{'above' if zc > 0 else 'below'} (flag threshold {A_Z}).",
                    zc, ["Unusual attendance (event, exam, holiday).", "Food ran out (if far below usual with 0 leftovers).",
                         "A data-entry slip."])
            if i >= 2:
                last3 = m[i - 2:i + 1]
                ref3 = m[max(0, i - 2 - A_REF_WINDOW):i - 2]
                if len(ref3) >= A_REF_MIN:
                    r3 = [_rate(x) for x in ref3]
                    med3, sc3 = _mad_scale(r3)
                    avg3 = mean(_rate(x) for x in last3)
                    z3 = (avg3 - med3) / (sc3 / math.sqrt(3))
                    if (z3 >= A_Z and avg3 - med3 >= A_MIN_RATE_GAP and all(_rate(x) > med3 for x in last3)
                            and sum(x["leftovers"] for x in last3) >= 3 * A_MIN_LEFTOVERS):
                        flags.append({**base, "kind": "sudden_change", "title": "Leftover rate has jumped over three sessions",
                                      "reason": (f"The last three {meal} sessions ({_iso(last3[0]['date'])} to {_iso(last3[-1]['date'])}) "
                                                 f"averaged {_pct(avg3)}% leftovers, all above the median of {_pct(med3)}% from "
                                                 f"the {len(ref3)} sessions before them (z {z3:.1f}, flag threshold {A_Z})."),
                                      "z": round(z3, 1),
                                      "reference": {"sessions": len(ref3), "from": _iso(ref3[0]["date"]), "to": _iso(ref3[-1]["date"]),
                                                    "median_leftover_rate_pct": _pct(med3), "median_prepared": median(x["prepared"] for x in ref3),
                                                    "median_eaten": median(_cons(x) for x in ref3)},
                                      "possible_explanations": ["A real change in demand or menu.", "Quantities raised without matching demand."]})
    for f in flags:
        rv = reviews.get((f["date"], f["meal"], f["kind"]))
        if rv:
            f["review"] = {**rv, "stale": (rv["snapshot_prepared"], rv["snapshot_leftovers"]) != (f["prepared"], f["leftovers"])}
        else:
            f["review"] = None
    flags.sort(key=lambda f: (f["date"], ml.MEALS.index(f["meal"])), reverse=True)
    return {"records_checked": len(rs), "flags": flags, "open": sum(1 for f in flags if not f["review"] or f["review"]["stale"]),
            "synthetic_records_included": sum(1 for r in rs if r["synthetic"]),
            "method": {"name": "robust z-score against the previous sessions of the same meal",
                       "reference_window": A_REF_WINDOW, "min_reference": A_REF_MIN, "z_threshold": A_Z,
                       "min_rate_gap_pp": A_MIN_RATE_GAP * 100, "min_leftover_portions": A_MIN_LEFTOVERS,
                       "why_not_ml": ("An ML anomaly detector (for example Isolation Forest) is not used: with a few dozen "
                                      "to a few hundred sessions per meal it would be harder to explain than a median-based "
                                      "threshold and cannot be validated without labelled examples. The review log below "
                                      "collects those labels so the question can be revisited with real data.")},
            "note": "A flag is a prompt to look at a record, not proof that it is wrong. Nothing is changed or deleted automatically."}


# ------------------------------------------------------------------ calendar context
def calendar_effects(records, context_rows, meal, target):
    """Describe flagged days (holiday/exam/event) on the target date and how past flagged days compared.
    This is CONTEXT ONLY: it never changes the forecast number. context_rows: [{date, kind, meal, label}] with
    ISO date strings."""
    tds = target.isoformat()
    today_flags = [c for c in context_rows if c["date"] == tds and (c["meal"] in (None, meal))]
    out = []
    mrs = [r for r in records if r["meal"] == meal and r["date"] < target]
    for c in today_flags:
        same_dates = {date.fromisoformat(x["date"]) for x in context_rows
                      if x["kind"] == c["kind"] and x["date"] < tds and (x["meal"] in (None, meal))}
        flagged = [r for r in mrs if r["date"] in same_dates]
        other = [r for r in mrs if r["date"] not in same_dates and r["date"].weekday() == target.weekday()]
        item = {"kind": c["kind"], "label": c["label"], "flagged_days_in_history": len(flagged)}
        if len(flagged) >= 3 and len(other) >= 4:
            f_avg, o_avg = mean(_cons(r) for r in flagged), mean(_cons(r) for r in other[-12:])
            item.update({"avg_eaten_on_flagged_days": round(f_avg, 1),
                         "avg_eaten_on_other_same_weekdays": round(o_avg, 1),
                         "difference_pct": round(100 * (f_avg - o_avg) / o_avg, 1) if o_avg else None,
                         "message": (f"On {len(flagged)} earlier {c['kind']} day(s) {meal} averaged {f_avg:.0f} portions eaten, "
                                     f"versus {o_avg:.0f} on other {ml.WEEKDAYS[target.weekday()]}s. This is context; it "
                                     "is not applied to the forecast. Adjust the attendance field if you expect a change.")})
        else:
            item["message"] = (f"This date is marked '{c['kind']}'{(' (' + c['label'] + ')') if c['label'] else ''}, but only "
                               f"{len(flagged)} earlier {c['kind']} day(s) have {meal} records (at least 3 needed), so no "
                               "effect can be estimated. It is not applied to the forecast.")
        out.append(item)
    return out
