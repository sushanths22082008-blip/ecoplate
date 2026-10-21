"""Deterministic TEST FIXTURES (invented data with fixed dates). They exist only to exercise the three model
outcomes; they say nothing about real canteen accuracy."""
import os
import random
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ecoplate_ml as ml  # noqa: E402

TODAY = date(2026, 6, 1)          # fixed "today" for model-level tests
START = date(2025, 1, 6)          # a Monday
BASE = {"Breakfast": 120, "Lunch": 300, "Dinner": 220}


def gen(days, seed=1, kind="weekday", synthetic=True, meals=ml.MEALS):
    """kind='weekday': strong weekday pattern + drifting level (learnable) -> Gradient Boosting wins.
       kind='noise'  : iid noise around a constant (nothing to learn)       -> Gradient Boosting loses."""
    rnd = random.Random(seed)
    out, level = [], {m: 1.0 for m in ml.MEALS}
    for i in range(days):
        d = START + timedelta(days=i)
        if d.weekday() == 6:
            continue
        for m in meals:
            if kind == "weekday":
                f = [1.0, 1.1, 0.95, 1.05, 1.2, 0.6][d.weekday()]
                level[m] = max(0.6, level[m] + rnd.gauss(0, 0.03))
                mult = f * level[m] * (0.95 + rnd.random() * 0.1)
            else:
                mult = 0.8 + rnd.random() * 0.4
            prep = round(BASE[m] * mult * 1.15)
            cons = round(BASE[m] * mult)
            out.append({"date": d.isoformat(), "meal": m, "prepared": prep,
                        "leftovers": max(0, prep - cons), "synthetic": synthetic})
    return out


def day_after(records):
    return max(date.fromisoformat(r["date"]) for r in records) + timedelta(days=1)
