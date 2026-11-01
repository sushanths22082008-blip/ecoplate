"""EcoPlate surplus-food register: validation, status workflow and review recommendations.

Principles (these are enforced in code, not just described):
* A status only changes when a person records it. EcoPlate never marks food safe, never says a recipient accepted
  or received food on its own, and never invents recipients: recipients exist only if staff entered them.
* "available" means a person ticked the food-safety confirmation. It is the person's statement, stored with who/when.
* Recommendations only say what should be REVIEWED, in what order, and what information is missing.
* Every change writes an audit event in the same transaction.

Statuses:  pending_review -> available -> reserved -> collected
           (any open status) -> withdrawn
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta

MEALS = ("Breakfast", "Lunch", "Dinner")
STORAGE = ("hot_held", "refrigerated", "room_temp", "frozen", "unknown")
STORAGE_LABEL = {"hot_held": "kept hot", "refrigerated": "refrigerated", "room_temp": "room temperature",
                 "frozen": "frozen", "unknown": "not recorded"}
OPEN_STATUSES = ("pending_review", "available", "reserved")
TRANSITIONS = {
    "confirm_safety": ({"pending_review"}, "available"),
    "reserve": ({"available"}, "reserved"),
    "release": ({"reserved"}, "available"),
    "collect": ({"available", "reserved"}, "collected"),
    "withdraw": ({"pending_review", "available", "reserved"}, "withdrawn"),
}
TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")
MAX_BACKDATE_DAYS = 2
MAX_PORTIONS = 10000


def clean_text(v, label, maxlen, required=False):
    if v is None:
        v = ""
    if not isinstance(v, str):
        return None, f"{label} must be text"
    v = " ".join(CTRL_RE.sub(" ", v).split())
    if required and not v:
        return None, f"{label} is required"
    if len(v) > maxlen:
        return None, f"{label} must be at most {maxlen} characters"
    return v, None


def norm_item(name):
    return " ".join(name.casefold().split())


def validate_new(obj, today, now):
    """-> (clean dict, errors). `today`/`now` are local date / datetime."""
    errs = []
    if not isinstance(obj, dict):
        return None, ["Request must be a JSON object"]
    item, e = clean_text(obj.get("item"), "Food item", 80, required=True)
    errs += [e] if e else []
    loc, e = clean_text(obj.get("location"), "Pickup location", 80)
    errs += [e] if e else []
    p = obj.get("portions")
    if isinstance(p, bool) or not isinstance(p, (int, float)) or (isinstance(p, float) and p != int(p)):
        errs.append("Portions must be a whole number")
        p = None
    else:
        p = int(p)
        if not 1 <= p <= MAX_PORTIONS:
            errs.append(f"Portions must be between 1 and {MAX_PORTIONS:,}")
    meal = obj.get("meal") or None
    if meal is not None and meal not in MEALS:
        errs.append("Meal must be Breakfast, Lunch or Dinner (or left blank)")
    storage = obj.get("storage") or "unknown"
    if storage not in STORAGE:
        errs.append("Storage must be one of: " + ", ".join(STORAGE))
    times = {}
    for key, label in (("prepared_time", "Preparation time"), ("use_by_time", "Safe-until time")):
        v = obj.get(key) or None
        if v is not None and (not isinstance(v, str) or not TIME_RE.match(v)):
            errs.append(f"{label} must be a time written HH:MM")
            v = None
        times[key] = v
    sd = obj.get("service_date") or today.isoformat()
    try:
        sdate = date.fromisoformat(sd) if isinstance(sd, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", sd) else None
    except ValueError:
        sdate = None
    if sdate is None:
        errs.append("Service date must be a real date written YYYY-MM-DD")
    elif sdate > today:
        errs.append("Service date cannot be in the future")
    elif sdate < today - timedelta(days=MAX_BACKDATE_DAYS):
        errs.append(f"Service date is more than {MAX_BACKDATE_DAYS} days ago; surplus food from earlier days "
                    "should not be registered for redistribution")
    if times["prepared_time"] and times["use_by_time"] and sdate and times["use_by_time"] <= times["prepared_time"]:
        errs.append("Safe-until time must be later than the preparation time")
    for flag in ("safety_confirmed", "confirm_duplicate"):
        if flag in obj and not isinstance(obj[flag], bool):
            errs.append(f"{flag} must be true or false")
    if errs:
        return None, errs
    return {"item": item, "location": loc, "portions": p, "meal": meal, "storage": storage,
            "service_date": sdate.isoformat(), "prepared_time": times["prepared_time"], "use_by_time": times["use_by_time"],
            "safety_confirmed": obj.get("safety_confirmed") is True,
            "confirm_duplicate": obj.get("confirm_duplicate") is True}, []


def use_by_dt(row):
    """Local naive datetime of the safe-until time, or None if it was not recorded."""
    if not row["use_by_time"]:
        return None
    return datetime.fromisoformat(f"{row['service_date']}T{row['use_by_time']}:00")


def missing_fields(row):
    miss = []
    if not row["use_by_time"]:
        miss.append("safe-until time")
    if row["storage"] == "unknown":
        miss.append("storage condition")
    if not row["prepared_time"]:
        miss.append("preparation time")
    return miss


def is_expired(row, now):
    ub = use_by_dt(row)
    if ub is not None:
        return now.replace(tzinfo=None) > ub
    return date.fromisoformat(row["service_date"]) < now.date()      # no time recorded: older than today is not offered


def item_view(row, now):
    """API representation. `now` is the local datetime."""
    ub = use_by_dt(row)
    mins = None if ub is None else int((ub - now.replace(tzinfo=None)).total_seconds() // 60)
    open_ = row["status"] in OPEN_STATUSES
    return {
        "id": row["id"], "item": row["item"], "meal": row["meal"], "portions": row["portions"],
        "service_date": row["service_date"], "prepared_time": row["prepared_time"], "use_by_time": row["use_by_time"],
        "storage": row["storage"], "location": row["location"], "status": row["status"],
        "safety_confirmed_by": row["safety_confirmed_by"], "safety_confirmed_at": row["safety_confirmed_at"],
        "recipient_id": row["recipient_id"], "created_at": row["created_at"], "updated_at": row["updated_at"],
        "legacy": row["legacy_id"] is not None,
        "minutes_to_use_by": mins if open_ else None,
        "expired": bool(open_ and is_expired(row, now)),
        "missing_information": missing_fields(row) if open_ else [],
    }


def check_transition(action, row, now, recipient=None, override=False):
    """-> (new_status, errors, warnings). Pure: does not touch the database."""
    if action not in TRANSITIONS:
        return None, [f"Unknown action '{action}'"], []
    allowed, new = TRANSITIONS[action]
    if row["status"] not in allowed:
        return None, [f"Cannot {action.replace('_', ' ')}: the item is '{row['status']}'"], []
    errs, warns = [], []
    # Expired food is never offered. (Recording a pickup of food that was already reserved is a record of fact,
    # so it is not blocked.)
    if action == "reserve" or (action == "collect" and row["status"] == "available" and row["legacy_id"] is None):
        if is_expired(row, now):
            errs.append("The recorded safe-until time has passed. Expired food must not be offered; withdraw the entry.")
    if action == "reserve":
        miss = [m for m in missing_fields(row) if m != "preparation time"]
        if miss:
            errs.append("This food cannot be offered to a recipient yet: it has no " + ", no ".join(miss) +
                        ". Withdraw this entry and register it again with that information.")
    if action == "reserve":
        if recipient is None:
            errs.append("Choose one of the recorded recipients.")
        else:
            lo, hi = recipient["min_portions"], recipient["max_portions"]
            if lo and row["portions"] < lo or hi and row["portions"] > hi:
                msg = (f"{recipient['name']} stated a range of {lo or 1}–{hi or 'any'} portions; this item has "
                       f"{row['portions']}.")
                if override:
                    warns.append(msg + " Recorded with override.")
                else:
                    errs.append(msg + " Confirm the override if the recipient agreed.")
    return (None if errs else new), errs, warns


def recommendations(rows, recipients, now):
    """Review suggestions for open items, most urgent first. Says what to REVIEW, never that food is safe."""
    out = []
    for row in rows:
        if row["status"] not in ("pending_review", "available"):
            continue
        v = item_view(row, now)
        miss = missing_fields(row)
        if v["expired"]:
            readiness, msg = "expired", ("The recorded safe-until time has passed. Do not redistribute; withdraw this entry "
                                         "after following your canteen's disposal rules.")
        elif [m for m in miss if m != "preparation time"]:
            readiness, msg = "incomplete", ("Cannot be offered yet: no " +
                                            ", no ".join(m for m in miss if m != "preparation time") +
                                            " recorded. Withdraw and register it again with that information.")
        else:
            readiness = "ready_for_human_review"
            left = v["minutes_to_use_by"]
            msg = (f"{row['portions']} portion(s) of {row['item']} ({STORAGE_LABEL[row['storage']]}) with a recorded "
                   f"safe-until time of {row['use_by_time']} ({left} min from now). A staff member must check the food and "
                   "confirm it is safe before it is offered.")
            if row["status"] == "pending_review":
                msg += " Safety has not been confirmed yet."
        fits = []
        for r in recipients:
            if not r["active"]:
                continue
            lo, hi = r["min_portions"], r["max_portions"]
            if (lo and row["portions"] < lo) or (hi and row["portions"] > hi):
                continue
            fits.append({"id": r["id"], "name": r["name"], "requirements": r["requirements"],
                         "stated_range": [lo, hi]})
        out.append({"id": row["id"], "item": row["item"], "readiness": readiness, "message": msg,
                    "minutes_to_use_by": v["minutes_to_use_by"], "missing_information": [m for m in miss],
                    "recipients_matching_stated_range": fits if readiness == "ready_for_human_review" else [],
                    "recipient_note": ("These recipients' stated portion ranges fit. Their availability is NOT recorded: "
                                       "contact them. Nothing has been offered or accepted."
                                       if fits and readiness == "ready_for_human_review" else None)})
    rank = {"ready_for_human_review": 0, "incomplete": 1, "expired": 2}
    out.sort(key=lambda x: (rank[x["readiness"]], x["minutes_to_use_by"] if x["minutes_to_use_by"] is not None else 10 ** 9))
    return out


def validate_recipient(obj):
    errs = []
    if not isinstance(obj, dict):
        return None, ["Request must be a JSON object"]
    name, e = clean_text(obj.get("name"), "Name", 80, required=True)
    errs += [e] if e else []
    contact, e = clean_text(obj.get("contact"), "Contact", 120)
    errs += [e] if e else []
    req, e = clean_text(obj.get("requirements"), "Requirements", 200)
    errs += [e] if e else []
    rng = {}
    for k in ("min_portions", "max_portions"):
        v = obj.get(k)
        if v in (None, ""):
            rng[k] = None
        elif isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= MAX_PORTIONS:
            errs.append(f"{k} must be a whole number from 1 to {MAX_PORTIONS:,}")
        else:
            rng[k] = v
    if rng.get("min_portions") and rng.get("max_portions") and rng["min_portions"] > rng["max_portions"]:
        errs.append("min_portions cannot be more than max_portions")
    if errs:
        return None, errs
    return {"name": name, "contact": contact, "requirements": req, **rng}, []
