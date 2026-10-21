"""EcoPlate backend (Flask + SQLite + scikit-learn).

* Serves the existing frontend (index.html) from the same origin, so there are no CORS problems.
* Stores meal records in a local SQLite file (ecoplate.db) with a UNIQUE(date, meal) rule.
* Trains / evaluates / forecasts through the supplied ecoplate_ml module (HistGradientBoostingRegressor).

Run:  python app.py        then open the address it prints (default http://127.0.0.1:5000)
"""
from __future__ import annotations

import logging
import math
import os
import platform
import re
import socket
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone

from flask import Flask, g, jsonify, request, send_from_directory

import ecoplate_ml as ml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.environ.get("ECOPLATE_DB") or os.path.join(BASE_DIR, "ecoplate.db")
MAX_BULK_ROWS = 50_000
MAX_COUNT = 1_000_000
MAX_FORECAST_DAYS_AHEAD = 365
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SOURCE_RE = re.compile(r"^[A-Za-z_]{1,20}$")
BATCH_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
REPLACE_TOKEN = "REPLACE_ALL_RECORDS"

SCHEMA = """
CREATE TABLE IF NOT EXISTS meal_records (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    date       TEXT    NOT NULL,
    meal       TEXT    NOT NULL CHECK (meal IN ('Breakfast','Lunch','Dinner')),
    prepared   INTEGER NOT NULL CHECK (prepared >= 1),
    leftovers  INTEGER NOT NULL CHECK (leftovers >= 0 AND leftovers <= prepared),
    source     TEXT    NOT NULL DEFAULT 'manual',
    batch      TEXT,
    synthetic  INTEGER NOT NULL DEFAULT 0 CHECK (synthetic IN (0,1)),
    created_at TEXT    NOT NULL,
    UNIQUE (date, meal)
);
CREATE INDEX IF NOT EXISTS idx_records_batch ON meal_records(batch);
"""


# ----------------------------------------------------------------------------- validation
def _as_count(v, label, minimum):
    """Whole number from JSON. Accepts ints and integer-valued floats; rejects bools, strings, NaN."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None, f"{label} must be a whole number"
    if isinstance(v, float) and (not math.isfinite(v) or v != int(v)):
        return None, f"{label} must be a whole number"
    v = int(v)
    if v < minimum:
        return None, f"{label} must be at least {minimum}"
    if v > MAX_COUNT:
        return None, f"{label} is unrealistically large"
    return v, None


def parse_iso_date(s):
    if not isinstance(s, str) or not DATE_RE.match(s.strip()):
        return None
    try:
        d = date.fromisoformat(s.strip())
    except ValueError:
        return None
    return d if d.year >= 2000 else None


def validate_record(obj, today=None, defaults=None):
    """Return (record_dict, errors). Strict: dates are YYYY-MM-DD, never in the future."""
    today = today or date.today()
    defaults = defaults or {}
    errs = []
    if not isinstance(obj, dict):
        return None, ["Record is not an object"]
    d = parse_iso_date(obj.get("date"))
    if d is None:
        errs.append("date must be a real calendar date written YYYY-MM-DD (year 2000 or later)")
    elif d > today:
        errs.append(f"date {d.isoformat()} is in the future")
    meal = obj.get("meal")
    if meal not in ml.MEALS:
        errs.append("meal must be exactly Breakfast, Lunch or Dinner")
    p, e = _as_count(obj.get("prepared"), "prepared", 1)
    if e:
        errs.append(e)
    l, e = _as_count(obj.get("leftovers"), "leftovers", 0)
    if e:
        errs.append(e)
    if p is not None and l is not None and l > p:
        errs.append(f"leftovers ({l}) cannot be more than prepared ({p})")
    source = obj.get("source", defaults.get("source", "manual"))
    if not isinstance(source, str) or not SOURCE_RE.match(source):
        errs.append("source must be 1-20 letters")
    batch = obj.get("batch", defaults.get("batch"))
    if batch is not None and (not isinstance(batch, str) or not BATCH_RE.match(batch)):
        errs.append("batch must be 1-40 letters, digits, - or _")
    syn = obj.get("synthetic", defaults.get("synthetic", False))
    if not isinstance(syn, bool):
        errs.append("synthetic must be true or false")
    if errs:
        return None, errs
    return {"date": d.isoformat(), "meal": meal, "prepared": p, "leftovers": l,
            "source": source, "batch": batch, "synthetic": 1 if syn else 0}, []


# ----------------------------------------------------------------------------- helpers
def json_safe(o):
    """Make a structure strictly JSON-safe (NaN/inf -> None, dates -> ISO strings)."""
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    try:                                    # numpy scalars
        import numpy as np
        if isinstance(o, np.generic):
            return json_safe(o.item())
    except ImportError:                     # pragma: no cover
        pass
    return o


def err(message, status=400, **extra):
    return jsonify(json_safe({"ok": False, "error": message, **extra})), status


def row_to_api(r):
    return {"date": r["date"], "meal": r["meal"], "prepared": r["prepared"], "leftovers": r["leftovers"],
            "source": r["source"], "batch": r["batch"], "synthetic": bool(r["synthetic"])}


# ----------------------------------------------------------------------------- app factory
def create_app(db_path=None):
    app = Flask(__name__, static_folder=None)
    app.config["DB_PATH"] = db_path or DEFAULT_DB
    app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
    app.json.sort_keys = False
    app.extensions["model_store"] = ml.ModelStore()

    # ---- database ----
    def connect():
        con = sqlite3.connect(app.config["DB_PATH"], isolation_level=None, timeout=15)
        con.row_factory = sqlite3.Row
        return con

    init_con = connect()
    try:
        init_con.executescript(SCHEMA)
        try:
            init_con.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:       # pragma: no cover
            pass
    finally:
        init_con.close()

    def db():
        if "db" not in g:
            g.db = connect()
        return g.db

    @app.teardown_appcontext
    def close_db(_exc):
        con = g.pop("db", None)
        if con is not None:
            con.close()

    def count_records():
        return db().execute("SELECT COUNT(*) FROM meal_records").fetchone()[0]

    def model_bundle(force=False):
        rows = db().execute("SELECT date, meal, prepared, leftovers, synthetic FROM meal_records").fetchall()
        raw = [{"date": r["date"], "meal": r["meal"], "prepared": r["prepared"],
                "leftovers": r["leftovers"], "synthetic": bool(r["synthetic"])} for r in rows]
        return app.extensions["model_store"].get(raw, force=force)

    def json_body():
        data = request.get_json(silent=True)
        return data if isinstance(data, dict) else None

    # ---- headers ----
    @app.after_request
    def no_cache(resp):
        resp.headers["Cache-Control"] = "no-store"
        return resp

    # ---- frontend ----
    @app.get("/")
    def index():
        return send_from_directory(BASE_DIR, "index.html")

    @app.get("/favicon.ico")
    def favicon():
        return ("", 204)

    # ---- health ----
    @app.get("/api/health")
    def health():
        import numpy
        import sklearn
        return jsonify({"ok": True, "app": "EcoPlate", "records": count_records(),
                        "database": os.path.basename(app.config["DB_PATH"]),
                        "model": "HistGradientBoostingRegressor", "model_version": ml.MODEL_VERSION,
                        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                                     "numpy": numpy.__version__}})

    # ---- records ----
    @app.get("/api/records")
    def list_records():
        rows = db().execute("SELECT * FROM meal_records ORDER BY date, "
                            "CASE meal WHEN 'Breakfast' THEN 0 WHEN 'Lunch' THEN 1 ELSE 2 END").fetchall()
        return jsonify({"ok": True, "count": len(rows), "records": [row_to_api(r) for r in rows]})

    @app.post("/api/records")
    def add_record():
        body = json_body()
        if body is None:
            return err("Request body must be a JSON object")
        rec, errs = validate_record(body)
        if errs:
            return err("; ".join(errs), errors=errs)
        con = db()
        old = con.execute("SELECT prepared, leftovers FROM meal_records WHERE date=? AND meal=?",
                          (rec["date"], rec["meal"])).fetchone()
        if old:
            return err(f"A record for {rec['meal']} on {rec['date']} already exists "
                       f"(prepared {old['prepared']}, leftovers {old['leftovers']}). Nothing was changed.",
                       409, existing={"prepared": old["prepared"], "leftovers": old["leftovers"]})
        try:
            con.execute("INSERT INTO meal_records(date,meal,prepared,leftovers,source,batch,synthetic,created_at) "
                        "VALUES(?,?,?,?,?,?,?,?)",
                        (rec["date"], rec["meal"], rec["prepared"], rec["leftovers"], rec["source"], rec["batch"],
                         rec["synthetic"], datetime.now(timezone.utc).isoformat(timespec="seconds")))
        except sqlite3.IntegrityError:      # lost a race with a parallel request
            return err("A record for that date and meal already exists. Nothing was changed.", 409)
        return jsonify({"ok": True, "record": {**rec, "synthetic": bool(rec["synthetic"])}, "records": count_records()}), 201

    @app.post("/api/records/bulk")
    def bulk_records():
        """Add many records. Invalid rows and duplicates are skipped and reported; existing rows are NEVER
        overwritten. dry_run=true validates and reports without saving."""
        body = json_body()
        if body is None or not isinstance(body.get("records"), list):
            return err("Body must be a JSON object with a 'records' list")
        items = body["records"]
        if len(items) > MAX_BULK_ROWS:
            return err(f"At most {MAX_BULK_ROWS} records per request")
        defaults = {"source": body.get("source", "import"), "batch": body.get("batch"),
                    "synthetic": body.get("synthetic", False)}
        dry = bool(body.get("dry_run", False))
        con = db()
        existing = {(r["date"], r["meal"]): (r["prepared"], r["leftovers"])
                    for r in con.execute("SELECT date, meal, prepared, leftovers FROM meal_records")}
        seen, to_add, invalid, conflicts, dup_same = {}, [], [], [], 0
        for i, obj in enumerate(items):
            rec, errs = validate_record(obj, defaults=defaults)
            if errs:
                invalid.append({"index": i, "errors": errs})
                continue
            key = (rec["date"], rec["meal"])
            vals = (rec["prepared"], rec["leftovers"])
            prior = existing.get(key) or seen.get(key)
            if prior is not None:
                if prior == vals:
                    dup_same += 1
                else:
                    conflicts.append({"index": i, "date": rec["date"], "meal": rec["meal"],
                                      "prepared": vals[0], "leftovers": vals[1],
                                      "kept": {"prepared": prior[0], "leftovers": prior[1]}})
                continue
            seen[key] = vals
            to_add.append(rec)
        if to_add and not dry:
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            con.execute("BEGIN IMMEDIATE")
            try:
                con.executemany(
                    "INSERT INTO meal_records(date,meal,prepared,leftovers,source,batch,synthetic,created_at) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    [(r["date"], r["meal"], r["prepared"], r["leftovers"], r["source"], r["batch"],
                      r["synthetic"], now) for r in to_add])
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
        return jsonify(json_safe({
            "ok": True, "dry_run": dry, "received": len(items), "added": 0 if dry else len(to_add),
            "would_add": len(to_add), "invalid_count": len(invalid), "invalid": invalid,
            "duplicate_identical": dup_same, "conflict_count": len(conflicts), "conflicts": conflicts,
            "records": count_records()}))

    @app.delete("/api/records")
    def delete_record():
        d, meal = request.args.get("date", ""), request.args.get("meal", "")
        if parse_iso_date(d) is None or meal not in ml.MEALS:
            return err("Provide ?date=YYYY-MM-DD&meal=Breakfast|Lunch|Dinner")
        cur = db().execute("DELETE FROM meal_records WHERE date=? AND meal=?", (d, meal))
        if cur.rowcount == 0:
            return err(f"No {meal} record for {d}", 404)
        return jsonify({"ok": True, "deleted": cur.rowcount, "records": count_records()})

    @app.delete("/api/records/batch/<batch>")
    def delete_batch(batch):
        if not BATCH_RE.match(batch):
            return err("Invalid batch id")
        cur = db().execute("DELETE FROM meal_records WHERE batch=?", (batch,))
        return jsonify({"ok": True, "deleted": cur.rowcount, "records": count_records()})

    @app.delete("/api/records/synthetic")
    def delete_synthetic():
        cur = db().execute("DELETE FROM meal_records WHERE synthetic=1")
        return jsonify({"ok": True, "deleted": cur.rowcount, "records": count_records()})

    @app.post("/api/records/replace")
    def replace_records():
        """Atomically replace ALL records (used by backup-restore and reset). All-or-nothing: if any record is
        invalid or duplicated, nothing changes. Requires confirm == 'REPLACE_ALL_RECORDS'."""
        body = json_body()
        if body is None or not isinstance(body.get("records"), list):
            return err("Body must be a JSON object with a 'records' list")
        if body.get("confirm") != REPLACE_TOKEN:
            return err(f"This replaces every stored record. Send \"confirm\": \"{REPLACE_TOKEN}\" to proceed.")
        items = body["records"]
        if len(items) > MAX_BULK_ROWS:
            return err(f"At most {MAX_BULK_ROWS} records per request")
        recs, problems, seen = [], [], {}
        for i, obj in enumerate(items):
            rec, errs = validate_record(obj, defaults={"source": "restored"})
            if errs:
                problems.append(f"record #{i + 1}: " + "; ".join(errs))
                continue
            k = (rec["date"], rec["meal"])
            if k in seen:
                problems.append(f"record #{i + 1}: repeats {rec['meal']} on {rec['date']} (same as record #{seen[k]})")
                continue
            seen[k] = i + 1
            recs.append(rec)
        if problems:
            return err(f"{len(problems)} problem(s); nothing was changed. First: " + " | ".join(problems[:5]),
                       400, problems=problems[:50])
        con = db()
        before = count_records()
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        con.execute("BEGIN IMMEDIATE")
        try:
            con.execute("DELETE FROM meal_records")
            con.executemany(
                "INSERT INTO meal_records(date,meal,prepared,leftovers,source,batch,synthetic,created_at) "
                "VALUES(?,?,?,?,?,?,?,?)",
                [(r["date"], r["meal"], r["prepared"], r["leftovers"], r["source"], r["batch"],
                  r["synthetic"], now) for r in recs])
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        return jsonify({"ok": True, "removed": before, "stored": len(recs), "records": len(recs)})

    # ---- model ----
    @app.get("/api/model/report")
    def model_report():
        bundle, cached = model_bundle(force=request.args.get("force") == "1")
        return jsonify(json_safe({"ok": True, "cached": cached, "report": bundle["report"]}))

    @app.post("/api/model/train")
    def model_train():
        bundle, cached = model_bundle(force=True)
        return jsonify(json_safe({"ok": True, "cached": cached, "report": bundle["report"]}))

    @app.post("/api/forecast")
    def forecast():
        body = json_body()
        if body is None:
            return err("Request body must be a JSON object")
        meal = body.get("meal")
        if meal not in ml.MEALS:
            return err("meal must be Breakfast, Lunch or Dinner")
        target = parse_iso_date(body.get("date"))
        if target is None:
            return err("date must be a real calendar date written YYYY-MM-DD")
        if target > date.today() + timedelta(days=MAX_FORECAST_DAYS_AHEAD):
            return err(f"Forecast date is more than {MAX_FORECAST_DAYS_AHEAD} days ahead")

        def num(key, default, lo, hi, label):
            v = body.get(key, default)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
                return None, f"{label} must be a number between {lo} and {hi}"
            return float(v), None

        att, e1 = num("attendance_pct", 100, 10, 300, "Attendance")
        buf, e2 = num("buffer_pct", 10, 0, 50, "Safety buffer")
        lo_raw = body.get("usable_leftovers", 0)
        lo, e3 = _as_count(lo_raw, "usable_leftovers", 0)
        if e1 or e2 or e3:
            return err("; ".join(x for x in (e1, e2, e3) if x))

        bundle, cached = model_bundle()
        res = ml.forecast(bundle, meal, target, att, buf, lo)
        if res.get("method") == "none":
            return err(res["error"], 422, code="no_history")
        rep = res["model_report"]
        if res["method"] == "ml_gradient_boosting":
            mode = "ml_active"
        elif rep["status"] == "not_better_than_baseline":
            mode = "ml_not_selected"
        elif rep["status"] == "trained":
            mode = "insufficient_meal_data"      # model passed validation overall but not enough rows for THIS meal
        else:
            mode = "insufficient_data"
        res.update({"ok": True, "mode": mode, "ml_used": res["method"] == "ml_gradient_boosting",
                    "model_cached": cached, "meal": meal, "date": target.isoformat()})
        return jsonify(json_safe(res))

    # ---- errors (always JSON under /api) ----
    @app.errorhandler(404)
    def not_found(_e):
        if request.path.startswith("/api/"):
            return err("Unknown API endpoint", 404)
        return ("Not found", 404)

    @app.errorhandler(405)
    def bad_method(_e):
        return err("Method not allowed for this endpoint", 405)

    @app.errorhandler(413)
    def too_big(_e):
        return err("Request is too large (limit 25 MB)", 413)

    @app.errorhandler(Exception)
    def crash(e):
        from werkzeug.exceptions import HTTPException
        if isinstance(e, HTTPException):
            return err(e.description, e.code or 500)
        logging.getLogger("ecoplate").exception("Unhandled error")
        return err("Internal server error. See the server console for details.", 500)

    return app


# ----------------------------------------------------------------------------- launcher
def _free_port(host, start):
    for port in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise SystemExit(f"No free port found between {start} and {start + 19}. Set ECOPLATE_PORT to another number.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    host = os.environ.get("ECOPLATE_HOST", "127.0.0.1")
    wanted = int(os.environ.get("ECOPLATE_PORT", "5000"))
    port = _free_port(host, wanted)
    application = create_app()
    url = f"http://{host}:{port}"
    print("\n" + "=" * 62)
    print("  EcoPlate is running.")
    print(f"  Open this address in your browser:  {url}")
    if port != wanted:
        print(f"  (port {wanted} was busy, so {port} is being used)")
    print(f"  Database file: {application.config['DB_PATH']}")
    print("  Press Ctrl+C in this window to stop the server.")
    print("=" * 62 + "\n")
    if "--open" in sys.argv:
        import threading
        import webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    application.run(host=host, port=port, debug=False, threaded=True)
