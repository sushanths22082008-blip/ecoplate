"""New EcoPlate API routes (auth, insights, anomalies, assistant, surplus, recipients, calendar notes).

Registered from app.create_app(). Every route here is either read-only or writes ONLY to its own table
(surplus_*, recipients, day_context, anomaly_reviews, forecast_log). None of them can modify or delete a meal record.
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime

from flask import current_app, jsonify, request

import ecoplate_assistant as assistant
import ecoplate_dbapi as dbapi
import ecoplate_insights as insights
import ecoplate_ml as ml
import ecoplate_surplus as sp
from ecoplate_security import (COOKIE, SESSION_SECONDS, check_login, client_ip, current_role, make_token, require)
from ecoplate_util import local_now, local_today

log = logging.getLogger("ecoplate")
ANOMALY_KINDS = ("high_leftovers", "all_left_over", "unusual_prepared", "demand_discrepancy", "sudden_change")
REVIEW_STATUSES = ("reviewed_ok", "confirmed_error", "dismissed")
CONTEXT_KINDS = ("holiday", "exam", "event", "other")
MAX_CONTEXT_ROWS = 5000
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def register(app, db, load_records, model_bundle, err, json_body, json_safe):
    S = lambda: current_app.extensions["ecoplate_settings"]          # noqa: E731
    limiter = app.extensions["rate_limiter"]
    budget = app.extensions["llm_budget"]

    def iso_or_none(s):
        if not isinstance(s, str) or not DATE_RE.match(s):
            return None
        try:
            return date.fromisoformat(s)
        except ValueError:
            return None

    def role_name():
        return current_role(S()) or "staff"

    # ------------------------------------------------------------ auth
    @app.get("/api/auth/status")
    def auth_status():
        s = S()
        return jsonify({"ok": True, "auth_enabled": s.auth_enabled, "role": current_role(s),
                        "admin_separate": bool(s.admin_code), "llm_configured": s.llm_configured})

    @app.post("/api/auth/login")
    def auth_login():
        s = S()
        if not s.auth_enabled:
            return jsonify({"ok": True, "auth_enabled": False, "role": "admin"})
        okrl, wait = limiter.allow("login", client_ip(), 10, 900)
        if not okrl:
            return err(f"Too many login attempts. Try again in {wait} seconds.", 429)
        body = json_body()
        role = check_login(s, (body or {}).get("code"))
        if role is None:
            return err("That access code is not correct.", 401)
        resp = jsonify({"ok": True, "auth_enabled": True, "role": role})
        resp.set_cookie(COOKIE, make_token(s, role), max_age=SESSION_SECONDS, httponly=True,
                        samesite="Lax", secure=s.cookie_secure)
        return resp

    @app.post("/api/auth/logout")
    def auth_logout():
        resp = jsonify({"ok": True})
        resp.delete_cookie(COOKIE)
        return resp

    # ------------------------------------------------------------ helpers
    def forecast_pairs():
        rows = db().execute("SELECT target_date, meal, made_at, expected_demand, recommended FROM forecast_log").fetchall()
        out = []
        for r in rows:
            try:
                out.append({"date": date.fromisoformat(r["target_date"]), "meal": r["meal"],
                            "made_at": date.fromisoformat(r["made_at"][:10]),
                            "expected_demand": r["expected_demand"], "recommended": r["recommended"]})
            except ValueError:
                continue
        return out

    def review_map():
        out = {}
        for r in db().execute("SELECT * FROM anomaly_reviews"):
            out[(r["date"], r["meal"], r["kind"])] = {"status": r["status"], "note": r["note"], "reviewed_at": r["reviewed_at"],
                                                      "snapshot_prepared": r["snapshot_prepared"],
                                                      "snapshot_leftovers": r["snapshot_leftovers"]}
        return out

    def truthy(name):
        return request.args.get(name) in ("1", "true", "yes")

    # ------------------------------------------------------------ insights
    @app.get("/api/insights")
    @require("staff")
    def get_insights():
        recs = load_records()
        res = insights.build_insights(recs, forecast_pairs(), exclude_synthetic=truthy("exclude_synthetic"), today=local_today())
        return jsonify(json_safe({"ok": True, **res}))

    # ------------------------------------------------------------ anomalies
    @app.get("/api/anomalies")
    @require("staff")
    def get_anomalies():
        try:
            days = max(1, min(3650, int(request.args.get("days", "90"))))
        except ValueError:
            return err("days must be a whole number")
        since = local_today().fromordinal(local_today().toordinal() - days)
        res = insights.detect_anomalies(load_records(), exclude_synthetic=truthy("exclude_synthetic"),
                                        reviews=review_map(), since=since)
        res["period_days"] = days
        return jsonify(json_safe({"ok": True, **res}))

    @app.post("/api/anomalies/review")
    @require("staff")
    def review_anomaly():
        body = json_body()
        if body is None:
            return err("Request body must be a JSON object")
        d, meal, kind, status = body.get("date"), body.get("meal"), body.get("kind"), body.get("status")
        if iso_or_none(d) is None or meal not in ml.MEALS or kind not in ANOMALY_KINDS or status not in REVIEW_STATUSES:
            return err("Provide date (YYYY-MM-DD), meal, kind and status (" + ", ".join(REVIEW_STATUSES) + ")")
        note, e = sp.clean_text(body.get("note"), "Note", 300)
        if e:
            return err(e)
        rec = db().execute("SELECT prepared, leftovers FROM meal_records WHERE date=? AND meal=?", (d, meal)).fetchone()
        if not rec:
            return err(f"No {meal} record for {d}", 404)
        db().execute("INSERT INTO anomaly_reviews(date,meal,kind,status,note,snapshot_prepared,snapshot_leftovers,reviewed_at) "
                     "VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(date,meal,kind) DO UPDATE SET status=excluded.status, "
                     "note=excluded.note, snapshot_prepared=excluded.snapshot_prepared, "
                     "snapshot_leftovers=excluded.snapshot_leftovers, reviewed_at=excluded.reviewed_at",
                     (d, meal, kind, status, note, rec["prepared"], rec["leftovers"], local_now().isoformat(timespec="seconds")))
        return jsonify({"ok": True, "message": "Review saved. The meal record itself was not changed."})

    @app.delete("/api/anomalies/review")
    @require("staff")
    def reopen_anomaly():
        d, meal, kind = request.args.get("date", ""), request.args.get("meal", ""), request.args.get("kind", "")
        cur = db().execute("DELETE FROM anomaly_reviews WHERE date=? AND meal=? AND kind=?", (d, meal, kind))
        if cur.rowcount == 0:
            return err("No review found for that flag", 404)
        return jsonify({"ok": True, "message": "Flag re-opened."})

    # ------------------------------------------------------------ calendar notes
    @app.get("/api/context")
    @require("staff")
    def list_context():
        rows = db().execute("SELECT id,date,kind,meal,label FROM day_context ORDER BY date DESC, id DESC LIMIT 500").fetchall()
        return jsonify({"ok": True, "notes": [dict(r) for r in rows]})

    @app.post("/api/context")
    @require("staff")
    def add_context():
        body = json_body()
        if body is None:
            return err("Request body must be a JSON object")
        d = iso_or_none(body.get("date"))
        kind, meal = body.get("kind"), body.get("meal") or None
        label, e = sp.clean_text(body.get("label"), "Label", 80)
        if d is None or d.year < 2000 or kind not in CONTEXT_KINDS or (meal is not None and meal not in ml.MEALS) or e:
            return err(e or "Provide date (YYYY-MM-DD), kind (" + ", ".join(CONTEXT_KINDS) + "), optional meal and label")
        if db().scalar("SELECT COUNT(*) FROM day_context") >= MAX_CONTEXT_ROWS:
            return err("Too many calendar notes stored")
        try:
            db().execute("INSERT INTO day_context(date,kind,meal,label,created_at) VALUES(?,?,?,?,?)",
                         (d.isoformat(), kind, meal, label, local_now().isoformat(timespec="seconds")))
        except dbapi.INTEGRITY_ERRORS:
            return err("That calendar note already exists.", 409)
        return jsonify({"ok": True}), 201

    @app.delete("/api/context/<int:cid>")
    @require("staff")
    def delete_context(cid):
        cur = db().execute("DELETE FROM day_context WHERE id=?", (cid,))
        return jsonify({"ok": True, "deleted": cur.rowcount}) if cur.rowcount else err("No such note", 404)

    # ------------------------------------------------------------ surplus
    def surplus_rows(where="", args=()):
        return db().execute("SELECT * FROM surplus_items " + where, args).fetchall()

    def recipients_rows():
        return db().execute("SELECT * FROM recipients ORDER BY active DESC, lower(name)").fetchall()

    def surplus_summary():
        now = local_now()
        rows = surplus_rows("ORDER BY id DESC LIMIT 1000")
        counts = {}
        for r in rows:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        return {"rows": rows, "counts": counts,
                "recommendations": sp.recommendations(rows, [dict(x) for x in recipients_rows()], now)}

    @app.get("/api/surplus")
    @require("staff")
    def list_surplus():
        now = local_now()
        s = surplus_summary()
        avail = sum(r["portions"] for r in s["rows"] if r["status"] == "available")
        return jsonify({"ok": True, "items": [sp.item_view(r, now) for r in s["rows"]], "counts": s["counts"],
                        "available_portions": avail, "recommendations": s["recommendations"],
                        "recipients": [{k: r[k] for k in ("id", "name", "contact", "requirements", "min_portions",
                                                          "max_portions", "active")} for r in recipients_rows()],
                        "server_time": now.isoformat(timespec="seconds")})

    def log_event(con, sid, action, old, new, note=""):
        con.execute("INSERT INTO surplus_events(surplus_id,at,action,from_status,to_status,actor,note) VALUES(?,?,?,?,?,?,?)",
                    (sid, local_now().isoformat(timespec="seconds"), action, old, new, role_name(), note))

    @app.post("/api/surplus")
    @require("staff")
    def add_surplus():
        body = json_body()
        now = local_now()
        rec, errs = sp.validate_new(body, now.date(), now)
        if errs:
            return err("; ".join(errs), errors=errs)
        con = db()
        dups = [r for r in surplus_rows("WHERE service_date=? AND status!='withdrawn'", (rec["service_date"],))
                if sp.norm_item(r["item"]) == sp.norm_item(rec["item"])]
        if dups and not rec["confirm_duplicate"]:
            return err(f"'{rec['item']}' is already in the register for {rec['service_date']} "
                       f"({', '.join('#%d: %d portions, %s' % (d['id'], d['portions'], d['status']) for d in dups)}). "
                       "Register it again only if this is a separate batch.", 409, code="possible_duplicate",
                       existing=[{"id": d["id"], "portions": d["portions"], "status": d["status"]} for d in dups])
        stamp = now.isoformat(timespec="seconds")
        status = "available" if rec["safety_confirmed"] else "pending_review"
        con.begin()
        try:
            sid = con.insert_id(
                "INSERT INTO surplus_items(item,meal,portions,service_date,prepared_time,use_by_time,storage,location,status,"
                "safety_confirmed_by,safety_confirmed_at,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (rec["item"], rec["meal"], rec["portions"], rec["service_date"], rec["prepared_time"], rec["use_by_time"],
                 rec["storage"], rec["location"], status,
                 f"{role_name()} (tick-box at registration)" if rec["safety_confirmed"] else None,
                 stamp if rec["safety_confirmed"] else None, stamp, stamp))
            log_event(con, sid, "registered", None, "pending_review")
            if rec["safety_confirmed"]:
                log_event(con, sid, "confirm_safety", "pending_review", "available", "Confirmed by a person at registration")
            con.commit()
        except Exception:
            con.rollback()
            raise
        row = con.execute("SELECT * FROM surplus_items WHERE id=?", (sid,)).fetchone()
        return jsonify({"ok": True, "item": sp.item_view(row, now)}), 201

    @app.post("/api/surplus/<int:sid>/action")
    @require("staff")
    def surplus_action(sid):
        body = json_body()
        if body is None:
            return err("Request body must be a JSON object")
        action = body.get("action")
        note, e = sp.clean_text(body.get("note"), "Note", 200)
        if e:
            return err(e)
        now = local_now()
        con = db()
        con.begin()
        try:
            # FOR UPDATE (PostgreSQL) locks this row until COMMIT, so two people cannot change the same item at once.
            row = con.execute("SELECT * FROM surplus_items WHERE id=?", (sid,), for_update=True).fetchone()
            if not row:
                con.rollback()
                return err("No such surplus item", 404)
            if action == "confirm_safety" and body.get("attest") is not True:
                con.rollback()
                return err("A person must tick the food-safety confirmation (attest: true). EcoPlate cannot decide this.", 422)
            recipient = None
            rid = body.get("recipient_id")
            if rid is not None:
                if isinstance(rid, bool) or not isinstance(rid, int):
                    con.rollback()
                    return err("recipient_id must be a number")
                recipient = con.execute("SELECT * FROM recipients WHERE id=? AND active=1", (rid,)).fetchone()
                if recipient is None:
                    con.rollback()
                    return err("That recipient is not recorded (or is inactive).", 404)
            new, errs, warns = sp.check_transition(action, row, now, dict(recipient) if recipient else None,
                                                   override=body.get("override") is True)
            if errs:
                con.rollback()
                return err(" ".join(errs), 422, errors=errs)
            sets = {"status": new, "updated_at": now.isoformat(timespec="seconds")}
            if action == "confirm_safety":
                sets.update(safety_confirmed_by=f"{role_name()} (person confirmed)", safety_confirmed_at=sets["updated_at"])
            if action == "reserve":
                sets["recipient_id"] = recipient["id"]
            if action == "release":
                sets["recipient_id"] = None
            if action == "collect" and recipient is not None:
                sets["recipient_id"] = recipient["id"]
            cur = con.execute(f"UPDATE surplus_items SET {', '.join(k + '=?' for k in sets)} WHERE id=? AND status=?",
                              (*sets.values(), sid, row["status"]))
            if cur.rowcount != 1:
                con.rollback()
                return err("This item was changed by someone else. Refresh and try again.", 409)
            log_event(con, sid, action, row["status"], new,
                      (note + " " if note else "") + " ".join(warns) + (f" recipient={recipient['name']}" if recipient else ""))
            con.commit()
        except Exception:
            try:
                con.rollback()
            except dbapi.OPERATIONAL_ERRORS:
                pass
            raise
        row = con.execute("SELECT * FROM surplus_items WHERE id=?", (sid,)).fetchone()
        msg = {"collect": "Recorded as collected by staff. EcoPlate cannot verify receipt.",
               "reserve": "Recorded as reserved for the chosen recipient. This is not a confirmation that they accepted."
               }.get(action, "Saved.")
        return jsonify({"ok": True, "message": msg, "warnings": warns, "item": sp.item_view(row, now)})

    @app.get("/api/surplus/<int:sid>/events")
    @require("staff")
    def surplus_events(sid):
        rows = db().execute("SELECT at,action,from_status,to_status,actor,note FROM surplus_events WHERE surplus_id=? "
                            "ORDER BY id", (sid,)).fetchall()
        return jsonify({"ok": True, "events": [dict(r) for r in rows]})

    @app.post("/api/surplus/import")
    @require("staff")
    def import_surplus():
        """One-time, additive copy of the old browser-only register. Never overwrites; skips ids already imported."""
        body = json_body()
        if body is None or not isinstance(body.get("entries"), list):
            return err("Body must be a JSON object with an 'entries' list")
        entries = body["entries"]
        if len(entries) > 2000:
            return err("At most 2000 entries per request")
        now = local_now()
        stamp = now.isoformat(timespec="seconds")
        invalid, added, skipped, ok_rows = [], 0, 0, []
        have = {r["legacy_id"] for r in surplus_rows("WHERE legacy_id IS NOT NULL")}
        same = {(sp.norm_item(r["item"]), r["portions"], r["service_date"]) for r in surplus_rows("")}
        for i, o in enumerate(entries):
            if not isinstance(o, dict):
                invalid.append({"index": i, "errors": ["not an entry"]}); continue
            lid = o.get("id")
            if not isinstance(lid, str) or not 0 < len(lid) <= 60:
                invalid.append({"index": i, "errors": ["missing id"]}); continue
            if lid in have:
                skipped += 1; continue
            item, e1 = sp.clean_text(o.get("item"), "Food item", 80, required=True)
            loc, e2 = sp.clean_text(o.get("location"), "Location", 80)
            p = o.get("portions")
            ub = o.get("safeUntil") or None
            errs = [x for x in (e1, e2) if x]
            if isinstance(p, bool) or not isinstance(p, (int, float)) or int(p) != p or not 1 <= int(p) <= sp.MAX_PORTIONS:
                errs.append("portions must be a whole number from 1 to 10,000")
            if ub is not None and (not isinstance(ub, str) or not sp.TIME_RE.match(ub)):
                ub = None
            status = "collected" if o.get("status") == "Collected" else "available"
            if errs:
                invalid.append({"index": i, "errors": errs}); continue
            m = re.search(r"(\d{4})-(\d{2})-(\d{2})", str(o.get("created") or ""))
            d = None
            if m:
                try:
                    d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                except ValueError:
                    d = None
            sdate = (d or now.date()).isoformat()
            if (sp.norm_item(item), int(p), sdate) in same:          # already in the register (e.g. restoring a backup)
                skipped += 1; continue
            same.add((sp.norm_item(item), int(p), sdate))
            ok_rows.append((item, int(p), sdate, ub, loc, status, lid))
        con = db()
        if ok_rows:
            con.begin()
            try:
                for (item, p, sd, ub, loc, status, lid) in ok_rows:
                    new_id = con.insert_id(
                        "INSERT INTO surplus_items(item,portions,service_date,use_by_time,location,status,safety_confirmed_by,"
                        "safety_confirmed_at,legacy_id,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        (item, p, sd, ub, loc, status, "old browser register (tick-box)", stamp, lid, stamp, stamp))
                    log_event(con, new_id, "imported", None, status, "Copied from the old browser-only register")
                    added += 1
                con.commit()
            except Exception:
                con.rollback()
                raise
        return jsonify({"ok": True, "added": added, "skipped_already_imported": skipped, "invalid": invalid})

    # ------------------------------------------------------------ recipients (only what staff entered)
    @app.post("/api/recipients")
    @require("staff")
    def add_recipient():
        rec, errs = sp.validate_recipient(json_body())
        if errs:
            return err("; ".join(errs), errors=errs)
        try:
            db().execute("INSERT INTO recipients(name,contact,requirements,min_portions,max_portions,active,created_at) "
                         "VALUES(?,?,?,?,?,1,?)", (rec["name"], rec["contact"], rec["requirements"], rec["min_portions"],
                                                   rec["max_portions"], local_now().isoformat(timespec="seconds")))
        except dbapi.INTEGRITY_ERRORS:
            return err("A recipient with that name already exists.", 409)
        return jsonify({"ok": True}), 201

    @app.delete("/api/recipients/<int:rid>")
    @require("staff")
    def deactivate_recipient(rid):
        """Deactivates (history keeps the name). Never deletes."""
        cur = db().execute("UPDATE recipients SET active=0 WHERE id=?", (rid,))
        return jsonify({"ok": True}) if cur.rowcount else err("No such recipient", 404)

    # ------------------------------------------------------------ assistant
    @app.get("/api/assistant/status")
    @require("staff")
    def assistant_status():
        s = S()
        return jsonify({"ok": True, "ai_wording": s.llm_configured and s.llm_daily_limit > 0,
                        "mode": "ai_wording" if (s.llm_configured and s.llm_daily_limit > 0) else "data_only"})

    @app.post("/api/assistant/ask")
    @require("staff")
    def assistant_ask():
        s = S()
        okrl, wait = limiter.allow("assistant", client_ip(), s.assistant_per_minute, 60)
        if not okrl:
            return err(f"You are asking too quickly. Try again in {wait} seconds.", 429)
        q, e = assistant.validate_question(json_body())
        if e:
            return err(e)
        cache = {}
        def memo(key, fn):
            if key not in cache:
                cache[key] = fn()
            return cache[key]
        today = local_today()
        def fc(meal, target):
            bundle, _ = model_bundle()
            return ml.forecast(bundle, meal, target)
        ctx = {
            "today": today,
            "records": lambda: memo("recs", load_records),
            "insights": lambda: memo("ins", lambda: insights.build_insights(memo("recs", load_records), forecast_pairs(), today=today)),
            "anomalies": lambda: memo("an", lambda: insights.detect_anomalies(memo("recs", load_records), reviews=review_map())),
            "forecast": fc,
            "model_report": lambda: model_bundle()[0]["report"],
            "surplus": lambda: memo("sur", lambda: (lambda s_: {"counts": s_["counts"], "recommendations": s_["recommendations"]})(surplus_summary())),
        }
        res = assistant.ask(ctx, q, s, budget, transport=current_app.config.get("LLM_TRANSPORT"))
        log.info("assistant question length=%d intents=%s source=%s llm=%s", len(q), res.get("intents"), res["source"],
                 res["llm"]["status"])
        return jsonify(json_safe(res))
