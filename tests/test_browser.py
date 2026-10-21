"""End-to-end browser test (OPTIONAL): drives the real page in headless Chromium against a live server.

Needs:  pip install playwright   and   playwright install chromium
Run:    python tests/test_browser.py
"""
import csv
import io
import json
import os
import sys
import tempfile
import threading
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fixtures import gen  # noqa: E402
import app as appmod  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402
from werkzeug.serving import make_server  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(("PASS  " if cond else "FAIL  ") + name + ((" -> " + str(detail)) if (detail and not cond) else ""))


def csv_text(rows, synthetic=False):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    w.writerow(["Date", "Meal", "Prepared", "Leftovers"] + (["Synthetic"] if synthetic else []))
    for r in rows:
        w.writerow([r["date"], r["meal"], r["prepared"], r["leftovers"]] + (["yes"] if synthetic else []))
    return buf.getvalue()


class Server:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = appmod.create_app(os.path.join(self.tmp.name, "e2e.db"))
        self.srv = make_server("127.0.0.1", 0, self.app, threaded=True)
        self.url = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def api(self, path):
        return self.app.test_client().get(path).get_json()

    def stop(self):
        self.srv.shutdown()
        self.tmp.cleanup()


def main():
    srv = Server()
    work = tempfile.mkdtemp()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(accept_downloads=True, viewport={"width": 1200, "height": 900})
        page = ctx.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append("pageerror: " + str(e)))
        page.on("console", lambda m: errors.append("console: " + m.text)
                if m.type == "error" and "422" not in m.text else None)   # 422 = deliberate "no history" answer
        page.on("dialog", lambda d: d.accept("RESET") if d.type == "prompt" else d.accept())

        def tab(name):
            page.click(f'#tabs button[data-tab="{name}"]')

        def settle():
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(250)

        # ---------- 1. first load, empty database ----------
        page.goto(srv.url)
        settle()
        check("page loads and header shows EcoPlate", "EcoPlate" in page.inner_text("header"))
        check("dashboard model card shows insufficient-data status",
              "Insufficient data" in page.inner_text("#dashModel"), page.inner_text("#dashModel"))
        tab("forecast")
        page.click('#fcForm button[type="submit"]')
        settle()
        check("forecast with no records says there is no history", "no history" in page.inner_text("#fcResult").lower(),
              page.inner_text("#fcResult"))

        # ---------- 2. manual record + duplicate ----------
        tab("record")
        day = (date.today() - timedelta(days=1)).isoformat()
        page.fill("#recDate", day); page.select_option("#recMeal", "Lunch")
        page.fill("#recPrep", "300"); page.fill("#recLeft", "25")
        page.click('#recForm button[type="submit"]'); settle()
        check("manual record saved", "Saved" in page.inner_text("#recMsg"), page.inner_text("#recMsg"))
        check("record reached the database", srv.api("/api/records")["count"] == 1)
        page.fill("#recPrep", "999"); page.fill("#recLeft", "1")
        page.click('#recForm button[type="submit"]'); settle()
        check("duplicate manual record refused", "already exists" in page.inner_text("#recMsg"), page.inner_text("#recMsg"))
        check("duplicate did not overwrite", srv.api("/api/records")["records"][0]["prepared"] == 300)
        page.fill("#recPrep", "10"); page.fill("#recLeft", "50")
        page.click('#recForm button[type="submit"]')
        check("leftovers > prepared refused", "cannot be more" in page.inner_text("#recMsg"), page.inner_text("#recMsg"))

        # ---------- 3. forecast with 1 record: labelled fallback ----------
        tab("forecast")
        page.fill("#fcDate", date.today().isoformat())
        page.click('#fcForm button[type="submit"]'); settle()
        txt = page.inner_text("#fcResult")
        check("tiny history -> rule-based fallback is clearly labelled", "NOT machine learning" in txt and "ACTIVE" not in txt, txt[:300])
        check("fallback shows predicted-surplus warning (not safe for donation)", "not</b> verified" in page.inner_html("#fcResult") or "not verified" in txt.lower(), txt[-400:])

        # ---------- 4. CSV bulk import with invalid + duplicate rows (weekday fixture, flagged synthetic) ----------
        fx = gen(200, seed=1, kind="weekday")
        rows = fx + [{"date": "2025-02-30", "meal": "Lunch", "prepared": 5, "leftovers": 1},       # invalid date
                     {"date": "2025-03-03", "meal": "Lunch", "prepared": 5, "leftovers": 9}]       # leftovers > prepared
        path = os.path.join(work, "fixture.csv")
        open(path, "w", newline="").write(csv_text(rows, synthetic=True))
        tab("import")
        page.set_input_files("#impFile", path); settle()
        prev = page.inner_text("#impPreview")
        check("import preview shows invalid rows", "Invalid rows" in prev and "does not exist" in prev, prev[:400])
        check("import preview flags sample rows", "marked as sample" in prev)
        page.click("#impConfirm"); settle()
        check("import completes", "Import complete" in page.inner_text("#impResult"), page.inner_text("#impResult"))
        check("imported records reached the database", srv.api("/api/records")["count"] == 1 + len(fx))
        check("sample flag stored", all(r["synthetic"] for r in srv.api("/api/records")["records"] if r["source"] == "import"))
        page.set_input_files("#impFile", path); settle()
        check("re-importing the same file adds nothing (duplicates skipped)",
              "nothing new to import" in page.inner_text("#impPreview"), page.inner_text("#impPreview")[:300])
        page.click("#impCancel")

        # ---------- 5. Gradient Boosting ACTIVE shown in UI ----------
        tab("forecast")
        page.select_option("#fcMeal", "Lunch")
        nxt = (date.fromisoformat(max(r["date"] for r in fx)) + timedelta(days=1)).isoformat()
        page.fill("#fcDate", nxt)
        page.click('#fcForm button[type="submit"]'); settle()
        txt = page.inner_text("#fcResult")
        check("UI says Gradient Boosting is ACTIVE", "Gradient Boosting is ACTIVE" in txt, txt[:300])
        check("UI shows MAE and RMSE", "MAE" in txt and "RMSE" in txt)
        check("UI shows sample-data warning", "Sample data warning" in txt)
        check("UI shows predicted-surplus separation", "prediction" in txt and "Surplus Sharing register" in txt)
        # the number on screen must equal the backend's answer, not a browser-side calculation
        api_res = srv.app.test_client().post("/api/forecast", json={"meal": "Lunch", "date": nxt, "attendance_pct": 100, "buffer_pct": 10, "usable_leftovers": 0}).get_json()
        check("recommended number on screen equals server response", str(api_res["recommended"]) in page.inner_text("#fcResult .big").replace(",", ""), page.inner_text("#fcResult .big"))
        tab("model")
        settle()
        mt = page.inner_text("#modelBody")
        check("Model tab shows active status, baselines and split", "Gradient Boosting is active" in mt and "best baseline" in mt and "chronological" in mt, mt[:300])
        page.click("#btnRetrain")
        try:
            page.wait_for_selector("#modelMsg.ok", timeout=20000)      # wait for the server to finish retraining
        except Exception:
            pass
        check("retrain button works", "retrained just now" in page.inner_text("#modelMsg"), repr(page.inner_text("#modelMsg")))

        # ---------- 6. changing data re-evaluates the open forecast ----------
        tab("history")
        check("history shows synthetic badge", page.locator("#hBody .syn").count() > 0)
        page.select_option("#hMeal", "Lunch")
        check("history filter works", all("Lunch" in r for r in page.locator("#hBody tr td:nth-child(2)").all_inner_texts()))
        page.select_option("#hMeal", "")

        # ---------- 7. Gradient Boosting trained but NOT selected (noise fixture) ----------
        tab("data")
        page.click("#btnRmSample"); settle()
        check("removing sample records leaves the manual record", srv.api("/api/records")["count"] == 1)
        tab("history"); page.fill("#hSearch", day); settle()
        page.locator("#hBody button[data-del]").first.click(); settle()
        page.fill("#hSearch", "")
        check("manual record deleted through the UI so the next scenario is isolated", srv.api("/api/records")["count"] == 0)
        noise = gen(200, seed=1, kind="noise")
        npath = os.path.join(work, "noise.csv")
        open(npath, "w", newline="").write(csv_text(noise, synthetic=True))
        tab("import"); page.set_input_files("#impFile", npath); settle()
        page.click("#impConfirm"); settle()
        tab("forecast")
        nxt2 = (date.fromisoformat(max(r["date"] for r in noise)) + timedelta(days=1)).isoformat()
        page.fill("#fcDate", nxt2)
        page.click('#fcForm button[type="submit"]'); settle()
        txt = page.inner_text("#fcResult")
        check("UI says trained but NOT used, and method is rule-based", "trained but NOT used" in txt and "NOT machine learning" in txt and "ACTIVE" not in txt, txt[:400])

        # put one real manual record back for the backup/merge scenarios
        tab("record"); page.fill("#recDate", day); page.select_option("#recMeal", "Lunch")
        page.fill("#recPrep", "300"); page.fill("#recLeft", "25")
        page.click('#recForm button[type="submit"]'); settle()

        # ---------- 8. backup JSON, destructive change, restore ----------
        tab("data")
        with page.expect_download() as dl:
            page.click("#btnJson")
        bpath = os.path.join(work, "backup.json"); dl.value.save_as(bpath)
        backup = json.load(open(bpath))
        n_before = srv.api("/api/records")["count"]
        check("JSON backup contains every database record", len(backup["records"]) == n_before and all("synthetic" in r for r in backup["records"]))
        with page.expect_download() as dl2:
            page.click("#btnCsv3")
        cpath = os.path.join(work, "export.csv"); dl2.value.save_as(cpath)
        head = open(cpath, encoding="utf-8-sig").readline().strip()
        check("CSV export has the Synthetic column when sample rows exist", head == "Date,Meal,Prepared,Leftovers,Synthetic", head)
        page.click("#btnRmSample"); settle()
        check("sample rows removed", srv.api("/api/records")["count"] == 1)
        page.set_input_files("#rsFile", bpath); settle()
        check("restore from backup brings every record back", srv.api("/api/records")["count"] == n_before, page.inner_text("#dataMsg"))
        bad = os.path.join(work, "bad.json")
        bj = json.load(open(bpath)); bj["records"][3]["leftovers"] = 10 ** 6
        json.dump(bj, open(bad, "w"))
        page.set_input_files("#rsFile", bad); settle()
        check("invalid backup is refused and nothing changes", "Restore refused" in page.inner_text("#dataMsg") and srv.api("/api/records")["count"] == n_before, page.inner_text("#dataMsg"))

        # ---------- 9. merge an old backup (additive) ----------
        old = {"app": "EcoPlate", "backupVersion": 2, "records": [
            {"date": "2024-01-02", "meal": "Dinner", "prepared": 90, "leftovers": 9, "source": "manual"},
            {"date": day, "meal": "Lunch", "prepared": 111, "leftovers": 1, "source": "manual"}],   # conflicts with the manual record
            "surplus": [{"id": "sOLD1", "item": "Old biryani", "portions": 4, "location": "Gate", "safeUntil": "14:00", "created": "x", "status": "Available"}]}
        opath = os.path.join(work, "old.json"); json.dump(old, open(opath, "w"))
        tab("data");
        page.set_input_files("#mgFile", opath); settle()
        recs = {(r["date"], r["meal"]): r for r in srv.api("/api/records")["records"]}
        check("merge added the new record", ("2024-01-02", "Dinner") in recs)
        check("merge did NOT overwrite the conflicting record", recs[(day, "Lunch")]["prepared"] == 300)
        check("merge keeps the conflicting entry in the unconverted list", page.is_visible("#unmapBox"))
        tab("surplus")
        check("merged surplus entry appears", "Old biryani" in page.inner_text("#sBody"))

        # ---------- 10. surplus register + food-safety confirmation + persistence ----------
        page.fill("#sItem", "Veg biryani"); page.fill("#sQty", "12"); page.fill("#sLoc", "Back door")
        page.click('#sForm button[type="submit"]')
        check("surplus needs the food-safety confirmation", "confirm that the food is safe" in page.inner_text("#sMsg"), page.inner_text("#sMsg"))
        page.check("#sSafe"); page.click('#sForm button[type="submit"]')
        check("surplus entry added", "Veg biryani" in page.inner_text("#sBody"))
        page.locator("#sBody tr", has_text="Veg biryani").locator('button[data-act="collect"]').click()
        check("mark collected works", "Collected" in page.locator("#sBody tr", has_text="Veg biryani").inner_text())
        page.reload(); settle(); tab("surplus")
        check("surplus register persists after reload", "Veg biryani" in page.inner_text("#sBody") and "Collected" in page.locator("#sBody tr", has_text="Veg biryani").inner_text())
        tab("dashboard")
        check("predicted surplus is not counted as available surplus food", page.inner_text("#sSurplus") == "4", page.inner_text("#sSurplus"))   # only the 4 uncollected, human-registered portions

        # ---------- 11. delete a history record ----------
        tab("history")
        page.fill("#hSearch", "2024-01-02"); settle()
        page.locator("#hBody button[data-del]").first.click(); settle()
        check("history delete removes it from the database", ("2024-01-02", "Dinner") not in {(r["date"], r["meal"]) for r in srv.api("/api/records")["records"]})
        page.fill("#hSearch", "")

        # ---------- 12. reset ----------
        tab("data"); page.click("#btnReset"); settle()
        check("reset empties the database", srv.api("/api/records")["count"] == 0, page.inner_text("#dataMsg"))

        # ---------- 13. migrate OLD browser data (pre-backend format) ----------
        legacy = {"version": 2, "records": [
            {"date": "2025-04-01", "meal": "Lunch", "prepared": 200, "leftovers": 20, "source": "manual"},
            {"date": "2025-04-02", "meal": "Lunch", "prepared": 210, "leftovers": 22, "source": "import", "batch": "bold1"},
            {"date": "2025-04-03", "meal": "Lunch", "prepared": 5, "leftovers": 50, "source": "manual"}],      # invalid (kept safe)
            "surplus": [{"id": "sMIG1", "item": "Legacy rice", "portions": 7, "location": "", "safeUntil": "", "created": "c", "status": "Available"}],
            "unmapped": [], "lastImport": None}
        page.evaluate("(d) => { localStorage.clear(); localStorage.setItem('ecoplateDemo', JSON.stringify(d)); }", legacy)
        page.reload(); settle()
        recs = srv.api("/api/records")["records"]
        check("old browser records were copied to the database", {r["date"] for r in recs} == {"2025-04-01", "2025-04-02"}, recs)
        check("batch/source were preserved by the migration", [r for r in recs if r["date"] == "2025-04-02"][0]["batch"] == "bold1")
        stored = json.loads(page.evaluate("localStorage.getItem('ecoplateDemo')"))
        check("browser copy is now format v3 and records are not duplicated there", stored["version"] == 3 and "records" not in stored)
        check("raw safety copy of the old data was kept", page.evaluate("localStorage.getItem('ecoplateDemo_pre_backend_backup')") is not None)
        check("invalid old entry was kept, not dropped", any("2025-04-03" in json.dumps(u) for u in stored["unmapped"]))
        check("migration notice shown", "copied to the EcoPlate database" in page.inner_text("#notice"))
        tab("surplus")
        check("old surplus entry kept", "Legacy rice" in page.inner_text("#sBody"))
        page.reload(); settle()
        check("second load does not duplicate or re-migrate", srv.api("/api/records")["count"] == 2)

        # ---------- 14. sample data button ----------
        tab("import");
        page.click("#btnLoadSample"); settle()
        samples = [r for r in srv.api("/api/records")["records"] if r["synthetic"]]
        check("sample button loads flagged synthetic records", len(samples) > 200 and all(r["source"] == "sample" for r in samples), len(samples))
        tab("forecast"); page.fill("#fcDate", date.today().isoformat())
        page.click('#fcForm button[type="submit"]'); settle()
        txt = page.inner_text("#fcResult")
        labelled = ("Gradient Boosting is ACTIVE" in txt and "trained ML model" in txt) or \
                   ("NOT machine learning" in txt and "ACTIVE" not in txt)
        check("forecast on sample data carries an unambiguous method label", labelled, txt[:300])
        check("sample-data warning shown on forecast", "SAMPLE" in txt)
        page.screenshot(path=os.path.join(work, "forecast.png"), full_page=True)

        # ---------- 15. file:// mode ----------
        page2 = ctx.new_page()
        page2.goto("file://" + os.path.join(ROOT, "index.html"))
        page2.wait_for_timeout(500)
        check("opening as a file explains how to start the server", "cannot reach the EcoPlate server" in page2.inner_text("#notice"), page2.inner_text("#notice")[:200])
        check("file mode offers to download old browser data", page2.is_visible("#btnOldExport"))

        check("no JavaScript errors in the console during the whole run", not errors, errors[:5])
        browser.close()
    srv.stop()
    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} browser checks passed")
    if failed:
        print("FAILED:", *failed, sep="\n  - ")
        sys.exit(1)
    print("screenshot folder:", work)


if __name__ == "__main__":
    main()
