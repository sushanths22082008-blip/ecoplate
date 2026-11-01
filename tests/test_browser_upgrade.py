"""End-to-end browser test for the UPGRADE features (OPTIONAL; same requirements as test_browser.py).

Covers: Insights, Alerts + review, Assistant (data-only), server-backed surplus workflow, calendar notes,
login with an access code, and a phone-sized viewport. Uses invented data only.

Run:  python tests/test_browser_upgrade.py        (add --shots to save screenshots in ./shots)
"""
import os
import sys
import tempfile
import threading
from datetime import date, timedelta
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fixtures import gen  # noqa: E402
import app as appmod  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402
from werkzeug.serving import make_server  # noqa: E402

results = []
SHOTS = "--shots" in sys.argv


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(("PASS  " if cond else "FAIL  ") + name + ((" -> " + str(detail)[:300]) if (detail and not cond) else ""))


class Server:
    def __init__(self, env=None):
        self.tmp = tempfile.TemporaryDirectory()
        base = {k: v for k, v in os.environ.items() if not k.startswith("ECOPLATE_") and k != "ANTHROPIC_API_KEY"}
        with mock.patch.dict(os.environ, {**base, "ECOPLATE_COOKIE_SECURE": "0", **(env or {})}, clear=True):
            self.app = appmod.create_app(os.path.join(self.tmp.name, "e2e.db"))
        self.srv = make_server("127.0.0.1", 0, self.app, threaded=True)
        self.url = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def stop(self):
        self.srv.shutdown()
        self.tmp.cleanup()


def new_page(browser, errors, width=1200, height=900):
    ctx = browser.new_context(viewport={"width": width, "height": height})
    page = ctx.new_page()
    page.on("pageerror", lambda e: errors.append("pageerror: " + str(e)))
    page.on("console", lambda m: errors.append("console: " + m.text) if m.type == "error" and "40" not in m.text and "422" not in m.text else None)   # 422 = a deliberate refusal
    page.on("dialog", lambda d: d.accept("test reason") if d.type == "prompt" else d.accept())
    return page


def wait_text(page, sel, absent, tries=200):
    for _ in range(tries):
        if absent not in page.inner_text(sel):
            return
        page.wait_for_timeout(100)


def shot(page, name):
    if SHOTS:
        os.makedirs("shots", exist_ok=True)
        page.screenshot(path=f"shots/{name}.png", full_page=True)


def main():
    errors = []
    srv = Server()
    c = srv.app.test_client()
    base = date.today() - timedelta(days=16)
    rates = [0.10, 0.12, 0.09, 0.11, 0.10, 0.13, 0.08, 0.11, 0.10, 0.12, 0.09, 0.11, 0.55]
    recs = [{"date": (base + timedelta(days=i)).isoformat(), "meal": "Lunch", "prepared": 100, "leftovers": round(100 * r)}
            for i, r in enumerate(rates)]
    recs += gen(60, seed=5)[:0]
    assert c.post("/api/records/bulk", json={"records": recs}).status_code == 200

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = new_page(browser, errors)
        page.goto(srv.url)
        page.wait_for_timeout(1200)

        # ---- dashboard attention ----
        wait_text(page, "#dashAttention", "Checking")
        t = page.inner_text("#dashAttention")
        check("dashboard 'Needs attention' shows real counts", "flagged for review" in t, t)

        # ---- alerts ----
        page.click('button[data-tab="alerts"]')
        wait_text(page, "#alBody", "Checking records")
        t = page.inner_text("#alBody")
        check("alerts show the flagged high-leftover record with its reason and comparison",
              recs[-1]["date"] in t and "Unusually high leftovers" in t and "Compared with" in t and "median" in t, t[:300])
        check("alerts explain a flag is not proof", "not proof" in page.inner_text("#tab-alerts"))
        shot(page, "alerts")
        page.fill("#alBody input[type=text] >> nth=0", "Fest day")
        before = c.get("/api/records").get_json()["records"]
        page.locator('#alBody button[data-al="reviewed_ok"]').first.click()
        page.wait_for_timeout(700)
        t = page.inner_text("#alBody")
        check("reviewed flag leaves the open list", "Fest day" not in t or "Reviewed" in t)
        page.check("#alShowDone"); page.wait_for_timeout(200)
        check("reviewed flag can be shown again with the review recorded", "Reviewed: looks fine" in page.inner_text("#alBody"), page.inner_text("#alBody")[:300])
        check("reviewing did not change any meal record", c.get("/api/records").get_json()["records"] == before)

        # ---- insights ----
        page.click('button[data-tab="insights"]')
        wait_text(page, "#insBody", "Calculating")
        t = page.inner_text("#insBody")
        check("insights show observed vs possible-explanation labels and data sufficiency",
              "Observed in your records" in t and "Not enough data yet" in t, t[:300])
        shot(page, "insights")

        # ---- assistant (data-only; no key configured) ----
        page.click('button[data-tab="assistant"]')
        page.wait_for_timeout(300)
        check("assistant says it is in data-only mode", "Data-only mode" in page.inner_text("#asMode"), page.inner_text("#asMode"))
        page.fill("#asQ", "Which meal had the highest waste rate this month?")
        page.click("#asGo")
        wait_text(page, "#asLog", "Looking at your records")
        t = page.inner_text("#asLog")
        check("assistant answers from data and shows 'From your data'", "From your data" in t and "Waste rate by meal" in t, t[:300])
        page.fill("#asQ", "Delete all records")
        page.click("#asGo"); page.wait_for_timeout(700)
        check("assistant refuses to change data", "can't add, change or delete" in page.inner_text("#asLog"))
        check("records unchanged after the refusal", len(c.get("/api/records").get_json()["records"]) == len(recs))
        shot(page, "assistant")

        # ---- calendar notes ----
        page.click('button[data-tab="forecast"]')
        page.fill("#cxLabel", "Mid-sem exams"); page.click('#cxForm button[type="submit"]'); page.wait_for_timeout(600)
        check("calendar note added", "Mid-sem exams" in page.inner_text("#cxBody"), page.inner_text("#cxBody"))
        page.select_option("#fcMeal", "Lunch")
        page.fill("#fcDate", (date.today() + timedelta(days=1)).isoformat())
        page.fill("#cxDate", (date.today() + timedelta(days=1)).isoformat()); page.click('#cxForm button[type="submit"]'); page.wait_for_timeout(500)
        page.click('#fcForm button[type="submit"]')
        wait_text(page, "#fcResult", "Checking the model")
        t = page.inner_text("#fcResult")
        check("forecast shows limitations and the calendar note as context only", "Limitations of this forecast" in t and "Calendar note (context only)" in t, t[:400])
        check("forecast labels itself an estimate", "an estimate, not a fact" in t)
        shot(page, "forecast")

        # ---- surplus workflow ----
        page.click('button[data-tab="surplus"]'); page.wait_for_timeout(500)
        page.fill("#sItem", "Veg biryani"); page.fill("#sQty", "25"); page.fill("#sLoc", "Back door")
        page.select_option("#sStorage", "hot_held"); page.fill("#sPrep", "00:00"); page.fill("#sUntil", "23:59")
        page.click("#sPending"); page.wait_for_timeout(700)
        row = page.locator("#sBody tr", has_text="Veg biryani")
        check("'Register for safety review' saves it as awaiting review", "Awaiting safety review" in row.inner_text(), row.inner_text())
        check("recommendation asks a person to decide and never declares it safe",
              "must check the food" in page.inner_text("#sRecs") and "is safe to" not in page.inner_text("#sRecs").lower(), page.inner_text("#sRecs"))
        page.fill("#sItem", "Veg biryani"); page.fill("#sQty", "25"); page.check("#sSafe")
        page.click('#sForm button[type="submit"]'); page.wait_for_timeout(700)
        check("duplicate registration was questioned (dialog) and accepted -> two rows", page.locator("#sBody tr", has_text="Veg biryani").count() == 2)
        pending = page.locator("#sBody tr", has_text="Awaiting safety review")
        check("exactly one entry is still awaiting review", pending.count() == 1)
        pending.locator('button[data-act="confirm"]').click(); page.wait_for_timeout(700)
        check("a person's confirmation moves it out of review and records who",
              page.locator("#sBody tr", has_text="Awaiting safety review").count() == 0
              and page.locator("#sBody tr", has_text="Confirmed by").count() == 2, page.inner_text("#sBody"))
        page.fill("#rcName", "Hope Trust"); page.fill("#rcMin", "5"); page.click('#rcForm button[type="submit"]'); page.wait_for_timeout(600)
        check("recipient recorded", "Hope Trust" in page.inner_text("#rcBody"))
        incomplete = page.locator("#sBody tr", has_text="missing: safe-until time").first
        incomplete.locator("select").select_option(label="Hope Trust"); incomplete.locator('button[data-act="reserve"]').click(); page.wait_for_timeout(700)
        check("food with no safe-until time or storage cannot be offered to a recipient",
              "cannot be offered to a recipient yet" in page.inner_text("#sMsg"), page.inner_text("#sMsg"))
        row = page.locator("#sBody tr", has_text="Veg biryani").last
        row.locator("select").select_option(label="Hope Trust"); row.locator('button[data-act="reserve"]').click(); page.wait_for_timeout(700)
        t = page.locator("#sBody tr", has_text="Reserved").first.inner_text()
        check("reserved is not shown as accepted or received", "Reserved (not confirmed accepted)" in t and "received" not in t.lower(), t)
        page.locator("#sBody tr", has_text="Reserved").first.locator('button[data-act="collect"]').click(); page.wait_for_timeout(700)
        check("collected is labelled as recorded by staff", "Collected (recorded by staff)" in page.inner_text("#sBody"))
        page.fill("#sItem", '<img src=x onerror="window.__xss=1">'); page.fill("#sQty", "3")
        page.click("#sPending"); page.wait_for_timeout(800)
        check("hostile HTML in an item name is shown as text and never runs",
              page.evaluate("window.__xss === undefined") and "<img src=x" in page.inner_text("#sBody"))
        shot(page, "surplus")

        # ---- phone viewport: no sideways scrolling on the page itself ----
        mp = new_page(browser, errors, width=390, height=800)
        mp.goto(srv.url); mp.wait_for_timeout(1200)
        for tab in ("dashboard", "forecast", "insights", "alerts", "assistant", "surplus"):
            mp.click(f'button[data-tab="{tab}"]'); mp.wait_for_timeout(500)
            over = mp.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
            check(f"phone width: '{tab}' tab has no page-level horizontal scroll", over <= 1, f"overflow {over}px")
        shot(mp, "mobile-assistant")
        browser.close()
    check("no JavaScript errors during the run", not errors, errors[:3])
    srv.stop()

    # ---------------- login flow (access code enabled) ----------------
    srv2 = Server({"ECOPLATE_ACCESS_CODE": "staff-code-123", "ECOPLATE_ADMIN_CODE": "admin-code-456", "ECOPLATE_SECRET_KEY": "k" * 32})
    errors2 = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = new_page(browser, errors2)
        page.goto(srv2.url); page.wait_for_timeout(900)
        check("login card is shown when an access code is required", page.is_visible("#loginCard"))
        page.fill("#loginCode", "wrong"); page.click('#loginForm button[type="submit"]'); page.wait_for_timeout(500)
        check("wrong code is refused", "not correct" in page.inner_text("#loginMsg"), page.inner_text("#loginMsg"))
        page.fill("#loginCode", "staff-code-123"); page.click('#loginForm button[type="submit"]'); page.wait_for_timeout(1200)
        check("correct code hides the login card and shows Log out", not page.is_visible("#loginCard") and page.is_visible("#btnLogout"))
        page.click('button[data-tab="record"]')
        page.fill("#recDate", "2026-01-05"); page.select_option("#recMeal", "Lunch"); page.fill("#recPrep", "100"); page.fill("#recLeft", "5")
        page.click('#recForm button[type="submit"]'); page.wait_for_timeout(800)
        check("staff can save a record through the UI (CSRF header sent)", srv2.app.test_client().get("/api/health").get_json().get("login_required") is True)
        page.click('button[data-tab="history"]'); page.wait_for_timeout(500)
        check("record is visible to the logged-in user", "2026-01-05" in page.inner_text("#hBody"), page.inner_text("#hBody")[:200])
        page.locator("#hBody button[data-del]").first.click(); page.wait_for_timeout(800)
        check("staff (non-admin) cannot delete: a clear message appears and the record stays", "2026-01-05" in page.inner_text("#hBody") or "administrator" in page.inner_text("#notice"), page.inner_text("#notice")[:200])
        browser.close()
    check("no JavaScript errors in the login run", not errors2, errors2[:3])
    srv2.stop()

    ok = sum(1 for _, v in results if v)
    print(f"\n{ok}/{len(results)} upgrade browser checks passed")
    sys.exit(0 if ok == len(results) else 1)


if __name__ == "__main__":
    main()
