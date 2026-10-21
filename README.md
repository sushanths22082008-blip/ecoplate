# EcoPlate — AI-based canteen food-waste reduction (Dexathon '26)

EcoPlate is your original web app, now backed by a small Python server that runs a **real Gradient Boosting
model** (scikit-learn `HistGradientBoostingRegressor`) to forecast how many portions a canteen will need.

Everything you had is still there: dashboard and charts, record meals, CSV bulk import/export, history and
filters, surplus-sharing register with the food-safety confirmation, backup/restore, reset, and the original
look. What changed is *where the forecast comes from* and *where meal records are stored*.

```
Browser (index.html)  --HTTP/JSON-->  Flask (app.py)  -->  SQLite (ecoplate.db)   meal records
                                           |
                                           +-->  ecoplate_ml.py  (train / evaluate / forecast)
```

* The page is served by the same server (`http://127.0.0.1:5000`), so there are no CORS problems.
* Meal records live in `ecoplate.db` (SQLite) next to `app.py`.
* The surplus register still lives in the browser (`localStorage`). Full backups (JSON) include both.
* The forecast is calculated **by the server**. The browser only displays what the server returns. The old
  JavaScript averaging code was removed from the page.

---

## 1. Windows quick start (beginner)

**One-time setup**

1. Install Python 3.11, 3.12 or 3.13 from <https://www.python.org/downloads/>. On the first installer screen,
   **tick "Add python.exe to PATH"**, then click *Install Now*.
2. Unzip `ecoplate-project.zip`, for example to your Desktop. You get a folder called `ecoplate`.
3. Open that folder in File Explorer, click the address bar at the top, type `cmd` and press **Enter**.
   A black Command Prompt window opens *inside* the folder.
4. Type these commands one by one (press Enter after each):

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

   (If `python` is "not recognized", use `py -m venv .venv` for the first command, then continue as above.)

   The install step downloads scikit-learn and Flask and takes a minute or two. When the last command works you
   will see a box saying **"EcoPlate is running"** with an address.

5. **Open the address in your browser** (Chrome or Edge): <http://127.0.0.1:5000>
   (If port 5000 is busy the program tells you which port it chose instead. Use the address it prints.)

**Stopping the server:** click the Command Prompt window and press **Ctrl + C**. Then you can close the window.

**Starting it again later** (no need to reinstall):

```
cd path\to\ecoplate
.venv\Scripts\activate
python app.py
```

Your records are kept in `ecoplate.db`, so they are still there next time.

macOS / Linux: `python3 -m venv .venv`, `source .venv/bin/activate`, `pip install -r requirements.txt`, `python app.py`.

---

## 2. Bringing in your existing records

**Why it is not fully automatic.** Your old page stored records inside the browser, under the address
`file://…`. EcoPlate now runs at `http://127.0.0.1:5000`. Browsers never share stored data between two
different addresses, so the server cannot see your old records until you move them. Nothing is lost
in the meantime: the old data stays where it is.

Choose **one** of these (all are additive and never overwrite or silently delete anything):

| Method | Steps |
|---|---|
| **A. JSON backup (recommended, keeps surplus entries too)** | 1. Open your **old** `index.html` (double-click it) → **Backup & Reset → Full backup (JSON)**. 2. Start the new server and open `http://127.0.0.1:5000`. 3. **Backup & Reset → Merge an old backup** → choose the `.json` file. |
| **B. CSV** | Old page → **History → Export all history (CSV)**. New page → **Bulk Import** → choose the CSV → check the preview → **Confirm import**. (Surplus entries are not in CSV files.) |
| **C. Opened the *new* `index.html` by double-click** | You will see a red banner explaining that the server is needed, plus a button **"Download my old browser data (JSON)"**. Click it, then use method A step 3. |

Automatic copy: if the new page ever finds old-format records in the *same* browser address (for example if you
previously served the page from `http://127.0.0.1:5000`), it copies them to the database once, on first load,
without deleting anything: duplicates are skipped, conflicting or invalid entries are kept under
**Backup & Reset → Old entries that could not be converted**, and a raw copy of the old data is kept in the
browser as `ecoplateDemo_pre_backend_backup`.

Duplicates: the database allows only one record per **date + meal**. An existing record is never overwritten;
if the new data differs, the stored one is kept and the other is reported.

---

## 3. Using the forecast

1. Add history: **Record Meal**, **Bulk Import**, or **Bulk Import → Load sample data into EcoPlate** (invented
   data, clearly flagged as sample).
2. **Forecast** tab → choose meal, date, attendance %, usable leftovers, buffer % → **Calculate recommendation**.
3. The result always starts with a banner saying which method produced it:

| Banner | Meaning |
|---|---|
| ✅ **Gradient Boosting is ACTIVE** | A trained model produced the demand estimate. It was only allowed to because it beat every simple baseline on the most recent, unseen days. |
| ℹ️ **Gradient Boosting was trained but NOT used** | It was trained, but on recent unseen days it was not more accurate than the rule-based method. The forecast is rule-based, **not** machine learning. |
| ⚠️ **Not enough data: rule-based fallback** | Too little history to train and test honestly (or too little for this particular meal). The forecast is rule-based, **not** machine learning. |

The recommended quantity uses your original logic: `demand × attendance% × (1 + buffer%) − usable leftovers`,
rounded up, never below 0 (leftovers are subtracted once).

**Predicted surplus** is shown separately and labelled as a planning estimate. It is never added to the Surplus
Sharing register automatically and is never declared safe. Only food that really exists and passes your
food-safety check goes into that register.

The **Model** tab shows the full report: data counts, the chronological train/test split, **MAE and RMSE** for
Gradient Boosting and three baselines, per-meal errors and which inputs the model relies on.

**Sample data is not evidence.** Scores on sample (synthetic) records only describe that invented data. The page
and the API both say so. The result on sample data can also change from day to day, because the sample dates
move with the calendar; that is expected, not a fault.

---

## 4. What the ML module does (and what was left unchanged)

`ecoplate_ml.py` is **unmodified**. I inspected it, probed it, and found no defect that needed changing. Its
safeguards all remain in force:

* Predicts meals eaten (`prepared − leftovers`) from features that use only **earlier** dates (tested for leakage).
* **Chronological validation**: oldest 80 % of dates train, newest 20 % test. Nothing is tuned on the test rows.
* **MAE and RMSE**, compared with the original rule-based method, an average of the last 7 services, and a historical mean.
* **Minimum data**: ≥ 60 usable rows overall, ≥ 28 for the meal being forecast, ≥ 12 test rows.
* **Honest fallback**: if data is insufficient, or the model does not beat the best baseline, it is *not used*.
* **Retraining** happens automatically whenever any record is added, changed or deleted (data fingerprint), and
  never when nothing changed. **Model → Retrain model now** forces a fresh run.
* Month/season is only used once the history spans a full year.

Records with 0 leftovers may mean the food ran out, so recorded consumption can understate true demand; the
Model tab counts these.

---

## 5. API reference

All paths are under the same server. Requests and responses are JSON.

| Method & path | Purpose |
|---|---|
| `GET /api/health` | Status, record count, versions. |
| `GET /api/records` | All stored records. |
| `POST /api/records` | Add one record `{date, meal, prepared, leftovers}`. `201`, or `400` invalid, `409` duplicate. |
| `POST /api/records/bulk` | Add many: `{records:[…], source?, batch?, synthetic?, dry_run?}`. Skips and reports invalid rows, identical duplicates and conflicts. Never overwrites. |
| `DELETE /api/records?date=YYYY-MM-DD&meal=Lunch` | Delete one record. |
| `DELETE /api/records/batch/<batch>` | Undo one import batch. |
| `DELETE /api/records/synthetic` | Remove all sample records. |
| `POST /api/records/replace` | All-or-nothing replace (restore/reset). Needs `"confirm": "REPLACE_ALL_RECORDS"`. |
| `GET /api/model/report` | Training/validation report (cached until data changes). `?force=1` retrains. |
| `POST /api/model/train` | Force retraining and return the report. |
| `POST /api/forecast` | `{meal, date, attendance_pct, buffer_pct, usable_leftovers}` → recommendation. |

Forecast response fields that matter: `mode` (`ml_active`, `ml_not_selected`, `insufficient_data`,
`insufficient_meal_data`), `ml_used`, `method_label`, `fallback_reason`, `recommended`, `expected_demand`,
`expected_surplus`, `demand_range`, `reliability`, `model_report`. A meal with no history returns `422` with
`code: "no_history"`.

Validation (server-side, authoritative): dates `YYYY-MM-DD`, real, 2000 or later, not in the future; meal exactly
`Breakfast`/`Lunch`/`Dinner`; `prepared` whole number ≥ 1; `leftovers` whole number 0…prepared.

---

## 6. CSV format

`Date,Meal,Prepared,Leftovers` (dates `YYYY-MM-DD`, or `DD-MM-YYYY`/`DD/MM/YYYY` day-first on import). An
optional fifth column `Synthetic` (`yes` on invented rows) lets the model warn about sample data. Exports add that
column only when sample rows exist, so round-trips keep the flag.

---

## 7. Tests

Run from the project folder with the virtual environment active:

```
python -m unittest discover -s tests -v
```

39 tests: the ML module (all three outcomes, leakage, chronological split, retraining, sample warning, formula)
and the API (validation, duplicates, bulk, atomic replace, persistence, forecast modes, JSON strictness, and a
guard that every original page element/tab is still present).

Optional real-browser test (drives the actual page in headless Chromium against a live server):

```
pip install -r requirements-dev.txt
playwright install chromium
python tests/test_browser.py
```

58 checks: manual records, duplicates, CSV preview/import, all three forecast banners, Model tab, backup/restore,
merge, surplus register + food-safety confirmation, delete, reset, old-localStorage migration, sample data,
`file://` behaviour, and no JavaScript errors.

The test fixtures are deterministic **invented** data with fixed dates, used only to exercise code paths.

### What has and has not been verified

Verified in the build sandbox (Linux, Python 3.12.3, Flask 3.1.3, scikit-learn 1.8.0, numpy 2.4.4, headless
Chromium 141 via Playwright): all 39 unit/API tests and all 58 browser checks pass; `python app.py` starts from a
fresh virtual environment, serves the page and health endpoint, and moves to the next port if 5000 is busy.

**Not** verified: running on Windows itself, a first-time `pip install` from the internet (the sandbox had no
network, so declared version ranges were checked against already-installed packages), Firefox/Safari, and other
Python or library versions. If something fails on your machine, see Troubleshooting.

### Manual check on your own computer (5 minutes)

1. Start the server and open the address. The header and tabs appear; the Dashboard card says
   "Insufficient data".
2. **Forecast** → Calculate: it says there is no history.
3. **Bulk Import → Load sample data into EcoPlate**. Dashboard counts update.
4. **Forecast** → Calculate: a banner states the method; the Model tab shows MAE/RMSE and the sample warning.
5. **Record Meal**: add one record, then add it again. The second attempt is refused.
6. **Backup & Reset → Full backup (JSON)**, then **Remove all sample records**, then **Restore** the backup:
   everything returns.
7. Stop with Ctrl+C, start again: your records are still there.

---

## 8. Troubleshooting (Windows)

| Problem | Fix |
|---|---|
| `'python' is not recognized` | Reinstall Python and tick **Add python.exe to PATH**, or use `py` instead of `python`. Open a **new** Command Prompt afterwards. |
| In PowerShell: *"running scripts is disabled"* when activating | Use **Command Prompt** (`cmd`) instead, or run `Set-ExecutionPolicy -Scope Process Bypass` in that PowerShell window first, then `.venv\Scripts\Activate.ps1`. |
| `No module named flask` / `sklearn` | The environment is not active or packages are not installed. Run `.venv\Scripts\activate` then `pip install -r requirements.txt`. The prompt should start with `(.venv)`. |
| `pip install` fails or is slow | Check the internet connection and try `python -m pip install --upgrade pip`, then repeat. Very new Python versions may lack ready-made scikit-learn packages; use Python 3.11–3.13. |
| Browser shows "This site can't be reached" | The server is not running. Start `python app.py` and use the exact address it prints. |
| Page says "cannot reach the EcoPlate server" | You opened `index.html` by double-clicking. Open `http://127.0.0.1:5000` instead (see section 2 to rescue old data). |
| "Port already in use" | The program picks the next free port automatically; use the printed address. Or set `set ECOPLATE_PORT=5050` before `python app.py`. |
| Windows Firewall prompt | The server only listens on your own computer (`127.0.0.1`); you can click Cancel. |
| `database is locked` | Close any second copy of the server or a database viewer, then retry. |
| Forecast says "Insufficient data" | Add more history: at least about 4 weeks per meal (see section 4 for exact rules), or load the sample data to try the interface. |
| Starting over | Stop the server, then delete `ecoplate.db` (and `ecoplate.db-wal`, `ecoplate.db-shm` if present). **Make a Full backup (JSON) first.** |

Advanced: `ECOPLATE_PORT`, `ECOPLATE_DB` (database path) and `ECOPLATE_HOST` can be set as environment variables.
Do not set `ECOPLATE_HOST=0.0.0.0` unless you understand that the app has no login and would then be open to your network.

---

## 9. Known limitations

* Single-computer prototype: no user accounts or access control, and the Flask development server is used.
* The surplus register is stored in the browser, not in the database (full backups include it).
* Forecast quality depends entirely on record quality and amount. Forecasts for dates far after the latest record
  are marked low reliability.
* Evaluation on sample data is not real-world accuracy. Real validation needs real canteen records.

## Files

`index.html` frontend · `app.py` Flask server and API · `ecoplate_ml.py` model (supplied, unmodified) ·
`requirements.txt` · `requirements-dev.txt` (optional browser test) · `tests/` · `ecoplate.db` is created on first run.
