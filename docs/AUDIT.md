# EcoPlate audit (Stage 1)

Audit of the uploaded repository (`ecoplate-wsgi-updated.zip`) **before** any change. Everything below was checked
against the actual files, not assumed from the brief. Baseline: 39 automated tests, all passing; 58 browser checks
(Playwright) all passing.

## Existing functionality (verified)
- **Backend (`app.py`)**: Flask factory `create_app()` + lazy WSGI object `app` (`gunicorn app:app`). SQLite with one
  table, `meal_records` (date, meal, prepared, leftovers, source, batch, synthetic, created_at; unique on date+meal).
  Routes: health, list/add/bulk-add records, delete one / by import batch / all sample rows, full replace (with a typed
  confirmation token), model report/train, forecast. Validation and JSON error handling are good; replace/bulk use
  transactions.
- **Frontend (`index.html`, one file, ~1450 lines)**: dashboard, forecast, model report, record meal, bulk CSV import
  (preview + validation), history, surplus register, JSON/CSV backup, restore, merge, reset, migration of an older
  browser-only format.
- **Keep-alive**: a GitHub Actions workflow pings the Render URL.

## Existing ML (verified)
`ecoplate_ml.py`: `HistGradientBoostingRegressor` predicting meals eaten (prepared − leftovers) from meal, weekday,
recent history and optional month; compared with three baselines on a chronological 80/20 hold-out; the model is used
only if it beats the best baseline; rule-based fallback otherwise; per-meal minimum rows; permutation importance;
cached by a data fingerprint (no retrain per request). This is a sound design and was **kept**, not replaced.

## Problems found
| # | Finding | Severity | Resolution |
|---|---|---|---|
| 1 | **No authentication anywhere.** Anyone with the URL can read, add, replace or delete all records (including `DELETE /api/records`, `POST /api/records/replace`). | High | Optional access code + separate admin code; destructive routes admin-only; CSRF header; rate-limited login. Opt-in so the live app is not locked out on upgrade. |
| 2 | **Surplus register lived only in the browser (`localStorage`)** — not shared between staff, lost with the browser, no audit trail, nothing the server or an assistant could read. | High | Server tables with status workflow and audit log; one-time additive copy of the old browser register. |
| 3 | **SQLite on Render's default filesystem is ephemeral**; a restart/redeploy erases the database. | High | Cannot be fixed in code. See `DEPLOYMENT.md` (persistent disk on a paid plan, `ECOPLATE_DB`, regular backups). |
| 4 | **"Today" used the server's UTC date.** For a canteen in India, records for "today" are rejected as future dates for the first 5.5 hours after midnight IST. | Medium | `ECOPLATE_TIMEZONE` (default `Asia/Kolkata`). |
| 5 | Accuracy gate was a single hold-out block (small and noisy on a few hundred rows). Uncertainty band came from ~12+ rows. | Medium | Added rolling-origin evaluation as a second gate, per-meal gate, finite-sample band needing ≥ 20 out-of-sample errors. |
| 6 | Demand is measured as *eaten* quantity. If food ran out, true demand is higher (censoring); nothing warned about it. | Medium | Warning when ≥ 20 % of recent services ended with 0 leftovers. |
| 7 | No security headers / CSP. | Low | Added (CSP forbids external scripts, `eval`, framing). |
| 8 | `/api/health` exposed record counts publicly. | Low | Counts shown only to logged-in users when auth is on. |
| 9 | Stale `.pytest_cache/lastfailed` in the zip. | Info | Ignored; `.gitignore` extended. |

Not verifiable from the repo: Render start command and environment (no `render.yaml`/`Procfile` in the zip). The
WSGI object works; I assume `gunicorn app:app`.

## Dependencies and costs
- **No new production dependencies.** `itsdangerous` (login cookies) already ships with Flask. The LLM call uses
  Python's standard library (`urllib`). `requirements.txt` is unchanged.
- **Optional AI wording** (Anthropic API, `claude-haiku-5-5`): $0.10 per million input tokens and $0.50 per million
  output tokens for prompts up to 100K tokens (checked October 2026; confirm on the current pricing page). One question
  uses about 3,000 input tokens (system prompt + facts, hard-capped at 9,000 characters) and at most 450 output tokens,
  i.e. roughly $0.0005 or less. With the default cap of 200 AI answers/day the worst case is about $0.10/day.
  Without an API key the assistant still works in data-only mode at zero cost.
- **Hosting**: a Render persistent disk needs a paid instance (see Render pricing); that is the only recurring cost
  that real use effectively requires.

## ML anomaly detector: evaluated, not adopted
With tens to a few hundred sessions per meal and no labelled examples, Isolation Forest / one-class SVM cannot be
validated and cannot explain *why* a record was flagged. A median/MAD threshold against the previous 28 sessions of
the same meal is explainable and leakage-free. The review log now records "looks fine / recorded in error / dismissed"
so labelled data accumulates and the question can be revisited.
