# Deployment guide (Render + GitHub)

## 0. What changed for deployment
- Entry point unchanged: `gunicorn app:app` (lazy WSGI object).
- `requirements.txt` unchanged (no new production dependency). `requirements-dev.txt` is for tests only.
- On first start the database is **migrated in place** (adds six tables; never touches `meal_records`). If the
  database already contains records, a consistent copy is written first, next to it:
  `<db>.pre-migration-<timestamp>.bak`.
- The app is **open exactly as before** until you set `ECOPLATE_ACCESS_CODE`. Do that before using real records.

## 1. Environment variables
| Variable | Required? | Purpose |
|---|---|---|
| `ECOPLATE_ACCESS_CODE` | Strongly recommended | Turns on login. Anyone logged in is "staff". Use 12+ characters. |
| `ECOPLATE_SECRET_KEY` | **Required if the access code is set** (16+ chars; the app refuses to start otherwise) | Signs login cookies. Generate: `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `ECOPLATE_ADMIN_CODE` | Optional | If set, only this code can delete/replace/reset data; the access code becomes staff-only. |
| `ECOPLATE_DB` | **Required for persistence** | Full path of the SQLite file, e.g. `/var/data/ecoplate.db` (inside the Render disk mount). |
| `ECOPLATE_TIMEZONE` | Optional (default `Asia/Kolkata`) | Defines "today" and "now". |
| `ANTHROPIC_API_KEY` | Optional | Enables AI-worded assistant answers. Never put it in GitHub or `index.html`. |
| `ECOPLATE_LLM_MODEL` | Optional (default `claude-haiku-5-5`) | Model used for wording. |
| `ECOPLATE_LLM_DAILY_LIMIT` | Optional (default 200; 0 disables AI wording) | Cost cap, counted per server process per day. |
| `ECOPLATE_LLM_TIMEOUT` | Optional (default 20 s) | Seconds to wait for the AI service. |
| `ECOPLATE_ASSISTANT_PER_MINUTE` | Optional (default 6) | Questions per minute per client. |
| `ECOPLATE_COOKIE_SECURE` | Optional (default 1) | Set `0` only for plain-http local testing. |

## 2. Persistence (read this)
Render services have an **ephemeral filesystem** by default: any file written after boot, including `ecoplate.db`,
is **lost on every redeploy or restart**. Persistent disks are available only on paid instance types, are attached
to a single instance, and only the mount path is preserved (Render docs, checked Nov 2026).

Recommended for a real canteen:
1. Upgrade the service to a paid instance type.
2. Dashboard → your service → **Disks** → add a disk, mount path `/var/data`.
3. Set `ECOPLATE_DB=/var/data/ecoplate.db`. Redeploy.
4. Keep one instance and use `gunicorn app:app --workers 1 --threads 4 --timeout 60` (the model cache, rate limiter and
   AI daily counter live in memory per worker; SQLite is happiest with a single writer process).
5. Download a **JSON backup** from the Backup & Reset tab regularly (e.g. weekly). The disk also has daily snapshots.

On the free plan, treat the app as a demo: data will disappear. Do not switch databases silently; if you later
outgrow SQLite, Render Postgres is the natural move, but it needs a code migration that is **not** part of this upgrade.

## 3. Upgrading the live site safely
1. **Before anything**: open the live site → Backup & Reset → *Download full backup (JSON)*. Keep the file.
2. Create a branch, copy the new files in, run the tests locally (section 5 of the README), commit, push the branch.
3. Merge to `main` (Render auto-deploys). Watch the deploy logs for `Database migrated to schema version 1`.
4. Open the site, confirm *Dashboard* shows your records and the *Insights/Alerts/Assistant* tabs open.
5. Set `ECOPLATE_ACCESS_CODE`, `ECOPLATE_SECRET_KEY` (and optionally the admin code) → the service restarts → log in.
6. Open the *Surplus* tab once: any entries saved in that browser are copied to the server automatically (additive;
   the browser copy is kept).
7. Roll back: redeploy the previous commit. The migration only added tables, so the old code still reads
   `meal_records`. (On a persistent disk the `.bak` copy is also there.)

## 4. Never commit
`*.db`, `*.db-wal`, `*.db-shm`, `*.bak`, `.env`. They are in `.gitignore`. Rotate any key that was ever pasted into
a chat, issue or commit.

## 5. Startup/shutdown behaviour
- Config problems are logged once at startup. Fatal ones (access code without a secret) stop the app with a clear
  message in the Render logs (first request returns HTTP 500 because the WSGI object is created lazily).
- SIGTERM from Render ends in-flight requests normally; SQLite uses WAL; multi-step writes are transactions.
- If the AI service is down, slow or out of credit, only the *wording* degrades: the assistant shows your data directly.
