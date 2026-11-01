"""EcoPlate configuration, optional access control and rate limiting.

Access control is OPT-IN so an existing deployment keeps working after an upgrade:
* ECOPLATE_ACCESS_CODE not set  -> the app behaves as before (open). A warning is shown in /api/health and the UI.
* ECOPLATE_ACCESS_CODE set      -> every /api route except /api/health and /api/auth/* needs a login.
* ECOPLATE_ADMIN_CODE set       -> destructive routes (delete, restore, reset) additionally need the admin code.
                                   If it is not set, the access code also grants admin rights.

Secrets are read from environment variables only. Nothing here logs a secret or a code.
"""
from __future__ import annotations

import hmac
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from functools import wraps

from flask import current_app, jsonify, request
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

COOKIE = "ecoplate_session"
SESSION_SECONDS = 12 * 3600
ROLE_RANK = {"staff": 1, "admin": 2}


def _int(name, default, lo, hi):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        v = int(raw)
    except ValueError:
        return default
    return max(lo, min(hi, v))


@dataclass(frozen=True)
class Settings:
    access_code: str = field(repr=False)
    admin_code: str = field(repr=False)
    secret_key: str = field(repr=False)
    llm_api_key: str = field(repr=False)
    llm_model: str
    llm_timeout: int
    llm_daily_limit: int
    assistant_per_minute: int
    cookie_secure: bool

    @property
    def auth_enabled(self):
        return bool(self.access_code)

    @property
    def llm_configured(self):
        return bool(self.llm_api_key)


def load_settings(env=None):
    e = os.environ if env is None else env
    def g(k, d=""):
        return (e.get(k) or d).strip()
    def gi(k, d, lo, hi):
        try:
            return max(lo, min(hi, int(g(k, str(d)))))
        except ValueError:
            return d
    return Settings(
        access_code=g("ECOPLATE_ACCESS_CODE"), admin_code=g("ECOPLATE_ADMIN_CODE"),
        secret_key=g("ECOPLATE_SECRET_KEY"), llm_api_key=g("ANTHROPIC_API_KEY"),
        llm_model=g("ECOPLATE_LLM_MODEL", "claude-haiku-5-5"),
        llm_timeout=gi("ECOPLATE_LLM_TIMEOUT", 20, 3, 60),
        llm_daily_limit=gi("ECOPLATE_LLM_DAILY_LIMIT", 200, 0, 100000),
        assistant_per_minute=gi("ECOPLATE_ASSISTANT_PER_MINUTE", 6, 1, 120),
        cookie_secure=g("ECOPLATE_COOKIE_SECURE", "1") not in ("0", "false", "no"))


def validate_settings(s):
    """Returns (fatal_problems, warnings). Fatal problems stop the app from starting."""
    fatal, warn = [], []
    if s.auth_enabled:
        if len(s.secret_key) < 16:
            fatal.append("ECOPLATE_ACCESS_CODE is set but ECOPLATE_SECRET_KEY is missing or shorter than 16 characters. "
                         "Set ECOPLATE_SECRET_KEY to a long random string (it signs login cookies).")
        if len(s.access_code) < 8:
            warn.append("ECOPLATE_ACCESS_CODE is shorter than 8 characters; use a longer code.")
        if s.admin_code and hmac.compare_digest(s.admin_code, s.access_code):
            warn.append("ECOPLATE_ADMIN_CODE equals ECOPLATE_ACCESS_CODE, so there is no separation of roles.")
    else:
        warn.append("ECOPLATE_ACCESS_CODE is not set: anyone who can open this site can read, change and delete data. "
                    "Set it before using EcoPlate with real canteen records.")
    if s.admin_code and not s.auth_enabled:
        warn.append("ECOPLATE_ADMIN_CODE is ignored because ECOPLATE_ACCESS_CODE is not set.")
    if not s.llm_configured:
        warn.append("ANTHROPIC_API_KEY is not set: the assistant will answer from data only (no AI wording).")
    return fatal, warn


# ------------------------------------------------------------------ rate limiting
class RateLimiter:
    """Sliding-window limiter, in memory (per server process). Good enough to stop a runaway browser or a
    casual abuser; it is not a substitute for a gateway-level limit on a large public service."""

    def __init__(self):
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, bucket, key, limit, window_seconds, now=None):
        now = time.monotonic() if now is None else now
        with self._lock:
            q = self._hits[(bucket, key)]
            while q and now - q[0] >= window_seconds:
                q.popleft()
            if len(q) >= limit:
                return False, max(1, int(window_seconds - (now - q[0])) + 1)
            q.append(now)
            if len(self._hits) > 5000:                       # bound memory
                for k in [k for k, v in self._hits.items() if not v][:2500]:
                    self._hits.pop(k, None)
            return True, 0


def client_ip():
    """Render (and most proxies) APPEND the real client address to X-Forwarded-For, so the right-most entry is the
    one added by the trusted proxy; the left-most entries can be forged by the client."""
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        last = xff.split(",")[-1].strip()
        if last:
            return last[:64]
    return request.remote_addr or "unknown"


# ------------------------------------------------------------------ sessions
def _serializer(settings):
    return URLSafeTimedSerializer(settings.secret_key, salt="ecoplate-session")


def make_token(settings, role):
    return _serializer(settings).dumps({"role": role})


def current_role(settings):
    """'admin' / 'staff' / None. When auth is disabled everyone is effectively admin (original behaviour)."""
    if not settings.auth_enabled:
        return "admin"
    tok = request.cookies.get(COOKIE)
    if not tok:
        return None
    try:
        data = _serializer(settings).loads(tok, max_age=SESSION_SECONDS)
    except (BadSignature, SignatureExpired):
        return None
    role = data.get("role") if isinstance(data, dict) else None
    return role if role in ROLE_RANK else None


def check_login(settings, code):
    """Returns the role for a code, or None. Constant-time comparison."""
    if not isinstance(code, str) or not code:
        return None
    if settings.admin_code and hmac.compare_digest(code.encode(), settings.admin_code.encode()):
        return "admin"
    if hmac.compare_digest(code.encode(), settings.access_code.encode()):
        return "admin" if not settings.admin_code else "staff"
    return None


def require(role="staff"):
    """Route decorator: 401 if not logged in, 403 if the role is too low. No-op when auth is disabled."""
    def deco(fn):
        @wraps(fn)
        def wrapper(*a, **kw):
            settings = current_app.extensions["ecoplate_settings"]
            have = current_role(settings)
            if have is None:
                return jsonify({"ok": False, "error": "Login required.", "code": "login_required"}), 401
            if ROLE_RANK[have] < ROLE_RANK[role]:
                return jsonify({"ok": False, "error": "This action needs the administrator code.",
                                "code": "admin_required"}), 403
            return fn(*a, **kw)
        return wrapper
    return deco
