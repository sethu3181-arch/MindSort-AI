"""
MindSort AI - sign-up, login and sessions.

Passwords are never stored. Only a salted PBKDF2 hash is kept.
Optional Render environment variables:
  MINDSORT_SECRET_KEY      long random text that signs login cookies
  MINDSORT_INVITE_CODE     if set, sign-up requires this code
  MINDSORT_OWNER_USERNAME  this user adopts messages saved before multi-user
"""
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
from datetime import datetime
from html import escape

_connect = None
USERNAME_RE = re.compile(r"^[a-z0-9_.-]{3,30}$")
MIN_PASSWORD, MAX_PASSWORD = 10, 200
ITERATIONS = 600_000
MAX_FAILS, WINDOW = 8, 600          # 8 wrong passwords per 10 minutes per username
_failed = {}


def init(connect_db):
    global _connect
    _connect = connect_db
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()


# ------------------------------------------------------------
# Passwords
# ------------------------------------------------------------
def hash_password(password):
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    return f"pbkdf2_sha256${ITERATIONS}${salt.hex()}${digest.hex()}"


def check_password(password, stored):
    try:
        _, iters, salt, digest = stored.split("$")
        calc = hashlib.pbkdf2_hmac("sha256", password.encode(),
                                   bytes.fromhex(salt), int(iters))
        return hmac.compare_digest(calc.hex(), digest)
    except (ValueError, TypeError):
        return False


_DUMMY_HASH = hash_password("not-a-real-password")   # keeps timing equal for unknown users


# ------------------------------------------------------------
# Users
# ------------------------------------------------------------
def create_user(username, password, confirm, invite):
    """Returns (user_id, None) or (None, error message)."""
    username = (username or "").strip().lower()
    password, confirm = password or "", confirm or ""
    required = os.environ.get("MINDSORT_INVITE_CODE", "")
    if required and not hmac.compare_digest((invite or "").strip().encode(), required.encode()):
        return None, "That invite code is not correct."
    if not USERNAME_RE.match(username):
        return None, "Username must be 3 to 30 characters: letters, numbers, dot, dash or underscore."
    if len(password) < MIN_PASSWORD:
        return None, f"Password must be at least {MIN_PASSWORD} characters."
    if len(password) > MAX_PASSWORD:
        return None, "Password is too long."
    if password != confirm:
        return None, "The two passwords do not match."

    conn = _connect()
    try:
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, hash_password(password), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        user_id = cur.lastrowid
        owner = os.environ.get("MINDSORT_OWNER_USERNAME", "").strip().lower()
        if owner and owner == username:
            conn.execute("UPDATE messages SET user_id = ? WHERE user_id IS NULL", (user_id,))
        conn.commit()
        return user_id, None
    except sqlite3.IntegrityError:
        return None, "That username is already taken."
    finally:
        conn.close()


def verify_login(username, password):
    """Returns (user_id, None) or (None, error message)."""
    username = (username or "").strip().lower()
    now = time.time()
    recent = [t for t in _failed.get(username, []) if now - t < WINDOW]
    _failed[username] = recent
    if len(recent) >= MAX_FAILS:
        return None, "Too many failed attempts. Please wait a few minutes and try again."

    conn = _connect()
    row = conn.execute("SELECT id, password_hash FROM users WHERE username = ?",
                       (username,)).fetchone()
    conn.close()
    ok = check_password(password or "", row[1] if row else _DUMMY_HASH)
    if row and ok:
        _failed.pop(username, None)
        return row[0], None
    _failed.setdefault(username, []).append(now)
    return None, "Wrong username or password."


def session_user_id(request):
    """The logged-in user's id, or None. The username must match, so an old
    cookie can never log in as a different person after a database reset."""
    try:
        user_id = request.session.get("user_id")
        name = request.session.get("username")
    except Exception:
        return None
    if not user_id or not name:
        return None
    conn = _connect()
    row = conn.execute("SELECT id FROM users WHERE id = ? AND username = ?",
                       (user_id, name)).fetchone()
    conn.close()
    return row[0] if row else None


def require_user(request):
    """Used by Gradio as auth_dependency. Not logged in -> go to the login page."""
    from fastapi import HTTPException
    user_id = session_user_id(request)
    if user_id:
        return str(user_id)
    raise HTTPException(status_code=307, headers={"Location": "/login"})


# ------------------------------------------------------------
# Pages
# ------------------------------------------------------------
PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MindSort AI - {title}</title>
<style>
:root{{--bg:#f6f7fb;--card:#fff;--text:#1d2433;--muted:#667085;--line:#d9dde7;--accent:#3b5bdb;--bad:#b42318}}
@media (prefers-color-scheme: dark){{:root{{--bg:#12151c;--card:#1b1f2a;--text:#e8eaf0;--muted:#98a2b3;--line:#2d3342;--accent:#7c93ff;--bad:#ff8b7e}}}}
*{{box-sizing:border-box}}
body{{margin:0;min-height:100vh;display:grid;place-items:center;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
main{{width:min(400px,92vw);background:var(--card);border:1px solid var(--line);border-radius:14px;padding:28px}}
h1{{font-size:1.35rem;margin:0 0 4px}} p.sub{{color:var(--muted);margin:0 0 20px;font-size:.95rem}}
label{{display:block;font-size:.88rem;margin:14px 0 6px}}
input{{width:100%;padding:10px 12px;border:1px solid var(--line);border-radius:8px;background:transparent;color:var(--text);font-size:1rem}}
input:focus{{outline:2px solid var(--accent);outline-offset:1px}}
button{{width:100%;margin-top:20px;padding:11px;border:0;border-radius:8px;background:var(--accent);color:#fff;font-size:1rem;font-weight:600;cursor:pointer}}
.err{{background:rgba(180,35,24,.1);color:var(--bad);border-radius:8px;padding:10px 12px;margin-bottom:6px;font-size:.92rem}}
.alt{{margin-top:18px;text-align:center;font-size:.92rem;color:var(--muted)}} a{{color:var(--accent)}}
</style></head><body><main>
<h1>MindSort AI</h1><p class="sub">{subtitle}</p>
{error}{form}
<div class="alt">{alt}</div>
</main></body></html>"""


def _page(title, subtitle, form, alt, error=""):
    err = f'<div class="err">{escape(error)}</div>' if error else ""
    return PAGE.format(title=title, subtitle=subtitle, form=form, alt=alt, error=err)


def login_page(error="", username=""):
    form = f"""<form method="post" action="/login">
<label for="u">Username</label>
<input id="u" name="username" value="{escape(username)}" autocomplete="username" required autofocus>
<label for="p">Password</label>
<input id="p" name="password" type="password" autocomplete="current-password" required>
<button type="submit">Log in</button></form>"""
    return _page("Log in", "Log in to your own private inbox organizer.", form,
                 'New here? <a href="/signup">Create an account</a>', error)


def signup_page(error="", username=""):
    invite = ""
    if os.environ.get("MINDSORT_INVITE_CODE"):
        invite = ('<label for="i">Invite code</label>'
                  '<input id="i" name="invite" autocomplete="off" required>')
    form = f"""<form method="post" action="/signup">
<label for="u">Choose a username</label>
<input id="u" name="username" value="{escape(username)}" autocomplete="username" required autofocus>
<label for="p">Password (at least {MIN_PASSWORD} characters)</label>
<input id="p" name="password" type="password" autocomplete="new-password" required>
<label for="c">Repeat password</label>
<input id="c" name="confirm" type="password" autocomplete="new-password" required>
{invite}
<button type="submit">Create account</button></form>"""
    return _page("Sign up", "Create an account. Your emails and folders stay private to you.",
                 form, 'Already have an account? <a href="/login">Log in</a>', error)


# ------------------------------------------------------------
# Routes (called once from app.py)
# ------------------------------------------------------------
def register_routes(server):
    from fastapi import Request
    from fastapi.responses import HTMLResponse, RedirectResponse

    def start_session(request, user_id, username):
        request.session.clear()                      # new session on every login
        request.session["user_id"] = user_id
        request.session["username"] = username

    @server.get("/")
    def home(request: Request):
        return RedirectResponse("/app/" if session_user_id(request) else "/login")

    @server.get("/login", response_class=HTMLResponse)
    def login_form(request: Request):
        if session_user_id(request):
            return RedirectResponse("/app/")
        return HTMLResponse(login_page())

    @server.post("/login")
    async def login_submit(request: Request):
        form = await request.form()
        username = str(form.get("username", "")).strip().lower()
        user_id, error = verify_login(username, str(form.get("password", "")))
        if error:
            return HTMLResponse(login_page(error, username), status_code=401)
        start_session(request, user_id, username)
        return RedirectResponse("/app/", status_code=303)

    @server.get("/signup", response_class=HTMLResponse)
    def signup_form(request: Request):
        if session_user_id(request):
            return RedirectResponse("/app/")
        return HTMLResponse(signup_page())

    @server.post("/signup")
    async def signup_submit(request: Request):
        form = await request.form()
        username = str(form.get("username", "")).strip().lower()
        user_id, error = create_user(username, str(form.get("password", "")),
                                     str(form.get("confirm", "")), str(form.get("invite", "")))
        if error:
            return HTMLResponse(signup_page(error, username), status_code=400)
        start_session(request, user_id, username)
        return RedirectResponse("/app/", status_code=303)

    @server.get("/logout")
    def logout(request: Request):
        request.session.clear()
        return RedirectResponse("/login")
