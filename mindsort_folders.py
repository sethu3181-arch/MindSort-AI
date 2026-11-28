"""
MindSort AI - adaptive Gmail folders (multi-user version).

Every card belongs to one user_id, and every read filters by it.
Put this file next to app.py. See the three edits in the chat.
Nothing here writes to Gmail. It only reads data your app already fetched.
"""
import html
import re
import sqlite3
from datetime import date, datetime, timedelta
from urllib.parse import quote

_connect = None


def init(connect_db):
    """Give this module your app's connect_db function and create the table."""
    global _connect
    _connect = connect_db
    conn = _connect()
    cols = [r[1] for r in conn.execute("PRAGMA table_info(folder_cards)").fetchall()]
    if cols and "user_id" not in cols:
        # Old single-user table: drop it. Cards are rebuilt on the next inbox fetch.
        conn.execute("DROP TABLE folder_cards")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS folder_cards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            gmail_account TEXT NOT NULL,
            gmail_message_id TEXT NOT NULL,
            subject TEXT,
            sender TEXT,
            folder TEXT,
            key_points TEXT,
            open_date TEXT,
            close_date TEXT,
            received_at TEXT,
            link TEXT,
            search_text TEXT,
            UNIQUE(user_id, gmail_account, gmail_message_id)
        )
    """)
    conn.commit()
    conn.close()


# ------------------------------------------------------------
# 1. ADAPTIVE FOLDERS
# Folders only appear when emails land in them, so a student sees
# Study / Hackathon and a professional sees Work / Jobs / Finance.
# Add your own folder by adding a line here.
# ------------------------------------------------------------
FOLDER_RULES = {
    "Hackathon": ["hackathon", "hack-a-thon", "buildathon", "devpost", "devfolio",
                  "unstop", "coding competition", "innovation challenge"],
    "Study": ["assignment", "exam", "semester", "syllabus", "lecture", "course",
              "coursework", "homework", "professor", "university", "college",
              "quiz", "timetable", "internal marks", "lab record"],
    "Work": ["meeting", "standup", "sprint", "client", "stakeholder", "agenda",
             "action items", "quarterly", "jira", "confluence", "pull request",
             "deployment", "onboarding", "project update", "performance review"],
    "Jobs & Career": ["interview", "recruiter", "job opening", "your application",
                      "offer letter", "hiring", "resume", "candidate", "naukri",
                      "linkedin", "position"],
    "Finance & Bills": ["invoice", "payment", "bill", "receipt", "statement",
                        "due date", "emi", "subscription", "refund", "transaction",
                        "payslip", "salary", "tax"],
    "Events": ["webinar", "workshop", "conference", "meetup", "seminar", "summit",
               "rsvp", "masterclass", "bootcamp", "event"],
    "Travel": ["flight", "itinerary", "pnr", "hotel", "boarding pass", "train",
               "booking"],
    "Orders & Shopping": ["order", "shipped", "delivered", "delivery", "tracking",
                          "amazon", "flipkart"],
    "Security & Accounts": ["password", "verification code", "otp", "security alert",
                            "new login", "two-factor", "sign-in"],
    "Newsletters & Promotions": ["unsubscribe", "newsletter", "% off", "discount",
                                 "promo", "sale"],
}
WEIGHT = {"Newsletters & Promotions": 0.5}
MIN_SCORE = 2  # below this, the email goes to Uncategorized


def _hits(text, keywords):
    return sum(1 for k in keywords if re.search(r"\b" + re.escape(k), text))


def classify_folder(subject, body):
    subject, body = (subject or "").lower(), (body or "").lower()[:3000]
    best, best_score = "Uncategorized", 0.0
    for folder, words in FOLDER_RULES.items():
        score = (2 * _hits(subject, words) + _hits(body, words)) * WEIGHT.get(folder, 1)
        if score > best_score:
            best, best_score = folder, score
    return best if best_score >= MIN_SCORE else "Uncategorized"


# ------------------------------------------------------------
# 2. DATE DETECTION (never invents dates)
# ------------------------------------------------------------
MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
          "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
MON = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|"
       r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
DMY = re.compile(rf"\b(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<m>{MON})\b\.?"
                 rf"(?:,?\s+(?P<y>\d{{4}}))?", re.I)
MDY = re.compile(rf"\b(?P<m>{MON})\b\.?\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\b"
                 rf"(?:,?\s+(?P<y>\d{{4}}))?", re.I)
ISO = re.compile(r"\b(?P<y>\d{4})-(?P<mo>\d{2})-(?P<d>\d{2})\b")
SHORT_RANGE = re.compile(rf"\b(?P<d1>\d{{1,2}})(?:st|nd|rd|th)?\s*(?:-|–|—|to|till|until)\s*"
                         rf"(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<m>{MON})\b\.?"
                         rf"(?:,?\s+(?P<y>\d{{4}}))?", re.I)
JOINER = re.compile(r"^\s*(?:-|–|—|to|till|until|through)\s*$", re.I)

OPEN_WORDS = ["opens", "open from", "opening", "starts", "start date", "begins",
              "commences", "from", "launch"]
CLOSE_WORDS = ["deadline", "last date", "closes", "closing", "close on", "due",
               "ends", "end date", "submit by", "submissions by", "register by",
               "registrations by", "apply by", "on or before", "until", "before"]


def _make_date(day, month, year, received):
    try:
        month = MONTHS[month.lower()[:3]] if isinstance(month, str) else month
        if year:
            return date(int(year), month, int(day))
        d = date(received.year, month, int(day))   # year taken from the email's own date
        if d < received - timedelta(days=30):
            d = date(received.year + 1, month, int(day))
        return d
    except (ValueError, KeyError):
        return None


def find_dates(text, received):
    """Returns sorted list of (start, end, date, short_range_start_date_or_None)."""
    found, taken = [], []

    def free(a, b):
        return all(b <= s or a >= e for s, e in taken)

    for m in SHORT_RANGE.finditer(text):
        end = _make_date(m["d"], m["m"], m["y"], received)
        start = _make_date(m["d1"], m["m"], m["y"], received)
        if end and start and free(m.start(), m.end()):
            found.append((m.start(), m.end(), end, start))
            taken.append((m.start(), m.end()))
    for rx in (DMY, MDY):
        for m in rx.finditer(text):
            d = _make_date(m["d"], m["m"], m["y"], received)
            if d and free(m.start(), m.end()):
                found.append((m.start(), m.end(), d, None))
                taken.append((m.start(), m.end()))
    for m in ISO.finditer(text):
        try:
            d = date(int(m["y"]), int(m["mo"]), int(m["d"]))
        except ValueError:
            continue
        if free(m.start(), m.end()):
            found.append((m.start(), m.end(), d, None))
            taken.append((m.start(), m.end()))
    return sorted(found, key=lambda x: x[0])


def _nearest(window, words):
    return max((window.rfind(w) for w in words), default=-1)


def detect_open_close(text, received):
    """Returns (open_iso or None, close_iso or None). Only dates found in the text."""
    text = text or ""
    items = find_dates(text, received)
    opens, closes = [], []
    i = 0
    while i < len(items):
        s, e, d, range_start = items[i]
        if range_start:                                  # "10 - 30 November 2026"
            opens.append(range_start)
            closes.append(d)
        elif i + 1 < len(items) and JOINER.match(text[e:items[i + 1][0]]):
            opens.append(d)                              # "10 Nov to 30 Nov 2026"
            closes.append(items[i + 1][2])
            i += 1
        else:
            window = text[max(0, s - 50):s].lower()
            o, c = _nearest(window, OPEN_WORDS), _nearest(window, CLOSE_WORDS)
            if c > o:
                closes.append(d)
            elif o > c:
                opens.append(d)
        i += 1
    today = date.today()
    upcoming = [d for d in closes if d >= today]
    close = min(upcoming) if upcoming else (max(closes) if closes else None)
    opening = min(opens) if opens else None
    return (opening.isoformat() if opening else None,
            close.isoformat() if close else None)


# ------------------------------------------------------------
# 3. KEY POINTS (sentences taken from the email, not written by AI)
# ------------------------------------------------------------
ACTION_WORDS = ["register", "registration", "submit", "submission", "deadline",
                "apply", "join", "eligible", "eligibility", "prize", "invited",
                "required", "must", "open", "exam", "assignment", "meeting",
                "payment", "due", "participants", "finalist", "team", "venue",
                "schedule", "fee", "certificate", "attend", "bring"]


def extract_key_points(body, limit=3):
    lines = []
    for line in (body or "").splitlines():
        low = line.strip().lower()
        if not low or low.startswith(">") or "unsubscribe" in low or "view in browser" in low:
            continue
        if re.fullmatch(r"\S*https?://\S+", low):
            continue
        lines.append(line.strip())
    text = " ".join(lines)
    parts = re.split(r"(?<=[.!?])\s+|\s[•\-*]\s", text)
    cands = []
    for p in parts:
        p = p.strip(" -•*\t")
        if 25 <= len(p) <= 300 and not re.match(r"(hi|hello|dear|thanks|regards|best)\b", p.lower()):
            cands.append(p)
    scored = [(sum(w in c.lower() for w in ACTION_WORDS), i, c) for i, c in enumerate(cands)]
    top = sorted([s for s in scored if s[0] > 0], key=lambda s: (-s[0], s[1]))[:limit]
    chosen = [c for _, _, c in sorted(top, key=lambda s: s[1])] or cands[:2]
    return [c if len(c) <= 160 else c[:157].rstrip() + "…" for c in chosen]


# ------------------------------------------------------------
# 4. SAVE A CARD (call from the Gmail import loop)
# ------------------------------------------------------------
def save_folder_card(cursor, user_id, account, message_id, subject, sender, body, received):
    """received: a datetime. Uses INSERT OR IGNORE so re-imports never duplicate."""
    body = body or ""
    full_text = f"{subject}\n{body}"
    open_iso, close_iso = detect_open_close(full_text, received.date())
    link = f"https://mail.google.com/mail/u/?authuser={quote(account)}#all/{message_id}"
    cursor.execute("""
        INSERT OR IGNORE INTO folder_cards
        (user_id, gmail_account, gmail_message_id, subject, sender, folder,
         key_points, open_date, close_date, received_at, link, search_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, account, message_id, subject, sender, classify_folder(subject, body),
          "\n".join(extract_key_points(body)), open_iso, close_iso,
          received.isoformat(), link, body[:1500]))


# ------------------------------------------------------------
# 5. READING FOLDERS AND DRAWING CARDS
# ------------------------------------------------------------
SORT_OPTIONS = ["Closing date (earliest first)", "Closing date (latest first)",
                "Opening date (earliest first)", "Opening date (latest first)",
                "Newest emails"]


def _rows(sql, params=()):
    conn = _connect()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return rows


def has_card(cursor, user_id, account, message_id):
    return cursor.execute(
        "SELECT 1 FROM folder_cards WHERE user_id = ? AND gmail_account = ? "
        "AND gmail_message_id = ?", (user_id, account, message_id)).fetchone() is not None


def list_accounts(user_id):
    if not user_id:
        return []
    return [r[0] for r in _rows(
        "SELECT DISTINCT gmail_account FROM folder_cards WHERE user_id = ? "
        "ORDER BY gmail_account", (user_id,))]


def folder_choices(user_id, account):
    if not user_id or not account:
        return []
    rows = _rows("""SELECT folder, COUNT(*) AS n FROM folder_cards
                    WHERE user_id = ? AND gmail_account = ? GROUP BY folder""",
                 (user_id, account))
    rows = sorted(rows, key=lambda r: (r["folder"] == "Uncategorized", -r["n"]))
    return [f"{r['folder']} ({r['n']})" for r in rows]


def _fmt(iso):
    if not iso:
        return "Not mentioned"
    d = date.fromisoformat(iso)
    return f"{d.day} {d:%B %Y}"


def _badge(close_iso):
    if not close_iso:
        return ""
    left = (date.fromisoformat(close_iso) - date.today()).days
    if left < 0:
        return '<span class="ms-badge ms-closed">Closed</span>'
    if left == 0:
        return '<span class="ms-badge ms-soon">Closes today</span>'
    cls = "ms-soon" if left <= 3 else "ms-open"
    return f'<span class="ms-badge {cls}">{left} day{"s" if left != 1 else ""} left</span>'


STYLE = """<style>
.ms-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px}
.ms-card{border:1px solid var(--border-color-primary);border-radius:10px;padding:14px;
 background:var(--background-fill-secondary);color:var(--body-text-color)}
.ms-head{display:flex;justify-content:space-between;gap:8px;align-items:flex-start}
.ms-title{font-weight:600;font-size:1.02em;color:var(--body-text-color);text-decoration:none}
.ms-title:hover{text-decoration:underline}
.ms-from{font-size:.82em;opacity:.7;margin:4px 0 8px}
.ms-card ul{margin:0 0 10px 18px;padding:0;font-size:.9em}
.ms-card li{margin-bottom:4px}
.ms-dates{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.ms-box{border:1px solid var(--border-color-primary);border-radius:8px;padding:8px;
 font-size:.88em;background:var(--background-fill-primary)}
.ms-box span{display:block;font-size:.78em;opacity:.7;margin-bottom:2px}
.ms-badge{font-size:.75em;padding:2px 8px;border-radius:999px;white-space:nowrap;
 border:1px solid currentColor}
.ms-open{color:#1a7f4b}.ms-soon{color:#b45309}.ms-closed{color:#9a3333}
.ms-empty{padding:18px;opacity:.7}
</style>"""


def message_html(text):
    return STYLE + f'<div class="ms-empty">{html.escape(text)}</div>'


def render_folder(user_id, account, folder_label, keyword, sort_by):
    if not user_id or not account or not folder_label:
        return STYLE + '<div class="ms-empty">Choose an account and a folder to see its emails.</div>'
    folder = folder_label.rsplit(" (", 1)[0]
    rows = [dict(r) for r in _rows(
        "SELECT * FROM folder_cards WHERE user_id = ? AND gmail_account = ? "
        "AND folder = ?", (user_id, account, folder))]
    kw = (keyword or "").strip().lower()
    if kw:
        rows = [r for r in rows if kw in " ".join(
            [r["subject"] or "", r["sender"] or "", r["key_points"] or "",
             r["search_text"] or ""]).lower()]

    col, rev = {"Closing date (earliest first)": ("close_date", False),
                "Closing date (latest first)": ("close_date", True),
                "Opening date (earliest first)": ("open_date", False),
                "Opening date (latest first)": ("open_date", True),
                "Newest emails": ("received_at", True)}.get(sort_by, ("close_date", False))
    dated = sorted([r for r in rows if r[col]], key=lambda r: r[col], reverse=rev)
    rows = dated + [r for r in rows if not r[col]]       # missing dates always last

    if not rows:
        return STYLE + '<div class="ms-empty">No emails match in this folder.</div>'

    cards = []
    for r in rows:
        points = [p for p in (r["key_points"] or "").split("\n") if p]
        lis = "".join(f"<li>{html.escape(p)}</li>" for p in points) or "<li>No key points detected</li>"
        received = (r["received_at"] or "")[:10]
        cards.append(f"""
<div class="ms-card">
  <div class="ms-head">
    <a class="ms-title" href="{html.escape(r['link'] or '#')}" target="_blank" rel="noopener">{html.escape(r['subject'] or '(No subject)')}</a>
    {_badge(r['close_date'])}
  </div>
  <div class="ms-from">{html.escape(r['sender'] or '')} · received {html.escape(received)}</div>
  <ul>{lis}</ul>
  <div class="ms-dates">
    <div class="ms-box"><span>Opening date</span>{_fmt(r['open_date'])}</div>
    <div class="ms-box"><span>Closing date</span>{_fmt(r['close_date'])}</div>
  </div>
</div>""")
    return STYLE + f'<div class="ms-grid">{"".join(cards)}</div>'
