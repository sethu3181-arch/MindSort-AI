# ============================================================
# MINDSORT AI - SECURED DEPLOYMENT VERSION
# ============================================================
# All eight existing tabs are preserved.
#
# SECURITY:
# Set these environment variables in Render:
#   MINDSORT_USERNAME = your chosen login username
#   MINDSORT_PASSWORD = a long, unique password
#
# Never put your password in this file or commit it to GitHub.
# ============================================================

import os
import sqlite3
import tempfile
import base64
import secrets
import time
from email.utils import parsedate_to_datetime
from datetime import datetime, timedelta

import gradio as gr
import pandas as pd
from fastapi import FastAPI, Request
import uvicorn
from fastapi.responses import HTMLResponse
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# ============================================================
# 1. CONFIGURATION
# ============================================================

DB = os.environ.get("MINDSORT_DB_PATH", "mindsort.db")
MAX_MESSAGE_LENGTH = 5000
GOOGLE_REDIRECT_URI = "https://mindsort-ai.onrender.com/oauth2callback"
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# OAuth credentials stay in server memory only. They are never written to
# GitHub or displayed in the browser. Reconnect after a server restart.
pending_oauth = {}
gmail_connections = {}


# ============================================================
# 2. DATABASE
# ============================================================

def connect_db():
    return sqlite3.connect(DB, check_same_thread=False, timeout=30)


def setup_database():
    conn = connect_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT,
            category TEXT,
            is_task TEXT,
            deadline TEXT,
            priority TEXT DEFAULT 'Normal',
            completed INTEGER DEFAULT 0,
            created_at TEXT
        )
    """)

    existing_columns = [
        row[1] for row in cursor.execute(
            "PRAGMA table_info(messages)"
        ).fetchall()
    ]

    required_columns = {
        "priority": "TEXT DEFAULT 'Normal'",
        "completed": "INTEGER DEFAULT 0",
        "created_at": "TEXT"
    }

    for column, definition in required_columns.items():
        if column not in existing_columns:
            cursor.execute(
                f"ALTER TABLE messages ADD COLUMN {column} {definition}"
            )

    existing_columns = [
        row[1] for row in cursor.execute(
            "PRAGMA table_info(messages)"
        ).fetchall()
    ]
    for column, definition in {
        "gmail_account": "TEXT",
        "gmail_message_id": "TEXT"
    }.items():
        if column not in existing_columns:
            cursor.execute(
                f"ALTER TABLE messages ADD COLUMN {column} {definition}"
            )

    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_gmail_message_account
        ON messages(gmail_account, gmail_message_id)
        WHERE gmail_message_id IS NOT NULL
    """)

    conn.commit()
    conn.close()


setup_database()


import mindsort_folders as folders
folders.init(connect_db)

    


# ============================================================
# 3. AI CLASSIFICATION MODEL
# ============================================================

training_messages = [
    "Complete mathematics homework",
    "Submit physics assignment",
    "Prepare for chemistry exam",
    "Download class notes",
    "Study for semester examination",
    "Register for AI hackathon",
    "Submit hackathon project",
    "Attend hackathon team meeting",
    "Prepare project presentation",
    "Join coding competition",
    "Attend technical workshop",
    "Pay electricity bill",
    "Buy groceries",
    "Attend family function",
    "Prepare monthly report",
    "Attend office meeting",
    "Complete company project"
]

training_labels = [
    "Study", "Study", "Study", "Study", "Study",
    "Hackathon", "Hackathon", "Hackathon", "Hackathon",
    "Events", "Events", "Home", "Home", "Home",
    "Work", "Work", "Work"
]

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", MultinomialNB())
])

model.fit(training_messages, training_labels)


def classify_message(message):
    prediction = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]
    confidence = max(probabilities)
    return prediction, round(float(confidence) * 100, 2)


# ============================================================
# 4. PRIORITY, TASK AND DEADLINE DETECTION
# ============================================================

def detect_priority(message):
    text = message.lower()

    if any(word in text for word in [
        "urgent", "asap", "immediately", "critical"
    ]):
        return "High"

    if any(word in text for word in [
        "tomorrow", "important", "soon", "upcoming"
    ]):
        return "Medium"

    if any(word in text for word in [
        "optional", "whenever", "later", "no hurry"
    ]):
        return "Low"

    return "Normal"


def detect_task(message):
    words = [
        "complete", "submit", "prepare", "attend",
        "register", "pay", "buy", "finish",
        "download", "meeting", "deadline"
    ]

    text = message.lower()
    return "Yes" if any(word in text for word in words) else "No"


def detect_deadline(message):
    text = message.lower()

    if "tomorrow" in text:
        return (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    if "today" in text:
        return datetime.now().strftime("%Y-%m-%d")

    return "Not specified"


# ============================================================
# 5. ORGANIZE AND SAVE MESSAGE
# ============================================================

def organize_message(message):
    if not message or not message.strip():
        return "Please enter a message."

    message = message.strip()

    if len(message) > MAX_MESSAGE_LENGTH:
        return f"Please keep the message under {MAX_MESSAGE_LENGTH} characters."

    category_name, confidence = classify_message(message)
    priority = detect_priority(message)
    task = detect_task(message)
    deadline = detect_deadline(message)

    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO messages
        (message, category, is_task, deadline, priority, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        message,
        category_name,
        task,
        deadline,
        priority,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()

    return f"""
### Message organized successfully!

- **Category:** {category_name}
- **AI confidence:** {confidence}%
- **Task:** {task}
- **Priority:** {priority}
- **Deadline:** {deadline}
"""


# ============================================================
# 6. TASKS AND REMINDERS
# ============================================================

def get_reminders():
    conn = connect_db()

    df = pd.read_sql_query("""
        SELECT id, message, deadline, priority
        FROM messages
        WHERE is_task = 'Yes' AND completed = 0
        ORDER BY id DESC
    """, conn)

    conn.close()

    if df.empty:
        return "No pending tasks found."

    return df.to_string(index=False)


def complete_task(task_id):
    if task_id is None:
        return "Please enter a task ID."

    try:
        task_id = int(task_id)
    except (TypeError, ValueError, OverflowError):
        return "Please enter a valid task ID."

    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE messages
        SET completed = 1
        WHERE id = ?
    """, (task_id,))

    conn.commit()
    changed = cursor.rowcount
    conn.close()

    if changed:
        return "Task marked as completed."

    return "Task ID not found."


# ============================================================
# 7. SEARCH AND FILTERS
# ============================================================

def advanced_search(keyword, category_name, priority_name):
    conn = connect_db()

    query = """
        SELECT id, message, category, is_task,
               deadline, priority, completed
        FROM messages
        WHERE 1=1
    """

    values = []

    if keyword and keyword.strip():
        query += " AND message LIKE ?"
        values.append("%" + keyword.strip() + "%")

    if category_name != "All":
        query += " AND category = ?"
        values.append(category_name)

    if priority_name != "All":
        query += " AND priority = ?"
        values.append(priority_name)

    query += " ORDER BY id DESC"

    df = pd.read_sql_query(query, conn, params=values)
    conn.close()

    return df


# ============================================================
# 8. DASHBOARD
# ============================================================

def dashboard():
    conn = connect_db()
    df = pd.read_sql_query("SELECT * FROM messages", conn)
    conn.close()

    if df.empty:
        return "No messages available yet. Organize a message first."

    total = len(df)
    tasks = len(df[df["is_task"] == "Yes"])
    completed = len(df[df["completed"] == 1])
    pending = tasks - completed

    category_counts = df["category"].value_counts()
    priority_counts = df["priority"].value_counts()

    return f"""
# MindSort AI Dashboard

| Metric | Count |
|---|---:|
| Total messages | {total} |
| Total tasks | {tasks} |
| Completed tasks | {completed} |
| Pending tasks | {pending} |

## Category distribution

{category_counts.to_string()}

## Priority distribution

{priority_counts.to_string()}
"""


# ============================================================
# 9. GMAIL OAUTH AND READ-ONLY INBOX IMPORT
# ============================================================

def make_google_flow(state=None):
    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise RuntimeError(
            "Google OAuth is not configured. Add GOOGLE_CLIENT_ID and "
            "GOOGLE_CLIENT_SECRET in Render environment variables."
        )

    client_config = {
        "web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [GOOGLE_REDIRECT_URI]
        }
    }

    flow = Flow.from_client_config(
        client_config,
        scopes=GMAIL_SCOPES,
        state=state
    )
    flow.redirect_uri = GOOGLE_REDIRECT_URI
    return flow


def start_gmail_connection(account_slot):
    if account_slot not in ("Gmail account 1", "Gmail account 2"):
        return "Please choose Gmail account 1 or Gmail account 2."

    try:
        flow = make_google_flow()
        authorization_url, state = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent"
        )
        pending_oauth[state] = {
            "account_slot": account_slot,
            "created_at": time.time(),
            "flow": flow
        }
        return (
            f"### Connect {account_slot}\n\n"
            f"[Continue to Google to authorize MindSort AI]({authorization_url})\n\n"
            "Google will ask you to sign in and approve read-only Gmail access. "
            "Return here after approval."
        )
    except Exception as error:
        
        return f"Could not start Google authorization: {str(error)}"


def gmail_oauth_callback(request: Request):
    state = request.query_params.get("state")
    code = request.query_params.get("code")
    error = request.query_params.get("error")

    if error:
        return HTMLResponse(
            "<h2>Gmail connection was not approved.</h2>"
            "<p>Return to MindSort AI and try again if you want to connect.</p>",
            status_code=400
        )

    pending = pending_oauth.pop(state, None) if state else None
    if not pending or time.time() - pending["created_at"] > 600 or not code:
        return HTMLResponse(
            "<h2>This authorization link has expired.</h2>"
            "<p>Return to MindSort AI and start the connection again.</p>",
            status_code=400
        )
    try:
        flow = pending["flow"]
        flow.fetch_token(code=code)
        credentials = flow.credentials

        service = build(
            "gmail", "v1", credentials=credentials,
            cache_discovery=False
        )

        profile = service.users().getProfile(userId="me").execute()
        gmail_address = profile.get("emailAddress", "Connected Google account")

        gmail_connections[pending["account_slot"]] = {
            "credentials": credentials,
            "email": gmail_address
        }

        return HTMLResponse(
            "<html><body style='font-family:Arial;padding:32px'>"
            "<h2>Gmail connected successfully!</h2>"
            f"<p>{pending['account_slot']} is connected as "
            f"<b>{gmail_address}</b>.</p>"
            "<p>You can close this tab and return to MindSort AI.</p>"
            "</body></html>"
        )

    except Exception as error:
        print("GMAIL OAUTH CALLBACK ERROR:", repr(error), flush=True)
        return HTMLResponse(
            "<h2>Could not complete Gmail connection.</h2>"
            "<p>Check the Render logs for the error.</p>",
            status_code=400
        )
        
def gmail_connection_status():
    lines = ["## Gmail connections"]
    for slot in ("Gmail account 1", "Gmail account 2"):
        connection = gmail_connections.get(slot)
        if connection:
            lines.append(f"- **{slot}:** Connected ({connection['email']})")
        else:
            lines.append(f"- **{slot}:** Not connected")
    lines.append(
        "\nConnections are held in memory for this running server. "
        "If Render restarts or redeploys, connect again."
    )
    return "\n".join(lines)


def decode_gmail_body(payload):
    """Extract plain-text email content; attachments are never downloaded."""
    chunks = []

    def walk(part):
        mime_type = part.get("mimeType", "")
        body = part.get("body", {})
        data = body.get("data")
        if mime_type == "text/plain" and data:
            try:
                decoded = base64.urlsafe_b64decode(data + "==").decode(
                    "utf-8", errors="replace"
                )
                chunks.append(decoded)
            except Exception:
                pass
        for child in part.get("parts", []):
            walk(child)

    walk(payload)
    return "\n".join(chunks).strip()


def fetch_and_organize_gmail(account_slot, max_emails):
    connection = gmail_connections.get(account_slot)
    if not connection:
        return f"**{account_slot} is not connected yet.** Use the Connect button first."

    try:
        max_emails = int(max_emails or 10)
        max_emails = max(1, min(max_emails, 100))
        service = build(
            "gmail", "v1", credentials=connection["credentials"],
            cache_discovery=False
        )
                service = build(
            "gmail", "v1", credentials=connection["credentials"],
            cache_discovery=False
        )
        message_refs = []
        page_token = None
        while len(message_refs) < max_emails:
            response = service.users().messages().list(
                userId="me", labelIds=["INBOX"],
                maxResults=min(100, max_emails - len(message_refs)),
                pageToken=page_token
            ).execute()
            message_refs.extend(response.get("messages", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                break

        if not message_refs:
            return f"No inbox messages found in {account_slot}."

        if not message_refs:
            return f"No inbox messages found in {account_slot}."

        imported = 0
        skipped = 0
        results = []
        conn = connect_db()
        cursor = conn.cursor()

        for ref in message_refs:
            item = service.users().messages().get(
                userId="me", id=ref["id"], format="full"
            ).execute()
            headers = {
                h.get("name", "").lower(): h.get("value", "")
                for h in item.get("payload", {}).get("headers", [])
            }
            subject = headers.get("subject", "(No subject)")
            sender = headers.get("from", "Unknown sender")
            body = decode_gmail_body(item.get("payload", {}))
            snippet = item.get("snippet", "")
            content = f"Subject: {subject}\nFrom: {sender}\n\n{body or snippet}"
            content = content[:MAX_MESSAGE_LENGTH]

            folders.save_folder_card(
                cursor, connection["email"], item["id"], subject, sender,
                body or snippet,
                datetime.fromtimestamp(int(item.get("internalDate", 0)) / 1000)
            )

            category_name, confidence = classify_message(content)
            
            priority = detect_priority(content)
            task = detect_task(content)
            deadline = detect_deadline(content)

            cursor.execute("""
                INSERT OR IGNORE INTO messages
                (message, category, is_task, deadline, priority,
                 created_at, gmail_account, gmail_message_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                content, category_name, task, deadline, priority,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                connection["email"], item["id"]
            ))

            if cursor.rowcount:
                imported += 1
                results.append(
                    f"- **{subject[:100]}** — {category_name}, "
                    f"{priority} priority, Task: {task} "
                    f"(AI confidence {confidence}%)"
                )
            else:
                skipped += 1

        conn.commit()
        conn.close()
        summary = [
            f"## Gmail organization complete: {account_slot}",
            f"- Account: **{connection['email']}**",
            f"- New messages organized: **{imported}**",
            f"- Already imported: **{skipped}**",
            "- Gmail mailbox changes: **None (read-only access)**",
            "\n### Results"
        ]
        return "\n".join(summary + results)

    except Exception as error:
        return (
            "Gmail could not be read. Check the connection and try again. "
            f"Technical detail: {str(error)[:300]}"
        )


def gmail_setup_information():
    return (
        "Gmail uses Google's official OAuth flow and requests only the "
        "gmail.readonly permission. Connect either account below."
    )


# ============================================================
# 10. APPLICATION TESTS
# ============================================================

def run_tests():
    tests = []

    try:
        category_name, confidence = classify_message(
            "Complete mathematics homework"
        )
        tests.append(("AI classification", category_name == "Study"))

        tests.append((
            "Priority detection",
            detect_priority("Urgent assignment") == "High"
        ))

        tests.append((
            "Task detection",
            detect_task("Submit assignment") == "Yes"
        ))

        tests.append((
            "Deadline detection",
            detect_deadline("Complete this tomorrow") != "Not specified"
        ))

        conn = connect_db()
        conn.execute("SELECT * FROM messages LIMIT 1")
        conn.close()
        tests.append(("Database connection", True))

    except Exception as error:
        tests.append((f"Testing error: {error}", False))

    return "\n".join(
        f"{'PASS' if result else 'FAIL'} - {name}"
        for name, result in tests
    )


# ============================================================
# 11. CSV EXPORT
# ============================================================

def export_messages():
    conn = connect_db()

    df = pd.read_sql_query(
        "SELECT * FROM messages ORDER BY id DESC",
        conn
    )

    conn.close()

    if df.empty:
        return None

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        prefix="MindSort_AI_Export_",
        delete=False,
        encoding="utf-8-sig",
        newline=""
    ) as file:
        df.to_csv(file.name, index=False)
        return file.name


# ============================================================
# 12. DEPLOYMENT INFORMATION
# ============================================================

def deployment_information():
    return """
## Deployment checklist

- Login protection is enabled using Render environment variables.
- Test all application features after deployment.
- Configure persistent database storage before relying on saved messages.
- Gmail uses official Google OAuth with read-only permission.
- OAuth credentials are kept in server memory, not source code.
- Reconnect Gmail accounts after a Render restart or redeploy.
- Configure persistent database storage before relying on saved messages.
- Keep API keys and secrets out of source code.

Gmail import processes up to 25 recent inbox messages per run. It does not
send, delete, archive, or modify Gmail messages. WhatsApp is not connected.
"""


# ============================================================
# 13. GRADIO USER INTERFACE
# ============================================================

with gr.Blocks(
    title="MindSort AI",
    theme=gr.themes.Soft()
) as app:

    gr.Markdown("""
    # 🧠 MindSort AI
    ### Your intelligent message organizer

    Organize messages, manage tasks and explore your productivity.
    """)

    with gr.Tabs():

        # ORGANIZE
        with gr.Tab("📩 Organize"):
            message = gr.Textbox(
                label="Enter your message",
                placeholder="Example: Submit mathematics assignment tomorrow",
                lines=4,
                max_lines=8
            )

            organize_button = gr.Button(
                "Organize Message",
                variant="primary"
            )

            organize_output = gr.Markdown()

            organize_button.click(
                organize_message,
                inputs=message,
                outputs=organize_output
            )

        # TASKS
        with gr.Tab("📋 Tasks"):
            gr.Markdown("## Pending tasks")

            task_button = gr.Button("Refresh Tasks")
            task_output = gr.Textbox(lines=12)

            task_button.click(
                get_reminders,
                outputs=task_output
            )

            task_id = gr.Number(
                label="Task ID to complete",
                precision=0
            )

            complete_button = gr.Button("Mark Completed")
            complete_output = gr.Textbox()

            complete_button.click(
                complete_task,
                inputs=task_id,
                outputs=complete_output
            )

        # SEARCH
        with gr.Tab("🔍 Search"):
            keyword = gr.Textbox(label="Search keyword")

            category = gr.Dropdown(
                ["All", "Study", "Hackathon", "Events", "Work", "Home"],
                value="All",
                label="Category"
            )

            priority = gr.Dropdown(
                ["All", "High", "Medium", "Low", "Normal"],
                value="All",
                label="Priority"
            )

            search_button = gr.Button("Search")
            search_output = gr.Dataframe()

            search_button.click(
                advanced_search,
                inputs=[keyword, category, priority],
                outputs=search_output
            )

        # DASHBOARD
        with gr.Tab("📊 Dashboard"):
            dashboard_button = gr.Button("Refresh Dashboard")
            dashboard_output = gr.Markdown()

            dashboard_button.click(
                dashboard,
                outputs=dashboard_output
            )

        # EXPORT
        with gr.Tab("📥 Export"):
            export_button = gr.Button("Export CSV")
            export_output = gr.File()

            export_button.click(
                export_messages,
                outputs=export_output
            )

        # GMAIL
        with gr.Tab("📧 Gmail"):
            gr.Markdown("""
            ## Connect your Gmail accounts
            Connect up to two accounts. MindSort AI reads recent inbox messages,
            categorizes them, and detects tasks, priority, and deadlines.

            **Permission:** Read-only. MindSort AI cannot send, delete, archive,
            or modify your Gmail messages.
            """)

            gmail_account = gr.Radio(
                ["Gmail account 1", "Gmail account 2"],
                value="Gmail account 1",
                label="Choose which account to connect"
            )
            connect_gmail_button = gr.Button(
                "Connect selected Gmail account", variant="primary"
            )
            gmail_connect_output = gr.Markdown()
            connect_gmail_button.click(
                start_gmail_connection,
                inputs=gmail_account,
                outputs=gmail_connect_output
            )

            status_button = gr.Button("Refresh connection status")
            gmail_status_output = gr.Markdown()
            status_button.click(
                gmail_connection_status,
                outputs=gmail_status_output
            )

            gr.Markdown("---\n## Organize inbox messages")
            fetch_account = gr.Dropdown(
                ["Gmail account 1", "Gmail account 2"],
                value="Gmail account 1",
                label="Select connected inbox"
            )
            email_count = gr.Slider(
                minimum=1, maximum=100, value=25, step=1,
                label="Number of recent inbox messages to process"
            )
            fetch_button = gr.Button("Fetch and organize inbox")
            fetch_output = gr.Markdown()
            fetch_button.click(
                fetch_and_organize_gmail,
                inputs=[fetch_account, email_count],
                outputs=fetch_output
            )

        
        # FOLDERS
        with gr.Tab("📁 Folders"):
            gr.Markdown("Pick an account, then click a folder to open it.")
            with gr.Row():
                fc_account = gr.Dropdown(choices=folders.list_accounts(), label="Gmail account")
                fc_refresh = gr.Button("Refresh folders")
            fc_folders = gr.Radio(choices=[], label="Folders")
            with gr.Row():
                fc_keyword = gr.Textbox(label="Search keyword")
                fc_sort = gr.Dropdown(folders.SORT_OPTIONS, value=folders.SORT_OPTIONS[0], label="Sort by")
            fc_cards = gr.HTML()

            fc_refresh.click(folders.refresh_folders, fc_account, [fc_account, fc_folders, fc_cards])
            fc_account.change(folders.change_account, fc_account, [fc_folders, fc_cards])
            show = [fc_account, fc_folders, fc_keyword, fc_sort]
            fc_folders.change(folders.render_folder, show, fc_cards)
            fc_keyword.submit(folders.render_folder, show, fc_cards)
            fc_sort.change(folders.render_folder, show, fc_cards)
        # TESTING
        with gr.Tab("🧪 Testing"):
            test_button = gr.Button("Run Tests")
            test_output = gr.Textbox(lines=8)

            test_button.click(
                run_tests,
                outputs=test_output
            )

        # DEPLOYMENT
        with gr.Tab("🚀 Deployment"):
            deployment_button = gr.Button(
                "View Deployment Checklist"
            )

            deployment_output = gr.Markdown()

            deployment_button.click(
                deployment_information,
                outputs=deployment_output
            )

    gr.Markdown("""
    ---
    **MindSort AI | Smart Message Management**
    """)


# Register Google's OAuth callback on the same FastAPI app used by Gradio.
# Our own FastAPI server holds the OAuth callback; Gradio is mounted onto it.
server = FastAPI()
server.add_api_route(
    "/oauth2callback",
    gmail_oauth_callback,
    methods=["GET"],
    response_class=HTMLResponse
)


# ============================================================
# 14. START SERVER WITH LOGIN PROTECTION
# ============================================================
if __name__ == "__main__":
    username = os.environ.get("MINDSORT_USERNAME")
    password = os.environ.get("MINDSORT_PASSWORD")

    if not username or not password:
        raise RuntimeError(
            "Login credentials are missing. In Render, add the "
            "MINDSORT_USERNAME and MINDSORT_PASSWORD environment "
            "variables, then redeploy."
        )

    if len(password) < 12:
        raise RuntimeError(
            "MINDSORT_PASSWORD must be at least 12 characters long. "
            "Set a longer unique password in Render."
        )

    mounted = gr.mount_gradio_app(
        server, app, path="/", auth=(username, password)
    )

    uvicorn.run(
        mounted,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )

    
