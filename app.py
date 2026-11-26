# ============================================================
# MINDSORT AI - CLEAN DEPLOYMENT VERSION
# ============================================================

import os
import sqlite3
from datetime import datetime, timedelta

import gradio as gr
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# ============================================================
# 1. DATABASE
# ============================================================

DB = "mindsort.db"


def connect_db():
    return sqlite3.connect(DB, check_same_thread=False)


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

    conn.commit()
    conn.close()


setup_database()


# ============================================================
# 2. AI CLASSIFICATION MODEL
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
# 3. PRIORITY, TASK AND DEADLINE DETECTION
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
# 4. ORGANIZE AND SAVE MESSAGE
# ============================================================

def organize_message(message):
    if not message or not message.strip():
        return "Please enter a message."

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
# 5. TASKS AND REMINDERS
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

    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE messages
        SET completed = 1
        WHERE id = ?
    """, (int(task_id),))

    conn.commit()
    changed = cursor.rowcount
    conn.close()

    if changed:
        return "Task marked as completed."

    return "Task ID not found."


# ============================================================
# 6. SEARCH AND FILTERS
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

    # Always return a DataFrame because the Gradio output is a Dataframe.
    return df


# ============================================================
# 7. DASHBOARD
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
# 8. GMAIL INFORMATION
# ============================================================

def gmail_setup_information():
    return """
## Gmail integration setup

Gmail is not connected in this version of MindSort AI.

A future Gmail connection should use Google's official Gmail API
and OAuth authentication.

1. Create a Google Cloud project.
2. Enable the Gmail API.
3. Configure OAuth consent.
4. Create OAuth client credentials.
5. Request only the permissions needed.

Never enter your Google password or private credentials into this demo.
"""


# ============================================================
# 9. APPLICATION TESTS
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
# 10. CSV EXPORT
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

    filename = "MindSort_AI_Export.csv"
    df.to_csv(filename, index=False, encoding="utf-8-sig")

    return filename


# ============================================================
# 11. DEPLOYMENT INFORMATION
# ============================================================

def deployment_information():
    return """
## Deployment checklist

- Test all application features.
- Use access control before storing personal messages.
- Configure persistent database storage.
- Use official OAuth for any future Gmail integration.
- Keep API keys and secrets out of source code.

This demo does not currently connect to Gmail or WhatsApp.
"""


# ============================================================
# 12. GRADIO USER INTERFACE
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
                lines=4
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
            gmail_button = gr.Button("View Gmail Setup")
            gmail_output = gr.Markdown()

            gmail_button.click(
                gmail_setup_information,
                outputs=gmail_output
            )

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


# ============================================================
# START SERVER FOR RENDER
# ============================================================

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000))
    )
