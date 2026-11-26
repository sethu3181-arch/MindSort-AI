
def display_results(records):

    print("\n========== FILTER RESULTS ==========")

    if not records:
        print("No matching messages found.")
        return

    for row in records:

        print("\nID:", row[0])
        print("Message:", row[1])
        print("Category:", row[2])
        print("Priority:", row[3])
        print("-----------------------------------")


# -----------------------------------------------------
# 5. FILTER MENU
# -----------------------------------------------------

while True:

    print("\n========== MINDSORT FILTER ==========")
    print("1. Filter by Category")
    print("2. Filter by Priority")
    print("3. Show All Tasks")
    print("4. Exit")

    choice = input("\nEnter your choice: ")

    if choice == "1":

        category = input("Enter category (Study, Work, Home): ")
        filter_category(category)

    elif choice == "2":

        priority = input("Enter priority (High, Medium, Low, Normal): ")
        filter_priority(priority)

    elif choice == "3":

        filter_tasks()

    elif choice == "4":

        print("Filtering completed!")
        break

    else:

        print("Invalid choice. Try again.")

print("\nStage 16 completed successfully!")

# =====================================================
# MINDSORT AI - STAGE 17
# UNIFIED DASHBOARD
# =====================================================


import gradio as gr
import sqlite3
import re

# Database connection
conn = sqlite3.connect("mindsort.db", check_same_thread=False)

# -----------------------------------------------------
# 1. CLASSIFICATION
# -----------------------------------------------------

def classify_message(message):

    embedding = semantic_model.encode(
        message,
        convert_to_tensor=True
    )

    similarities = util.cos_sim(
        embedding,
        example_embeddings
    )[0]

    best_index = similarities.argmax().item()
    score = similarities[best_index].item()

    if score < 0.35:
        return "Needs review", score

    return labels[best_index], score


# -----------------------------------------------------
# 2. TASK DETECTION
# -----------------------------------------------------

def detect_task(message):

    text = message.lower()

    task_words = [
        "submit", "complete", "finish", "pay",
        "attend", "prepare", "register",
        "upload", "send", "buy", "review"
    ]

    deadline_words = [
        "today", "tomorrow", "tonight",
        "monday", "tuesday", "wednesday",
        "thursday", "friday", "saturday", "sunday"
    ]

    task = any(word in text for word in task_words)

    deadlines = [
        word for word in deadline_words
        if re.search(r"\b" + word + r"\b", text)
    ]

    return (
        "Yes" if task else "No",
        ", ".join(deadlines) if deadlines else "Not detected"
    )


# -----------------------------------------------------
# 3. PRIORITY
# -----------------------------------------------------

def detect_priority(message):

    text = message.lower()

    if any(word in text for word in
           ["urgent", "asap", "immediately", "critical"]):
        return "High"

    if any(word in text for word in
           ["tomorrow", "soon", "important", "upcoming"]):
        return "Medium"

    if any(word in text for word in
           ["whenever", "optional", "no hurry", "later"]):
        return "Low"

    return "Normal"


# -----------------------------------------------------
# 4. ORGANIZE AND SAVE
# -----------------------------------------------------

def organize_message(message):

    if not message or not message.strip():
        return "Please enter a message."

    category, score = classify_message(message)
    task, deadline = detect_task(message)
    priority = detect_priority(message)

    with conn:
        conn.execute("""
        INSERT INTO messages
        (message, category, is_task, deadline, priority)
        VALUES (?, ?, ?, ?, ?)
        """, (message, category, task, deadline, priority))

    return f"""
## Message Organized Successfully!

**Category:** {category}

**Task:** {task}

**Deadline:** {deadline}

**Priority:** {priority}

**Similarity:** {score:.2%}

*Similarity is not a calibrated confidence score.*
"""


# -----------------------------------------------------
# 5. SEARCH
# -----------------------------------------------------

def search_messages(keyword):

    with conn:
        records = conn.execute("""
        SELECT id, message, category, is_task, deadline, priority
        FROM messages
        WHERE message LIKE ? OR category LIKE ?
        ORDER BY id DESC
        """, (f"%{keyword}%", f"%{keyword}%")).fetchall()

    if not records:
        return "No matching messages found."

    output = ""

    for row in records:
        output += f"""
ID: {row[0]}
Message: {row[1]}
Category: {row[2]}
Task: {row[3]}
Deadline: {row[4]}
Priority: {row[5]}
-----------------------------
"""

    return output


# -----------------------------------------------------
# 6. VIEW ALL MESSAGES
# -----------------------------------------------------

def view_all_messages():

    with conn:
        records = conn.execute("""
        SELECT id, message, category, is_task, deadline, priority
        FROM messages
        ORDER BY id DESC
        """).fetchall()

    if not records:
        return "No messages saved yet."

    output = ""

    for row in records:
        output += f"""
ID: {row[0]}
Message: {row[1]}
Category: {row[2]}
Task: {row[3]}
Deadline: {row[4]}
Priority: {row[5]}
-----------------------------
"""

    return output


# -----------------------------------------------------
# 7. DASHBOARD STATISTICS
# -----------------------------------------------------

def dashboard():

    with conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM messages"
        ).fetchone()[0]

        tasks = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE is_task='Yes'"
        ).fetchone()[0]

        high = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE priority='High'"
        ).fetchone()[0]

        categories = conn.execute("""
        SELECT category, COUNT(*)
        FROM messages
        GROUP BY category
        """).fetchall()

    result = f"""
# 🧠 MindSort AI Dashboard

### Overview

- **Total Messages:** {total}
- **Total Tasks:** {tasks}
- **High Priority Messages:** {high}

### Category Statistics
"""

    for category, count in categories:
        result += f"\n- {category}: {count}"

    return result


# -----------------------------------------------------
# 8. GRADIO INTERFACE
# -----------------------------------------------------

with gr.Blocks(title="MindSort AI") as app:

    gr.Markdown("""
    # 🧠 MindSort AI
    ### Your Personal Intelligent Message Organizer

    Organize your messages. Manage your tasks. Stay organized.
    """)

    with gr.Tab("Organize"):

        message = gr.Textbox(
            label="Enter Message",
            placeholder="Submit my AI project by Friday",
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

    with gr.Tab("Search"):

        keyword = gr.Textbox(label="Search Keyword")

        search_button = gr.Button("Search")

        search_output = gr.Textbox(lines=15)

        search_button.click(
            search_messages,
            inputs=keyword,
            outputs=search_output
        )

    with gr.Tab("Saved Messages"):

        history_button = gr.Button("Refresh Messages")

        history_output = gr.Textbox(lines=20)

        history_button.click(
            view_all_messages,
            outputs=history_output
        )

    with gr.Tab("Dashboard"):

        dashboard_button = gr.Button("Refresh Dashboard")

        dashboard_output = gr.Markdown()

        dashboard_button.click(
            dashboard,
            outputs=dashboard_output
        )

    gr.Markdown("""
    ---
    **MindSort AI | Smart Message Management**
    """)

# Launch application
app.launch(share=False, debug=True)

# =====================================================
# MINDSORT AI - STAGE 18
# PROFESSIONAL UI DESIGN
# =====================================================

import gradio as gr

custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}

#main-title {
    text-align: center;
    padding: 25px;
    border-radius: 15px;
}

#main-title h1 {
    font-size: 36px;
    font-weight: 800;
}

#main-title p {
    font-size: 16px;
}

button {
    border-radius: 10px !important;
}

textarea, input {
    border-radius: 10px !important;
}

footer {
    display: none !important;
}
"""

with gr.Blocks(
    title="MindSort AI",
    theme=gr.themes.Soft(),
    css=custom_css
) as app:

    gr.HTML("""
    <div id="main-title">
        <h1>🧠 MindSort AI</h1>
        <p>Your Personal Intelligent Message Organizer</p>
        <p>Organize • Prioritize • Simplify</p>
    </div>
    """)

    with gr.Tabs():

        # ORGANIZE
        with gr.Tab("📩 Organize"):

            gr.Markdown("## Organize a New Message")

            message = gr.Textbox(
                label="Your Message",
                placeholder="Enter your message here...",
                lines=4
            )

            button = gr.Button(
                "✨ Organize Message",
                variant="primary"
            )

            output = gr.Markdown()

            button.click(
                organize_message,
                inputs=message,
                outputs=output
            )

        # SEARCH
        with gr.Tab("🔍 Search"):

            gr.Markdown("## Find Saved Messages")

            keyword = gr.Textbox(
                label="Search",
                placeholder="Enter a keyword..."
            )

            search_btn = gr.Button("Search")

            search_result = gr.Textbox(lines=15)

            search_btn.click(
                search_messages,
                inputs=keyword,
                outputs=search_result
            )

        # HISTORY
        with gr.Tab("📚 History"):

            gr.Markdown("## Your Saved Messages")

            history_btn = gr.Button("Refresh History")

            history = gr.Textbox(lines=20)

            history_btn.click(
                view_all_messages,
                outputs=history
            )

        # DASHBOARD
        with gr.Tab("📊 Dashboard"):

            gr.Markdown("## Your Activity Overview")

            refresh = gr.Button("Refresh Statistics")

            stats = gr.Markdown()

            refresh.click(
                dashboard,
                outputs=stats
            )

    gr.Markdown("""
    ---
    <center>
    MindSort AI | Intelligent Message Management
    </center>
    """)

app.launch(share=False, debug=True)

# =====================================================
# MINDSORT AI - STAGE 19
# EXPORT MESSAGES TO CSV
# =====================================================

import csv
import sqlite3
import os
import gradio as gr

def export_messages():

    conn = sqlite3.connect("mindsort.db")

    cursor = conn.cursor()

    cursor.execute("""
    SELECT id, message, category, is_task, deadline, priority
    FROM messages
    ORDER BY id DESC
    """)

    records = cursor.fetchall()

    conn.close()

    if not records:
        return None

    filename = "MindSort_AI_Messages.csv"

    with open(filename, "w", newline="", encoding="utf-8-sig") as file:

        writer = csv.writer(file)

        writer.writerow([
            "ID",
            "Message",
            "Category",
            "Task",
            "Deadline",
            "Priority"
        ])

        writer.writerows(records)

    return filename


# -----------------------------------------------------
# TEST EXPORT
# -----------------------------------------------------

exported_file = export_messages()

if exported_file:
    print("Messages exported successfully!")
    print("File:", exported_file)
else:
    print("No messages available to export.")


# -----------------------------------------------------
# DOWNLOAD INTERFACE
# -----------------------------------------------------

with gr.Blocks(title="MindSort AI Export") as export_app:

    gr.Markdown("""
    # 📥 MindSort AI - Export Center

    Download your organized messages.
    """)

    export_button = gr.Button(
        "Export Messages",
        variant="primary"
    )

    download_file = gr.File(label="Your CSV File")

    export_button.click(
        export_messages,
        outputs=download_file
    )

export_app.launch(share=False)

# =====================================================
# MINDSORT AI - STAGE 20
# GOOGLE DRIVE BACKUP
# =====================================================

from google.colab import drive
import os
import sqlite3
import shutil
from datetime import datetime

# 1. Connect Google Drive
drive.mount('/content/drive')

# 2. Create backup folder
backup_folder = "/content/drive/MyDrive/MindSortAI/Backups"

os.makedirs(backup_folder, exist_ok=True)

# 3. Check database
database = "mindsort.db"

if os.path.exists(database):

    # Save database changes
    conn = sqlite3.connect(database)
    conn.commit()
    conn.close()

    # 4. Create timestamped backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    backup_name = f"MindSort_Backup_{timestamp}.db"

    backup_path = os.path.join(
        backup_folder,
        backup_name
    )

    shutil.copy2(database, backup_path)

    # 5. Update latest backup
    latest_path = os.path.join(
        backup_folder,
        "MindSort_Latest.db"
    )

    shutil.copy2(database, latest_path)

    print("Google Drive backup completed!")
    print("Backup location:", backup_folder)
    print("Backup file:", backup_name)

else:
    print("Database not found!")
    print("Save at least one message before creating a backup.")

# =====================================================
# MINDSORT AI - STAGE 21
# RESTORE DATABASE FROM GOOGLE DRIVE
# =====================================================

from google.colab import drive
import os
import shutil
import sqlite3

# 1. Connect Google Drive
drive.mount('/content/drive')

# 2. Locate backup folder
backup_folder = "/content/drive/MyDrive/MindSortAI/Backups"

latest_backup = os.path.join(
    backup_folder,
    "MindSort_Latest.db"
)

# 3. Restore database
if os.path.exists(latest_backup):

    # Keep a copy of the current database, if present
    if os.path.exists("mindsort.db"):
        shutil.copy2(
            "mindsort.db",
            "mindsort_before_restore.db"
        )

    shutil.copy2(latest_backup, "mindsort.db")

    print("Database restored successfully!")

    # 4. Check restored messages
    conn = sqlite3.connect("mindsort.db")
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM messages")

    total = cursor.fetchone()[0]

    conn.close()

    print("Total restored messages:", total)

else:
    print("No backup found!")
    print("Create a backup using Stage 20 first.")

# =====================================================
# MINDSORT AI - STAGE 22
# BACKUP AND RESTORE CONTROL PANEL
# =====================================================

import gradio as gr
import sqlite3
import os
import shutil
from datetime import datetime
from google.colab import drive

backup_folder = "/content/drive/MyDrive/MindSortAI/Backups"
latest_backup = os.path.join(
    backup_folder,
    "MindSort_Latest.db"
)

def create_backup():

    if not os.path.exists("/content/drive/MyDrive"):
        drive.mount("/content/drive")

    os.makedirs(backup_folder, exist_ok=True)

    if not os.path.exists("mindsort.db"):
        return "Database not found!"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    backup_name = f"MindSort_Backup_{timestamp}.db"

    conn = sqlite3.connect("mindsort.db")
    backup_conn = sqlite3.connect(latest_backup)

    with backup_conn:
        conn.backup(backup_conn)

    backup_conn.close()
    conn.close()

    shutil.copy2(latest_backup, os.path.join(
        backup_folder, backup_name
    ))

    return "Backup created successfully!\n" + backup_name


def restore_backup():

    if not os.path.exists("/content/drive/MyDrive"):
        drive.mount("/content/drive")

    if not os.path.exists(latest_backup):
        return "No backup found!"

    if os.path.exists("mindsort.db"):
        shutil.copy2(
            "mindsort.db",
            "mindsort_before_restore.db"
        )

    shutil.copy2(latest_backup, "mindsort.db")

    return "Database restored successfully!"


# -----------------------------------------------------
# CONTROL PANEL
# -----------------------------------------------------

with gr.Blocks(
    title="MindSort AI - Backup Manager",
    theme=gr.themes.Soft()
) as backup_app:

    gr.Markdown("""
    # 🧠 MindSort AI
    ## Backup Manager

    Protect your organized messages.
    """)

    backup_button = gr.Button(
        "Create Backup",
        variant="primary"
    )

    restore_button = gr.Button(
        "Restore Backup",
        variant="secondary"
    )

    result = gr.Textbox(
        label="Status",
        lines=3
    )

    backup_button.click(
        create_backup,
        outputs=result
    )

    restore_button.click(
        restore_backup,
        outputs=result
    )

backup_app.launch(share=False)

# ============================================================
# MINDSORT AI - STAGES 23 TO 30
# COMPLETE INTEGRATED APPLICATION
# ============================================================

!pip -q install gradio scikit-learn pandas matplotlib

import os
import re
import csv
import sqlite3
import shutil
import gradio as gr
import pandas as pd

from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# ============================================================
# STAGE 23 - DATABASE AND UNIFIED APPLICATION
# ============================================================

DB = "mindsort.db"

def connect_db():
    return sqlite3.connect(DB, check_same_thread=False)

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

for column, definition in [
    ("priority", "TEXT DEFAULT 'Normal'"),
    ("completed", "INTEGER DEFAULT 0"),
    ("created_at", "TEXT")
]:
    if column not in existing_columns:
        cursor.execute(
            f"ALTER TABLE messages ADD COLUMN {column} {definition}"
        )

conn.commit()
conn.close()


# ============================================================
# STAGE 24 - IMPROVED AI CLASSIFICATION
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
    "Study", "Study", "Study", "Study", "Study", "Hackathon",
    "Hackathon", "Hackathon", "Hackathon", "Events",
    "Events", "Home", "Home", "Home", "Work", "Work", "Work"
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
# STAGE 25 - TASK AND REMINDER MANAGEMENT
# ============================================================

def organize_message(message):
    if not message or not message.strip():
        return "Please enter a message."

    category, confidence = classify_message(message)
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
        message, category, task, deadline,
        priority, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()

    return f"""
### Message organized successfully!

- **Category:** {category}
- **AI confidence:** {confidence}%
- **Task:** {task}
- **Priority:** {priority}
- **Deadline:** {deadline}
"""


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

    return "Task marked completed." if changed else "Task ID not found."


# ============================================================
# STAGE 26 - ADVANCED SEARCH AND FILTERS
# ============================================================

def advanced_search(keyword, category, priority):
    conn = connect_db()

    query = """
        SELECT id, message, category, is_task,
               deadline, priority, completed
        FROM messages WHERE 1=1
    """

    values = []

    if keyword.strip():
        query += " AND message LIKE ?"
        values.append("%" + keyword + "%")

    if category != "All":
        query += " AND category = ?"
        values.append(category)

    if priority != "All":
        query += " AND priority = ?"
        values.append(priority)

    query += " ORDER BY id DESC"

    df = pd.read_sql_query(query, conn, params=values)

    conn.close()

    return df if not df.empty else "No matching messages found."


# ============================================================
# STAGE 27 - DASHBOARD AND ANALYTICS
# ============================================================

def dashboard():
    conn = connect_db()

    df = pd.read_sql_query(
        "SELECT * FROM messages", conn
    )

    conn.close()

    if df.empty:
        return "No messages available yet."

    total = len(df)
    tasks = len(df[df["is_task"] == "Yes"])
    completed = len(df[df["completed"] == 1])
    pending = tasks - completed

    category_counts = df["category"].value_counts()

    priority_counts = df["priority"].value_counts()

    report = f"""
# MindSort AI Analytics

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

    return report


# ============================================================
# STAGE 28 - OPTIONAL GMAIL INTEGRATION
# ============================================================

def gmail_setup_information():
    return """
## Gmail integration setup

To connect Gmail securely:

1. Create a Google Cloud project.
2. Enable the Gmail API.
3. Configure OAuth consent.
4. Create OAuth client credentials.
5. Use Google's official authentication flow.
6. Request only the permissions needed.

Gmail is not connected by this demonstration.
No emails are accessed or downloaded.

Never paste your Google password or OAuth secret into this app.
"""


# ============================================================
# STAGE 29 - TESTING
# ============================================================

def run_tests():
    tests = []

    try:
        category, confidence = classify_message(
            "Complete mathematics homework"
        )

        tests.append(
            ("AI classification", category == "Study")
        )

        tests.append(
            ("Priority detection",
             detect_priority("Urgent assignment") == "High")
        )

        tests.append(
            ("Task detection",
             detect_task("Submit assignment") == "Yes")
        )

        tests.append(
            ("Deadline detection",
             detect_deadline("Complete this tomorrow")
             != "Not specified")
        )

        conn = connect_db()
        conn.execute("SELECT * FROM messages LIMIT 1")
        conn.close()

        tests.append(("Database connection", True))

    except Exception as error:
        tests.append(("Testing error", str(error)))

    return "\n".join(
        f"{'PASS' if result is True else 'FAIL'} - {name}"
        for name, result in tests
    )


# ============================================================
# STAGE 30 - CSV EXPORT AND DEPLOYMENT PREPARATION
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


def deployment_information():
    return """
# Deployment preparation

Before deployment:

1. Test all application features.
2. Store secrets securely.
3. Configure persistent database storage.
4. Configure Gmail OAuth if required.
5. Choose a hosting provider.
6. Set up access control.

This notebook runs locally in the Colab runtime.
It has not been deployed to a public server.
"""


# ============================================================
# FINAL - SINGLE MINDSORT AI APPLICATION
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

        # ANALYTICS
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
import os

app.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000))
)

from google.colab import files

files.download("mindsort.db")
