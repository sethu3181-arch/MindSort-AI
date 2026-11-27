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
from datetime import datetime, timedelta

import gradio as gr
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# ============================================================
# 1. CONFIGURATION
# ============================================================

DB = os.environ.get("MINDSORT_DB_PATH", "mindsort.db")
MAX_MESSAGE_LENGTH = 5000


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

    conn.commit()
    conn.close()


setup_database()


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
