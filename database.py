
"""
MindSort AI - PostgreSQL database connection
Uses Neon PostgreSQL through the DATABASE_URL environment variable.
"""

import os
import psycopg
from psycopg.rows import dict_row


def connect_db():
    """
    Open a PostgreSQL connection.

    DATABASE_URL must be configured in Render.
    """

    database_url = os.environ.get("DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is missing. "
            "Add your Neon PostgreSQL connection string "
            "to Render environment variables."
        )

    return psycopg.connect(
        database_url,
        row_factory=dict_row,
        connect_timeout=15
    )


def test_connection():
    """Test whether PostgreSQL is reachable."""

    conn = connect_db()

    try:
        result = conn.execute(
            "SELECT version() AS version"
        ).fetchone()

        return result["version"]

    finally:
        conn.close()
