import json
import os
import sqlite3
from pathlib import Path

DB_PATH = Path(os.getenv("DATABASE_PATH", ".data/mold-agent.sqlite3"))


def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS providers (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                provider_type TEXT NOT NULL,
                base_url TEXT,
                api_key_encrypted TEXT NOT NULL,
                enabled INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                provider_id TEXT NOT NULL REFERENCES providers(id) ON DELETE CASCADE,
                display_name TEXT NOT NULL,
                model_id TEXT NOT NULL,
                enabled INTEGER DEFAULT 1,
                is_default INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(provider_id, model_id)
            );

            CREATE TABLE IF NOT EXISTS app_config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )


def row(row):
    return dict(row) if row else None


def rows(items):
    return [dict(x) for x in items]


def get_config(key: str, default):
    with connect() as conn:
        value = conn.execute("SELECT value FROM app_config WHERE key = ?", (key,)).fetchone()
    return json.loads(value["value"]) if value else default


def set_config(key: str, value):
    with connect() as conn:
        conn.execute(
            "INSERT INTO app_config (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value)),
        )
