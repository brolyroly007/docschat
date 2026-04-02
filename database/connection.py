"""SQLite connection manager using aiosqlite — singleton pattern."""

from pathlib import Path

import aiosqlite
from loguru import logger

from config import settings

DB_PATH = Path(settings.database_url.replace("sqlite:///", ""))

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    collection TEXT NOT NULL DEFAULT 'default',
    file_type TEXT NOT NULL,
    file_size INTEGER,
    chunk_count INTEGER DEFAULT 0,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    collection TEXT NOT NULL DEFAULT 'default',
    title TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources TEXT,
    tokens_used INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conversations_collection ON conversations(collection);
"""

_connection: aiosqlite.Connection | None = None


async def init_db():
    """Create the singleton connection, enable WAL mode, and run schema."""
    global _connection
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    _connection = await aiosqlite.connect(str(DB_PATH))
    _connection.row_factory = aiosqlite.Row
    await _connection.execute("PRAGMA journal_mode=WAL")
    await _connection.execute("PRAGMA foreign_keys=ON")
    await _connection.executescript(SCHEMA)
    await _connection.commit()
    logger.info(f"Database initialized at {DB_PATH}")


def get_db() -> aiosqlite.Connection:
    """Return the existing singleton connection (sync — no await needed)."""
    if _connection is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _connection


async def close_db():
    """Close the singleton connection."""
    global _connection
    if _connection is not None:
        await _connection.close()
        _connection = None
        logger.info("Database connection closed")
