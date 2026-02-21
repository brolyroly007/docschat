"""CRUD operations for documents, conversations, and messages."""

import json
import uuid

import aiosqlite

# --- Documents ---


async def create_document(
    db: aiosqlite.Connection,
    filename: str,
    collection: str,
    file_type: str,
    file_size: int,
    chunk_count: int,
) -> int:
    """Insert a document record and return its ID."""
    cursor = await db.execute(
        """INSERT INTO documents (filename, collection, file_type, file_size, chunk_count)
           VALUES (?, ?, ?, ?, ?)""",
        (filename, collection, file_type, file_size, chunk_count),
    )
    await db.commit()
    return cursor.lastrowid


async def list_documents(db: aiosqlite.Connection, collection: str | None = None) -> list[dict]:
    """List all documents, optionally filtered by collection."""
    if collection:
        rows = await db.execute_fetchall(
            "SELECT * FROM documents WHERE collection = ? ORDER BY ingested_at DESC",
            (collection,),
        )
    else:
        rows = await db.execute_fetchall("SELECT * FROM documents ORDER BY ingested_at DESC")
    return [dict(row) for row in rows]


async def delete_documents_by_collection(db: aiosqlite.Connection, collection: str) -> int:
    """Delete all documents in a collection. Returns count deleted."""
    cursor = await db.execute("DELETE FROM documents WHERE collection = ?", (collection,))
    await db.commit()
    return cursor.rowcount


# --- Conversations ---


async def create_conversation(
    db: aiosqlite.Connection, collection: str = "default", title: str | None = None
) -> str:
    """Create a new conversation and return its UUID."""
    conv_id = str(uuid.uuid4())
    await db.execute(
        """INSERT INTO conversations (id, collection, title)
           VALUES (?, ?, ?)""",
        (conv_id, collection, title),
    )
    await db.commit()
    return conv_id


async def get_conversation(db: aiosqlite.Connection, conversation_id: str) -> dict | None:
    """Get a conversation by ID."""
    cursor = await db.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
    row = await cursor.fetchone()
    return dict(row) if row else None


async def list_conversations(db: aiosqlite.Connection) -> list[dict]:
    """List all conversations."""
    rows = await db.execute_fetchall("SELECT * FROM conversations ORDER BY updated_at DESC")
    return [dict(row) for row in rows]


async def update_conversation(
    db: aiosqlite.Connection, conversation_id: str, title: str | None = None
):
    """Update conversation title and timestamp."""
    if title:
        await db.execute(
            "UPDATE conversations SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (title, conversation_id),
        )
    else:
        await db.execute(
            "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (conversation_id,),
        )
    await db.commit()


async def delete_conversation(db: aiosqlite.Connection, conversation_id: str) -> bool:
    """Delete a conversation and its messages. Returns True if found."""
    cursor = await db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    await db.commit()
    return cursor.rowcount > 0


# --- Messages ---


async def add_message(
    db: aiosqlite.Connection,
    conversation_id: str,
    role: str,
    content: str,
    sources: list[dict] | None = None,
    tokens_used: int = 0,
) -> int:
    """Add a message to a conversation."""
    sources_json = json.dumps(sources) if sources else None
    cursor = await db.execute(
        """INSERT INTO messages (conversation_id, role, content, sources, tokens_used)
           VALUES (?, ?, ?, ?, ?)""",
        (conversation_id, role, content, sources_json, tokens_used),
    )
    # Update conversation timestamp
    await db.execute(
        "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (conversation_id,),
    )
    await db.commit()
    return cursor.lastrowid


async def get_messages(db: aiosqlite.Connection, conversation_id: str) -> list[dict]:
    """Get all messages in a conversation, ordered by creation time."""
    rows = await db.execute_fetchall(
        "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
        (conversation_id,),
    )
    messages = []
    for row in rows:
        msg = dict(row)
        if msg["sources"]:
            msg["sources"] = json.loads(msg["sources"])
        messages.append(msg)
    return messages
