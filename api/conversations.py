"""Conversation management endpoints."""

from fastapi import APIRouter, HTTPException

from database import get_db
from database.repositories import (
    delete_conversation,
    get_conversation,
    get_messages,
    list_conversations,
)

router = APIRouter()


@router.get("/conversations")
async def list_convs():
    """List all conversations."""
    db = get_db()
    convs = await list_conversations(db)
    return {"conversations": convs}


@router.get("/conversations/{conversation_id}")
async def get_conv(conversation_id: str):
    """Get a conversation with its message history."""
    db = get_db()
    conv = await get_conversation(db, conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    msgs = await get_messages(db, conversation_id)
    return {"conversation": conv, "messages": msgs}


@router.delete("/conversations/{conversation_id}")
async def delete_conv(conversation_id: str):
    """Delete a conversation and its messages."""
    db = get_db()
    deleted = await delete_conversation(db, conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"deleted": conversation_id}
