"""Collection management endpoints."""

import hashlib
import re

from fastapi import APIRouter, HTTPException
from loguru import logger

from core.ingestion import get_chroma_client
from database import get_db
from database.repositories import (
    delete_document,
    delete_documents_by_collection,
    get_document,
    list_documents,
)

router = APIRouter()

COLLECTION_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


def _validate_collection_name(name: str) -> None:
    """Raise HTTPException if collection name is invalid."""
    if not COLLECTION_PATTERN.match(name):
        raise HTTPException(
            status_code=422,
            detail="Collection name must contain only alphanumeric characters, underscores, "
            "and hyphens.",
        )


@router.get("/collections")
async def list_collections():
    """List all collections."""
    client = get_chroma_client()
    collections = client.list_collections()
    result = []
    for col in collections:
        result.append(
            {
                "name": col.name,
                "count": col.count(),
            }
        )
    return {"collections": result}


@router.delete("/collections/{name}")
async def delete_collection(name: str):
    """Delete a collection and its document records."""
    _validate_collection_name(name)
    client = get_chroma_client()

    try:
        client.delete_collection(name=name)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found") from exc

    db = get_db()
    deleted = await delete_documents_by_collection(db, name)

    logger.info(f"Deleted collection '{name}' ({deleted} document records)")
    return {"deleted": name, "documents_removed": deleted}


@router.get("/collections/{name}/documents")
async def collection_documents(name: str):
    """List documents in a collection."""
    db = get_db()
    docs = await list_documents(db, collection=name)
    return {"collection": name, "documents": docs}


@router.delete("/collections/{collection}/documents/{document_id}")
async def delete_collection_document(collection: str, document_id: int):
    """Delete a single document from a collection (ChromaDB chunks + SQLite metadata)."""
    _validate_collection_name(collection)

    db = get_db()
    doc = await get_document(db, document_id)
    if not doc or doc["collection"] != collection:
        raise HTTPException(
            status_code=404,
            detail=f"Document {document_id} not found in collection '{collection}'",
        )

    # Delete chunks from ChromaDB
    client = get_chroma_client()
    try:
        chroma_collection = client.get_collection(name=collection)
    except Exception as exc:
        raise HTTPException(
            status_code=404, detail=f"Collection '{collection}' not found in ChromaDB"
        ) from exc

    chunk_ids = [
        hashlib.md5(f"{doc['filename']}:{i}".encode()).hexdigest()
        for i in range(doc["chunk_count"])
    ]
    chroma_collection.delete(ids=chunk_ids)

    # Delete metadata from SQLite
    await delete_document(db, document_id)

    logger.info(
        f"Deleted document {document_id} ('{doc['filename']}') "
        f"from collection '{collection}' ({doc['chunk_count']} chunks)"
    )
    return {
        "deleted_document_id": document_id,
        "filename": doc["filename"],
        "collection": collection,
        "chunks_removed": doc["chunk_count"],
    }
