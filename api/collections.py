"""Collection management endpoints."""

from fastapi import APIRouter, HTTPException
from loguru import logger

from core.ingestion import get_chroma_client
from database import get_db
from database.repositories import delete_documents_by_collection, list_documents

router = APIRouter()


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
    client = get_chroma_client()

    try:
        client.delete_collection(name=name)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found") from exc

    db = await get_db()
    try:
        deleted = await delete_documents_by_collection(db, name)
    finally:
        await db.close()

    logger.info(f"Deleted collection '{name}' ({deleted} document records)")
    return {"deleted": name, "documents_removed": deleted}


@router.get("/collections/{name}/documents")
async def collection_documents(name: str):
    """List documents in a collection."""
    db = await get_db()
    try:
        docs = await list_documents(db, collection=name)
    finally:
        await db.close()
    return {"collection": name, "documents": docs}
