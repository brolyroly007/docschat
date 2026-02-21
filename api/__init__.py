"""API router registration."""

from fastapi import APIRouter

from api.chat import router as chat_router
from api.collections import router as collections_router
from api.conversations import router as conversations_router
from api.health import router as health_router
from api.ingest import router as ingest_router

router = APIRouter()
router.include_router(health_router, tags=["health"])
router.include_router(chat_router, tags=["chat"])
router.include_router(ingest_router, tags=["ingest"])
router.include_router(collections_router, tags=["collections"])
router.include_router(conversations_router, tags=["conversations"])
