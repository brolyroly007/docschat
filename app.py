"""
DocsChat — RAG Chat System.

Ingest documents, embed in ChromaDB, and chat with any LLM.

Run:  python app.py
Docs: http://localhost:8000/docs
"""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api import router as api_router
from config import settings
from database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    logger.info("Starting DocsChat...")
    await init_db()
    logger.info("Database initialized")
    yield
    logger.info("Shutting down DocsChat")


app = FastAPI(
    title="DocsChat",
    description="RAG chat system: ingest documents and chat with any LLM",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(api_router, prefix="/api")


if __name__ == "__main__":
    host = settings.host
    port = settings.port
    debug = settings.debug
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run("app:app", host=host, port=port, reload=debug, log_level="info")
