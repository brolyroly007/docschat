"""Document ingestion pipeline: parse → chunk → embed → store in ChromaDB."""

import hashlib
from pathlib import Path

import chromadb
from loguru import logger

from config import settings
from core.chunker import RecursiveCharacterSplitter
from core.embeddings import get_embedding_provider
from database import get_db
from database.repositories import create_document

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def parse_file(file_path: Path) -> str:
    """Extract text from a file based on its extension."""
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        import pymupdf4llm

        return pymupdf4llm.to_markdown(str(file_path))

    elif ext == ".docx":
        import docx

        doc = docx.Document(str(file_path))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())

    elif ext in (".txt", ".md"):
        return file_path.read_text(encoding="utf-8")

    else:
        raise ValueError(f"Unsupported file type: {ext}")


def get_chroma_client() -> chromadb.ClientAPI:
    """Get a persistent ChromaDB client."""
    persist_dir = Path(settings.chroma_persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


async def ingest_file(file_path: Path, collection_name: str | None = None) -> dict:
    """Ingest a single file: parse, chunk, embed, store."""
    collection_name = collection_name or settings.default_collection

    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {file_path.suffix}. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    logger.info(f"Ingesting {file_path.name} into collection '{collection_name}'")

    # Parse
    text = parse_file(file_path)
    if not text.strip():
        raise ValueError(f"No text extracted from {file_path.name}")

    # Chunk
    splitter = RecursiveCharacterSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split(text, metadata={"source": file_path.name})

    # Embed
    embedding_provider = get_embedding_provider()
    chunk_texts = [c.text for c in chunks]
    embeddings = await embedding_provider.embed(chunk_texts)

    # Store in ChromaDB
    client = get_chroma_client()
    collection = client.get_or_create_collection(name=collection_name)

    ids = [hashlib.md5(f"{file_path.name}:{c.index}".encode()).hexdigest() for c in chunks]
    metadatas = [{"source": file_path.name, "chunk_index": c.index} for c in chunks]

    collection.upsert(
        ids=ids,
        documents=chunk_texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    # Record in SQLite
    db = await get_db()
    try:
        doc_id = await create_document(
            db,
            filename=file_path.name,
            collection=collection_name,
            file_type=file_path.suffix.lower(),
            file_size=file_path.stat().st_size,
            chunk_count=len(chunks),
        )
    finally:
        await db.close()

    result = {
        "document_id": doc_id,
        "filename": file_path.name,
        "collection": collection_name,
        "chunks": len(chunks),
        "file_size": file_path.stat().st_size,
    }
    logger.info(f"Ingested {file_path.name}: {len(chunks)} chunks")
    return result


async def ingest_directory(dir_path: Path, collection_name: str | None = None) -> list[dict]:
    """Ingest all supported files in a directory."""
    results = []
    files = [
        f for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        logger.warning(f"No supported files found in {dir_path}")
        return results

    for file_path in sorted(files):
        try:
            result = await ingest_file(file_path, collection_name)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to ingest {file_path.name}: {e}")
            results.append({"filename": file_path.name, "error": str(e)})

    return results
