"""File upload and ingestion endpoint."""

import re
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger

from core.ingestion import SUPPORTED_EXTENSIONS, ingest_file

router = APIRouter()

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
COLLECTION_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


@router.post("/ingest")
async def ingest(
    files: list[UploadFile] = File(...),
    collection: str = Form(default="default"),
):
    """Upload and ingest one or more files."""
    if not COLLECTION_PATTERN.match(collection):
        raise HTTPException(
            status_code=422,
            detail="Collection name must contain only alphanumeric characters, underscores, "
            "and hyphens.",
        )

    results = []

    for upload in files:
        ext = Path(upload.filename or "").suffix.lower()

        # Validate file extension
        if ext not in SUPPORTED_EXTENSIONS or ext not in ALLOWED_EXTENSIONS:
            results.append(
                {
                    "filename": upload.filename,
                    "error": f"Unsupported file type: {ext}. "
                    f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
                }
            )
            continue

        # Read content and validate file size
        content = await upload.read()
        if len(content) > MAX_FILE_SIZE:
            results.append(
                {
                    "filename": upload.filename,
                    "error": f"File exceeds maximum size of "
                    f"{MAX_FILE_SIZE // (1024 * 1024)}MB.",
                }
            )
            continue

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            result = await ingest_file(tmp_path, collection_name=collection)
            result["filename"] = upload.filename  # Use original filename
            results.append(result)
        except Exception as e:
            logger.error(f"Ingestion failed for {upload.filename}: {e}")
            results.append({"filename": upload.filename, "error": str(e)})
        finally:
            tmp_path.unlink(missing_ok=True)

    return {"results": results}
