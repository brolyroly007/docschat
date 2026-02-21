"""File upload and ingestion endpoint."""

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile
from loguru import logger

from core.ingestion import SUPPORTED_EXTENSIONS, ingest_file

router = APIRouter()


@router.post("/ingest")
async def ingest(
    files: list[UploadFile] = File(...),
    collection: str = Form(default="default"),
):
    """Upload and ingest one or more files."""
    results = []

    for upload in files:
        ext = Path(upload.filename or "").suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            results.append(
                {
                    "filename": upload.filename,
                    "error": f"Unsupported file type: {ext}",
                }
            )
            continue

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await upload.read()
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
