"""Health check endpoint."""

from fastapi import APIRouter

from providers import list_providers

router = APIRouter()


@router.get("/health")
async def health():
    """System health check."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "providers": list_providers(),
    }
