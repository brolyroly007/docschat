"""Optional API key authentication middleware."""

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

from config import settings


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Require API key header if API_KEY is set in config."""

    async def dispatch(self, request: Request, call_next):
        # Skip auth if no API key configured
        if not settings.api_key:
            return await call_next(request)

        # Skip auth for health endpoint and docs
        if request.url.path in ("/api/health", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        api_key = request.headers.get("X-API-Key", "")
        if api_key != settings.api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

        return await call_next(request)
