"""Optional API key authentication and rate limiting middleware."""

import time
from collections import defaultdict

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

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


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory sliding window rate limiter per client IP."""

    def __init__(self, app):
        super().__init__(app)
        # {ip: [timestamp, timestamp, ...]}
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health endpoint
        if request.url.path == "/api/health":
            return await call_next(request)

        # Skip if rate limiting is disabled (0 = unlimited)
        if settings.rate_limit_rpm <= 0:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window = 60.0  # 1 minute

        # Remove timestamps older than the window
        timestamps = self._requests[client_ip]
        self._requests[client_ip] = [t for t in timestamps if now - t < window]

        # Check limit
        if len(self._requests[client_ip]) >= settings.rate_limit_rpm:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
            )

        # Record this request
        self._requests[client_ip].append(now)

        return await call_next(request)
