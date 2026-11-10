"""
rate_limiter.py
---------------
Enterprise In-Memory Sliding-Window Rate Limiter for FraudGuard AI.
Guards sensitive authentication and inference endpoints against brute-force
and denial-of-service (DoS) floods.

Enforces HTTP 429 Too Many Requests with standard 'Retry-After' response headers.
Zero external infrastructure required.
"""

import os
import time
import threading
from collections import defaultdict, deque
from typing import Dict, Optional, Tuple
from fastapi import Request, HTTPException, status
import logging

logger = logging.getLogger("fraudguard.rate_limiter")


class SlidingWindowRateLimiter:
    """
    Thread-safe in-memory sliding window rate limiter.
    
    Tracks request timestamps within a sliding window (default 60 seconds).
    Reads capacity limits dynamically from environment variables, allowing
    hot-reconfiguration without code changes.
    """

    def __init__(
        self,
        name: str,
        env_var: str,
        default_limit: int,
        window_seconds: int = 60,
    ):
        self.name = name
        self.env_var = env_var
        self.default_limit = default_limit
        self.window_seconds = window_seconds
        self._history: Dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def get_limit(self) -> int:
        """Reads current limit configuration from environment or fallback default."""
        try:
            val = os.getenv(self.env_var)
            if val is not None and val.strip():
                return int(val.strip())
        except (ValueError, TypeError):
            pass
        return self.default_limit

    def get_client_identifier(self, request: Request) -> str:
        """Extracts client IP from proxy headers (X-Forwarded-For) or raw connection."""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            # First IP in comma-separated proxy list is the original client
            return forwarded.split(",")[0].strip()
        if request.client and request.client.host:
            return request.client.host
        return "127.0.0.1"

    def check_rate_limit(self, client_key: str) -> Tuple[bool, int, int]:
        """
        Evaluates whether a request from client_key is within the allowed quota.
        Returns: (is_allowed: bool, current_count: int, retry_after_seconds: int)
        """
        limit = self.get_limit()
        if limit <= 0:
            # Non-positive limit disables rate limiting
            return True, 0, 0

        now = time.time()
        cutoff = now - self.window_seconds

        with self._lock:
            q = self._history[client_key]
            # Prune timestamps older than current sliding window
            while q and q[0] <= cutoff:
                q.popleft()

            if len(q) >= limit:
                oldest_timestamp = q[0]
                retry_after = max(1, int(oldest_timestamp + self.window_seconds - now + 0.999))
                return False, len(q), retry_after

            # Quota available: record current request timestamp
            q.append(now)
            return True, len(q), 0

    def __call__(self, request: Request) -> None:
        """
        FastAPI dependency callable.
        Raises HTTP 429 Too Many Requests if client quota is exhausted.
        """
        client_id = self.get_client_identifier(request)
        client_key = f"{self.name}:{client_id}"
        allowed, count, retry_after = self.check_rate_limit(client_key)

        if not allowed:
            limit = self.get_limit()
            logger.warning(
                f"Rate limit exceeded on '{self.name}' by client {client_id} "
                f"({count}/{limit} reqs in {self.window_seconds}s). Retry after {retry_after}s."
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Too Many Requests: Rate limit for '{self.name}' exceeded. "
                    f"Maximum allowed is {limit} requests per {self.window_seconds} seconds. "
                    f"Please retry after {retry_after} seconds."
                ),
                headers={"Retry-After": str(retry_after)},
            )

    def reset(self) -> None:
        """Clears all tracked client history. Useful for deterministic testing."""
        with self._lock:
            self._history.clear()


# Pre-configured production rate limiters
# Login endpoint: 5 attempts per 60 seconds (brute force protection)
rate_limit_login = SlidingWindowRateLimiter(
    name="auth_login",
    env_var="RATE_LIMIT_LOGIN",
    default_limit=5,
    window_seconds=60,
)

# Inference endpoints: 60 requests per 60 seconds (anti-scraping / DoS protection)
rate_limit_predict = SlidingWindowRateLimiter(
    name="predict_inference",
    env_var="RATE_LIMIT_PREDICT",
    default_limit=60,
    window_seconds=60,
)
