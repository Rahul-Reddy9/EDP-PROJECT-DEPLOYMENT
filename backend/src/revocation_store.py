"""
revocation_store.py
-------------------
Zero-cost, persistent token revocation registry backed by local SQLite.
Survives backend process restarts, eliminates in-memory state loss, and
automatically prunes expired token entries.

Enforces fail-closed semantics: raises RevocationStoreError on database
unavailability, corruption, or query failures so authentication dependencies
can reject requests with HTTP 503 instead of treating errors as unrevoked.
"""

import os
import sqlite3
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger("fraudguard.revocation")


class RevocationStoreError(Exception):
    """Raised when the token revocation database is unavailable, corrupted, or encounters I/O failure."""
    pass


# Locate persistent revocation database in project data directory
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DB_PATH = os.path.join(_PROJECT_ROOT, "data", "revoked_tokens.db")
DB_PATH = os.getenv("REVOCATION_DB_PATH", DEFAULT_DB_PATH)


class BaseRevocationStore(ABC):
    """Abstract interface defining the contract for token revocation storage."""

    @abstractmethod
    def record_revocation(self, jti: str, expires_at: float) -> bool:
        """Persists a revoked token identifier and its expiration epoch."""
        pass

    @abstractmethod
    def is_token_revoked(self, jti: str) -> bool:
        """Returns True if the token is revoked and not yet expired."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clears all stored revocation records (primarily for testing)."""
        pass


class SQLiteRevocationStore(BaseRevocationStore):
    """Zero-cost local persistent revocation store backed by SQLite (WAL mode)."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.getenv("REVOCATION_DB_PATH", DEFAULT_DB_PATH)
        self.init_db()

    def init_db(self) -> None:
        target_path = self.db_path
        try:
            os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
            with sqlite3.connect(target_path, timeout=5.0) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS revoked_tokens (
                        jti TEXT PRIMARY KEY,
                        revoked_at REAL NOT NULL,
                        expires_at REAL NOT NULL
                    );
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_revoked_expiry ON revoked_tokens(expires_at);")
                conn.commit()
        except Exception as exc:
            logger.error(f"Failed to initialize revocation SQLite database at {target_path}: {exc}")
            raise RevocationStoreError(f"Revocation database initialization failed at {target_path}: {exc}") from exc

    def record_revocation(self, jti: str, expires_at: float) -> bool:
        if not jti:
            return False
        try:
            self.init_db()
            now = time.time()
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO revoked_tokens (jti, revoked_at, expires_at) VALUES (?, ?, ?);",
                    (jti, now, float(expires_at)),
                )
                conn.execute("DELETE FROM revoked_tokens WHERE expires_at < ?;", (now,))
                conn.commit()
            return True
        except Exception as exc:
            logger.error(f"Error persisting token revocation for jti={jti}: {exc}")
            raise RevocationStoreError(f"Failed to persist token revocation for jti={jti}: {exc}") from exc

    def is_token_revoked(self, jti: str) -> bool:
        if not jti:
            return False
        try:
            self.init_db()
            now = time.time()
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT 1 FROM revoked_tokens WHERE jti = ? AND expires_at >= ? LIMIT 1;",
                    (jti, now),
                )
                return cursor.fetchone() is not None
        except Exception as exc:
            logger.error(f"Error querying token revocation for jti={jti}: {exc}")
            raise RevocationStoreError(f"Revocation database lookup failed for jti={jti}: {exc}") from exc

    def clear(self) -> None:
        try:
            self.init_db()
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                conn.execute("DELETE FROM revoked_tokens;")
                conn.commit()
        except Exception as exc:
            logger.error(f"Error clearing revocation DB: {exc}")
            raise RevocationStoreError(f"Failed to clear revocation database at {self.db_path}: {exc}") from exc


class RedisRevocationStore(BaseRevocationStore):
    """Distributed token revocation store adapter backed by Redis.
    Uses native SET with TTL (EXAT) for automatic, zero-overhead record expiration.
    Activates when REDIS_URL environment variable is set.
    """

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import redis
                self._client = redis.from_url(self.redis_url, socket_timeout=2.0)
            except ImportError:
                raise RevocationStoreError(
                    "redis package is not installed. Run 'pip install redis' to use RedisRevocationStore."
                )
            except Exception as exc:
                raise RevocationStoreError(f"Failed to connect to Redis at {self.redis_url}: {exc}") from exc
        return self._client

    def record_revocation(self, jti: str, expires_at: float) -> bool:
        if not jti:
            return False
        try:
            client = self._get_client()
            remaining_seconds = max(1, int(expires_at - time.time()))
            client.set(f"revoked_token:{jti}", "1", ex=remaining_seconds)
            return True
        except Exception as exc:
            logger.error(f"Redis revocation record failure for jti={jti}: {exc}")
            raise RevocationStoreError(f"Redis revocation record failure for jti={jti}: {exc}") from exc

    def is_token_revoked(self, jti: str) -> bool:
        if not jti:
            return False
        try:
            client = self._get_client()
            return client.exists(f"revoked_token:{jti}") > 0
        except Exception as exc:
            logger.error(f"Redis revocation check failure for jti={jti}: {exc}")
            raise RevocationStoreError(f"Redis revocation check failure for jti={jti}: {exc}") from exc

    def clear(self) -> None:
        try:
            client = self._get_client()
            keys = client.keys("revoked_token:*")
            if keys:
                client.delete(*keys)
        except Exception as exc:
            logger.error(f"Redis revocation clear failure: {exc}")
            raise RevocationStoreError(f"Redis revocation clear failure: {exc}") from exc


def get_revocation_store() -> BaseRevocationStore:
    """Factory resolving the active token revocation backend."""
    redis_url = os.getenv("REDIS_URL")
    if redis_url and redis_url.strip():
        return RedisRevocationStore(redis_url.strip())
    return SQLiteRevocationStore()


def get_db_path(custom_path: Optional[str] = None) -> str:
    """Dynamically resolves the active SQLite database path."""
    if custom_path:
        return custom_path
    return os.getenv("REVOCATION_DB_PATH", DEFAULT_DB_PATH)


def init_revocation_db(db_path: Optional[str] = None) -> None:
    """Initializes SQLite schema for token revocation persistence."""
    store = SQLiteRevocationStore(db_path=get_db_path(db_path))
    store.init_db()


def record_revocation(jti: str, expires_at: float, db_path: Optional[str] = None) -> bool:
    """Persists a revoked token's unique identifier (jti) and expiry timestamp."""
    store = SQLiteRevocationStore(db_path=get_db_path(db_path))
    return store.record_revocation(jti, expires_at)


def is_token_revoked_in_db(jti: str, db_path: Optional[str] = None) -> bool:
    """Checks whether a token's JTI is recorded as revoked and has not yet expired."""
    store = SQLiteRevocationStore(db_path=get_db_path(db_path))
    return store.is_token_revoked(jti)


def clear_revocation_db(db_path: Optional[str] = None) -> None:
    """Clears all revoked tokens. Primarily for testing purposes."""
    store = SQLiteRevocationStore(db_path=get_db_path(db_path))
    store.clear()
