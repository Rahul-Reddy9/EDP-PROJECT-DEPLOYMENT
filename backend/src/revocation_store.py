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
from typing import Optional

logger = logging.getLogger("fraudguard.revocation")


class RevocationStoreError(Exception):
    """Raised when the token revocation database is unavailable, corrupted, or encounters I/O failure."""
    pass


# Locate persistent revocation database in project data directory
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DB_PATH = os.path.join(_PROJECT_ROOT, "data", "revoked_tokens.db")
DB_PATH = os.getenv("REVOCATION_DB_PATH", DEFAULT_DB_PATH)


def get_db_path(custom_path: Optional[str] = None) -> str:
    """Dynamically resolves the active SQLite database path."""
    if custom_path:
        return custom_path
    return os.getenv("REVOCATION_DB_PATH", DEFAULT_DB_PATH)


def init_revocation_db(db_path: Optional[str] = None) -> None:
    """Initializes SQLite schema for token revocation persistence."""
    target_path = get_db_path(db_path)
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


def record_revocation(jti: str, expires_at: float, db_path: Optional[str] = None) -> bool:
    """
    Persists a revoked token's unique identifier (jti) and expiry timestamp.
    Also opportunistically cleans up expired records.
    Raises RevocationStoreError if database persistence fails.
    """
    if not jti:
        return False
    target_path = get_db_path(db_path)
    try:
        init_revocation_db(target_path)
        now = time.time()
        with sqlite3.connect(target_path, timeout=5.0) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO revoked_tokens (jti, revoked_at, expires_at) VALUES (?, ?, ?);",
                (jti, now, float(expires_at)),
            )
            # Prune stale expired records to maintain lean storage
            conn.execute("DELETE FROM revoked_tokens WHERE expires_at < ?;", (now,))
            conn.commit()
        return True
    except Exception as exc:
        logger.error(f"Error persisting token revocation for jti={jti}: {exc}")
        raise RevocationStoreError(f"Failed to persist token revocation for jti={jti}: {exc}") from exc


def is_token_revoked_in_db(jti: str, db_path: Optional[str] = None) -> bool:
    """
    Checks whether a token's JTI is recorded as revoked and has not yet expired.
    Raises RevocationStoreError on database lookup failure (fail-closed policy).
    """
    if not jti:
        return False
    target_path = get_db_path(db_path)
    try:
        init_revocation_db(target_path)
        now = time.time()
        with sqlite3.connect(target_path, timeout=5.0) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM revoked_tokens WHERE jti = ? AND expires_at >= ? LIMIT 1;",
                (jti, now),
            )
            return cursor.fetchone() is not None
    except Exception as exc:
        logger.error(f"Error querying token revocation for jti={jti}: {exc}")
        raise RevocationStoreError(f"Revocation database lookup failed for jti={jti}: {exc}") from exc


def clear_revocation_db(db_path: Optional[str] = None) -> None:
    """Clears all revoked tokens. Primarily for testing purposes."""
    target_path = get_db_path(db_path)
    try:
        init_revocation_db(target_path)
        with sqlite3.connect(target_path, timeout=5.0) as conn:
            conn.execute("DELETE FROM revoked_tokens;")
            conn.commit()
    except Exception as exc:
        logger.error(f"Error clearing revocation DB: {exc}")
        raise RevocationStoreError(f"Failed to clear revocation database at {target_path}: {exc}") from exc
