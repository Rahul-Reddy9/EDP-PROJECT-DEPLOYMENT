"""
auth.py
-------
Enterprise Authentication, Session/Token Management, and Server-Side Role-Based
Access Control (RBAC) for FraudGuard AI.

Implements:
1. Secure password hashing and verification using salted bcrypt (cost factor 12).
2. Cryptographic JWT access tokens with configurable expiration (HS256).
3. Server-side RBAC distinguishing 'independent_user' and 'mlops_admin'.
4. Persistent token revocation backed by local SQLite (zero-cost, survives restarts).
5. Strict production validation rejecting missing, short, or known default secrets.
6. FastAPI dependency injection guards enforcing 401 Unauthorized and 403 Forbidden.
"""

import os
import sys
import uuid
import time
import secrets
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger("fraudguard.auth")

try:
    from src.revocation_store import record_revocation, is_token_revoked_in_db, RevocationStoreError
except ImportError:
    try:
        from backend.src.revocation_store import record_revocation, is_token_revoked_in_db, RevocationStoreError
    except ImportError:
        # Fallback if executed directly
        from revocation_store import record_revocation, is_token_revoked_in_db, RevocationStoreError

# ---------------------------------------------------------------------------
# Cryptographic Configuration & Secret Validation
# ---------------------------------------------------------------------------

ALGORITHM = "HS256"
DEFAULT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

KNOWN_INSECURE_SECRETS = {
    "fraudguard-ai-enterprise-secret-key-32bytes-secure-2026",
    "secret",
    "secretkey",
    "changeme",
    "admin",
    "password",
    "test",
    "default",
    "12345678",
}

KNOWN_INSECURE_PASSWORDS = {
    "admin",
    "password",
    "12345678",
    "changeme",
    "root",
    "toor",
}

# Generate a single fallback secret per process instance for development mode
_DEV_DYNAMIC_SECRET = secrets.token_hex(32)


def is_production_mode() -> bool:
    """Returns True if the runtime environment is designated as production."""
    env = os.getenv("ENV", os.getenv("ENVIRONMENT", "development")).lower().strip()
    return env in ("production", "prod")


def resolve_jwt_secret() -> str:
    """
    Resolves and validates the JWT signing secret.
    In production: strictly requires JWT_SECRET_KEY >= 32 chars and not in known default list.
    In development: uses JWT_SECRET_KEY if provided, or dynamic secure session secret.
    """
    raw_secret = os.getenv("JWT_SECRET_KEY")

    if is_production_mode():
        if not raw_secret or not raw_secret.strip():
            raise RuntimeError(
                "Production startup halted: 'JWT_SECRET_KEY' environment variable is missing. "
                "A secure, cryptographically random key of >= 32 characters is required."
            )
        secret = raw_secret.strip()
        if secret in KNOWN_INSECURE_SECRETS:
            raise RuntimeError(
                "Production startup halted: 'JWT_SECRET_KEY' matches a known default or insecure value. "
                "Provide a unique, high-entropy secret key."
            )
        if len(secret) < 32:
            raise RuntimeError(
                f"Production startup halted: 'JWT_SECRET_KEY' is only {len(secret)} characters long. "
                "Cryptographic signing key must be at least 32 characters long."
            )
        return secret
    else:
        if raw_secret and raw_secret.strip():
            return raw_secret.strip()
        return _DEV_DYNAMIC_SECRET


def get_secret_key() -> str:
    """Retrieves current validated secret key."""
    return resolve_jwt_secret()


# Validate immediately on module load
SECRET_KEY = resolve_jwt_secret()

# OAuth2 / Bearer Scheme for Swagger UI & programmatic clients
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)
http_bearer = HTTPBearer(auto_error=False)


class UserRole:
    INDEPENDENT_USER = "independent_user"
    MLOPS_ADMIN = "mlops_admin"


class User(BaseModel):
    username: str
    display_name: str
    role: str
    is_active: bool = True


class UserInDB(User):
    hashed_password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    role: str
    username: str
    display_name: str


class LoginPayload(BaseModel):
    username: str = Field(..., description="Username (e.g. 'analyst' or 'admin')")
    password: str = Field(..., description="Plaintext password")


# ---------------------------------------------------------------------------
# Password Utility Functions (Bcrypt)
# ---------------------------------------------------------------------------

def hash_password(plain_password: str) -> str:
    """Hashes a plaintext password using bcrypt with a random salt (cost 12)."""
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(plain_password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plaintext password against a stored bcrypt hash."""
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8")
        )
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Dynamic Seed Users Database & Credential Provisioning
# ---------------------------------------------------------------------------

def resolve_admin_password() -> str:
    """Resolves administrator password from environment with production enforcement."""
    raw = os.getenv("FRAUDGUARD_ADMIN_PASSWORD")
    if is_production_mode():
        if not raw or not raw.strip():
            raise RuntimeError(
                "Production startup halted: 'FRAUDGUARD_ADMIN_PASSWORD' environment variable is missing. "
                "A strong administrator password of >= 12 characters is required in production."
            )
        pwd = raw.strip()
        if pwd in KNOWN_INSECURE_PASSWORDS:
            raise RuntimeError(
                "Production startup halted: 'FRAUDGUARD_ADMIN_PASSWORD' matches a known insecure default."
            )
        if len(pwd) < 12:
            raise RuntimeError(
                f"Production startup halted: 'FRAUDGUARD_ADMIN_PASSWORD' is only {len(pwd)} characters long. "
                "Minimum required length is 12."
            )
        return pwd
    return raw.strip() if raw and raw.strip() else "AdminPassword@2026!"


def resolve_user_password() -> str:
    """Resolves analyst user password from environment with production enforcement."""
    raw = os.getenv("FRAUDGUARD_USER_PASSWORD")
    if is_production_mode():
        if not raw or not raw.strip():
            raise RuntimeError(
                "Production startup halted: 'FRAUDGUARD_USER_PASSWORD' environment variable is missing. "
                "An analyst user password of >= 8 characters is required in production."
            )
        pwd = raw.strip()
        if pwd in KNOWN_INSECURE_PASSWORDS:
            raise RuntimeError(
                "Production startup halted: 'FRAUDGUARD_USER_PASSWORD' matches a known insecure default."
            )
        if len(pwd) < 8:
            raise RuntimeError(
                f"Production startup halted: 'FRAUDGUARD_USER_PASSWORD' is only {len(pwd)} characters long. "
                "Minimum required length is 8."
            )
        return pwd
    return raw.strip() if raw and raw.strip() else "UserPassword@2026!"


USERS_DB: Dict[str, UserInDB] = {}


def provision_user(username: str, password_plain: str, role: str, display_name: str) -> None:
    """Securely provisions or updates a user in the user registry with bcrypt hash."""
    USERS_DB[username.lower()] = UserInDB(
        username=username.lower(),
        display_name=display_name,
        role=role,
        hashed_password=hash_password(password_plain),
        is_active=True,
    )


def initialize_default_users() -> None:
    """Initializes user registry from environment variables."""
    admin_pwd = resolve_admin_password()
    user_pwd = resolve_user_password()

    provision_user("admin", admin_pwd, UserRole.MLOPS_ADMIN, "MLOps Administrator")
    provision_user("analyst", user_pwd, UserRole.INDEPENDENT_USER, "Risk Analyst (Independent User)")
    provision_user("mlops_admin", admin_pwd, UserRole.MLOPS_ADMIN, "MLOps Administrator")
    provision_user("user", user_pwd, UserRole.INDEPENDENT_USER, "Risk Analyst (Independent User)")


# Initialize user accounts
initialize_default_users()


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticates a user by username and password."""
    user = USERS_DB.get(username.lower().strip())
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


# ---------------------------------------------------------------------------
# JWT Token Creation & Persistent Revocation
# ---------------------------------------------------------------------------

def create_access_token(
    user: User,
    expires_delta: Optional[timedelta] = None,
) -> Tuple[str, int]:
    """Generates a cryptographically signed JWT with expiration and claims."""
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
        expires_seconds = int(expires_delta.total_seconds())
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=DEFAULT_ACCESS_TOKEN_EXPIRE_MINUTES)
        expires_seconds = DEFAULT_ACCESS_TOKEN_EXPIRE_MINUTES * 60

    jti = str(uuid.uuid4())
    to_encode = {
        "sub": user.username,
        "role": user.role,
        "name": user.display_name,
        "jti": jti,
        "iat": datetime.now(timezone.utc),
        "exp": expire,
    }
    secret = get_secret_key()
    encoded_jwt = jwt.encode(to_encode, secret, algorithm=ALGORITHM)
    return encoded_jwt, expires_seconds


def revoke_token(token: str) -> bool:
    """
    Revokes a JWT token by adding its jti claim to the persistent SQLite blacklist.
    Survives backend process restarts.
    Raises RevocationStoreError if database persistence fails.
    """
    secret = get_secret_key()
    try:
        payload = jwt.decode(token, secret, algorithms=[ALGORITHM], options={"verify_exp": False})
    except Exception as exc:
        logger.warning(f"Failed to decode token for revocation: {exc}")
        return False

    jti = payload.get("jti")
    exp = payload.get("exp", time.time() + (DEFAULT_ACCESS_TOKEN_EXPIRE_MINUTES * 60))
    if not jti:
        return False
    return record_revocation(jti, exp)


def is_token_revoked(jti: str) -> bool:
    """
    Checks whether a token's jti is in the persistent SQLite blacklist.
    Raises RevocationStoreError on database lookup failure (fail-closed).
    """
    return is_token_revoked_in_db(jti)


# ---------------------------------------------------------------------------
# FastAPI Dependency Resolvers & RBAC Enforcers
# ---------------------------------------------------------------------------

def get_current_user(
    bearer_header: Optional[HTTPAuthorizationCredentials] = Depends(http_bearer),
    token_param: Optional[str] = Depends(oauth2_scheme),
) -> User:
    """
    Extracts and validates the JWT Bearer token from headers.
    Raises HTTP 401 Unauthorized for missing, invalid, expired, or revoked tokens.
    Raises HTTP 503 Service Unavailable if token revocation status cannot be verified (fail-closed).
    """
    raw_token = None
    if bearer_header and bearer_header.credentials:
        raw_token = bearer_header.credentials
    elif token_param:
        raw_token = token_param

    if not raw_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided. Include 'Authorization: Bearer <token>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        secret = get_secret_key()
        payload = jwt.decode(raw_token, secret, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        jti: str = payload.get("jti")

        if username is None or role is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token claims.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if jti:
            try:
                revoked = is_token_revoked(jti)
            except RevocationStoreError as exc:
                logger.error(f"Revocation verification failed (fail-closed): {exc}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Authentication service temporarily unavailable: unable to verify token revocation status.",
                    headers={"Retry-After": "30"},
                )

            if revoked:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked/logged out. Please authenticate again.",
                    headers={"WWW-Authenticate": "Bearer"},
                )

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or malformed authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = USERS_DB.get(username.lower())
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user account does not exist or has been disabled.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return User(
        username=user.username,
        display_name=user.display_name,
        role=user.role,
        is_active=user.is_active,
    )


def require_role(allowed_roles: List[str]):
    """
    Dependency factory enforcing Server-Side Role-Based Access Control (RBAC).
    Raises HTTP 403 Forbidden if user's role is not within allowed_roles.
    """
    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Access Forbidden: User '{current_user.username}' has role '{current_user.role}', "
                    f"which lacks required permission. Required role(s): {', '.join(allowed_roles)}."
                ),
            )
        return current_user

    return role_checker


# Named RBAC dependencies
require_admin_role = require_role([UserRole.MLOPS_ADMIN])
require_user_or_admin = require_role([UserRole.INDEPENDENT_USER, UserRole.MLOPS_ADMIN])
