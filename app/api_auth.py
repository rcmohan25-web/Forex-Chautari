"""
ForexChautari — FastAPI Authentication & Authorization Layer
app/api_auth.py

Features:
  - JWT bearer tokens (access + refresh)
  - Role-based dependency guards  (require_user, require_admin, require_plan)
  - Plan-gated pair access        (pair_allowed_for_plan)
  - Per-IP / per-user rate limiting (in-memory, pluggable)
  - Token blacklist for logout    (in-memory; swap for Redis in production)
  - All errors speak RFC-7807 Problem JSON format

Quick-start:
  pip install python-jose[cryptography] passlib[bcrypt]

Environment variables (add to .env):
  JWT_SECRET=<long-random-string>          # required — generate with: openssl rand -hex 32
  JWT_ALGORITHM=HS256                      # optional, default HS256
  JWT_ACCESS_EXPIRE_MINUTES=60            # optional, default 60
  JWT_REFRESH_EXPIRE_DAYS=7               # optional, default 7
"""

from __future__ import annotations

import os
import time
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Annotated, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel

# ── re-use the existing database layer ───────────────────────────────────────
from src.database import authenticate_user, get_user_by_id
from config.settings import PLAN_LIMITS, ACTIVE_PAIRS

logger = logging.getLogger("api_auth")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

_SECRET = os.getenv("JWT_SECRET", "")
if not _SECRET:
    # Fail loudly at import time so the developer notices immediately.
    raise RuntimeError(
        "JWT_SECRET environment variable is not set.\n"
        "Generate one with:  openssl rand -hex 32\n"
        "Then add it to your .env file:  JWT_SECRET=<value>"
    )

ALGORITHM               = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_EXPIRE_MINUTES   = int(os.getenv("JWT_ACCESS_EXPIRE_MINUTES", "60"))
REFRESH_EXPIRE_DAYS     = int(os.getenv("JWT_REFRESH_EXPIRE_DAYS", "7"))

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────

class TokenResponse(BaseModel):
    access_token:  str
    refresh_token: str
    token_type:    str = "bearer"
    expires_in:    int           # seconds until access token expires


class RefreshRequest(BaseModel):
    refresh_token: str


class LoginRequest(BaseModel):
    username: str
    password: str


class CurrentUser(BaseModel):
    """Validated user payload attached to every authenticated request."""
    id:         int
    username:   str
    email:      str
    full_name:  str
    role:       str              # "admin" | "user"
    plan:       str              # "free" | "basic" | "pro" | "enterprise"
    auto_trade: bool
    max_pairs:  int

    # convenience helpers
    def is_admin(self) -> bool:
        return self.role == "admin"

    def allowed_pairs(self) -> list[str]:
        n = PLAN_LIMITS.get(self.plan, {}).get("pairs", 1)
        return ACTIVE_PAIRS[:n]

    def can_trade(self) -> bool:
        return bool(PLAN_LIMITS.get(self.plan, {}).get("auto_trade", False))


# ─────────────────────────────────────────────────────────────────────────────
# Token blacklist  (swap for Redis in production)
# ─────────────────────────────────────────────────────────────────────────────

_blacklist: set[str] = set()


def blacklist_token(jti: str) -> None:
    """Add a token JTI to the blacklist (logout / forced expiry)."""
    _blacklist.add(jti)


def is_blacklisted(jti: str) -> bool:
    return jti in _blacklist


# ─────────────────────────────────────────────────────────────────────────────
# Rate limiter  (in-memory; replace with slowapi + Redis for production)
# ─────────────────────────────────────────────────────────────────────────────

_rate_windows: dict[str, list[float]] = defaultdict(list)

# limits:  (max_requests, window_seconds)
_RATE_LIMITS: dict[str, tuple[int, int]] = {
    "login":    (10, 60),    # 10 login attempts per minute per IP
    "api":      (300, 60),   # 300 API calls per minute per user/IP
    "retrain":  (3, 3600),   # 3 retrain calls per hour
}


def _check_rate_limit(key: str, bucket: str) -> None:
    """
    Sliding-window rate limiter.
    Raises HTTP 429 if the caller exceeds the bucket's limit.
    key    — identifies the caller (IP or user-id string)
    bucket — one of the keys in _RATE_LIMITS
    """
    max_req, window = _RATE_LIMITS.get(bucket, (300, 60))
    composite = f"{bucket}:{key}"
    now = time.monotonic()
    hits = _rate_windows[composite]

    # drop expired entries
    _rate_windows[composite] = [t for t in hits if now - t < window]
    if len(_rate_windows[composite]) >= max_req:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "type":    "rate_limit_exceeded",
                "title":   "Too Many Requests",
                "detail":  f"Limit: {max_req} requests per {window}s.",
                "bucket":  bucket,
            },
        )
    _rate_windows[composite].append(now)


# ─────────────────────────────────────────────────────────────────────────────
# JWT helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_token(
    payload: dict,
    expire_delta: timedelta,
    token_type: str = "access",
) -> str:
    """Sign a JWT with standard claims."""
    import secrets as _secrets
    now     = datetime.now(timezone.utc)
    expires = now + expire_delta
    data    = {
        **payload,
        "iat":  now.timestamp(),
        "exp":  expires.timestamp(),
        "type": token_type,
        "jti":  _secrets.token_hex(16),   # unique token id — needed for blacklist
    }
    return jwt.encode(data, _SECRET, algorithm=ALGORITHM)


def _decode_token(token: str) -> dict:
    """
    Decode and validate a JWT.
    Raises HTTPException on any problem so callers don't have to.
    """
    try:
        payload = jwt.decode(token, _SECRET, algorithms=[ALGORITHM])
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "type":   "invalid_token",
                "title":  "Unauthorized",
                "detail": f"Token validation failed: {exc}",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    if is_blacklisted(payload.get("jti", "")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"type": "token_revoked", "title": "Unauthorized", "detail": "Token has been revoked."},
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload


def create_token_pair(user: dict) -> TokenResponse:
    """
    Issue both an access token and a refresh token for a user dict
    (as returned by database.authenticate_user or get_user_by_id).
    """
    common = {
        "sub":      str(user["id"]),
        "username": user["username"],
        "role":     user["role"],
        "plan":     user.get("plan", "free"),
    }
    access  = _make_token(common, timedelta(minutes=ACCESS_EXPIRE_MINUTES), "access")
    refresh = _make_token({"sub": str(user["id"])}, timedelta(days=REFRESH_EXPIRE_DAYS), "refresh")
    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        expires_in=ACCESS_EXPIRE_MINUTES * 60,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI dependency: extract + validate current user
# ─────────────────────────────────────────────────────────────────────────────

_bearer = HTTPBearer(auto_error=False)


def _get_current_user_from_token(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(_bearer)],
) -> CurrentUser:
    """
    Core dependency — decode the Bearer token and hydrate a CurrentUser.
    Always re-fetches the user row so deactivated accounts are rejected
    without waiting for the token to expire.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"type": "missing_token", "title": "Unauthorized", "detail": "Bearer token required."},
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = _decode_token(credentials.credentials)

    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"type": "wrong_token_type", "title": "Unauthorized",
                    "detail": "Access token required (not refresh token)."},
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = int(payload.get("sub", 0))
    db_user = get_user_by_id(user_id)

    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"type": "user_not_found", "title": "Unauthorized", "detail": "User no longer exists."},
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not db_user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"type": "account_deactivated", "title": "Forbidden", "detail": "Account has been deactivated."},
        )

    return CurrentUser(
        id=db_user["id"],
        username=db_user["username"],
        email=db_user["email"],
        full_name=db_user.get("full_name") or "",
        role=db_user["role"],
        plan=db_user.get("plan") or "free",
        auto_trade=bool(db_user.get("auto_trade", False)),
        max_pairs=db_user.get("max_pairs") or 1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public dependency aliases  (import these in api.py)
# ─────────────────────────────────────────────────────────────────────────────

# Any authenticated user
AuthUser = Annotated[CurrentUser, Depends(_get_current_user_from_token)]


def require_user(user: AuthUser) -> CurrentUser:
    """Dependency: any active, authenticated user."""
    return user


def require_admin(user: AuthUser) -> CurrentUser:
    """Dependency: admin role only."""
    if not user.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "type":   "insufficient_permissions",
                "title":  "Forbidden",
                "detail": "Admin role required.",
            },
        )
    return user


def require_plan(*plans: str):
    """
    Dependency factory — restrict an endpoint to specific subscription plans.

    Usage:
        @router.get("/pro-feature")
        def pro_feature(user: CurrentUser = Depends(require_plan("pro", "enterprise"))):
            ...
    """
    def _check(user: AuthUser) -> CurrentUser:
        if not user.is_admin() and user.plan not in plans:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "type":     "plan_required",
                    "title":    "Forbidden",
                    "detail":   f"This endpoint requires one of: {list(plans)}.",
                    "your_plan": user.plan,
                    "upgrade":  "Contact your admin or visit the Account tab.",
                },
            )
        return user
    return _check


def pair_allowed_for_plan(pair: str, user: CurrentUser) -> None:
    """
    Check that the requested pair is accessible under the user's plan.
    Admins bypass all pair restrictions.
    Raises HTTP 403 otherwise.

    Call this inside any endpoint that accepts a `pair` query parameter.
    """
    if user.is_admin():
        return
    allowed = user.allowed_pairs()
    if pair not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "type":          "pair_not_in_plan",
                "title":         "Forbidden",
                "detail":        f"Pair '{pair}' is not available on your '{user.plan}' plan.",
                "allowed_pairs": allowed,
                "upgrade":       "Upgrade to Pro to access all 4 pairs.",
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# Rate-limit helper dependency
# ─────────────────────────────────────────────────────────────────────────────

def rate_limit(bucket: str = "api"):
    """
    Dependency factory for rate limiting.

    Usage:
        @router.post("/retrain", dependencies=[Depends(rate_limit("retrain"))])
        def retrain_endpoint(...):
            ...
    """
    def _check(request: Request, user: AuthUser) -> None:
        key = str(user.id)   # rate-limit per authenticated user
        _check_rate_limit(key, bucket)
    return _check


def rate_limit_by_ip(bucket: str = "login"):
    """
    Rate-limit dependency keyed on client IP (for pre-auth endpoints like /login).
    """
    def _check(request: Request) -> None:
        ip = request.client.host if request.client else "unknown"
        _check_rate_limit(ip, bucket)
    return _check


# ─────────────────────────────────────────────────────────────────────────────
# Auth endpoint handlers  (mount these as an APIRouter in api.py)
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import APIRouter

auth_router = APIRouter(prefix="/auth", tags=["Auth"])


@auth_router.post(
    "/login",
    response_model=TokenResponse,
    summary="Exchange credentials for JWT tokens",
    dependencies=[Depends(rate_limit_by_ip("login"))],
)
def login(body: LoginRequest, request: Request) -> TokenResponse:
    """
    Authenticate with username/password and receive access + refresh tokens.

    The access token is short-lived (default 60 min).
    The refresh token is long-lived (default 7 days) and should be stored
    securely — use it with POST /auth/refresh to get a new access token.
    """
    user = authenticate_user(body.username.strip(), body.password)
    if not user:
        # Uniform error — don't reveal whether the username exists
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "type":   "invalid_credentials",
                "title":  "Unauthorized",
                "detail": "Incorrect username or password.",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )
    logger.info(f"Login OK: user_id={user['id']} username={user['username']} ip={request.client.host if request.client else '?'}")
    return create_token_pair(user)


@auth_router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Exchange a refresh token for a new access token",
)
def refresh_token(body: RefreshRequest) -> TokenResponse:
    """
    Use a valid refresh token to obtain a fresh access token (and a rotated
    refresh token — the old one is immediately blacklisted).
    """
    payload = _decode_token(body.refresh_token)

    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"type": "wrong_token_type", "title": "Bad Request",
                    "detail": "A refresh token is required here."},
        )

    # Rotate: blacklist the old refresh token
    blacklist_token(payload["jti"])

    user_id = int(payload.get("sub", 0))
    db_user = get_user_by_id(user_id)
    if not db_user or not db_user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"type": "user_not_found", "title": "Unauthorized",
                    "detail": "User not found or deactivated."},
        )
    return create_token_pair(db_user)


@auth_router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke the current access token",
)
def logout(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(_bearer)],
) -> None:
    """
    Blacklist the presented token so it can no longer be used.
    Clients should also discard their locally stored refresh token.
    """
    if credentials:
        payload = _decode_token(credentials.credentials)
        blacklist_token(payload.get("jti", ""))


@auth_router.get(
    "/me",
    response_model=CurrentUser,
    summary="Return the current authenticated user",
)
def me(user: AuthUser) -> CurrentUser:
    """Introspect endpoint — useful for frontends to validate a stored token."""
    return user
