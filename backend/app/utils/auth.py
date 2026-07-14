from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from cryptography.fernet import Fernet
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.config import settings
import base64
import hashlib
import os

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Fernet key for RTSP URL encryption
_fernet: Optional[Fernet] = None


def get_fernet() -> Fernet:
    global _fernet
    if _fernet is None:
        key = settings.FERNET_KEY
        if not key:
            # Keep encrypted camera URLs decryptable across restarts.
            digest = hashlib.sha256(
                ("bevp:rtsp-url:" + settings.SECRET_KEY).encode("utf-8")
            ).digest()
            key = base64.urlsafe_b64encode(digest).decode("ascii")
        try:
            key_bytes = key.encode() if isinstance(key, str) else key
            _fernet = Fernet(key_bytes)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("FERNET_KEY must be a valid urlsafe base64 Fernet key") from exc
    return _fernet


def encrypt_url(url: str) -> str:
    """Encrypt RTSP URL for storage."""
    f = get_fernet()
    return f.encrypt(url.encode()).decode()


def decrypt_url(encrypted: str) -> str:
    """Decrypt stored RTSP URL."""
    f = get_fernet()
    return f.decrypt(encrypted.encode()).decode()


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode["exp"] = expire
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    payload = decode_token(credentials.credentials)
    user_id = payload.get("sub")
    role = payload.get("role", "viewer")

    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid token")

    return {"id": int(user_id), "role": role, "email": payload.get("email")}


def require_role(*roles: str):
    async def _check(user=Depends(get_current_user)):
        if user["role"] not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user['role']}' is not allowed. Required: {roles}"
            )
        return user
    return _check


# Convenience role deps
require_admin = require_role("admin")
require_operator = require_role("admin", "operator")
require_viewer = require_role("admin", "operator", "viewer")
