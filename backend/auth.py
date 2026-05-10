"""
auth.py — Simple JWT-based authentication for RescueAI.

Users are stored in-memory (replace with DB in production).
Roles: "admin" | "ngo_operator"
"""

import hashlib
import time
import json
import base64
import hmac
from typing import Optional

SECRET_KEY = "rescueai-secret-key-change-in-production"

# ── In-memory user store ───────────────────────────────────────────────────────
# password stored as sha256 hex
USERS = {
    "admin": {
        "password": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "admin",
        "name": "System Administrator",
        "ngo": "RescueAI HQ",
    },
    "ngo1": {
        "password": hashlib.sha256("ngo123".encode()).hexdigest(),
        "role": "ngo_operator",
        "name": "Todd's Welfare Operator",
        "ngo": "Todd's Welfare",
    },
    "ngo2": {
        "password": hashlib.sha256("ngo456".encode()).hexdigest(),
        "role": "ngo_operator",
        "name": "ACF Operator",
        "ngo": "Animal Care Foundation",
    },
}


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _sign(msg: str) -> str:
    return _b64url(hmac.new(SECRET_KEY.encode(), msg.encode(), hashlib.sha256).digest())


def create_token(username: str, role: str) -> str:
    header  = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64url(json.dumps({
        "sub": username,
        "role": role,
        "iat": int(time.time()),
        "exp": int(time.time()) + 86400,   # 24 h
    }).encode())
    sig = _sign(f"{header}.{payload}")
    return f"{header}.{payload}.{sig}"


def verify_token(token: str) -> Optional[dict]:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header, payload, sig = parts
        expected = _sign(f"{header}.{payload}")
        if not hmac.compare_digest(sig, expected):
            return None
        # decode payload
        pad = 4 - len(payload) % 4
        data = json.loads(base64.urlsafe_b64decode(payload + "=" * pad))
        if data.get("exp", 0) < time.time():
            return None
        return data
    except Exception:
        return None


def authenticate(username: str, password: str) -> Optional[dict]:
    user = USERS.get(username)
    if not user:
        return None
    pw_hash = hashlib.sha256(password.encode()).hexdigest()
    if not hmac.compare_digest(pw_hash, user["password"]):
        return None
    return {
        "username": username,
        "role": user["role"],
        "name": user["name"],
        "ngo": user["ngo"],
    }
