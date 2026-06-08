"""QWeather (和风天气) API client — JWT (EdDSA) auth + in-memory cache.

Centralizes all 和风天气 API access. Per docs/技术准备.md, auth is JWT with:
  - alg = "EdDSA" (Ed25519)
  - kid = credential id (QWEATHER_KID)
  - sub = project id  (QWEATHER_PROJECT_ID)
  - iat, exp = 15-minute window

Configuration is read from env vars at call time (cached on the key object);
if QWEATHER_PRIVATE_KEY is missing/blank, ``is_configured()`` returns False and
callers should fall back to mocks.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

import httpx
import jwt

log = logging.getLogger(__name__)

# Defaults from docs/技术准备.md
_DEFAULT_HOST = "https://pr5khwpqpj.re.qweatherapi.com"
_DEFAULT_PROJECT_ID = "268BCGC4RX"
_DEFAULT_KID = "C45F5YBK9B"

# JWT lifetime: stay well under the 24h hard cap; long enough to amortize.
_JWT_TTL_SECONDS = 15 * 60
# Refresh JWT this many seconds before expiry.
_JWT_SAFETY_MARGIN = 60

# HTTP request timeout (seconds).
_HTTP_TIMEOUT = 5.0

# In-memory cache entry: (value, expires_at_monotonic).
_CacheVal = Tuple[Any, float]
_cache: Dict[str, _CacheVal] = {}
_cache_lock = threading.Lock()

# Lazy-loaded private key + cached JWT.
_lock = threading.Lock()
_private_key: Optional[Any] = None  # cryptography Ed25519PrivateKey
_jwt: Optional[str] = None
_jwt_expires_at: float = 0.0


class QWeatherError(RuntimeError):
    """Raised on any QWeather API failure (network, HTTP, JSON, missing config)."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def _private_key_pem() -> str:
    """Return the raw PEM from env, supporting multi-line values via \\n escapes."""
    raw = os.getenv("QWEATHER_PRIVATE_KEY", "").strip()
    if not raw:
        return ""
    # Allow .env users to write literal "\n" instead of real newlines.
    return raw.replace("\\n", "\n")


def is_configured() -> bool:
    """True when a non-empty private key is available for signing."""
    return bool(_private_key_pem())


def host() -> str:
    return os.getenv("QWEATHER_HOST", _DEFAULT_HOST).rstrip("/")


def _project_id() -> str:
    return os.getenv("QWEATHER_PROJECT_ID", _DEFAULT_PROJECT_ID)


def _kid() -> str:
    return os.getenv("QWEATHER_KID", _DEFAULT_KID)


# ---------------------------------------------------------------------------
# JWT (EdDSA) signing
# ---------------------------------------------------------------------------
def _load_private_key() -> Any:
    """Load the Ed25519 private key from PEM, cached after first call."""
    global _private_key
    if _private_key is not None:
        return _private_key
    pem = _private_key_pem()
    if not pem:
        raise QWeatherError(
            "QWEATHER_PRIVATE_KEY 未配置。请在 backend/.env 设置 Ed25519 私钥（PEM 格式）。"
        )
    try:
        from cryptography.hazmat.primitives import serialization
        _private_key = serialization.load_pem_private_key(pem.encode(), password=None)
    except Exception as e:
        raise QWeatherError(
            f"QWEATHER_PRIVATE_KEY 解析失败，请检查是否为有效的 Ed25519 PEM 私钥：{e}"
        ) from e
    if not hasattr(_private_key, "private_bytes"):
        raise QWeatherError("加载的私钥不是 Ed25519 类型，请确认 alg=EdDSA 对应的密钥")
    return _private_key


def _build_jwt() -> str:
    """Return a freshly-signed JWT. Caller must hold _lock."""
    global _jwt, _jwt_expires_at
    now = int(time.time())
    payload = {
        "sub": _project_id(),
        "iat": now,
        "exp": now + _JWT_TTL_SECONDS,
    }
    headers = {"alg": "EdDSA", "kid": _kid()}
    key = _load_private_key()
    token = jwt.encode(payload, key, algorithm="EdDSA", headers=headers)
    _jwt = token
    _jwt_expires_at = time.monotonic() + _JWT_TTL_SECONDS - _JWT_SAFETY_MARGIN
    return token


def _get_jwt() -> str:
    """Return a non-expired JWT, refreshing if needed. Thread-safe."""
    global _jwt, _jwt_expires_at
    with _lock:
        if _jwt is None or time.monotonic() >= _jwt_expires_at:
            return _build_jwt()
        return _jwt


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
def _cache_get(key: str) -> Optional[Any]:
    with _cache_lock:
        entry = _cache.get(key)
        if not entry:
            return None
        value, expires_at = entry
        if time.monotonic() >= expires_at:
            _cache.pop(key, None)
            return None
        return value


def _cache_set(key: str, value: Any, ttl_seconds: float) -> None:
    with _cache_lock:
        _cache[key] = (value, time.monotonic() + ttl_seconds)


# ---------------------------------------------------------------------------
# HTTP request
# ---------------------------------------------------------------------------
def _request(
    path: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    cache_ttl: float = 0.0,
) -> Dict[str, Any]:
    """Authenticated GET against the QWeather API.

    Raises QWeatherError on any failure. If ``cache_ttl`` > 0, the parsed JSON
    response is cached by ``(path, frozenset(params))`` for that many seconds.
    """
    if not is_configured():
        raise QWeatherError("QWeather 未配置 QWEATHER_PRIVATE_KEY")
    cache_key = f"{path}?{frozenset((params or {}).items())}"
    if cache_ttl > 0:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

    url = f"{host()}{path}"
    headers = {"Authorization": f"Bearer {_get_jwt()}"}
    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            resp = client.get(url, params=params or {}, headers=headers)
    except httpx.HTTPError as e:
        raise QWeatherError(f"QWeather 网络错误 {path}: {e}") from e

    if resp.status_code != 200:
        raise QWeatherError(
            f"QWeather HTTP {resp.status_code} {path}: {resp.text[:200]}"
        )
    try:
        data = resp.json()
    except ValueError as e:
        raise QWeatherError(f"QWeather 响应不是合法 JSON {path}: {e}") from e

    code = str(data.get("code", ""))
    if code and code != "200":
        raise QWeatherError(
            f"QWeather 业务错误 {path}: code={code}, body={str(data)[:200]}"
        )

    if cache_ttl > 0:
        _cache_set(cache_key, data, cache_ttl)
    return data
