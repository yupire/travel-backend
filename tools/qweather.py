"""QWeather (和风天气) API 客户端 — JWT (EdDSA) 认证 + 内存缓存

这是和风天气 API 的统一访问层，负责：
1. JWT (EdDSA/Ed25519) 签名认证
2. HTTP 请求封装
3. 响应数据内存缓存

认证方式（参考 docs/技术准备.md）：
  - alg = "EdDSA" (Ed25519 签名算法)
  - kid = credential id (QWEATHER_KID)
  - sub = project id (QWEATHER_PROJECT_ID)
  - iat, exp = 15 分钟有效期窗口

配置通过环境变量读取；如果 QWEATHER_PRIVATE_KEY 未配置，
is_configured() 返回 False，调用方应降级到 mock 数据。
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
    """QWeather API 错误异常类

    当以下情况发生时抛出：
    - 网络请求失败
    - HTTP 状态码非 200
    - JSON 解析失败
    - 业务 code 非 200
    - 私钥未配置或无效
    """


# ---------------------------------------------------------------------------
# 配置读取 (Configuration)
# ---------------------------------------------------------------------------
def _private_key_pem() -> str:
    """从环境变量读取 Ed25519 私钥 (PEM 格式)

    支持 .env 文件中用 "\\n" 转义表示换行，方便配置多行私钥。
    """
    raw = os.getenv("QWEATHER_PRIVATE_KEY", "").strip()
    if not raw:
        return ""
    # 允许 .env 用户用字面量 "\n" 代替真实换行
    return raw.replace("\\n", "\n")


def is_configured() -> bool:
    """检查 QWeather 私钥是否已配置

    Returns:
        bool: 私钥非空返回 True，否则返回 False
    """
    return bool(_private_key_pem())


def host() -> str:
    """获取 QWeather API 主机地址"""
    return os.getenv("QWEATHER_HOST", _DEFAULT_HOST).rstrip("/")


def _project_id() -> str:
    """获取 QWeather 项目 ID (JWT payload 中的 sub)"""
    return os.getenv("QWEATHER_PROJECT_ID", _DEFAULT_PROJECT_ID)


def _kid() -> str:
    """获取 QWeather 凭证 ID (JWT header 中的 kid)"""
    return os.getenv("QWEATHER_KID", _DEFAULT_KID)


# ---------------------------------------------------------------------------
# JWT (EdDSA) 签名认证
# ---------------------------------------------------------------------------
def _load_private_key() -> Any:
    """加载 Ed25519 私钥（首次调用后缓存）

    Raises:
        QWeatherError: 私钥未配置、解析失败或密钥类型错误时
    """
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
    """生成新签名的 JWT token

    调用者必须持有 _lock。

    JWT 结构：
    - Header: {"alg": "EdDSA", "kid": credential_id}
    - Payload: {"sub": project_id, "iat": now, "exp": now + 15min}

    Returns:
        str: 签名后的 JWT token
    """
    global _jwt, _jwt_expires_at
    now = int(time.time())
    payload = {
        "sub": _project_id(),
        "iat": now,
        "exp": now + _JWT_TTL_SECONDS,
    }
    headers = {"alg": "EdDSA", "kid": _kid()}
    key = _load_private_key()
    # 使用 EdDSA 算法签名生成 JWT
    token = jwt.encode(payload, key, algorithm="EdDSA", headers=headers)
    _jwt = token
    # 使用 monotonic 时间避免系统时间调整影响，提前 60 秒刷新确保安全
    _jwt_expires_at = time.monotonic() + _JWT_TTL_SECONDS - _JWT_SAFETY_MARGIN
    return token


def _get_jwt() -> str:
    """获取未过期的 JWT，如需要则自动刷新（线程安全）

    Returns:
        str: 有效的 JWT token
    """
    global _jwt, _jwt_expires_at
    with _lock:
        # JWT 不存在或即将过期时重新生成
        if _jwt is None or time.monotonic() >= _jwt_expires_at:
            return _build_jwt()
        return _jwt


# ---------------------------------------------------------------------------
# 内存缓存 (Caching)
# ---------------------------------------------------------------------------
def _cache_get(key: str) -> Optional[Any]:
    """从内存缓存获取数据

    如果数据已过期，自动删除并返回 None。

    Args:
        key: 缓存键

    Returns:
        缓存的值，未命中或过期返回 None
    """
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
    """设置内存缓存

    Args:
        key: 缓存键
        value: 要缓存的值
        ttl_seconds: 过期时间（秒）
    """
    with _cache_lock:
        _cache[key] = (value, time.monotonic() + ttl_seconds)


# ---------------------------------------------------------------------------
# HTTP 请求封装 (HTTP request)
# ---------------------------------------------------------------------------
def _request(
    path: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    cache_ttl: float = 0.0,
) -> Dict[str, Any]:
    """向 QWeather API 发起认证的 GET 请求

    自动添加 JWT Bearer token，处理错误，支持缓存。

    Args:
        path: API 路径，如 "/v7/weather/3d"
        params: URL 查询参数
        cache_ttl: 缓存时间（秒），0 表示不缓存

    Returns:
        Dict: 解析后的 JSON 响应

    Raises:
        QWeatherError: 网络错误、HTTP 错误、JSON 解析错误、业务错误
    """
    if not is_configured():
        raise QWeatherError("QWeather 未配置 QWEATHER_PRIVATE_KEY")
    # 使用 (path, 参数) 作为缓存键，确保相同请求复用缓存
    cache_key = f"{path}?{frozenset((params or {}).items())}"
    if cache_ttl > 0:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

    url = f"{host()}{path}"
    # 添加 JWT Bearer token 认证头
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

    # QWeather 业务状态码：200 表示成功
    code = str(data.get("code", ""))
    if code and code != "200":
        raise QWeatherError(
            f"QWeather 业务错误 {path}: code={code}, body={str(data)[:200]}"
        )

    if cache_ttl > 0:
        _cache_set(cache_key, data, cache_ttl)
    return data
