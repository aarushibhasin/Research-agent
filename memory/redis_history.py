"""Append conversation turns to Redis LIST conv:{session_id} (JSON per element)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import redis

_client: redis.Redis | None = None


def _get_redis() -> redis.Redis:
    global _client
    if _client is None:
        url = os.getenv("REDIS_URL", "redis://localhost:6379")
        _client = redis.from_url(url, decode_responses=True)
    return _client


def conv_list_key(session_id: str) -> str:
    prefix = os.getenv("REDIS_CONV_PREFIX", "conv:")
    return f"{prefix}{session_id}"


def append_turn(
    session_id: str,
    *,
    query: str,
    answer: str,
    rewritten_query: str,
    query_type: str,
    keywords: list[str],
) -> int:
    """
    RPUSH one JSON object for this turn. Returns 1-based turn index (LLEN after push).
    """
    key = conv_list_key(session_id)
    payload: dict[str, Any] = {
        "query": query,
        "answer": answer,
        "rewritten_query": rewritten_query,
        "query_type": query_type,
        "keywords": list(keywords),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    r = _get_redis()
    r.rpush(key, json.dumps(payload, ensure_ascii=False))
    return int(r.llen(key))
