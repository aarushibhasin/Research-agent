from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

import redis

from agents.state import AgentState, ensure_state_defaults
from utils.console_trace import trace

_REDIS_CLIENT: redis.Redis | None = None


def _get_redis() -> redis.Redis:
    global _REDIS_CLIENT
    if _REDIS_CLIENT is None:
        url = os.getenv("REDIS_URL", "redis://localhost:6379")
        _REDIS_CLIENT = redis.from_url(url, decode_responses=True)
    return _REDIS_CLIENT


def _conv_list_key(session_id: str) -> str:
    prefix = os.getenv("REDIS_CONV_PREFIX", "conv:")
    return f"{prefix}{session_id}"


def _append_turn_to_redis(
    session_id: str,
    *,
    query: str,
    answer: str,
    rewritten_query: str,
    query_type: str,
    keywords: list[str],
) -> int:
    payload = {
        "query": query,
        "answer": answer,
        "rewritten_query": rewritten_query,
        "query_type": query_type,
        "keywords": list(keywords),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    r = _get_redis()
    key = _conv_list_key(session_id)
    r.rpush(key, json.dumps(payload, ensure_ascii=False))
    return int(r.llen(key))


async def memory_store_agent(state: AgentState) -> AgentState:
    ensure_state_defaults(state)
    start = time.perf_counter()
    try:
        prefix = os.getenv("REDIS_CONV_PREFIX", "conv:")
        session_id = state["session_id"]

        new_turn = int(state.get("turn_index") or 0) + 1

        trace(
            "Node4 memory_store -> writing",
            session_id=session_id,
            turn_index=new_turn,
            answer_sample=(state.get("answer") or "")[:80].replace("\n", " "),
        )

        redis_start = time.perf_counter()
        _append_turn_to_redis(
            session_id,
            query=state["query"],
            answer=state.get("answer") or "",
            rewritten_query=state.get("rewritten_query") or "",
            query_type=state.get("query_type") or "factual",
            keywords=list(state.get("keywords") or []),
        )
        redis_store_ms = (time.perf_counter() - redis_start) * 1000

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        hist = list(state.get("conversation_history") or [])
        hist.append({"role": "user", "content": state["query"], "timestamp": ts})
        hist.append(
            {
                "role": "assistant",
                "content": state.get("answer") or "",
                "timestamp": ts,
            }
        )
        state["conversation_history"] = hist
        state["turn_index"] = new_turn
        state["t_store_ms"] = (time.perf_counter() - start) * 1000
        state["t_redis_store_ms"] = redis_store_ms
        state["t_redis_store_total_ms"] = float(state.get("t_redis_store_total_ms") or 0.0) + redis_store_ms
        state["latency_breakdown"] = list(state.get("latency_breakdown") or []) + [
            {"node": "memory_store", "database": "redis", "metric": "store", "ms": redis_store_ms}
        ]
        state["stored_memory_id"] = f"{prefix}{session_id}:{new_turn}"

        trace(
            "Node4 memory_store -> done",
            stored_memory_id=state["stored_memory_id"],
            conversation_history_len=len(state.get("conversation_history") or []),
        )
        return state
    except Exception:
        state["t_store_ms"] = (time.perf_counter() - start) * 1000
        state["t_redis_store_ms"] = 0.0
        state["stored_memory_id"] = ""

        trace("Node4 memory_store -> error")
        return state
