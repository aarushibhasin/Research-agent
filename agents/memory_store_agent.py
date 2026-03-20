from __future__ import annotations

import os
import time
from datetime import datetime, timezone

from agents.state import AgentState, ensure_state_defaults
from memory import redis_history
from utils.console_trace import trace


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

        redis_history.append_turn(
            session_id,
            query=state["query"],
            answer=state.get("answer") or "",
            rewritten_query=state.get("rewritten_query") or "",
            query_type=state.get("query_type") or "factual",
            keywords=list(state.get("keywords") or []),
        )

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
        state["stored_memory_id"] = f"{prefix}{session_id}:{new_turn}"

        trace(
            "Node4 memory_store -> done",
            stored_memory_id=state["stored_memory_id"],
            conversation_history_len=len(state.get("conversation_history") or []),
        )
        return state
    except Exception:
        state["t_store_ms"] = (time.perf_counter() - start) * 1000
        state["stored_memory_id"] = ""

        trace("Node4 memory_store -> error")
        return state
