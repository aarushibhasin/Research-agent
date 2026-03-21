import os
from typing import Any, List, MutableMapping, TypedDict


class AgentState(TypedDict):
    # Input
    query: str
    session_id: str  # used by Redis checkpointer for thread identity

    # Session context (checkpointed; prior turns only before current inference)
    conversation_history: List[dict]  # each: {role, content, optional timestamp}
    turn_index: int  # increments after each stored turn

    # Node 1 — intent
    rewritten_query: str
    query_type: str  # "factual" | "comparative" | "exploratory"
    keywords: List[str]

    # Node 2 — retrieval
    retrieved_chunks: List[dict]  # each: {id, document, score, metadata}
    retrieval_context: str  # formatted string of retrieved chunks

    # Retrieval collection override (used by benchmark prebuilt collections).
    chroma_collection_name: str

    # Node 3 — inference
    answer: str

    # Node 4 — Redis conversation store
    stored_memory_id: str

    # Metrics (populated by each node, collected at end)
    t_retrieval_ms: float
    t_inference_ms: float
    t_store_ms: float
    t_e2e_ms: float  # set by main.py wrapping the full ainvoke call
    t_time_to_first_token_ms: float
    t_time_to_last_token_ms: float
    t_chroma_retrieve_ms: float
    t_redis_store_ms: float
    t_chroma_retrieve_total_ms: float
    t_redis_store_total_ms: float
    latency_breakdown: List[dict]

    # Debug / error propagation (kept in state so LangGraph doesn't drop them)
    retrieval_error: str
    inference_error: str
    inference_msg_type: str


def ensure_state_defaults(state: MutableMapping[str, Any]) -> None:
    """Fill missing keys when using partial ainvoke input + Redis checkpoint merge."""
    if state.get("conversation_history") is None:
        state["conversation_history"] = []
    if state.get("turn_index") is None:
        state["turn_index"] = 0
    if state.get("chroma_collection_name") is None:
        state["chroma_collection_name"] = os.getenv("CHROMA_COLLECTION_PAPERS", "research_memory")
    if state.get("retrieval_error") is None:
        state["retrieval_error"] = ""
    if state.get("inference_error") is None:
        state["inference_error"] = ""
    if state.get("inference_msg_type") is None:
        state["inference_msg_type"] = ""
    if state.get("t_time_to_first_token_ms") is None:
        state["t_time_to_first_token_ms"] = 0.0
    if state.get("t_time_to_last_token_ms") is None:
        state["t_time_to_last_token_ms"] = 0.0
    if state.get("t_chroma_retrieve_ms") is None:
        state["t_chroma_retrieve_ms"] = 0.0
    if state.get("t_redis_store_ms") is None:
        state["t_redis_store_ms"] = 0.0
    if state.get("t_chroma_retrieve_total_ms") is None:
        state["t_chroma_retrieve_total_ms"] = 0.0
    if state.get("t_redis_store_total_ms") is None:
        state["t_redis_store_total_ms"] = 0.0
    if state.get("latency_breakdown") is None:
        state["latency_breakdown"] = []


def default_state(query: str, session_id: str) -> AgentState:
    return {
        "query": query,
        "session_id": session_id,
        "conversation_history": [],
        "turn_index": 0,
        "rewritten_query": query,
        "query_type": "factual",
        "keywords": [],
        "retrieved_chunks": [],
        "retrieval_context": "",
        "chroma_collection_name": os.getenv("CHROMA_COLLECTION_PAPERS", "research_memory"),
        "answer": "",
        "stored_memory_id": "",
        "t_retrieval_ms": 0.0,
        "t_inference_ms": 0.0,
        "t_store_ms": 0.0,
        "t_e2e_ms": 0.0,
        "t_time_to_first_token_ms": 0.0,
        "t_time_to_last_token_ms": 0.0,
        "t_chroma_retrieve_ms": 0.0,
        "t_redis_store_ms": 0.0,
        "t_chroma_retrieve_total_ms": 0.0,
        "t_redis_store_total_ms": 0.0,
        "latency_breakdown": [],
        "retrieval_error": "",
        "inference_error": "",
        "inference_msg_type": "",
    }

