from __future__ import annotations

import os
import time

from agents.state import AgentState, ensure_state_defaults
from memory import chroma_client
from memory.embedder import embed_single
from utils.console_trace import trace


def _distance_to_score(distance: float) -> float:
    try:
        return 1.0 / (1.0 + float(distance))
    except Exception:
        return 0.0


def _build_chunks_from_chroma_result(result: dict) -> list[dict]:
    """
    Convert a Chroma `query()` response into a list of chunk dicts.
    Chroma returns nested lists; tolerate missing keys.
    """
    ids = (result.get("ids") or [[]])[0]
    docs = (result.get("documents") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]

    chunks: list[dict] = []
    n = min(len(docs), len(dists)) if dists else len(docs)
    for i in range(n):
        score = _distance_to_score(dists[i]) if dists else 0.0
        chunks.append(
            {
                "id": ids[i] if i < len(ids) else "",
                "document": docs[i] if i < len(docs) else "",
                "score": score,
                "metadata": metas[i] if i < len(metas) else {},
            }
        )
    return chunks


async def retrieval_agent(state: AgentState) -> AgentState:
    ensure_state_defaults(state)
    start = time.perf_counter()
    chroma_total_ms = 0.0
    try:
        top_k = int(os.getenv("TOP_K_CHUNKS", "5"))
        collection_name = state.get("chroma_collection_name") or os.getenv(
            "CHROMA_COLLECTION_PAPERS", "research_memory"
        )

        trace(
            "Node2 retrieval -> start",
            chroma_collection_name=collection_name,
            top_k=top_k,
            rewritten_query_sample=(state.get("rewritten_query") or "")[:80],
        )

        # Two-pass semantic retrieval:
        # - `rewritten_query` usually improves relevance.
        # - the original `query` is a safety net if the rewrite drifts.
        queries_to_try = [
            (state.get("rewritten_query") or "").strip(),
            (state.get("query") or "").strip(),
        ]

        # Merge results by `id` (or a fallback doc prefix key).
        merged_by_key: dict[str, dict] = {}

        for q_text in queries_to_try:
            if not q_text:
                continue
            q_emb = embed_single(q_text)
            chroma_start = time.perf_counter()
            result = chroma_client.query_collection(
                collection_name=collection_name,
                query_embedding=q_emb,
                n_results=top_k,
            )
            chroma_total_ms += (time.perf_counter() - chroma_start) * 1000
            for ch in _build_chunks_from_chroma_result(result):
                doc_key = ch.get("id") or (ch.get("document") or "")[:80]
                prev = merged_by_key.get(doc_key)
                if prev is None or float(ch.get("score") or 0.0) > float(prev.get("score") or 0.0):
                    merged_by_key[doc_key] = ch

        merged_chunks = sorted(
            merged_by_key.values(),
            key=lambda x: float(x.get("score") or 0.0),
            reverse=True,
        )[:top_k]

        lines = []
        for i, ch in enumerate(merged_chunks):
            lines.append(
                f"[{i+1}] (score: {float(ch.get('score') or 0.0):.2f}) {ch.get('document') or ''}"
            )

        state["t_retrieval_ms"] = (time.perf_counter() - start) * 1000
        state["t_chroma_retrieve_ms"] = chroma_total_ms
        state["t_chroma_retrieve_total_ms"] = float(state.get("t_chroma_retrieve_total_ms") or 0.0) + chroma_total_ms
        state["latency_breakdown"] = list(state.get("latency_breakdown") or []) + [
            {"node": "retrieval", "database": "chroma", "metric": "retrieve", "ms": chroma_total_ms}
        ]
        state["retrieved_chunks"] = merged_chunks
        state["retrieval_context"] = "\n".join(lines)

        # Print a small preview rather than dumping full context.
        first_doc = merged_chunks[0].get("document") if merged_chunks else ""
        trace(
            "Node2 retrieval -> done",
            retrieved_chunks=len(merged_chunks),
            retrieval_context_chars=len(state["retrieval_context"]),
            top_doc_sample=(first_doc or "")[:120].replace("\n", " "),
        )
        return state
    except Exception as e:
        state["t_retrieval_ms"] = (time.perf_counter() - start) * 1000
        state["t_chroma_retrieve_ms"] = chroma_total_ms
        state["t_chroma_retrieve_total_ms"] = float(state.get("t_chroma_retrieve_total_ms") or 0.0) + chroma_total_ms
        state["latency_breakdown"] = list(state.get("latency_breakdown") or []) + [
            {"node": "retrieval", "database": "chroma", "metric": "retrieve", "ms": chroma_total_ms}
        ]
        state["retrieved_chunks"] = []
        state["retrieval_context"] = ""
        state["retrieval_error"] = repr(e)

        trace("Node2 retrieval -> error", error=repr(e))
        return state

