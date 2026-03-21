from __future__ import annotations

import time

from langchain_core.prompts import ChatPromptTemplate

from agents.state import AgentState, ensure_state_defaults
from utils.llm_client import llm
from utils.console_trace import trace


def _format_conversation_history(history: list[dict]) -> str:
    if not history:
        return "(none)"
    lines: list[str] = []
    for m in history:
        role = str(m.get("role", ""))
        content = str(m.get("content", ""))
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


async def inference_agent(state: AgentState) -> AgentState:
    ensure_state_defaults(state)
    start = time.perf_counter()
    try:
        history_block = _format_conversation_history(state.get("conversation_history") or [])
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Answer the query using the conversation so far (if any) and the provided memory context. "
                    "If the answer is not supported by the memory context, say so explicitly. "
                    "For quantitative questions, use only the numeric values present in the memory context; do not invent.",
                ),
                (
                    "human",
                    "Conversation so far:\n{conversation_history}\n\n"
                    "Query: {rewritten_query}\n\n"
                    "Memory context:\n{retrieval_context}",
                ),
            ]
        )
        formatted_messages = prompt.format_messages(
            conversation_history=history_block,
            rewritten_query=state["rewritten_query"],
            retrieval_context=state["retrieval_context"],
        )
        first_token_ms = None
        chunks: list[str] = []
        msg_type = "AIMessageChunk"

        async for chunk in llm.astream(formatted_messages):
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - start) * 1000
            msg_type = type(chunk).__name__
            content = getattr(chunk, "content", "")
            if isinstance(content, str):
                if content:
                    chunks.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, str) and part:
                        chunks.append(part)
                    elif isinstance(part, dict):
                        part_text = str(part.get("text", "")).strip()
                        if part_text:
                            chunks.append(part_text)

        ttlt_ms = (time.perf_counter() - start) * 1000
        state["t_time_to_first_token_ms"] = float(first_token_ms or ttlt_ms)
        state["t_time_to_last_token_ms"] = ttlt_ms
        state["t_inference_ms"] = ttlt_ms
        state["inference_msg_type"] = msg_type
        state["answer"] = "".join(chunks).strip()
        if not state["answer"]:
            trace("Node3 inference -> empty", inference_msg_type=state["inference_msg_type"])
        else:
            trace(
                "Node3 inference -> done",
                inference_msg_type=state["inference_msg_type"],
                answer_sample=(state["answer"] or "")[:80].replace("\n", " "),
            )
        return state
    except Exception as e:
        state["t_inference_ms"] = (time.perf_counter() - start) * 1000
        state["t_time_to_first_token_ms"] = 0.0
        state["t_time_to_last_token_ms"] = state["t_inference_ms"]
        state["answer"] = ""
        state["inference_error"] = repr(e)
        trace("Node3 inference -> error", error=repr(e))
        return state

