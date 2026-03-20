from __future__ import annotations

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agents.state import AgentState, ensure_state_defaults
from utils.llm_client import llm
from utils.console_trace import trace


async def intent_agent(state: AgentState) -> AgentState:
    ensure_state_defaults(state)
    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    'Output only valid JSON with keys "rewritten_query" (string), '
                    '"query_type" (one of "factual","comparative","exploratory"), '
                    '"keywords" (list of 3-6 strings).',
                ),
                ("human", "{query}"),
            ]
        )
        parser = JsonOutputParser()
        chain = prompt | llm | parser
        result = await chain.ainvoke({"query": state["query"]})

        rewritten_query = result.get("rewritten_query")
        query_type = result.get("query_type")
        keywords = result.get("keywords")

        if not isinstance(rewritten_query, str) or not rewritten_query.strip():
            raise ValueError("Invalid rewritten_query")
        if query_type not in {"factual", "comparative", "exploratory"}:
            raise ValueError("Invalid query_type")
        if not isinstance(keywords, list):
            raise ValueError("Invalid keywords")

        state["rewritten_query"] = rewritten_query
        state["query_type"] = query_type
        state["keywords"] = [str(k) for k in keywords if str(k).strip()]

        trace(
            "Node1 intent -> done",
            query_type=state["query_type"],
            rewritten_query_sample=state["rewritten_query"][:80],
            keywords=state["keywords"],
        )
        return state
    except Exception:
        state["rewritten_query"] = state["query"]
        state["query_type"] = "factual"
        state["keywords"] = []

        trace(
            "Node1 intent -> fallback",
            query_type=state["query_type"],
            rewritten_query_sample=state["rewritten_query"][:80],
        )
        return state

