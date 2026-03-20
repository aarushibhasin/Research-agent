import os

from langgraph.checkpoint.redis import AsyncRedisSaver
from langgraph.graph import END, StateGraph

from agents.inference_agent import inference_agent
from agents.intent_agent import intent_agent
from agents.memory_store_agent import memory_store_agent
from agents.retrieval_agent import retrieval_agent
from agents.state import AgentState


_APP = None


def _build_graph(checkpointer):
    graph = StateGraph(AgentState)
    graph.add_node("intent", intent_agent)
    graph.add_node("retrieval", retrieval_agent)
    graph.add_node("inference", inference_agent)
    graph.add_node("memory_store", memory_store_agent)
    graph.set_entry_point("intent")
    graph.add_edge("intent", "retrieval")
    graph.add_edge("retrieval", "inference")
    graph.add_edge("inference", "memory_store")
    graph.add_edge("memory_store", END)
    return graph.compile(checkpointer=checkpointer)


async def get_app():
    """
    Lazily build the compiled graph so AsyncRedisSaver can be instantiated
    inside an active event loop.
    """
    global _APP
    if _APP is not None:
        return _APP
    checkpointer = AsyncRedisSaver(redis_url=os.getenv("REDIS_URL"))
    # Ensure Redis Search indexes exist before compiling/using the saver.
    # Otherwise `checkpoint_write` index may be missing on first run.
    await checkpointer.setup()
    _APP = _build_graph(checkpointer)
    return _APP

