from __future__ import annotations

import os
from typing import Any


def _enabled() -> bool:
    return os.getenv("AGENT_CONSOLE_TRACE", "0").strip().lower() in {"1", "true", "yes", "on"}


def trace(msg: str, **kwargs: Any) -> None:
    """
    Lightweight console tracing for node-by-node debugging.

    Controlled via env var `AGENT_CONSOLE_TRACE`.
    """
    if not _enabled():
        return

    if kwargs:
        details = " ".join([f"{k}={v!r}" for k, v in kwargs.items()])
        print(f"[AgentTrace] {msg} {details}")
    else:
        print(f"[AgentTrace] {msg}")

