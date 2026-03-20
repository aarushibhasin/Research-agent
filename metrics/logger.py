import json
import os
from typing import Any


def log_metrics(entry: dict[str, Any]) -> None:
    path = os.getenv("METRICS_LOG_PATH", "metrics_log.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

