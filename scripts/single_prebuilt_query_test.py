from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys

from dotenv import load_dotenv

load_dotenv()

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from benchmark.loader import load_benchmark  # noqa: E402
from graph.pipeline import get_app  # noqa: E402
from memory import chroma_client  # noqa: E402


def _count_collection_docs(collection_name: str) -> int:
    col = chroma_client.get_collection(collection_name)
    try:
        return int(col.count())
    except Exception:
        return len((col.get() or {}).get("ids", []) or [])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a single retrieval+inference query using prebuilt Chroma collections."
    )
    p.add_argument("--split", default=os.getenv("BENCHMARK_SPLIT", "Accurate_Retrieval"))
    p.add_argument(
        "--trace",
        action="store_true",
        help="Print node-by-node console traces (intent/retrieval/inference/redis store).",
    )
    p.add_argument(
        "--example-id",
        type=int,
        default=None,
        help="Which benchmark example to use. If omitted, we pick a random non-empty prebuilt example.",
    )
    p.add_argument(
        "--question-index",
        type=int,
        default=None,
        help="Which question index within the example. If omitted, we pick a random question.",
    )
    p.add_argument(
        "--max-example-scan",
        type=int,
        default=200,
        help="How many initial example ids to scan for non-empty prebuilt collections when --example-id is not set.",
    )
    return p.parse_args()


async def run_one(
    *,
    split: str,
    example_id: int,
    question_index: int,
) -> None:
    base_prefix = os.getenv("CHROMA_COLLECTION_PAPERS", "research_memory")
    chroma_collection_name = f"{base_prefix}__{split}__{example_id}"

    ds = load_benchmark(split)
    example = ds[example_id]
    questions = list(example.get("questions") or [])
    if not questions:
        raise ValueError(f"Example {example_id} has no questions.")
    question = questions[question_index]

    session_id = f"single_{example_id}_{question_index}"
    app = await get_app()

    state = await app.ainvoke(
        {
            "query": question,
            "session_id": session_id,
            "chroma_collection_name": chroma_collection_name,
        },
        config={"configurable": {"thread_id": session_id}},
    )

    print("\n--- Query ---")
    print(question)

    print("\n--- Answer ---")
    print(state.get("answer") or "")

    print("\n--- Timings ---")
    print(f"T_Retrieval: {float(state.get('t_retrieval_ms') or 0.0):.1f}ms")
    print(f"T_Inference: {float(state.get('t_inference_ms') or 0.0):.1f}ms")
    print(f"T_Store: {float(state.get('t_store_ms') or 0.0):.1f}ms")
    print(f"thread_id: {session_id}")

    print("\n--- Context Used ---")
    print(f"query_type: {state.get('query_type')}")
    print(f"retrieved_chunks_count: {len(state.get('retrieved_chunks') or [])}")
    ctx = state.get("retrieval_context") or ""
    print(f"retrieval_context_chars: {len(ctx)}")
    print(f"retrieval_context_sample: {ctx[:500]!r}")

    if state.get("retrieval_error"):
        print("\n--- Retrieval Error ---")
        print(state.get("retrieval_error"))
    if state.get("inference_error"):
        print("\n--- Inference Error ---")
        print(state.get("inference_error"))


def main() -> None:
    args = parse_args()
    split = args.split
    base_prefix = os.getenv("CHROMA_COLLECTION_PAPERS", "research_memory")

    # Enable lightweight node traces for this script.
    if args.trace:
        os.environ["AGENT_CONSOLE_TRACE"] = "1"

    example_id = args.example_id
    if example_id is None:
        candidates: list[int] = []
        for ex_id in range(0, max(1, args.max_example_scan)):
            cname = f"{base_prefix}__{split}__{ex_id}"
            try:
                if _count_collection_docs(cname) > 0:
                    candidates.append(ex_id)
            except Exception:
                # Collection may not exist yet.
                continue
        if not candidates:
            raise ValueError(
                "No non-empty prebuilt Chroma collections found. "
                "Run benchmark/build_chroma.py first."
            )
        example_id = random.choice(candidates)

    ds = load_benchmark(split)
    example = ds[example_id]
    questions = list(example.get("questions") or [])
    if not questions:
        raise ValueError(f"Example {example_id} has no questions.")

    question_index = args.question_index
    if question_index is None:
        question_index = random.randrange(0, len(questions))

    asyncio.run(
        run_one(
            split=split,
            example_id=example_id,
            question_index=question_index,
        )
    )


if __name__ == "__main__":
    main()

