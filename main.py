import argparse
import asyncio
import os
import random
import time
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

from benchmark.build_chroma import load_benchmark
from graph.pipeline import get_app
from memory import chroma_client
from metrics.logger import log_metrics


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _count_collection_docs(collection_name: str) -> int:
    col = chroma_client.get_collection(collection_name)
    try:
        return int(col.count())
    except Exception:
        return len((col.get() or {}).get("ids", []) or [])


def _print_state_metrics(*, state: dict[str, Any], t_e2e_ms: float, session_id: str) -> None:
    print("\n--- Timings ---")
    print(f"T_E2E: {t_e2e_ms:.1f}ms")
    print(f"T_time_to_first_token: {float(state.get('t_time_to_first_token_ms') or 0.0):.1f}ms")
    print(f"T_time_to_last_token: {float(state.get('t_time_to_last_token_ms') or 0.0):.1f}ms")
    print(f"T_chroma_retrieve: {float(state.get('t_chroma_retrieve_ms') or 0.0):.1f}ms")
    print(f"T_redis_store: {float(state.get('t_redis_store_ms') or 0.0):.1f}ms")
    print(f"T_chroma_retrieve_cumulative: {float(state.get('t_chroma_retrieve_total_ms') or 0.0):.1f}ms")
    print(f"T_redis_store_cumulative: {float(state.get('t_redis_store_total_ms') or 0.0):.1f}ms")
    print(f"T_Retrieval_node_total: {float(state.get('t_retrieval_ms') or 0.0):.1f}ms")
    print(f"T_Inference_node_total: {float(state.get('t_inference_ms') or 0.0):.1f}ms")
    print(f"T_Store_node_total: {float(state.get('t_store_ms') or 0.0):.1f}ms")
    print(f"thread_id: {session_id}")

    print("\n--- Latency Breakdown (node, database) ---")
    for entry in list(state.get("latency_breakdown") or []):
        print(
            f"({entry.get('node')}, {entry.get('database')}): "
            f"{float(entry.get('ms') or 0.0):.1f}ms [{entry.get('metric')}]"
        )


async def run_one(*, split: str, example_id: int, question_index: int) -> None:
    base_prefix = os.getenv("CHROMA_COLLECTION_PAPERS", "research_memory")
    chroma_collection_name = f"{base_prefix}__{split}__{example_id}"

    ds = load_benchmark(split)
    example = ds[example_id]
    questions = list(example.get("questions") or [])
    answers = list(example.get("answers") or [])
    if not questions:
        raise ValueError(f"Example {example_id} has no questions.")
    question = questions[question_index]
    ground_truth: Any = answers[question_index] if question_index < len(answers) else ""

    session_id = f"single_{example_id}_{question_index}"
    app = await get_app()
    t0 = time.perf_counter()
    state = await app.ainvoke(
        {"query": question, "session_id": session_id, "chroma_collection_name": chroma_collection_name},
        config={"configurable": {"thread_id": session_id}},
    )
    t_e2e_ms = (time.perf_counter() - t0) * 1000

    print("\n--- Query ---")
    print(question)
    print("\n--- Answer ---")
    print(state.get("answer") or "")
    print("\n--- Ground Truth ---")
    print(ground_truth)
    _print_state_metrics(state=state, t_e2e_ms=t_e2e_ms, session_id=session_id)

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

    predicted = (state.get("answer") or "").strip()
    log_metrics(
        {
            "timestamp": _now_utc_iso(),
            "split": split,
            "example_id": example_id,
            "question_index": question_index,
            "session_id": session_id,
            "query": question,
            "ground_truth": ground_truth,
            "predicted_answer": predicted,
            "query_type": state.get("query_type"),
            "t_e2e_ms": t_e2e_ms,
            "t_retrieval_ms": state.get("t_retrieval_ms"),
            "t_inference_ms": state.get("t_inference_ms"),
            "t_store_ms": state.get("t_store_ms"),
            "t_time_to_first_token_ms": state.get("t_time_to_first_token_ms"),
            "t_time_to_last_token_ms": state.get("t_time_to_last_token_ms"),
            "t_chroma_retrieve_ms": state.get("t_chroma_retrieve_ms"),
            "t_redis_store_ms": state.get("t_redis_store_ms"),
            "t_chroma_retrieve_total_ms": state.get("t_chroma_retrieve_total_ms"),
            "t_redis_store_total_ms": state.get("t_redis_store_total_ms"),
            "latency_breakdown": state.get("latency_breakdown"),
            "answer_length": len(predicted),
            "retrieved_chunks_count": len(state.get("retrieved_chunks") or []),
            "retrieval_error": state.get("retrieval_error") or "",
            "inference_error": state.get("inference_error") or "",
        }
    )


async def run_all_questions_for_example(*, split: str, example_id: int) -> None:
    ds = load_benchmark(split)
    example = ds[example_id]
    questions = list(example.get("questions") or [])
    if not questions:
        raise ValueError(f"Example {example_id} has no questions.")
    for question_index in range(len(questions)):
        print("\n" + "=" * 80)
        print(f"Example {example_id} | Question {question_index + 1}/{len(questions)}")
        print("=" * 80)
        await run_one(split=split, example_id=example_id, question_index=question_index)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run dataset-backed memory-agent queries using prebuilt Chroma.")
    p.add_argument("--split", default=os.getenv("BENCHMARK_SPLIT", "Accurate_Retrieval"))
    p.add_argument("--trace", action="store_true", help="Enable node-level console traces.")
    p.add_argument(
        "--example-id",
        type=int,
        default=None,
        help="Dataset example id. If omitted, choose a random non-empty prebuilt example.",
    )
    p.add_argument(
        "--question-index",
        type=int,
        default=None,
        help="Question index for single-question mode. If omitted, choose random question.",
    )
    p.add_argument(
        "--all-questions",
        action="store_true",
        help="Run all questions for the selected example (each with a new session/thread id).",
    )
    p.add_argument(
        "--max-example-scan",
        type=int,
        default=200,
        help="Max example ids to scan for non-empty prebuilt collections when --example-id is omitted.",
    )
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    if args.trace:
        os.environ["AGENT_CONSOLE_TRACE"] = "1"

    split = args.split
    base_prefix = os.getenv("CHROMA_COLLECTION_PAPERS", "research_memory")
    example_id = args.example_id
    if example_id is None:
        candidates: list[int] = []
        for ex_id in range(0, max(1, args.max_example_scan)):
            cname = f"{base_prefix}__{split}__{ex_id}"
            try:
                if _count_collection_docs(cname) > 0:
                    candidates.append(ex_id)
            except Exception:
                continue
        if not candidates:
            raise ValueError(
                "No non-empty prebuilt Chroma collections found. "
                "Run benchmark/build_chroma.py first."
            )
        example_id = random.choice(candidates)

    if args.all_questions:
        asyncio.run(run_all_questions_for_example(split=split, example_id=example_id))
        return

    ds = load_benchmark(split)
    example = ds[example_id]
    questions = list(example.get("questions") or [])
    if not questions:
        raise ValueError(f"Example {example_id} has no questions.")
    question_index = args.question_index
    if question_index is None:
        question_index = random.randrange(0, len(questions))
    asyncio.run(run_one(split=split, example_id=example_id, question_index=question_index))


if __name__ == "__main__":
    main()

