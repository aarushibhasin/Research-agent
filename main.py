import argparse
import asyncio
import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

from benchmark.runner import run_benchmark
from graph.pipeline import get_app
from metrics.logger import log_metrics


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


async def repl_mode() -> None:
    session_id = "repl_session"
    app = await get_app()
    while True:
        q = input("\nQuery: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            return

        t0 = time.perf_counter()
        # Partial input: conversation_history / turn_index come from Redis checkpoint merge.
        state = await app.ainvoke(
            {"query": q, "session_id": session_id},
            config={"configurable": {"thread_id": session_id}},
        )
        t_e2e_ms = (time.perf_counter() - t0) * 1000

        answer = state.get("answer") or ""
        print("\n--- Answer ---\n")
        print(answer)
        print("\n--- Timings ---")
        print(f"T_Retrieval: {float(state.get('t_retrieval_ms') or 0.0):.1f}ms")
        print(f"T_Inference: {float(state.get('t_inference_ms') or 0.0):.1f}ms")
        print(f"T_Store: {float(state.get('t_store_ms') or 0.0):.1f}ms")

        log_metrics(
            {
                "timestamp": _now_utc_iso(),
                "session_id": session_id,
                "query": q,
                "query_type": state.get("query_type"),
                "t_e2e_ms": t_e2e_ms,
                "t_retrieval_ms": state.get("t_retrieval_ms"),
                "t_inference_ms": state.get("t_inference_ms"),
                "t_store_ms": state.get("t_store_ms"),
                "answer_length": len(answer),
                "chunks_retrieved": len(state.get("retrieved_chunks") or []),
                "benchmark_split": None,
                "ground_truth": None,
                "is_correct": None,
            }
        )


async def benchmark_mode(split: str) -> None:
    await run_benchmark(split)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["benchmark", "repl"], required=True)
    p.add_argument("--split", default=os.getenv("BENCHMARK_SPLIT", "Accurate_Retrieval"))
    p.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Benchmark only first N examples (for quick iteration).",
    )
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    if args.max_examples is not None:
        os.environ["MAX_EXAMPLES"] = str(args.max_examples)
    if args.mode == "repl":
        asyncio.run(repl_mode())
    else:
        asyncio.run(benchmark_mode(args.split))


if __name__ == "__main__":
    main()

