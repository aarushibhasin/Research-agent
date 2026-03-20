from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
from typing import Any

from benchmark.loader import extract_haystack_chunks, load_benchmark
from graph.pipeline import get_app
from memory import chroma_client
from memory.embedder import embed
from metrics.logger import log_metrics


_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalise(s: Any) -> str:
    # HF datasets sometimes store multiple acceptable answers as a list.
    if s is None:
        return ""
    if isinstance(s, list):
        # Normalize each element separately to avoid ending up with "['...']".
        # Join with spaces so substring scoring still works.
        parts = [str(x) for x in s if x is not None and str(x).strip()]
        s = " ".join(parts)
    else:
        s = str(s)

    s = (s or "").lower()
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def score_answer(predicted: str, ground_truth: Any) -> bool:
    p = _normalise(predicted)
    if not p:
        return False

    # If multiple acceptable ground-truth answers exist, accept any of them.
    if isinstance(ground_truth, list):
        for gt in ground_truth:
            g = _normalise(gt)
            if g and g in p:
                return True
        return False

    g = _normalise(ground_truth)
    if not g:
        return False
    return g in p


async def run_benchmark(split: str) -> None:
    app = await get_app()
    benchmark_split = split
    base_collection_prefix = os.getenv("CHROMA_COLLECTION_PAPERS", "research_memory")
    use_prebuilt = os.getenv("BENCHMARK_USE_PREBUILT_CHROMA", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    log_every = int(os.getenv("BENCHMARK_LOG_EVERY", "0") or "0")  # 0 => default (no extra spam)
    ds = load_benchmark(split)

    # Optional fast mode for development/CI.
    # Example: MAX_EXAMPLES=5 python main.py --mode benchmark --split Accurate_Retrieval
    max_examples_raw = os.getenv("MAX_EXAMPLES", "").strip()
    max_examples = int(max_examples_raw) if max_examples_raw.isdigit() else 0

    totals: dict[str, Any] = {
        "n_questions": 0,
        "n_correct": 0,
        "t_e2e_ms": [],
        "t_retrieval_ms": [],
        "t_inference_ms": [],
        "t_store_ms": [],
    }

    built_non_empty_examples = 0
    for example_index, example in enumerate(ds):
        # If the dataset item has no haystack sessions, there's nothing for retrieval to use.
        example_chunks = extract_haystack_chunks(example)
        if not example_chunks:
            continue

        if max_examples and built_non_empty_examples >= max_examples:
            break

        built_non_empty_examples += 1
        example_id = str(example_index)

        # Evaluation mode:
        # - default: embed + ingest per example (slower, but self-contained)
        # - prebuilt: reuse collections created by `benchmark/build_chroma.py` (faster)
        chroma_collection_name = ""
        t_ingest_total_ms = 0.0
        if use_prebuilt:
            chroma_collection_name = f"{base_collection_prefix}__{benchmark_split}__{example_id}"
        else:
            chroma_collection_name = base_collection_prefix
            chroma_client.clear_collection(base_collection_prefix)

            # Ingest haystack sessions for this example only.
            ingest_start = time.perf_counter()
            embs = embed(example_chunks)
            ids = [f"{example_id}_{i}" for i in range(len(example_chunks))]
            metas = [{"chunk_index": i, "example_id": example_id} for i in range(len(example_chunks))]
            chroma_client.upsert_chunks(
                collection_name=base_collection_prefix,
                ids=ids,
                embeddings=embs,
                documents=example_chunks,
                metadatas=metas,
            )
            t_ingest_total_ms = (time.perf_counter() - ingest_start) * 1000

            log_metrics(
                {
                    "timestamp": _now_utc_iso(),
                    "benchmark_split": benchmark_split,
                    "example_id": example_id,
                    "t_ingest_total_ms": t_ingest_total_ms,
                }
            )

        questions = list(example.get("questions") or [])
        answers = list(example.get("answers") or [])

        for qi, question in enumerate(questions):
            ground_truth = answers[qi] if qi < len(answers) else ""
            session_id = f"{example_id}_{qi}"

            t_e2e_start = time.perf_counter()
            state = await app.ainvoke(
                {
                    "query": question,
                    "session_id": session_id,
                    "chroma_collection_name": chroma_collection_name,
                },
                config={"configurable": {"thread_id": session_id}},
            )
            t_e2e_ms = (time.perf_counter() - t_e2e_start) * 1000

            predicted = (state.get("answer") or "").strip()
            is_correct = score_answer(predicted, ground_truth)

            if log_every and (qi % log_every == 0):
                print(
                    f"[Benchmark] example={example_index+1} q={qi+1}/{len(questions)} "
                    f"thread_id={session_id} correct={is_correct}"
                )

            totals["n_questions"] += 1
            totals["n_correct"] += int(is_correct)
            totals["t_e2e_ms"].append(t_e2e_ms)
            totals["t_retrieval_ms"].append(float(state.get("t_retrieval_ms") or 0.0))
            totals["t_inference_ms"].append(float(state.get("t_inference_ms") or 0.0))
            totals["t_store_ms"].append(float(state.get("t_store_ms") or 0.0))

            log_metrics(
                {
                    "timestamp": _now_utc_iso(),
                    "session_id": session_id,
                    "query": question,
                    "query_type": state.get("query_type"),
                    "t_e2e_ms": t_e2e_ms,
                    "t_retrieval_ms": state.get("t_retrieval_ms"),
                    "t_inference_ms": state.get("t_inference_ms"),
                    "t_store_ms": state.get("t_store_ms"),
                    "answer_length": len(predicted),
                    "chunks_retrieved": len(state.get("retrieved_chunks") or []),
                    "benchmark_split": benchmark_split,
                    "ground_truth": ground_truth,
                    "is_correct": is_correct,
                }
            )

        print(f"Example {example_index+1}/{len(ds)} done. Ingest {t_ingest_total_ms:.1f}ms")

    n = totals["n_questions"]
    acc = (totals["n_correct"] / n * 100.0) if n else 0.0

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    print("\n=== Benchmark Summary ===")
    print(f"Total questions: {n}")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Mean T_E2E: {mean(totals['t_e2e_ms']):.1f}ms")
    print(f"Mean T_Retrieval: {mean(totals['t_retrieval_ms']):.1f}ms")
    print(f"Mean T_Inference: {mean(totals['t_inference_ms']):.1f}ms")
    print(f"Mean T_Store: {mean(totals['t_store_ms']):.1f}ms")

