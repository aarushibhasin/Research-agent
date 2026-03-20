import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv

# Legacy smoke test.
# Prefer `benchmark/build_chroma.py` + `scripts/single_prebuilt_query_test.py`
# for faster iteration using prebuilt Chroma collections.
load_dotenv()

# Ensure project root is on PYTHONPATH *before* importing local packages.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from benchmark.loader import extract_haystack_chunks, load_benchmark
from graph.pipeline import get_app
from memory import chroma_client
from memory.embedder import embed, embed_single


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick RAG smoke test using MemoryAgentBench benchmark data.")
    p.add_argument("--split", default=os.getenv("BENCHMARK_SPLIT", "Accurate_Retrieval"))
    p.add_argument("--example-index", type=int, default=0)
    p.add_argument("--question-index", type=int, default=0)
    p.add_argument("--max-scan", type=int, default=10, help="Scan forward to find first example with non-empty haystack_sessions.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ds = load_benchmark(args.split)
    selected_example_index = None
    selected_example = None

    start = max(0, args.example_index)
    end = min(len(ds), start + max(1, int(args.max_scan)))
    for i in range(start, end):
        ex = ds[i]
        chunks = extract_haystack_chunks(ex)
        if len(chunks) > 0:
            selected_example_index = i
            selected_example = ex
            break

    if selected_example is None:
        selected_example_index = start
        selected_example = ds[start]

    research_collection = os.getenv("CHROMA_COLLECTION_PAPERS", "research_memory")
    chroma_client.clear_collection(research_collection)

    chunks = extract_haystack_chunks(selected_example)
    print(
        f"[Ingest] example_index={selected_example_index} "
        f"haystack_sessions_chunks={len(chunks)} (scan_start={args.example_index}, max_scan={args.max_scan})"
    )
    if chunks:
        embs = embed(chunks)
        ids = [f"debug_{selected_example_index}_{i}" for i in range(len(chunks))]
        metas = [
            {"chunk_index": i, "example_id": str(selected_example_index)}
            for i in range(len(chunks))
        ]
        chroma_client.upsert_chunks(
            collection_name=research_collection,
            ids=ids,
            embeddings=embs,
            documents=chunks,
            metadatas=metas,
        )
    else:
        print(
            "[Ingest] WARNING: selected benchmark example has 0 haystack_sessions chunks. "
            "Ingestion into Chroma is skipped, so retrieval will return no context."
        )

    # Chroma sanity checks: ensure ingestion actually populated the collection,
    # and that raw retrieval returns documents (before running the LLM).
    try:
        col = chroma_client.get_collection(research_collection)
        # `count()` is the fastest if supported by this Chroma version.
        try:
            num_docs = col.count()
        except Exception:
            # fallback to `get()`
            num_docs = len((col.get() or {}).get("ids", []) or [])
        print(f"\n[Chroma sanity] collection '{research_collection}' doc_count: {num_docs}")
    except Exception as e:
        print(f"\n[Chroma sanity] collection count check failed: {e!r}")

    questions = list(selected_example.get("questions") or [])
    if not questions:
        raise ValueError(f"Selected benchmark example {selected_example_index} has no questions.")
    q_idx = min(max(0, args.question_index), len(questions) - 1)
    question = questions[q_idx]

    session_id = f"debug_{selected_example_index}_{q_idx}"

    # Raw Chroma query using the *original* question embedding (no intent rewrite).
    try:
        q_emb = embed_single(question)
        top_k = int(os.getenv("TOP_K_CHUNKS", "5"))
        raw = chroma_client.query_collection(
            collection_name=research_collection,
            query_embedding=q_emb,
            n_results=top_k,
        )
        docs = (raw.get("documents") or [[]])[0]
        ids = (raw.get("ids") or [[]])[0]
        print(f"[Chroma sanity] raw query returned docs_len={len(docs)} ids_len={len(ids)}")
        if docs:
            print(f"[Chroma sanity] first_doc_sample: {docs[0][:200]!r}")
    except Exception as e:
        print(f"[Chroma sanity] raw query failed: {e!r}")

    async def _run() -> None:
        app = await get_app()
        return await app.ainvoke(
            {"query": question, "session_id": session_id},
            config={"configurable": {"thread_id": session_id}},
        )

    state = asyncio.run(_run())

    print("\n--- Question ---")
    print(question)
    print("\n--- Answer ---")
    print(state.get("answer") or "")
    print("\n--- Timings ---")
    print(f"T_Retrieval: {float(state.get('t_retrieval_ms') or 0.0):.1f}ms")
    print(f"T_Inference: {float(state.get('t_inference_ms') or 0.0):.1f}ms")
    print(f"T_Store: {float(state.get('t_store_ms') or 0.0):.1f}ms")

    # Always show what query/context were actually used for inference.
    print("\n--- Context Used ---")
    print(f"query_type: {state.get('query_type')}")
    print(f"rewritten_query_sample: {(state.get('rewritten_query') or '')[:200]!r}")
    print(f"retrieved_chunks_count: {len(state.get('retrieved_chunks') or [])}")
    ctx = state.get("retrieval_context") or ""
    print(f"retrieval_context_chars: {len(ctx)}")
    print(f"retrieval_context_sample: {ctx[:500]!r}")
    ctx_lower = ctx.lower()
    for kw in ["peak", "campaign", "hours", "week"]:
        print(f"ctx_has_{kw}: {kw in ctx_lower}")

    if not (state.get("answer") or "").strip():
        print("\n--- Debug ---")
        print(f"query_type: {state.get('query_type')}")
        print(f"keywords: {state.get('keywords')}")
        print(f"retrieved_chunks_count: {len(state.get('retrieved_chunks') or [])}")
        print(f"retrieved_chunks_sample: {(state.get('retrieved_chunks') or [])[:1]}")
        ctx = state.get("retrieval_context") or ""
        print(f"retrieval_context_chars: {len(ctx)}")
        print(f"retrieval_context_sample: {ctx[:300]!r}")
        print(f"retrieval_error: {state.get('retrieval_error')}")
        print(f"inference_error: {state.get('inference_error')}")
        print(f"inference_msg_type: {state.get('inference_msg_type')}")
        raw = state.get("inference_msg_repr")
        if raw:
            print(f"inference_msg_repr: {raw!r}")


if __name__ == "__main__":
    main()

