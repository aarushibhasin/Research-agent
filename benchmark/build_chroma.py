from __future__ import annotations

import argparse
import os
import sys
import time

from datasets import load_dataset

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from memory import chroma_client
from memory.embedder import embed


def load_benchmark(split: str):
    dataset_name = os.getenv("BENCHMARK_DATASET", "ai-hyz/MemoryAgentBench")
    cache_dir = os.getenv(
        "BENCHMARK_CACHE_DIR",
        os.path.join(os.path.dirname(__file__), "..", ".cache", "hf_datasets"),
    )
    os.makedirs(cache_dir, exist_ok=True)
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    return dataset[split]


def _extract_texts_from_item(item) -> list[str]:
    if item is None:
        return []
    if isinstance(item, str):
        return [item]
    if isinstance(item, dict):
        content = item.get("content")
        if content is None:
            return []
        role = item.get("role")
        if role:
            return [f"{role}: {content}"]
        return [str(content)]
    if isinstance(item, list):
        out: list[str] = []
        for sub in item:
            out.extend(_extract_texts_from_item(sub))
        return out
    return []


def extract_haystack_chunks(example) -> list[str]:
    hs = None
    if isinstance(example, dict):
        md = example.get("metadata") or {}
        if isinstance(md, dict):
            hs = md.get("haystack_sessions")
        if hs is None:
            hs = example.get("haystack_sessions")
    if not isinstance(hs, list):
        return []

    chunks: list[str] = []
    for chunk_item in hs:
        texts = _extract_texts_from_item(chunk_item)
        for t in texts:
            t_str = str(t).strip()
            if t_str:
                chunks.append(t_str)
    return chunks


def _collection_name_for_example(
    base_prefix: str, split: str, example_id: str
) -> str:
    # Keep collection naming deterministic so the benchmark can reuse it.
    # Using `__` avoids accidental collisions with underscores in IDs.
    return f"{base_prefix}__{split}__{example_id}"

def build_chroma_for_split(
    split: str,
    max_examples: int,
    embed_batch_size: int,
) -> None:
    base_prefix = os.getenv("CHROMA_COLLECTION_PAPERS", "research_memory")
    ds = load_benchmark(split)

    # Ensure model is loaded before we start the timer-heavy loop.
    _ = embed(["warmup"])

    built_non_empty = 0
    for example_index, example in enumerate(ds):
        if max_examples and built_non_empty >= max_examples:
            break

        example_id = str(example_index)
        collection_name = _collection_name_for_example(base_prefix, split, example_id)

        chunks = extract_haystack_chunks(example)
        if not chunks:
            print(f"[Build] example_id={example_id}: no chunks, skipping")
            continue

        # Only count examples that actually produce embeddings.
        built_non_empty += 1

        chroma_client.clear_collection(collection_name)

        ids = [f"{example_id}_{i}" for i in range(len(chunks))]
        metas = [{"chunk_index": i, "example_id": example_id} for i in range(len(chunks))]

        t0 = time.perf_counter()
        # Embed in batches to control memory usage.
        for start in range(0, len(chunks), embed_batch_size):
            end = min(len(chunks), start + embed_batch_size)
            docs_batch = chunks[start:end]
            ids_batch = ids[start:end]
            metas_batch = metas[start:end]

            embs_batch = embed(docs_batch)
            chroma_client.upsert_chunks(
                collection_name=collection_name,
                ids=ids_batch,
                embeddings=embs_batch,
                documents=docs_batch,
                metadatas=metas_batch,
            )

        t_ms = (time.perf_counter() - t0) * 1000
        print(
            f"[Build] split={split} example_id={example_id} "
            f"chunks={len(chunks)} time_ms={t_ms:.1f} collection='{collection_name}'"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-time build of Chroma embeddings for benchmark examples.")
    p.add_argument("--split", default=os.getenv("BENCHMARK_SPLIT", "Accurate_Retrieval"))
    p.add_argument(
        "--max-examples",
        type=int,
        default=int(os.getenv("MAX_EXAMPLES", "0")) or 0,
        help="0 means all examples in the split.",
    )
    p.add_argument(
        "--embed-batch-size",
        type=int,
        default=int(os.getenv("EMBED_BATCH_SIZE", "64")),
        help="Batch size for embedding generation.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_chroma_for_split(
        split=args.split,
        max_examples=args.max_examples,
        embed_batch_size=args.embed_batch_size,
    )

