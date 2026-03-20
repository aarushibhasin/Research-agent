import os

from datasets import load_dataset


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
    """
    Robustly extract text fragments from the MemoryAgentBench haystack structure.

    Per dataset card, `metadata.haystack_sessions` can be deeply nested:
    list -> list -> list -> dict(content, has_answer, role).
    """
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
    """
    Convert an example's memory "haystack" into a list of text chunks
    suitable for embedding + upsert into Chroma.

    Returns a list[str]. If no haystack data exists, returns [].
    """
    # Preferred location (dataset card): metadata.haystack_sessions
    hs = None
    if isinstance(example, dict):
        md = example.get("metadata") or {}
        if isinstance(md, dict):
            hs = md.get("haystack_sessions")
        if hs is None:
            hs = example.get("haystack_sessions")

    if not isinstance(hs, list):
        return []

    # Flatten to message-level chunks to improve retrieval recall.
    # Each haystack session dict has `content` (and often `role`), and the
    # benchmark questions target specific statements inside those messages.
    chunks: list[str] = []
    for chunk_item in hs:
        texts = _extract_texts_from_item(chunk_item)
        for t in texts:
            t_str = str(t).strip()
            if t_str:
                chunks.append(t_str)
    return chunks

