import os
from typing import Any

import chromadb


def _client() -> chromadb.HttpClient:
    host = os.getenv("CHROMA_HOST", "localhost")
    port = int(os.getenv("CHROMA_PORT", "8001"))
    return chromadb.HttpClient(host=host, port=port)


def get_collection(name: str):
    client = _client()
    return client.get_or_create_collection(name=name)


def clear_collection(name: str) -> None:
    client = _client()
    try:
        client.delete_collection(name=name)
    except Exception:
        pass
    client.get_or_create_collection(name=name)


def upsert_chunks(
    collection_name: str,
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict[str, Any]] | None = None,
) -> None:
    collection = get_collection(collection_name)
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


def query_collection(
    collection_name: str,
    query_embedding: list[float],
    n_results: int,
) -> dict:
    collection = get_collection(collection_name)
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        # Chroma include options vary by version; `ids` is NOT a valid include item.
        include=["documents", "distances", "metadatas"],
    )

