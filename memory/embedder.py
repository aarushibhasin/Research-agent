import os

from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None


def embed(texts: list[str]) -> list[list[float]]:
    _ensure_model_loaded()
    return _model.encode(texts, convert_to_numpy=True).tolist()


def embed_single(text: str) -> list[float]:
    _ensure_model_loaded()
    return _model.encode([text], convert_to_numpy=True)[0].tolist()


def _ensure_model_loaded() -> None:
    global _model
    if _model is None:
        _model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))

