"""Cross-encoder reranker — precision pass after RRF recall.
Model: ms-marco-MiniLM-L-6-v2 (~80 MB, downloaded once then cached by HF).
Lazy-loaded so import cost is zero until the first rerank() call.
"""
from sentence_transformers import CrossEncoder

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_encoder: CrossEncoder | None = None


def _get_encoder() -> CrossEncoder:
    global _encoder
    if _encoder is None:
        _encoder = CrossEncoder(_MODEL_NAME)
    return _encoder


def rerank(query: str, candidates: list[tuple[str, str]], top_k: int = 5) -> list[tuple[str, str]]:
    """Score (chunk_id, text) pairs against query; return top_k by relevance."""
    if not candidates:
        return candidates
    encoder = _get_encoder()
    scores = encoder.predict([(query, text) for _, text in candidates])
    ranked = sorted(zip(scores, candidates), key=lambda x: -x[0])
    return [cand for _, cand in ranked[:top_k]]
