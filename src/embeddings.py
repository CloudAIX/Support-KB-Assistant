"""Embeddings — pluggable. Default OpenAI (matches the course demo); local fallback.

Lock your embedder version: query and chunk vectors MUST come from the same model,
or the vector space differs and results are garbage (Week 2, slide 12).
"""
import os
from dotenv import load_dotenv

load_dotenv()
PROVIDER = os.getenv("EMBED_PROVIDER", "openai")

if PROVIDER == "openai":
    from openai import OpenAI
    _client = OpenAI()
    _MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    def embed(texts: list[str]) -> list[list[float]]:
        resp = _client.embeddings.create(model=_MODEL, input=texts)
        return [d.embedding for d in resp.data]

elif PROVIDER == "local":
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    def embed(texts: list[str]) -> list[list[float]]:
        return _model.encode(texts, normalize_embeddings=True).tolist()

else:
    raise ValueError(f"Unknown EMBED_PROVIDER: {PROVIDER}")
