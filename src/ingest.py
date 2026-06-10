"""Ingest (build-time, run once / on updates).

Pipeline: load -> chunk (recursive char split, with overlap) -> embed -> store in
Chroma, carrying metadata through so we can filter on it at query time.

Course parallel: Arvind's notebook 1 (basic) + notebook 2 (metadata). Pinecone in the
course; local Chroma here so it runs with no account. Swapping to Pinecone is the
brief / June-9 lecture step — the moving parts are the same.

TODO (vibe session): tune chunk_size/overlap; add structural chunking by markdown
heading (slide 22); consider parent-doc "retrieve small, return big".
"""
import json
import pathlib
import chromadb
from embeddings import embed

CORPUS = pathlib.Path(__file__).parent.parent / "corpus"
CHUNK_SIZE = 600        # characters; ~400-800 tokens range from the deck
CHUNK_OVERLAP = 100     # ~10-20% overlap, the boundary safety net


def _split(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    # Simple recursive-ish splitter. TODO: replace with a heading-aware splitter.
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


def load_docs():
    docs = []  # each: {id, text, metadata}
    meta_map = json.loads((CORPUS / "metadata.json").read_text())

    # KB articles
    for rel, md in meta_map.items():
        text = (CORPUS / rel).read_text()
        for i, chunk in enumerate(_split(text)):
            docs.append({
                "id": f"{rel}::{i}",
                "text": chunk,
                "metadata": {**md, "source": rel},
            })

    # Past tickets (warm-start corpus) — one ticket = one chunk
    for line in (CORPUS / "tickets" / "tickets.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        t = json.loads(line)
        docs.append({
            "id": t["ticket_id"],
            "text": t["content"],
            "metadata": {k: v for k, v in t.items() if k != "content"} | {"source": "tickets.jsonl"},
        })
    return docs


def main():
    docs = load_docs()
    client = chromadb.PersistentClient(path=str(CORPUS.parent / "chroma_db"))
    col = client.get_or_create_collection("support_kb")
    col.add(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        embeddings=embed([d["text"] for d in docs]),
        metadatas=[d["metadata"] for d in docs],
    )
    print(f"Ingested {len(docs)} chunks into Chroma collection 'support_kb'.")


if __name__ == "__main__":
    main()
