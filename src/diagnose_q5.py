"""Step-by-step retrieval trace for a single question.
Run from the repo root:
    python src/diagnose_q5.py "Which items are not refundable?"
"""
import sys
import pathlib
import chromadb
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from embeddings import embed
from rerank import rerank
from rag import _rrf, answer

load_dotenv()

CHROMA = pathlib.Path(__file__).parent.parent / "chroma_db"
TARGET = "billing-and-refunds"


def diagnose(query: str):
    client = chromadb.PersistentClient(path=str(CHROMA))
    col = client.get_collection("support_kb")

    all_docs = col.get()
    total = len(all_docs["ids"])
    n = min(20, total)

    print(f"\n{'='*60}")
    print(f"QUERY: {query!r}")
    print(f"KB size: {total} chunks total  (n={n} = {'whole KB' if n == total else 'top-20'})")
    print(f"{'='*60}\n")

    # Dense
    dense = col.query(query_embeddings=embed([query]), n_results=n)
    dense_ids = dense["ids"][0]
    print(f"DENSE top-{n}:")
    for i, cid in enumerate(dense_ids[:10]):
        mark = f"  <-- {TARGET}" if TARGET in cid.lower() else ""
        print(f"  {i+1:2}. {cid}{mark}")

    # Sparse (BM25)
    corpus_ids = all_docs["ids"]
    tokenised = [d.split() for d in all_docs["documents"]]
    bm25 = BM25Okapi(tokenised)
    scored = sorted(zip(corpus_ids, bm25.get_scores(query.split())), key=lambda x: -x[1])
    sparse_ids = [cid for cid, _ in scored[:n]]
    print(f"\nSPARSE (BM25) top-{n}:")
    for i, (cid, sc) in enumerate(scored[:10]):
        mark = f"  <-- {TARGET}" if TARGET in cid.lower() else ""
        print(f"  {i+1:2}. {cid}  score={sc:.3f}{mark}")

    # RRF fusion
    fused = _rrf([dense_ids, sparse_ids])
    fused_ranked = sorted(fused, key=lambda x: -fused[x])
    print(f"\nFUSED (RRF) — all {len(fused_ranked)} candidates:")
    for i, cid in enumerate(fused_ranked):
        mark = f"  <-- {TARGET}" if TARGET in cid.lower() else ""
        print(f"  {i+1:2}. {cid}  rrf={fused[cid]:.5f}{mark}")

    # Reranker
    wide_ids = fused_ranked[:n]
    got = col.get(ids=wide_ids)
    by_id = dict(zip(got["ids"], got["documents"]))
    candidates = [(cid, by_id[cid]) for cid in wide_ids if cid in by_id]
    top5 = rerank(query, candidates, top_k=5)
    top5_ids = [cid for cid, _ in top5]
    print(f"\nAFTER RERANK top-5:")
    for i, (cid, text) in enumerate(top5):
        mark = f"  <-- {TARGET}" if TARGET in cid.lower() else ""
        print(f"  {i+1}. {cid}{mark}")
        print(f"     {text[:120]}...")

    # Answer
    ans = answer(query)
    print(f"\nANSWER:\n  {ans[:400]}")

    # Decision
    in_fused = any(TARGET in cid.lower() for cid in fused_ranked)
    in_top5 = any(TARGET in cid.lower() for cid in top5_ids)
    refused = any(s in ans.lower() for s in (
        "i don't know", "i don't have", "could not find",
        "not in the", "not mentioned",
        "i don't know based on the provided sources",
    ))

    print(f"\n{'='*60}")
    print("DIAGNOSIS")
    print(f"  {TARGET} in fused recall? {'YES' if in_fused else 'NO'}")
    print(f"  {TARGET} in top-5?        {'YES' if in_top5 else 'NO'}")
    print(f"  answer refused?           {'YES' if refused else 'NO'}")

    if not in_fused:
        verdict = "CHUNKING / RETRIEVAL problem — right chunk never reached the fused set."
        action  = "Re-ingest with smaller chunks or check how chunk IDs are named."
    elif not in_top5:
        verdict = "ORDERING problem — chunk in fused recall but reranker didn't surface it."
        action  = "Try a stronger reranker model or widen the recall set past 20."
    elif refused:
        verdict = "GENERATION problem — right chunk is in context but model over-refuses."
        action  = "Soften the refusal instruction or check the chunk's text quality."
    else:
        verdict = "PASS — chunk retrieved and answer grounded. Q5 fixed."
        action  = "Re-run eval.py to confirm the score improved."

    print(f"\n  VERDICT: {verdict}")
    print(f"  ACTION:  {action}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Which items are not refundable?"
    diagnose(q)
