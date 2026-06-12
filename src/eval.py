"""
eval.py — 20-question evaluation for the Support KB Assistant (Week 2).

Wired to rag.py:
  retrieve(query, top_k=5, where=None) -> list[(cid, text)]  (ranked, best first)
  answer(query, where=None) -> str

What it scores
--------------
  Precision@k  (PRIMARY for customer support): of the top-k sources retrieved,
               how many were relevant?  (Arjun/Pinecone: query-first, small set.)
  Recall@k     (guardrail): of the sources that should appear, how many did?
  Refusal      (guardrail): on unanswerable questions, did it say "I don't know"?

Run from the src/ folder, with the venv active and ANTHROPIC_API_KEY + the
embeddings key set in the same terminal:

    python3 eval.py

If the matching looks wrong (precision/recall all zero), run the diagnostic:

    python3 eval.py --show-ids

That prints the real chunk IDs each question retrieves, so you can fix the
`expected` strings below to match how your chunk IDs are actually named.
"""

import sys
from dataclasses import dataclass, field

from rag import retrieve, answer


def run_pipeline(question):
    hits = retrieve(question, top_k=5)        # [(cid, text), ...] ranked best-first
    source_ids = [cid for cid, _text in hits]
    ans = answer(question)
    return ans, source_ids


@dataclass
class TestCase:
    id: int
    category: str
    question: str
    expected: list = field(default_factory=list)
    should_refuse: bool = False


TESTS = [
    TestCase(1,  "happy", "How do I reset my password?", ["account-and-login"]),
    TestCase(2,  "happy", "How long does a refund take to process?", ["billing-and-refunds"]),
    TestCase(3,  "happy", "What does error E-402 mean?", ["troubleshooting"]),
    TestCase(4,  "happy", "My account is locked. How long until it unlocks?", ["account-and-login"]),
    TestCase(5,  "happy", "Which items are not refundable?", ["billing-and-refunds"]),
    TestCase(6,  "happy", "The app will not load. What should I try first?", ["troubleshooting"]),
    TestCase(7,  "happy", "How is my personal data handled and stored?", ["data-and-privacy"]),
    TestCase(8,  "happy", "How do I change or upgrade my plan?", ["plans-and-upgrades"]),

    TestCase(9,  "multi_doc", "I can't get into my account anymore",
             ["account-and-login", "troubleshooting"]),
    TestCase(10, "multi_doc", "I was charged but can't log in to use what I paid for",
             ["billing-and-refunds", "account-and-login"]),
    TestCase(11, "multi_doc", "If I upgrade my plan will I be refunded the difference?",
             ["plans-and-upgrades", "billing-and-refunds"]),
    TestCase(12, "multi_doc", "My order shipped but I think my data was wrong on the account",
             ["shipping-and-delivery", "data-and-privacy"]),

    TestCase(13, "ambiguous", "It's not working", []),
    TestCase(14, "ambiguous", "I have a billing problem", ["billing-and-refunds", "troubleshooting"]),
    TestCase(15, "ambiguous", "Something's wrong with my account", ["account-and-login"]),
    TestCase(16, "ambiguous", "When will it arrive?", ["shipping-and-delivery"]),

    TestCase(17, "unanswerable", "What's the refund policy for orders placed in New Zealand?",
             [], should_refuse=True),
    TestCase(18, "unanswerable", "Do you offer a student discount?", [], should_refuse=True),
    TestCase(19, "unanswerable", "What are your support hours on public holidays?",
             [], should_refuse=True),
    TestCase(20, "unanswerable", "Can I pay with cryptocurrency?", [], should_refuse=True),
]


REFUSAL_MARKERS = [
    "i don't know", "i do not know", "don't have", "do not have",
    "not in the", "could not find", "couldn't find", "no information",
    "not mentioned", "don't mention", "do not mention",
    "i don't know based on the provided sources",
]


def looks_like_refusal(ans):
    a = ans.lower()
    return any(m in a for m in REFUSAL_MARKERS)


def id_matches(expected, source_ids):
    joined = " ".join(source_ids).lower()
    return [e for e in expected if e.lower() in joined]


def show_ids():
    print("Diagnostic: the chunk ids each question retrieves.\n")
    for tc in TESTS[:8]:
        _, ids = run_pipeline(tc.question)
        print(f"Q{tc.id}: {tc.question}")
        print(f"   expected substring: {tc.expected}")
        print(f"   retrieved ids: {ids}\n")
    print("If 'expected' substrings don't appear in 'retrieved ids', edit the "
          "expected lists in eval.py to match your real id format.")


def main():
    K = 3
    rows, passes = [], 0
    refusal_total = refusal_correct = 0
    precision_scores, recall_scores = [], []

    for tc in TESTS:
        ans, sources = run_pipeline(tc.question)
        refused = looks_like_refusal(ans)

        if tc.expected:
            top = sources[:K]
            matched_top = id_matches(tc.expected, top)
            precision = len(matched_top) / max(1, len(top))
            recall = len(set(matched_top)) / len(tc.expected)
            precision_scores.append(precision)
            recall_scores.append(recall)

        hit = bool(id_matches(tc.expected, sources)) if tc.expected else True

        if tc.should_refuse:
            refusal_total += 1
            passed = refused
            if passed:
                refusal_correct += 1
            note = "refused correctly" if passed else "FAILED TO REFUSE"
        elif tc.category == "ambiguous":
            passed = hit or refused
            note = "handled" if passed else "answered with no supporting source"
        elif tc.category == "multi_doc":
            passed = hit
            both = len(id_matches(tc.expected, sources)) >= 2
            note = "both sources" if both else ("one source" if hit else "wrong sources")
        else:
            passed = hit and not refused
            note = "ok" if passed else ("refused answerable Q" if refused else "wrong/missing source")

        if passed:
            passes += 1
        rows.append((tc.id, tc.category, passed, note, sources[:K]))

    print("\n" + "=" * 72)
    print("  SUPPORT KB ASSISTANT — 20-QUESTION EVALUATION")
    print("=" * 72)
    print(f"{'ID':<4}{'CATEGORY':<14}{'RESULT':<8}{'NOTE'}")
    print("-" * 72)
    for rid, cat, passed, note, _ in rows:
        print(f"{rid:<4}{cat:<14}{'PASS' if passed else 'FAIL':<8}{note}")
    print("-" * 72)

    avg_p = sum(precision_scores) / len(precision_scores) if precision_scores else 0
    avg_r = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    print("PRIMARY METRIC (customer support)")
    print(f"  Precision@{K}: {avg_p*100:.0f}%   (of top {K} sources, share relevant)")
    print("GUARDRAILS")
    print(f"  Recall@{K}:    {avg_r*100:.0f}%   (of sources that should appear, share found)")
    if refusal_total:
        print(f"  Refusal accuracy: {refusal_correct}/{refusal_total} "
              f"= {refusal_correct/refusal_total*100:.0f}%")
    print(f"  Overall pass rate: {passes}/{len(TESTS)} = {passes/len(TESTS)*100:.0f}%")

    fails = [r for r in rows if not r[2]]
    if fails:
        print("\nFAILURES (for failure analysis):")
        for rid, cat, _, note, top in fails:
            print(f"  Q{rid} [{cat}] — {note} | top{K}: {top}")
    print()


if __name__ == "__main__":
    if "--show-ids" in sys.argv:
        show_ids()
    else:
        main()
