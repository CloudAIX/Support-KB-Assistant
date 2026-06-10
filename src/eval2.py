"""
eval.py — 20-question evaluation for the Support KB Assistant (Week 2).

Wired to rag.py:
  retrieve(query, top_k=5, where=None) -> list[(cid, text)]  (ranked, best first)
  answer(query, where=None) -> str            (Claude)
  answer_nebius(query, where=None) -> str      (Llama 3.3 70B via Nebius)

Scores
------
  Precision@k  (PRIMARY for customer support): of the top-k sources retrieved,
               how many were relevant?  (Pinecone session: query-first, small set.)
  Recall@k     (guardrail): of the sources that should appear, how many did?
  Refusal      (guardrail): on unanswerable questions, did it say "I don't know"?

Modes
-----
  python3 eval.py            scored eval on the default model (Claude)
  python3 eval.py --show-ids diagnostic: print the chunk ids each question gets
  python3 eval.py --compare  run BOTH models on all 20 and print a comparison

Retrieval is identical for both models, so the comparison isolates the model:
same chunks, same prompt, different generator.
"""

import sys
from dataclasses import dataclass, field

from rag import retrieve, answer, answer_nebius


def get_sources(question):
    hits = retrieve(question, top_k=5)        # [(cid, text), ...] ranked best-first
    return [cid for cid, _text in hits]


def run_pipeline(question, answer_fn=answer):
    source_ids = get_sources(question)
    ans = answer_fn(question)
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
    TestCase(14, "ambiguous", "I have a billing problem", ["billing-and-refunds"]),
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
    "not mentioned", "don't mention", "do not mention", "based on the provided sources",
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
        ids = get_sources(tc.question)
        print(f"Q{tc.id}: {tc.question}")
        print(f"   expected substring: {tc.expected}")
        print(f"   retrieved ids: {ids}\n")


def score_one(tc, ans, sources, K=3):
    """Return (passed, note, precision, recall, refused) for a single question."""
    refused = looks_like_refusal(ans)
    precision = recall = None
    if tc.expected:
        top = sources[:K]
        matched_top = id_matches(tc.expected, top)
        precision = len(matched_top) / max(1, len(top))
        recall = len(set(matched_top)) / len(tc.expected)
    hit = bool(id_matches(tc.expected, sources)) if tc.expected else True

    if tc.should_refuse:
        passed = refused
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
    return passed, note, precision, recall, refused


def summarise(label, results):
    """results: list of (tc, passed, note, precision, recall, refused)"""
    passes = sum(1 for r in results if r[1])
    ps = [r[3] for r in results if r[3] is not None]
    rs = [r[4] for r in results if r[4] is not None]
    ref_total = sum(1 for r in results if r[0].should_refuse)
    ref_ok = sum(1 for r in results if r[0].should_refuse and r[1])
    avg_p = sum(ps) / len(ps) * 100 if ps else 0
    avg_r = sum(rs) / len(rs) * 100 if rs else 0
    print(f"\n  {label}")
    print(f"    Pass rate:        {passes}/{len(results)} = {passes/len(results)*100:.0f}%")
    print(f"    Precision@3:      {avg_p:.0f}%")
    print(f"    Recall@3:         {avg_r:.0f}%")
    print(f"    Refusal accuracy: {ref_ok}/{ref_total} = {ref_ok/ref_total*100:.0f}%")


def run_model(answer_fn):
    out = []
    for tc in TESTS:
        sources = get_sources(tc.question)
        ans = answer_fn(tc.question)
        passed, note, p, r, refused = score_one(tc, ans, sources)
        out.append((tc, passed, note, p, r, refused))
    return out


def main():
    results = run_model(answer)
    print("\n" + "=" * 72)
    print("  SUPPORT KB ASSISTANT — 20-QUESTION EVALUATION (Claude)")
    print("=" * 72)
    print(f"{'ID':<4}{'CATEGORY':<14}{'RESULT':<8}{'NOTE'}")
    print("-" * 72)
    for tc, passed, note, *_ in results:
        print(f"{tc.id:<4}{tc.category:<14}{'PASS' if passed else 'FAIL':<8}{note}")
    print("-" * 72)
    summarise("Claude (claude-sonnet-4-6)", results)
    fails = [(tc, note) for tc, passed, note, *_ in results if not passed]
    if fails:
        print("\n  FAILURES:")
        for tc, note in fails:
            print(f"    Q{tc.id} [{tc.category}] — {note}")
    print()


def compare():
    print("\nRunning Claude on 20 questions...")
    claude = run_model(answer)
    print("Running Llama 3.3 70B (Nebius) on 20 questions...")
    llama = run_model(answer_nebius)

    print("\n" + "=" * 72)
    print("  MODEL COMPARISON — same retrieval + prompt, different generator")
    print("=" * 72)
    print(f"{'ID':<4}{'CATEGORY':<13}{'CLAUDE':<18}{'LLAMA 3.3 70B'}")
    print("-" * 72)
    for (tc, cp, cn, *_), (_, lp, ln, *_) in zip(claude, llama):
        c = ('PASS ' if cp else 'FAIL ') + cn
        l = ('PASS ' if lp else 'FAIL ') + ln
        print(f"{tc.id:<4}{tc.category:<13}{c:<18}{l}")
    print("-" * 72)
    summarise("Claude (claude-sonnet-4-6)", claude)
    summarise("Llama 3.3 70B (Nebius Token Factory)", llama)

    # Where they differ
    diffs = [(tc, cp, lp) for (tc, cp, *_), (_, lp, *_) in zip(claude, llama) if cp != lp]
    print("\n  WHERE THEY DIFFER:")
    if diffs:
        for tc, cp, lp in diffs:
            print(f"    Q{tc.id} [{tc.category}]: "
                  f"Claude {'PASS' if cp else 'FAIL'}, Llama {'PASS' if lp else 'FAIL'} "
                  f"— {tc.question}")
    else:
        print("    None — both models scored identically on all 20.")
    print()


if __name__ == "__main__":
    if "--show-ids" in sys.argv:
        show_ids()
    elif "--compare" in sys.argv:
        compare()
    else:
        main()
