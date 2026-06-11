# Support KB Assistant — Evaluation Report (Week 2)

All data is synthetic. This report tests the assistant against 20 questions
across four types: questions it should answer well (happy path), questions whose
answer spans two articles, vague questions, and questions it should refuse
because the knowledge base does not cover them.

The test set follows the approach taught in the Week 2 Pinecone guest session:
start with a small set of real queries you want the system to handle, define what
a good result looks like for each, and measure whether you are improving against
them. It is a diagnostic set, not a formal benchmark.

Result: **18 out of 20 passed (90%).** The two failures are described in detail
below, because they are more useful than the passes — each points at a real,
nameable retrieval issue.

## Headline metrics

| Metric | Role | Value |
|--------|------|-------|
| Overall pass rate | Headline | 18 / 20 = 90% |
| Refusal accuracy (correct refusals / 4) | Guardrail | 4 / 4 = 100% |
| Recall@3 (expected sources that appeared in top 3) | Guardrail | 87% |
| Precision@3 (expected sources as a share of top 3) | See note | 33% |
| Multi-article questions that pulled BOTH sources | Detail | 3 / 4 |

### Note on the Precision@3 number

Precision@3 reads as 33%, which looks low, but this is mostly a measurement
artefact, not a retrieval failure. Precision@3 asks: of the top 3 sources
returned, how many are the *expected* ones? Most happy-path questions have only
**one** expected article. So even a perfect retrieval that ranks the correct
document first scores 1 out of 3 (33%), because the other two slots are filled
with other chunks that were not on the answer key — they are not wrong, just not
the single expected source.

For this reason, **Recall@3 (87%) is the truer measure of retrieval quality
here**: when the right document exists, the system surfaced it in the top 3 in
almost every case. The headline quality signal is the 90% pass rate plus 100%
refusal accuracy. Precision@3 is reported honestly but should be read with this
context, not as a standalone quality score.

## Test questions and results

| # | Type | Question | Result | Notes |
|---|------|----------|--------|-------|
| 1 | Happy | How do I reset my password? | PASS | Correct article first. |
| 2 | Happy | How long does a refund take to process? | PASS | Cited billing/refunds. |
| 3 | Happy | What does error E-402 mean? | PASS | Session expired, not billing. |
| 4 | Happy | My account is locked. How long until it unlocks? | PASS | Correct unlock time. |
| 5 | Happy | Which items are not refundable? | **FAIL** | Refused an answerable question. See failure 1. |
| 6 | Happy | The app will not load. What should I try first? | PASS | Ordered steps. |
| 7 | Happy | How is my personal data handled and stored? | PASS | Cited data-and-privacy. |
| 8 | Happy | How do I change or upgrade my plan? | PASS | Cited plans-and-upgrades. |
| 9 | Multi | I can't get into my account anymore | PASS | Pulled login AND troubleshooting. |
| 10 | Multi | I was charged but can't log in to use what I paid for | **FAIL** | Pulled tickets, missed the articles. See failure 2. |
| 11 | Multi | If I upgrade my plan will I be refunded the difference? | PASS | Pulled plans AND billing. |
| 12 | Multi | My order shipped but I think my data was wrong on the account | PASS | Pulled shipping AND data-and-privacy. |
| 13 | Vague | It's not working | PASS | Handled (clarified / grounded). |
| 14 | Vague | I have a billing problem | PASS | Narrowed to billing. |
| 15 | Vague | Something's wrong with my account | PASS | Narrowed to account. |
| 16 | Vague | When will it arrive? | PASS | Narrowed to shipping. |
| 17 | Refuse | What's the refund policy for orders placed in New Zealand? | PASS | Refused, no invented policy. |
| 18 | Refuse | Do you offer a student discount? | PASS | Refused. |
| 19 | Refuse | What are your support hours on public holidays? | PASS | Refused. |
| 20 | Refuse | Can I pay with cryptocurrency? | PASS | Refused. |

## Failure analysis

### Failure 1 — Q5: refused a question it could have answered

**Question:** "Which items are not refundable?"

**What it did:** The system refused, saying it did not know, even though the
answer (used add-on credits, and annual plans cancelled after the 14-day window)
is in the billing-and-refunds article.

**Likely cause:** This is a chunking and retrieval-granularity issue. The correct
*document* was retrieved — billing-and-refunds ranked first in the top 3. But the
specific *chunk* that was fused to the top (`billing-and-refunds.md::0`) was the
refund-window chunk, not the "what is not refundable" chunk. So the model was
given the right document but the wrong section of it, had no grounded support for
the actual question, and correctly refused rather than guess. The refusal
behaviour worked exactly as designed; the retrieval handed it the wrong piece.

**What I would change:** Improve chunk granularity or retrieval depth so that
sibling chunks from the same article are pulled together, or increase top_k so
the non-refundable chunk is not crowded out. A parent-child chunking strategy
(retrieve the small chunk, return the surrounding section) would likely fix this
directly. This is the single clearest example in the eval of why chunking, not
the model, is where RAG quality is won.

### Failure 2 — Q10: past tickets crowded out the knowledge-base articles

**Question:** "I was charged but can't log in to use what I paid for"

**What it did:** The top three sources were all past tickets (T-1001, T-1008,
T-1005) and none of the expected articles (billing-and-refunds, account-and-login)
appeared in the top 3.

**Likely cause:** This is a warm-start over-trigger / corpus-balance issue. The
query is phrased like a real customer complaint, which makes it look very similar
to past tickets in both the dense and the keyword search. Both retrievers leaned
toward the tickets, and reciprocal rank fusion then ranked the tickets above the
canonical articles. The historical-ticket corpus, which is a strength for
known issues, can dominate on conversational queries and push the authoritative
documentation out of the top results.

**What I would change:** Balance the corpus at retrieval time — for example,
guarantee at least one knowledge-base article in the fused top results, or weight
articles slightly above tickets, or tag sources by type and ensure a mix. This is
a genuinely useful finding for the capstone: it shows that adding historical data
for warm-start has a cost (it can swamp canonical answers) that has to be managed,
not just enabled.

## Known issues (found before this eval, carried in)

- One source article is missing a full stop, so its last chunk reads as a
  fragment. The fix is in the source text, not the code.
- The sources panel shows every retrieved chunk, not only the ones actually
  cited in the answer. The next version will show only cited sources.

## What I would improve next (deliberately out of scope for this version)

- **Cross-encoder reranker** — a second pass to re-order the fused results for
  higher precision. Named in the code as the planned v1.1 change. This would also
  help Failure 1, by promoting the chunk that actually answers the question.
- **Parent-child / section-aware chunking** — directly addresses Failure 1.
- **Source-type balancing** — addresses Failure 2.
- **Metadata pre-filtering wired into the interface.**

These are named on purpose. They are real improvements I have chosen not to build
in this version, to keep it small and shippable, rather than gaps I missed.

## Model comparison — Claude vs Llama 3.3 70B (Nebius)

To meet the cohort model-comparison requirement, the same 20 questions were run
through two generators while holding everything else constant: the same hybrid
retrieval, the same retrieved chunks, and the same system prompt. The only
variable is the model that writes the answer. This isolates model behaviour from
retrieval behaviour. The comparison was run twice; the results below were stable
across both runs.

| Metric | Claude (claude-sonnet-4-6) | Llama 3.3 70B (Nebius) |
|--------|----------------------------|------------------------|
| Pass rate | 18/20 = 90% | 18/20 = 90% |
| Precision@3 | 33% | 33% |
| Recall@3 | 87% | 87% |
| Refusal accuracy | 4/4 = 100% | 4/4 = 100% |

The headline numbers are identical, which is expected: retrieval drives most
outcomes, and retrieval is shared. The interesting result is where the two models
disagree on the same context.

### Where they differ (stable across two runs)

- **Q5 — "Which items are not refundable?"**: Claude refused; Llama answered.
- **Q8 — "How do I change or upgrade my plan?"**: Llama refused; Claude answered.

On both questions the relevant article was retrieved but was not the top chunk.
Given that same marginal context, the two models made opposite judgement calls
about whether they had enough to answer. This shows the models have slightly
different **refusal thresholds**: Claude is more conservative on Q5, Llama is more
conservative on Q8. Neither is uniformly safer — they draw the "do I have enough
to answer?" line in different places on borderline context.

For a customer-support agent this is a real design consideration: the choice of
model changes *when the system declines to answer*, independently of retrieval.

### What the comparison confirms

Q10 failed for **both** models. Because both received the same retrieved context
and both failed, this confirms the Q10 failure is a retrieval problem (the
warm-start over-trigger described above), not a generation problem. The model
comparison therefore validates the failure analysis: retrieval-side failures fail
both models, while the only model-dependent differences are the two borderline
refusal cases.
