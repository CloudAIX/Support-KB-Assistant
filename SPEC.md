# Support KB Assistant — Build Spec (Week 2 RAG)

Plain-English spec for a Week 2 course build. Paste-ready for Confluence.
Note: this is the spec for this one build. It is not the reusable Confluence
template (that is parked until I confirm where the page style I liked came from).

## Problem

A customer-support team needs fast, correct answers from its own help articles
and past tickets. A plain language model cannot do this on its own. It was never
trained on this company's data, and it can make up details. We want answers that
are grounded in real text and that show their sources.

## What I built

A small assistant that:

- Searches a knowledge base of help articles and past support tickets.
- Answers a question using only the text it found.
- Shows a citation for each fact, so the answer can be checked.
- Says "I don't know" when the answer is not in the knowledge base.

The data is made up. There are no real company names or customer details.

## How it works

Build-time (done once, and again when documents change):

1. Load the help articles and past tickets.
2. Break each document into small pieces, with a small overlap between pieces so
   facts that sit on a boundary are not lost.
3. Turn each piece into numbers (an embedding) using one fixed model.
4. Store the pieces, their numbers, and their tags in a local vector store.

Run-time (every time someone asks a question):

1. Search by meaning (vector search) and by exact words (keyword search) at the
   same time.
2. Join the two result lists into one ranked list.
3. Keep the top few pieces.
4. Ask the model to write a grounded answer with citations, using only those
   pieces.

## Decisions

- Hybrid search from the start. For support questions you get both plain-English
  questions and exact terms like order numbers and error codes. Searching by
  meaning and by exact words together handles both. This matches the course
  advice for support use cases.
- Local vector store first. A local store runs with no account and no cost, which
  is enough for a small demo. Moving to a managed store (such as the one in the
  course) is a later step and lines up with the guest lecture on scale and cost.
- Embeddings set to the course's model by default, with a local no-key option, so
  results line up with the course demo but the project still runs offline if
  needed.
- This is the no-code / low-code submission track, built by vibe-coding (the same
  way as Week 1). The code-heavy LangChain version stays a side-learning task.

## Out of scope for now (tracked, not done)

- A reranker step that re-sorts the found pieces for higher precision. The course
  calls this the single biggest win. Planned as the next small follow-up.
- A managed vector store (the course uses one). Planned after the scale-and-cost
  lecture.
- A quality-checking step (evals). Covered in a later week.
- Memory across turns and multi-step retrieval. Covered in the agentic week.

## Next

1. Wait for the project brief, then match this build to what the brief asks for.
2. Build the index and confirm answers are grounded and cite sources.
3. Add the reranker as the first follow-up.
4. Write the short LinkedIn build-in-public post once the repo is public.

## Track 1 follow-up (side learning)

After the build works, spend one hour having the coding tool walk through the
retrieve-and-answer code, line by line. This closes the Python gap on the exact
code shape that matters for the evals and security weeks, with no extra build.
