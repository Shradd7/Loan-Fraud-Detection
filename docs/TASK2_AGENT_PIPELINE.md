# Task 2: Local Multi-Agent RAG Pipeline

Task 2 verifies likely borrower location using a local, auditable RAG pipeline. It does not require Groq, Selenium, browser automation, or live LinkedIn scraping.

## Why Local RAG

The dummy dataset mostly contains structured signals: branch code, address, DL number, vehicle number, UPI location, frequent location, last location, and ATM activity. For these fields, vectorless retrieval is more explainable than embedding-only retrieval.

## Retrieval Layer

The retrieval layer combines:

- exact branch-code lookup
- state-code lookup from DL and vehicle numbers
- city/state alias matching
- phone-prefix heuristic
- optional future embedding retrieval for messy unstructured documents

Knowledge source:

```text
data/rag/location_knowledge.json
```

## Four Agents

### Agent 1: Planner & Query Builder

Builds a retrieval query from the borrower row and decides which checks should run.

Outputs:

- static checks
- activity checks
- retrieval query
- retrieved context

### Agent 2: Static Identity Verifier

Checks stable identity signals:

- branch code
- address
- DL number
- vehicle number
- phone prefix

It answers: where does the identity profile appear to belong?

### Agent 3: Activity Location Verifier

Checks behavior/activity signals:

- UPI location
- frequent location
- last location
- ATM transaction signal

It answers: where does the person appear to be active?

### Agent 4: Conflict Resolver & Final Scorer

Compares static and activity evidence, detects cross-state conflicts, assigns confidence, and decides whether manual review is needed.

Outputs:

- predicted location
- confidence
- score
- manual review flag
- evidence trail

## Current Tech

- Python
- local JSON knowledge base
- vectorless retrieval
- deterministic local agents

## Future Upgrade

Embedding retrieval can be added later for:

- messy address documents
- city aliases
- branch policy documents
- manually supplied LinkedIn/public-profile location text

The current implementation intentionally avoids live scraping for reliability, repeatability, and portfolio safety.
