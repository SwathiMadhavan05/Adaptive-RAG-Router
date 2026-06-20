# Adaptive RAG Router

A RAG system that **routes each query to the cheapest strategy that can answer it correctly**, instead of always doing full retrieval. A trained classifier decides per-query whether to:

| Route | When | Relative cost |
|---|---|---|
| **PARAMETRIC** | Model already knows the answer (general knowledge, no doc lookup needed) | 1x |
| **SIMPLE_RAG** | A single retrieval pass answers it | 2x |
| **MULTI_HOP** | Needs retrieve → reason → retrieve again (comparative/multi-part questions) | 5x |
| **WEB_SEARCH** | Needs current info not in the indexed corpus | 3x |

Naive RAG retrieves on *every* query, even "what's 2+2." That wastes latency and cost on the queries that didn't need it. This project measures exactly how much that costs and shows the fix.

## Why this exists (the actual problem)

Most RAG demos retrieve unconditionally. This project asks: how much of that retrieval is wasted, and can a cheap classifier decide when to skip it — without hurting answer quality? See [eval/results.csv](eval/results.csv) and the summary below for the measured answer.

## Architecture

```
Query
  │
  ▼
Router (PyTorch classifier on sentence-transformer embeddings)
  │
  ├─ PARAMETRIC  → direct LLM call
  ├─ SIMPLE_RAG  → 1x retrieve (FAISS) + generate
  ├─ MULTI_HOP   → retrieve → LLM decides if more needed → retrieve again (capped at 3 hops)
  └─ WEB_SEARCH  → Tavily search + generate
  │
  ▼
Answer + metadata (route taken, latency, cost units, chunks used)
```

## Tech stack

- **Router classifier**: PyTorch MLP on frozen `sentence-transformers` (all-MiniLM-L6-v2) embeddings
- **Vector store**: FAISS (local, zero infra)
- **Orchestration**: LangChain-style manual multi-hop loop (kept explicit/manual rather than a black-box agent, so hop logic and stopping conditions are auditable — see "Design decisions" below)
- **LLM generation**: Groq (free tier, Llama 3.3 70B) — see `app/llm_provider.py` for why this is swappable
- **Web search fallback**: Tavily (free tier)
- **API**: FastAPI

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Get free API keys
#    - Groq: https://console.groq.com
#    - Tavily: https://tavily.com
cp .env.example .env
# edit .env and paste in your keys

# 3. Train the router classifier
python scripts/train_router.py

# 4. Build the document index
python scripts/build_index.py

# 5. Run the eval harness (this generates the headline numbers)
python eval/run_eval.py

# 6. Start the API
uvicorn app.main:app --reload --port 8000
# visit http://localhost:8000/docs
```

### Try it

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How long is the refund window for the Premium plan?"}'
```

## Results

*(Numbers below are filled in by running `python eval/run_eval.py` — placeholder structure shown; replace with your actual run output.)*

```
Router route-selection accuracy:  XX%

Answer correctness (router vs. always-RAG baseline):
  Router (adaptive):     XX%
  Baseline (always RAG): XX%

Avg latency per query:
  Router (adaptive):     X.XXXs
  Baseline (always RAG): X.XXXs
  Improvement:           +XX%

Avg cost units per query:
  Router (adaptive):     X.XX
  Baseline (always RAG): X.XX
  Improvement:           +XX%
```

**Headline claim**: routing cut average cost by **XX%** and latency by **XX%**, with **no drop** in answer correctness on the held-out eval set (`eval/eval_set.py`, kept separate from training data to avoid inflated numbers).

## Design decisions (and trade-offs I'd defend in an interview)

**Why a trained classifier instead of letting the LLM decide the route via function-calling?**
A separate lightweight classifier adds near-zero latency/cost per query (one tiny forward pass on a 384-dim vector). Letting the LLM decide via tool-calling is more flexible (no retraining needed for new categories) but adds a full LLM call to *every* query just to decide what to do next — which partially defeats the purpose of a router meant to save cost. I chose the classifier because the 4 categories are stable and well-defined; I'd reconsider if the routing logic needed to be more dynamic/extensible.

**Why paragraph-based chunking instead of fixed-size windows?**
The sample policy documents have one coherent rule per paragraph. Fixed-size chunking risks splitting a single policy clause across two chunks, which would hurt retrieval precision more than it helps. For longer, less structured documents I'd switch to a recursive splitter with overlap — this is a real trade-off, not a universal answer.

**Why cap multi-hop at 3 hops?**
Without a cap, a model that's unsure could loop indefinitely, burning cost with no guarantee of converging. Capping and falling back to a "best effort" answer at the cap is a deliberate latency/completeness trade-off — better to return a slightly incomplete answer fast than hang indefinitely.

**Why is the answer-correctness check just keyword matching (`must_contain`)?**
It's a simple, explainable proxy for "did the answer contain the right facts," not a substitute for human eval or an LLM-as-judge approach. At this project's scale, I preferred a metric I could fully explain over a black-box similarity score. With more time, I'd add an LLM-as-judge layer for nuance (e.g., correct facts stated wrong way).

## What I'd do with more time

- **Confidence-based fallback**: when the router's confidence is low, fall back to SIMPLE_RAG instead of trusting a shaky prediction — currently it always commits to the top class.
- **Hybrid search** (keyword + semantic) for retrieval, since pure embedding similarity can miss exact-term matches (e.g., specific policy section numbers).
- **Re-ranking** retrieved chunks with a cross-encoder before passing to the LLM, rather than trusting raw cosine similarity ranking.
- **LLM-as-judge** for answer correctness instead of keyword matching, to catch cases where the answer is right but phrased differently than expected.
- **Real cost tracking** using actual token counts/pricing instead of the relative `COST_UNITS` proxy used here.
- **Bedrock swap-in**: `app/llm_provider.py` is already structured so this is a single function implementation + one config change, not a refactor (see the `_generate_bedrock` stub).
- **SageMaker deployment** of the router classifier as a hosted endpoint, for when routing needs to happen outside this codebase (e.g., called from a different service).

## Repo structure

```
app/
  config.py              # central settings, API keys, paths
  llm_provider.py         # provider-agnostic LLM interface (Groq now, Bedrock-ready)
  vector_store.py          # FAISS wrapper: chunking, indexing, search
  router.py                # loads trained classifier, predicts route at inference time
  strategies.py             # the 4 execution paths (parametric/simple/multi-hop/web)
  orchestrator.py            # ties router + strategies together, tracks latency/cost
  main.py                     # FastAPI app
  models/
    router_classifier.py      # PyTorch MLP architecture
    router_weights.pt          # trained weights (generated by scripts/train_router.py)
    label_map.json              # class index -> label name
data/
  training_data.py           # labeled examples for training the router
  sample_documents.py         # sample policy docs forming the RAG corpus
  faiss_index/                 # saved FAISS index (generated by scripts/build_index.py)
scripts/
  train_router.py             # trains and saves the router classifier
  build_index.py                # builds and saves the FAISS index
eval/
  eval_set.py                  # held-out eval queries with expected routes + key facts
  run_eval.py                    # runs eval, compares router vs always-RAG baseline
  results.csv                     # generated eval output
```
