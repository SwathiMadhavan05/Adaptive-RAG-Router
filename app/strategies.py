"""
The four execution strategies the router can dispatch to. Each returns
a dict with the answer plus metadata (chunks used, hop count, etc.) so
the eval harness and API response can report exactly what happened.
"""
from app.llm_provider import generate
from app.vector_store import VectorStore
from app.config import MAX_HOPS, RETRIEVAL_TOP_K


def run_parametric(query: str) -> dict:
    answer = generate(
        prompt=query,
        system="You are a helpful, knowledgeable assistant. Answer directly and concisely.",
    )
    return {
        "answer": answer,
        "route": "PARAMETRIC",
        "hops": 0,
        "chunks_retrieved": [],
    }


def run_simple_rag(query: str, store: VectorStore, top_k: int = RETRIEVAL_TOP_K) -> dict:
    results = store.search(query, top_k=top_k)
    context = "\n\n".join(f"[{c.source}] {c.text}" for c, _ in results)

    prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say so clearly.

Context:
{context}

Question: {query}

Answer:"""

    answer = generate(prompt=prompt, system="You answer strictly based on provided context. Do not use outside knowledge.")

    return {
        "answer": answer,
        "route": "SIMPLE_RAG",
        "hops": 1,
        "chunks_retrieved": [{"source": c.source, "text": c.text, "score": s} for c, s in results],
    }


def run_multi_hop(query: str, store: VectorStore, max_hops: int = MAX_HOPS, top_k: int = RETRIEVAL_TOP_K) -> dict:
    all_chunks_seen = []
    seen_chunk_ids = set()
    accumulated_context = []
    current_query = query

    for hop in range(1, max_hops + 1):
        results = store.search(current_query, top_k=top_k)

        new_results = [(c, s) for c, s in results if c.chunk_id not in seen_chunk_ids]
        for c, s in new_results:
            seen_chunk_ids.add(c.chunk_id)
            accumulated_context.append(f"[{c.source}] {c.text}")
            all_chunks_seen.append({"source": c.source, "text": c.text, "score": s, "hop": hop})

        context_so_far = "\n\n".join(accumulated_context)

        decision_prompt = f"""You are answering a multi-part question that may require multiple rounds of information retrieval.

Original question: {query}

Context gathered so far:
{context_so_far}

Can you FULLY and ACCURATY answer the original question using only the context above?
Respond in this exact format:
DECISION: <ANSWER or NEED_MORE_INFO>
NEXT_QUERY: <if NEED_MORE_INFO, write a focused search query for the missing piece. If ANSWER, write NONE>
ANSWER: <if DECISION is ANSWER, write the full answer here. If NEED_MORE_INFO, write NONE>"""

        decision_raw = generate(prompt=decision_prompt, system="Follow the exact response format requested. Be precise.")

        decision, next_query, draft_answer = _parse_decision(decision_raw)

        if decision == "ANSWER" or hop == max_hops:
            final_answer = draft_answer if decision == "ANSWER" else _force_final_answer(query, context_so_far)
            return {
                "answer": final_answer,
                "route": "MULTI_HOP",
                "hops": hop,
                "chunks_retrieved": all_chunks_seen,
                "hop_cap_reached": hop == max_hops and decision != "ANSWER",
            }

        current_query = next_query if next_query and next_query != "NONE" else query

    # Should not reach here due to max_hops check above, but safe fallback
    final_answer = _force_final_answer(query, "\n\n".join(accumulated_context))
    return {
        "answer": final_answer,
        "route": "MULTI_HOP",
        "hops": max_hops,
        "chunks_retrieved": all_chunks_seen,
        "hop_cap_reached": True,
    }


def _parse_decision(raw: str) -> tuple[str, str, str]:
    decision, next_query, answer = "NEED_MORE_INFO", "NONE", "NONE"
    for line in raw.splitlines():
        line = line.strip()
        if line.upper().startswith("DECISION:"):
            decision = line.split(":", 1)[1].strip().upper()
        elif line.upper().startswith("NEXT_QUERY:"):
            next_query = line.split(":", 1)[1].strip()
        elif line.upper().startswith("ANSWER:"):
            answer = line.split(":", 1)[1].strip()
    return decision, next_query, answer


def _force_final_answer(query: str, context: str) -> str:
    """Used when hop cap is reached without an explicit ANSWER decision -- we
    still return the best possible answer rather than erroring out."""
    prompt = f"""Based on the context gathered below, give the best possible answer to the question, even if some information may be incomplete.

Context:
{context}

Question: {query}

Answer (note any gaps if relevant):"""
    return generate(prompt=prompt)


def run_web_search(query: str) -> dict:
    from tavily import TavilyClient
    from app.config import TAVILY_API_KEY

    client = TavilyClient(api_key=TAVILY_API_KEY)
    search_results = client.search(query=query, max_results=5)

    context = "\n\n".join(
        f"[{r['url']}] {r['content']}" for r in search_results.get("results", [])
    )

    prompt = f"""Answer the question using the web search results below. Cite sources where relevant.

Search results:
{context}

Question: {query}

Answer:"""

    answer = generate(prompt=prompt, system="You answer based on current web search results.")

    return {
        "answer": answer,
        "route": "WEB_SEARCH",
        "hops": 1,
        "chunks_retrieved": [{"source": r["url"], "text": r["content"][:300]} for r in search_results.get("results", [])],
    }
