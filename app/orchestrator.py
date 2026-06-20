"""
Main orchestration layer: takes a query, routes it via the trained
classifier, executes the corresponding strategy, and returns the answer
along with timing/cost metadata needed for the eval harness.
"""
import time

from app.router import Router
from app.vector_store import VectorStore
from app.strategies import run_parametric, run_simple_rag, run_multi_hop, run_web_search

# Rough relative cost units per strategy, used for the eval harness'
# cost comparison. These are NOT real dollar amounts -- they're a proxy
# based on number of LLM calls + retrieval calls each strategy makes,
# so we can compare "router on" vs "always full RAG" cost without
# needing real billing data. See eval/README for the full breakdown.
COST_UNITS = {
    "PARAMETRIC": 1,    # 1 LLM call, 0 retrieval
    "SIMPLE_RAG": 2,    # 1 retrieval + 1 LLM call
    "MULTI_HOP": 5,     # ~2-3 retrieval + 2-3 LLM calls (decision + answer)
    "WEB_SEARCH": 3,    # 1 web search + 1 LLM call
}


class Orchestrator:
    def __init__(self, vector_store: VectorStore):
        self.router = Router()
        self.store = vector_store

    def handle_query(self, query: str, force_route: str | None = None) -> dict:
        start = time.perf_counter()

        if force_route:
            # used by eval harness baseline runs (e.g., "always SIMPLE_RAG")
            route_decision = {"route": force_route, "confidence": 1.0, "all_probs": {}}
        else:
            route_decision = self.router.predict(query)

        route = route_decision["route"]

        if route == "PARAMETRIC":
            result = run_parametric(query)
        elif route == "SIMPLE_RAG":
            result = run_simple_rag(query, self.store)
        elif route == "MULTI_HOP":
            result = run_multi_hop(query, self.store)
        elif route == "WEB_SEARCH":
            result = run_web_search(query)
        else:
            raise ValueError(f"Unknown route: {route}")

        elapsed = time.perf_counter() - start

        result["latency_seconds"] = round(elapsed, 3)
        result["router_confidence"] = route_decision["confidence"]
        result["router_probs"] = route_decision["all_probs"]
        result["cost_units"] = COST_UNITS[result["route"]]
        result["query"] = query

        return result
