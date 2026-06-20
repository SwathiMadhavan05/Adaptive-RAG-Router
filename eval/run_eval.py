"""
Eval harness: runs the eval set through (a) the adaptive router and
(b) a baseline that always does SIMPLE_RAG (the naive default most
RAG tutorials use), and reports:

  - Router accuracy (did it pick the expected route?)
  - Latency: router vs always-RAG
  - Cost: router vs always-RAG (using COST_UNITS proxy)
  - Answer correctness: does the answer contain the expected key facts?

Usage:
    python eval/run_eval.py

Outputs:
    eval/results.csv      -- per-query results
    Prints summary stats to console
"""
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd

from app.vector_store import VectorStore
from app.orchestrator import Orchestrator
from app.config import FAISS_INDEX_DIR
from eval.eval_set import EVAL_SET


def answer_contains_facts(answer: str, must_contain: list[str]) -> bool:
    if not must_contain:
        return None  # not checkable (e.g., web search questions with no fixed answer)
    answer_lower = answer.lower()
    return all(fact.lower() in answer_lower for fact in must_contain)


def main():
    print("Loading vector store + router...")
    store = VectorStore()
    store.load(str(FAISS_INDEX_DIR / "index"))
    orchestrator = Orchestrator(vector_store=store)

    rows = []

    print(f"\nRunning {len(EVAL_SET)} eval queries through ADAPTIVE ROUTER...\n")
    for item in EVAL_SET:
        query = item["query"]
        expected_route = item["expected_route"]

        result = orchestrator.handle_query(query)
        correct_route = result["route"] == expected_route
        fact_check = answer_contains_facts(result["answer"], item["must_contain"])

        rows.append({
            "query": query,
            "expected_route": expected_route,
            "predicted_route": result["route"],
            "route_correct": correct_route,
            "router_confidence": result["router_confidence"],
            "latency_router": result["latency_seconds"],
            "cost_router": result["cost_units"],
            "answer_correct": fact_check,
            "answer": result["answer"][:200],
        })
        status = "✓" if correct_route else "✗"
        print(f"  [{status}] '{query[:60]}...' -> predicted={result['route']} (expected={expected_route})")

    print(f"\nRunning {len(EVAL_SET)} eval queries through BASELINE (always SIMPLE_RAG)...\n")
    for i, item in enumerate(EVAL_SET):
        query = item["query"]
        result = orchestrator.handle_query(query, force_route="SIMPLE_RAG")
        fact_check_baseline = answer_contains_facts(result["answer"], item["must_contain"])

        rows[i]["latency_baseline"] = result["latency_seconds"]
        rows[i]["cost_baseline"] = result["cost_units"]
        rows[i]["answer_correct_baseline"] = fact_check_baseline

    df = pd.DataFrame(rows)
    df.to_csv(Path(__file__).parent / "results.csv", index=False)

    # ---- Summary stats ----
    route_accuracy = df["route_correct"].mean()

    checkable = df[df["answer_correct"].notna()]
    router_answer_acc = checkable["answer_correct"].mean() if len(checkable) else float("nan")
    baseline_answer_acc = checkable["answer_correct_baseline"].mean() if len(checkable) else float("nan")

    avg_latency_router = df["latency_router"].mean()
    avg_latency_baseline = df["latency_baseline"].mean()
    latency_improvement_pct = (1 - avg_latency_router / avg_latency_baseline) * 100

    avg_cost_router = df["cost_router"].mean()
    avg_cost_baseline = df["cost_baseline"].mean()
    cost_improvement_pct = (1 - avg_cost_router / avg_cost_baseline) * 100

    print("\n" + "=" * 60)
    print("EVAL SUMMARY")
    print("=" * 60)
    print(f"Router route-selection accuracy:  {route_accuracy:.1%}")
    print(f"\nAnswer correctness (on checkable queries, n={len(checkable)}):")
    print(f"  Router (adaptive):    {router_answer_acc:.1%}")
    print(f"  Baseline (always RAG): {baseline_answer_acc:.1%}")
    print(f"\nAvg latency per query:")
    print(f"  Router (adaptive):     {avg_latency_router:.3f}s")
    print(f"  Baseline (always RAG): {avg_latency_baseline:.3f}s")
    print(f"  Improvement:           {latency_improvement_pct:+.1f}%")
    print(f"\nAvg cost units per query:")
    print(f"  Router (adaptive):     {avg_cost_router:.2f}")
    print(f"  Baseline (always RAG): {avg_cost_baseline:.2f}")
    print(f"  Improvement:           {cost_improvement_pct:+.1f}%")
    print("=" * 60)
    print(f"\nFull results saved to eval/results.csv")

    print("\n--- Routing breakdown by category ---")
    print(df.groupby("expected_route")["route_correct"].agg(["mean", "count"]))


if __name__ == "__main__":
    main()
