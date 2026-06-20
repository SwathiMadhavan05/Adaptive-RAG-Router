"""
Quick CLI for testing the router end-to-end without starting the API server.

Usage:
    python scripts/try_it.py "What is the refund policy for Premium customers?"
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from app.vector_store import VectorStore
from app.orchestrator import Orchestrator
from app.config import FAISS_INDEX_DIR


def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/try_it.py "your question here"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    store = VectorStore()
    store.load(str(FAISS_INDEX_DIR / "index"))
    orchestrator = Orchestrator(vector_store=store)

    result = orchestrator.handle_query(query)

    print(f"\nQuery: {query}")
    print(f"Route: {result['route']} (confidence: {result['router_confidence']:.2f})")
    print(f"Hops: {result['hops']}")
    print(f"Latency: {result['latency_seconds']}s")
    print(f"Cost units: {result['cost_units']}")
    print(f"\nAnswer:\n{result['answer']}")

    if result["chunks_retrieved"]:
        print(f"\n--- Retrieved {len(result['chunks_retrieved'])} chunk(s) ---")
        for c in result["chunks_retrieved"]:
            print(f"  [{c['source']}] {c['text'][:100]}...")


if __name__ == "__main__":
    main()
