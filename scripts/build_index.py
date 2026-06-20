"""
Build the FAISS vector index from the sample documents and save it to disk.

Usage:
    python scripts/build_index.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from app.vector_store import VectorStore
from data.sample_documents import DOCUMENTS
from app.config import FAISS_INDEX_DIR


def main():
    print(f"Building index from {len(DOCUMENTS)} documents...")
    store = VectorStore()
    store.build(DOCUMENTS)
    print(f"Indexed {len(store.chunks)} chunks total.")

    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    save_path = str(FAISS_INDEX_DIR / "index")
    store.save(save_path)
    print(f"Saved index -> {save_path}.index")
    print(f"Saved chunks -> {save_path}.chunks.pkl")


if __name__ == "__main__":
    main()
