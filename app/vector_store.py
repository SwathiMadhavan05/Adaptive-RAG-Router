"""
FAISS-backed vector store for document retrieval.

Chunking note: we chunk by paragraph/section rather than fixed character
windows. These sample docs are short policy documents where each
paragraph is a coherent semantic unit (e.g., one paragraph = one plan's
refund terms). Fixed-size chunking would risk splitting a single policy
rule across two chunks, losing context. For longer/messier real-world
docs, a recursive character splitter with overlap is the safer default --
see README for the trade-off discussion.
"""
from dataclasses import dataclass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL, RETRIEVAL_TOP_K


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


class VectorStore:
    def __init__(self, embedding_model_name: str = EMBEDDING_MODEL):
        self.embedder = SentenceTransformer(embedding_model_name)
        self.index: faiss.Index | None = None
        self.chunks: list[Chunk] = []

    def _chunk_document(self, text: str, source: str) -> list[str]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return paragraphs

    def build(self, documents: dict[str, str]) -> None:
        self.chunks = []
        for source, text in documents.items():
            for para in self._chunk_document(text, source):
                self.chunks.append(Chunk(text=para, source=source, chunk_id=len(self.chunks)))

        texts = [c.text for c in self.chunks]
        embeddings = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine sim via normalized inner product
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> list[tuple[Chunk, float]]:
        if self.index is None:
            raise RuntimeError("Vector store not built yet. Call build() first.")

        query_emb = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_emb)

        scores, indices = self.index.search(query_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, path: str) -> None:
        import pickle
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, path: str) -> None:
        import pickle
        self.index = faiss.read_index(f"{path}.index")
        with open(f"{path}.chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
