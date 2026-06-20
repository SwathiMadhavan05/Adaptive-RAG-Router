"""
Hand-crafted structural features to supplement sentence embeddings.

WHY THIS EXISTS: diagnostic testing (scripts/diagnose_router.py) showed
SIMPLE_RAG and MULTI_HOP class centroids have 0.775 cosine similarity in
embedding space -- by far the closest pair of any two classes (next
highest is 0.246). Sentence embeddings capture topical similarity, not
reasoning structure, so a question like "what is X" and "compare X and Y"
about the same policy topic land very close together even though they
need completely different retrieval strategies.

These features give the classifier an explicit signal for reasoning
structure that the embedding alone can't provide.
"""
import re

import numpy as np

COMPARISON_WORDS = [
    "compare", "comparison", "versus", " vs ", " vs.", "difference between",
    "differ", "interact", "relationship between", "combined effect",
    "contradict", "conflict",
]

MULTI_ENTITY_CONNECTORS = [
    " and ", " both ", " across ", " between ",
]


def extract_structural_features(query: str) -> np.ndarray:
    q = query.lower()

    has_comparison_word = float(any(word in q for word in COMPARISON_WORDS))
    has_multi_entity_connector = float(any(conn in q for conn in MULTI_ENTITY_CONNECTORS))
    question_mark_count = float(q.count("?"))
    word_count = float(len(q.split()))
    has_two_capitalized_terms = float(
        len(re.findall(r"\b[A-Z][a-zA-Z]+\b", query)) >= 2
    )

    return np.array([
        has_comparison_word,
        has_multi_entity_connector,
        word_count / 20.0,  # normalize roughly to 0-1ish range
        has_two_capitalized_terms,
    ], dtype=np.float32)


NUM_STRUCTURAL_FEATURES = 4
