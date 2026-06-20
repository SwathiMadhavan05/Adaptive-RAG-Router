"""
Loads the trained router classifier and predicts which RAG strategy
to use for an incoming query.
"""
import json

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL, ROUTER_WEIGHTS_PATH, LABEL_MAP_PATH
from app.models.router_classifier import RouterClassifier
from app.features import extract_structural_features, NUM_STRUCTURAL_FEATURES


class Router:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        with open(LABEL_MAP_PATH) as f:
            raw_label_map = json.load(f)
        self.label_map = {int(k): v for k, v in raw_label_map.items()}

        embedding_dim = self.embedder.get_sentence_embedding_dimension()
        self.model = RouterClassifier(
            embedding_dim=embedding_dim,
            num_classes=len(self.label_map),
            num_extra_features=NUM_STRUCTURAL_FEATURES,
        )
        self.model.load_state_dict(torch.load(ROUTER_WEIGHTS_PATH, map_location="cpu"))
        self.model.eval()

    def predict(self, query: str) -> dict:
        embedding = self.embedder.encode([query], convert_to_numpy=True)
        structural = extract_structural_features(query).reshape(1, -1)
        combined = np.concatenate([embedding, structural], axis=1)
        x = torch.tensor(combined, dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
            predicted_idx = int(torch.argmax(probs).item())

        return {
            "route": self.label_map[predicted_idx],
            "confidence": float(probs[predicted_idx].item()),
            "all_probs": {self.label_map[i]: float(probs[i].item()) for i in range(len(self.label_map))},
        }
