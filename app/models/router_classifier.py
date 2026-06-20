"""
Router classifier: takes a sentence embedding (+ optional hand-crafted
structural features) and predicts which RAG strategy to use
(PARAMETRIC, SIMPLE_RAG, MULTI_HOP, WEB_SEARCH).

Architecture: small MLP head on top of frozen sentence-transformer
embeddings. We don't fine-tune the embedding model itself -- that would
be overkill for ~100-200 training examples and risks overfitting. A
lightweight classifier head on frozen embeddings is the right-sized
solution here.

NOTE ON STRUCTURAL FEATURES: pure semantic embeddings struggle to
separate SIMPLE_RAG from MULTI_HOP, because both classes are topically
similar (same policy domains) -- the distinguishing signal is reasoning
*structure* (single lookup vs. combine-across-documents), which sentence
embeddings aren't trained to capture. See app/features.py for the
hand-crafted features that supplement the embedding to fix this.
"""
import torch
import torch.nn as nn


class RouterClassifier(nn.Module):
    def __init__(self, embedding_dim: int = 384, num_classes: int = 4, hidden_dim: int = 64, num_extra_features: int = 0):
        super().__init__()
        input_dim = embedding_dim + num_extra_features
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
