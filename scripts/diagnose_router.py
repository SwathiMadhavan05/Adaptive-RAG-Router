"""
Diagnostic: prints a confusion matrix and shows the actual misclassified
examples, so we can see exactly what MULTI_HOP queries are being
predicted as instead.

Usage:
    python scripts/diagnose_router.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sentence_transformers import SentenceTransformer

from data.training_data import TRAINING_DATA, LABELS
from app.models.router_classifier import RouterClassifier

SEED = 42
EPOCHS = 60
LR = 1e-3


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    queries = [q for q, _ in TRAINING_DATA]
    labels = [label for _, label in TRAINING_DATA]

    embeddings = embedder.encode(queries, convert_to_numpy=True)

    X_train, X_val, y_train, y_val, q_train, q_val = train_test_split(
        embeddings, labels, queries, test_size=0.2, random_state=SEED, stratify=labels
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    model = RouterClassifier(embedding_dim=embeddings.shape[1], num_classes=len(LABELS))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t).argmax(dim=1).numpy()

    print("Confusion matrix (rows=actual, cols=predicted):")
    label_names = [LABELS[i] for i in sorted(LABELS.keys())]
    cm = confusion_matrix(y_val, val_preds, labels=sorted(LABELS.keys()))
    print(f"{'':>12}" + "".join(f"{name:>12}" for name in label_names))
    for i, row in enumerate(cm):
        print(f"{label_names[i]:>12}" + "".join(f"{v:>12}" for v in row))

    print("\n--- Misclassified examples ---")
    for i, (q, actual, pred) in enumerate(zip(q_val, y_val, val_preds)):
        if actual != pred:
            print(f"  ACTUAL={LABELS[actual]:<12} PREDICTED={LABELS[pred]:<12} | {q}")

    # Also check semantic similarity between MULTI_HOP and SIMPLE_RAG class centroids
    print("\n--- Class centroid similarity (cosine) ---")
    from numpy.linalg import norm
    centroids = {}
    for label_id in sorted(LABELS.keys()):
        mask = np.array(labels) == label_id
        centroids[label_id] = embeddings[mask].mean(axis=0)

    for i in sorted(LABELS.keys()):
        for j in sorted(LABELS.keys()):
            if i < j:
                sim = np.dot(centroids[i], centroids[j]) / (norm(centroids[i]) * norm(centroids[j]))
                print(f"  {LABELS[i]:<12} vs {LABELS[j]:<12}: {sim:.3f}")


if __name__ == "__main__":
    main()
