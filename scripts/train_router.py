"""
Train the router classifier on labeled query examples.

Usage:
    python scripts/train_router.py

Saves:
    app/models/router_weights.pt   -- trained classifier weights
    app/models/label_map.json      -- class index -> label name
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).parent.parent))
from data.training_data import TRAINING_DATA, LABELS
from app.models.router_classifier import RouterClassifier
from app.features import extract_structural_features, NUM_STRUCTURAL_FEATURES

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_SAVE_PATH = Path(__file__).parent.parent / "app" / "models" / "router_weights.pt"
LABEL_MAP_PATH = Path(__file__).parent.parent / "app" / "models" / "label_map.json"

EPOCHS = 60
LR = 1e-3
SEED = 42


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    queries = [q for q, _ in TRAINING_DATA]
    labels = [label for _, label in TRAINING_DATA]

    print(f"Embedding {len(queries)} training queries...")
    embeddings = embedder.encode(queries, show_progress_bar=True, convert_to_numpy=True)

    print("Extracting structural features (comparison words, multi-entity connectors, etc.)...")
    structural_feats = np.array([extract_structural_features(q) for q in queries], dtype=np.float32)
    combined_features = np.concatenate([embeddings, structural_feats], axis=1)
    print(f"Embedding dim: {embeddings.shape[1]}, structural feature dim: {structural_feats.shape[1]}, combined: {combined_features.shape[1]}")

    X_train, X_val, y_train, y_val = train_test_split(
        combined_features, labels, test_size=0.2, random_state=SEED, stratify=labels
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    model = RouterClassifier(
        embedding_dim=embeddings.shape[1],
        num_classes=len(LABELS),
        num_extra_features=NUM_STRUCTURAL_FEATURES,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("\nTraining router classifier...")
    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == EPOCHS:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t)
                val_preds = val_logits.argmax(dim=1)
                val_acc = (val_preds == y_val_t).float().mean().item()
                best_val_acc = max(best_val_acc, val_acc)
            print(f"  Epoch {epoch:3d} | train_loss={loss.item():.4f} | val_acc={val_acc:.3f}")

    # Final eval report
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t).argmax(dim=1).numpy()
    print("\nValidation classification report:")
    print(classification_report(
        y_val, val_preds,
        target_names=[LABELS[i] for i in sorted(LABELS.keys())],
        zero_division=0,
    ))

    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(LABELS, f, indent=2)

    print(f"\nSaved model weights -> {MODEL_SAVE_PATH}")
    print(f"Saved label map     -> {LABEL_MAP_PATH}")
    print(f"Best validation accuracy observed: {best_val_acc:.3f}")
    print(f"\nNOTE: With only {len(queries)} examples, val accuracy is noisy.")
    print("Add more labeled examples (data/training_data.py) for a more robust router.")


if __name__ == "__main__":
    main()
