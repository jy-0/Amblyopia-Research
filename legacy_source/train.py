"""
Training script for the multi-modal amblyopia classifier (simplified: 2 features + 1 label).

Usage:
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate hospital
    python model_code/train.py
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
import json
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

from config import (
    OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR, BATCH_SIZE, NUM_EPOCHS,
    LEARNING_RATE, WEIGHT_DECAY, EARLY_STOP_PATIENCE,
    DEVICE, SEED,
)
from dataset import get_dataloaders
from model import MultiModalAmblyopiaClassifier, compute_pos_weight

# ── Setup ──────────────────────────────────────────────────────────────────

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

LABEL_NAMES = ["是否弱视"]


# ── Metrics ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate model on a given dataloader. Returns dict of metrics."""
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    for batch in dataloader:
        images = batch["images"].to(device)
        features = batch["features"].to(device)
        mask = batch["feature_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(images, features, mask)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0).squeeze(-1)  # (N,) after squeeze
    all_labels = torch.cat(all_labels, dim=0).squeeze(-1)  # (N,)
    all_probs = torch.sigmoid(all_logits)                   # probabilities
    all_preds = (all_probs >= 0.5).float()                  # hard predictions

    avg_loss = total_loss / len(dataloader.dataset)

    y_true = all_labels.numpy()
    y_pred = all_preds.numpy()
    y_prob = all_probs.numpy()

    metrics = {"loss": avg_loss}
    metrics["acc"] = accuracy_score(y_true, y_pred)
    metrics["prec"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["rec"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

    # AUC — only if both classes present
    if len(set(y_true)) > 1:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
        metrics["ap"] = average_precision_score(y_true, y_prob)
    else:
        metrics["auc"] = 0.5
        metrics["ap"] = 0.0

    return metrics, all_labels, all_probs


def format_metrics(metrics, prefix=""):
    """Pretty-print metrics."""
    line = (f"{prefix}loss={metrics['loss']:.4f}  "
            f"acc={metrics['acc']:.4f}  "
            f"prec={metrics['prec']:.4f}  "
            f"rec={metrics['rec']:.4f}  "
            f"f1={metrics['f1']:.4f}  "
            f"auc={metrics['auc']:.4f}")
    return line


# ── Training loop ──────────────────────────────────────────────────────────

def train():
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ── Data ──────────────────────────────────────────────────────────
    print("\nLoading data...")
    train_dl, val_dl, test_dl = get_dataloaders(
        batch_size=BATCH_SIZE, val_split=0.15, test_split=0.15,
    )

    # ── Model ─────────────────────────────────────────────────────────
    print("\nBuilding model...")
    model = MultiModalAmblyopiaClassifier()
    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # ── Loss (with class-balancing weight) ────────────────────────────
    pos_weight = compute_pos_weight(train_dl)
    pos_weight = pos_weight.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Optimizer & scheduler ─────────────────────────────────────────
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE,
                     weight_decay=WEIGHT_DECAY)  # Adam (aligned with paper ViT#5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                  patience=5)

    # ── Training loop ─────────────────────────────────────────────────
    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {"train": [], "val": []}

    for epoch in range(1, NUM_EPOCHS + 1):
        # ---- Training ----
        model.train()
        train_loss = 0.0

        for batch in train_dl:
            images = batch["images"].to(DEVICE)
            features = batch["features"].to(DEVICE)
            mask = batch["feature_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(images, features, mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        avg_train_loss = train_loss / len(train_dl.dataset)

        # ---- Validation ----
        val_metrics, _, _ = evaluate(model, val_dl, criterion, DEVICE)
        val_auc = val_metrics["auc"]

        # Scheduler step
        scheduler.step(val_auc)

        # Log
        print(f"\nEpoch {epoch:3d}/{NUM_EPOCHS} | "
              f"train_loss={avg_train_loss:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} | "
              f"val_auc={val_auc:.4f} | "
              f"val_f1={val_metrics['f1']:.4f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # Save history
        history["train"].append({"epoch": epoch, "loss": avg_train_loss})
        history["val"].append({k: v for k, v in val_metrics.items()
                               if isinstance(v, (int, float))})

        # Checkpoint best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "pos_weight": pos_weight,
            }, checkpoint_path)
            print(f"  ✓ Saved best model (val_auc={best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # ── Final evaluation on test set ──────────────────────────────────
    print("\n" + "=" * 70)
    print("Loading best model for test evaluation...")
    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pt"),
                            map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("\nTest set results:")
    test_metrics, test_labels, test_probs = evaluate(
        model, test_dl, criterion, DEVICE,
    )
    print(format_metrics(test_metrics, prefix="test_"))

    # Confusion matrix
    y_true = test_labels.numpy()
    y_pred = (test_probs >= 0.5).int().numpy()
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix (test set):")
    print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")

    # ── Save final results ────────────────────────────────────────────
    results = {
        "best_epoch": best_epoch,
        "best_val_auc": float(best_val_auc),
        "test_metrics": {k: float(v) if isinstance(v, (np.floating, float))
                         else v for k, v in test_metrics.items()},
        "label_names": LABEL_NAMES,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(LOG_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {os.path.join(LOG_DIR, 'results.json')}")

    return model, test_metrics


if __name__ == "__main__":
    train()
