"""Six-feature patient-level anisometropia experiment using the CSV splits."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model_pairwise_amblyopia import Block
from train_reflection_image import seed_everything

FEATURES = ["性别", "年龄", "右球镜", "右柱镜", "左球镜", "左柱镜"]
ROOT = Path(__file__).resolve().parent / "outputs"


class PairwiseAnisometropiaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.demo = nn.Sequential(Block(2, 24), nn.Linear(24, 16), nn.LayerNorm(16), nn.GELU())
        self.eye = nn.Sequential(Block(2, 48), nn.Linear(48, 32), nn.LayerNorm(32), nn.GELU())
        self.pair = nn.Sequential(Block(4, 96), nn.Linear(96, 48), nn.LayerNorm(48), nn.GELU())
        self.head = nn.Sequential(Block(128, 64), nn.Linear(64, 24), nn.LayerNorm(24), nn.GELU(), nn.Linear(24, 1))

    def forward(self, x):
        r, l = x[:, 2:4], x[:, 4:6]
        er, el = self.eye(r), self.eye(l)
        # A patient-level label should not depend on which eye is called R or L.
        pair = self.pair(torch.cat([(r-l).abs(), (r+l)/2], dim=1))
        return self.head(torch.cat([self.demo(x[:, :2]), pair, (er+el)/2, (er-el).abs()], dim=1)).squeeze(1)


def prepare(source):
    frame = pd.read_csv(source, keep_default_na=False, dtype=str)
    frame["屈光参差"] = frame["屈光参差"].str.strip()
    frame["split"] = frame["split"].str.strip()
    eligible = frame["屈光参差"].isin(["是", "否"]) & frame["split"].isin(["train", "val", "test"])
    selected = frame[eligible].copy()
    selected["patient_key"] = selected["patient_key"].str.strip()
    if selected.patient_key.eq("").any():
        raise ValueError("Missing patient ID")
    if selected.groupby("patient_key")["split"].nunique().max() > 1:
        raise ValueError("Patient overlaps across existing CSV splits")
    conflicting = selected.groupby("patient_key")["屈光参差"].nunique()
    conflict_keys = conflicting[conflicting > 1].index.tolist()
    excluded_conflicts = selected[selected.patient_key.isin(conflict_keys)]
    selected = selected[~selected.patient_key.isin(conflict_keys)].copy()
    for col in FEATURES[1:]:
        parsed = pd.to_numeric(selected[col], errors="coerce")
        bad = selected[col].str.strip().ne("") & ~np.isfinite(parsed)
        if bad.any():
            raise ValueError(f"Unrecognized numeric input: {col}")
        selected[col] = parsed
    selected["性别"] = selected["性别"].map({"女": 1., "男": 0.})
    if selected["性别"].isna().any():
        raise ValueError("Unknown gender encoding")
    selected["observed_count"] = selected[FEATURES].notna().sum(axis=1)
    ordered = selected.sort_values("observed_count", ascending=False, kind="stable")
    retained = ordered.drop_duplicates("patient_key", keep="first").sort_index()
    duplicates = selected[~selected.index.isin(retained.index)]
    x = retained[FEATURES].to_numpy(dtype=np.float32)
    y = retained["屈光参差"].map({"否": 0., "是": 1.}).to_numpy(dtype=np.float32)
    split = retained["split"].to_numpy()
    mean = np.zeros(6, dtype=np.float32)
    std = np.ones(6, dtype=np.float32)
    # Shared right/left statistics preserve eye-exchange invariance.
    for cols in ([0], [1], [2, 4], [3, 5]):
        values = x[split == "train"][:, cols].reshape(-1)
        values = values[np.isfinite(values)]
        if len(values):
            mean[cols] = values.mean(dtype=np.float64)
            std[cols] = max(values.std(dtype=np.float64), 1e-6)
            if std[cols[0]] <= 1e-6:
                std[cols] = 1.
    normalized = np.where(np.isfinite(x), (x-mean)/std, 0).astype(np.float32)
    assert np.isfinite(normalized).all()
    counts = {s: {"patients": int((split == s).sum()), "positive": int(y[split == s].sum()), "negative": int((y[split == s] == 0).sum())} for s in ("train", "val", "test")}
    assert all(v["positive"] > 0 and v["negative"] > 0 for v in counts.values())
    audit = {
        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_rows": len(frame), "source_label_counts": frame["屈光参差"].replace("", "unknown").value_counts().to_dict(),
        "explicit_label_existing_split_rows": int(eligible.sum()),
        "excluded_unknown_label_rows": int((~frame["屈光参差"].isin(["是", "否"])).sum()),
        "excluded_labeled_without_split": int((frame["屈光参差"].isin(["是", "否"]) & ~frame["split"].isin(["train", "val", "test"])).sum()),
        "excluded_conflicting_rows": excluded_conflicts[["record_id", "patient_key", "屈光参差"]].to_dict("records"),
        "removed_duplicate_rows": duplicates[["record_id", "patient_key", "split"]].to_dict("records"),
        "deduplication": "most observed input features per patient; ties keep earliest CSV row; labels not used for ranking",
        "split_counts": counts, "features": FEATURES, "normalization_mean": mean.tolist(), "normalization_std": std.tolist(),
        "normalization": "train-only observed values; shared bilateral sphere/cylinder statistics; missing values -> normalized zero; no mask input",
        "missing_by_split": {s: dict(zip(FEATURES, (~np.isfinite(x[split == s])).sum(0).tolist())) for s in counts},
        "retained_rows": retained[["record_id", "patient_key", "name", "split", "屈光参差"]].to_dict("records"),
    }
    return torch.from_numpy(normalized), torch.from_numpy(y), split, audit


def metrics(y, scores):
    pred = scores >= .5
    return {"threshold": .5, "auc": float(roc_auc_score(y, scores)), "acc": float(accuracy_score(y, pred)),
            "balanced_acc": float(balanced_accuracy_score(y, pred)), "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)), "f1": float(f1_score(y, pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/root/the_dataset 1.csv")
    ap.add_argument("--seed", type=int, default=46)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()
    seed_everything(args.seed)
    torch.set_num_threads(4)
    out = ROOT / f"anisometropia_six_features_csvsplit_seed{args.seed}"
    out.mkdir(exist_ok=True)
    x, y, splits, audit = prepare(Path(args.csv))
    (out / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(audit["split_counts"], ensure_ascii=False), flush=True)
    if args.prepare_only:
        return
    model = PairwiseAnisometropiaClassifier()
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(x), model(x[:, [0, 1, 4, 5, 2, 3]]), rtol=0, atol=0)
    train, val, test = [torch.from_numpy(np.flatnonzero(splits == s)) for s in ("train", "val", "test")]
    loader = DataLoader(TensorDataset(x[train], y[train]), batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    positive_weight = (len(train)-y[train].sum())/y[train].sum()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=.5)
    best, wait, history = -1., 0, []
    for epoch in range(1, args.epochs+1):
        model.train()
        total = 0.
        for bx, by in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += loss.item()*len(bx)
        model.eval()
        with torch.no_grad():
            vm = metrics(y[val].numpy(), model(x[val]).sigmoid().numpy())
        scheduler.step(vm["auc"])
        history.append({"epoch": epoch, "train_loss": total/len(train), "val_metrics": vm})
        print(f"epoch={epoch} train_loss={total/len(train):.4f} val_auc={vm['auc']:.4f} val_f1={vm['f1']:.4f}", flush=True)
        if vm["auc"] > best:
            best, wait, best_epoch, best_state = vm["auc"], 0, epoch, copy.deepcopy(model.state_dict())
        else:
            wait += 1
        if wait >= 15:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probabilities = model(x[test]).sigmoid().numpy()
        validation = metrics(y[val].numpy(), model(x[val]).sigmoid().numpy())
        torch.testing.assert_close(model(x[test]), model(x[test][:, [0, 1, 4, 5, 2, 3]]), rtol=0, atol=0)
    tm = metrics(y[test].numpy(), probabilities)
    result = {"task": "patient-level anisometropia", "target_column": "屈光参差", "target_mapping": {"否": 0, "是": 1},
              "model": type(model).__name__, "seed": args.seed, "input_dim": 6, "output_dim": 1, "features": FEATURES,
              "split_counts": audit["split_counts"], "best_epoch": best_epoch, "val_metrics": validation, "test_metrics": tm,
              "parameters": sum(p.numel() for p in model.parameters()), "pos_weight": positive_weight.item(), "history": history,
              "training": {"loss": "BCEWithLogitsLoss", "batch_size": 32, "lr": .001, "weight_decay": .0001, "patience": 15, "max_epochs": args.epochs,
                           "selection": "validation AUC", "ranking_loss": False, "mask_input": False, "image_input": False}, "audit": str(out / "data_audit.json")}
    (out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    torch.save({"model_state_dict": best_state, "normalization_mean": audit["normalization_mean"], "normalization_std": audit["normalization_std"], "features": FEATURES, "threshold": .5}, out / "best_model.pt")
    predictions = [{**audit["retained_rows"][i], "label": int(y[i]), "probability": float(p), "prediction": int(p >= .5)} for i, p in zip(test.tolist(), probabilities)]
    (out / "test_predictions.json").write_text(json.dumps(predictions, ensure_ascii=False, indent=2))
    print(json.dumps(tm, ensure_ascii=False, indent=2), flush=True)
    print(out, flush=True)


if __name__ == "__main__":
    main()
