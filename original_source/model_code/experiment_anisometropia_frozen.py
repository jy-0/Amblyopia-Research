"""Controlled frozen-ViT probes and training-only memorization diagnostics."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
import torch
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from config import IMG_MEAN, IMG_STD, VIT_MODEL_NAME
from train_anisometropia_images import ROOT, ReflectionAnisometropiaClassifier, prepare
from train_reflection_image import seed_everything


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class AuditedPairs(Dataset):
    """Use the verified examination, eye order and fundus crop, without augmentation."""
    def __init__(self, records, kind, size, normalization):
        self.records, self.kind, self.size = records, kind, size
        mean, std = ([.5]*3, [.5]*3) if normalization == "checkpoint" else (IMG_MEAN, IMG_STD)
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        eyes = self.records[index]["eyes"]
        sides = [eye.get("verified_side", eye.get("side") if self.kind == "raw" else None)
                 for eye in eyes]
        if sides != ["R", "L"]:
            raise ValueError("Expected verified right, left eye order")
        result = []
        for eye in eyes:
            prefix = "raw" if self.kind == "raw" else "reflection"
            path = Path(eye[prefix+"_path"])
            if sha256(path) != eye[prefix+"_sha256"]:
                raise ValueError(f"Image hash changed: {path}")
            with Image.open(path) as source:
                if self.kind == "raw":
                    image = ImageOps.exif_transpose(source).convert("RGB")
                    interpolation = Image.Resampling.BILINEAR
                else:
                    pixels = np.asarray(source.convert("L"))
                    if not set(np.unique(pixels).tolist()) <= {0, 1, 255}:
                        raise ValueError(f"Nonbinary reflection map: {path}")
                    image = Image.fromarray((pixels > 0).astype(np.uint8)*255).convert("RGB")
                    interpolation = Image.Resampling.NEAREST
                image = image.crop(eye["crop_box"])
                image = ImageOps.contain(image, (self.size, self.size), interpolation)
                canvas = Image.new("RGB", (self.size, self.size), 0)
                canvas.paste(image, ((self.size-image.width)//2, (self.size-image.height)//2))
            result.append(torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1).float()/255.)
        return (torch.stack(result)-self.mean)/self.std


def pair_features(eyes):
    if eyes.ndim != 3 or eyes.shape[1] != 2:
        raise ValueError("Expected [patients, 2, embedding_dim]")
    right, left = eyes[:, 0], eyes[:, 1]
    return np.concatenate([(right+left)/2, np.abs(right-left)], axis=1)


def evaluate(labels, probability, threshold=.5):
    pred = probability >= threshold
    return {"threshold": float(threshold), "auc": float(roc_auc_score(labels, probability)),
            "acc": float(accuracy_score(labels, pred)),
            "balanced_acc": float(balanced_accuracy_score(labels, pred)),
            "precision": float(precision_score(labels, pred, zero_division=0)),
            "recall": float(recall_score(labels, pred, zero_division=0)),
            "f1": float(f1_score(labels, pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(labels, pred, labels=[0, 1]).tolist()}


def validation_threshold(labels, scores):
    # Only validation scores enter this function. Ties favor proximity to 0.5.
    unique = np.unique(scores)
    candidates = np.unique(np.r_[0., .5, 1., (unique[:-1]+unique[1:])/2])
    return float(max(candidates, key=lambda t: (
        balanced_accuracy_score(labels, scores >= t), -abs(t-.5), -t)))


def load_encoder(name, weights, device):
    if name == "google":
        from transformers import ViTModel
        model = ViTModel.from_pretrained(VIT_MODEL_NAME, local_files_only=True)
        size, dim = 384, 768
    else:
        import timm
        model = timm.create_model("vit_small_patch14_reg4_dinov2", img_size=(392, 392),
                                  num_classes=0, pretrained=False)
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
        model.global_pool = "avg"
        size, dim = 392, 384
    model.requires_grad_(False).eval().to(device)
    return model, size, dim


def run_probe(audit, args, encoder_name, kind, normalization, root, device):
    out = root / f"{encoder_name}_{kind}_{normalization}"
    out.mkdir()
    seed_everything(args.seed)
    model, size, dim = load_encoder(encoder_name, args.weights, device)
    records = audit["records"]
    dataset = AuditedPairs(records, kind, size, normalization)
    embeddings = []
    with torch.inference_mode():
        for images in DataLoader(dataset, batch_size=4, shuffle=False):
            flat = images.flatten(0, 1).to(device)
            encoded = (model(pixel_values=flat, interpolate_pos_encoding=True).last_hidden_state[:, 0]
                       if encoder_name == "google" else model(flat))
            embeddings.append(encoded.cpu().numpy().reshape(len(images), 2, dim))
            if sum(len(e) for e in embeddings) % 100 == 0:
                print(f"{out.name}: encoded {sum(len(e) for e in embeddings)}/{len(records)}", flush=True)
    eyes = np.concatenate(embeddings)
    assert np.isfinite(eyes).all()
    x = pair_features(eyes).astype(np.float64)
    np.testing.assert_array_equal(x, pair_features(eyes[:, ::-1]).astype(np.float64))
    split = np.asarray([r["split"] for r in records])
    labels = np.asarray([r["label"] for r in records])
    train, val = split == "train", split == "val"
    scaler = StandardScaler().fit(x[train])
    z = scaler.transform(x)
    trials, best_auc, selected = [], -1., None
    for c in (.001, .01, .1, 1., 10.):
        head = LogisticRegression(C=c, class_weight="balanced", solver="lbfgs",
                                  max_iter=3000, tol=1e-8, random_state=args.seed)
        head.fit(z[train], labels[train])
        if int(head.n_iter_.max()) >= head.max_iter:
            raise RuntimeError("Logistic regression did not converge")
        val_scores = head.predict_proba(z[val])[:, 1]
        vm = evaluate(labels[val], val_scores)
        trials.append({"C": c, "train": evaluate(labels[train], head.predict_proba(z[train])[:, 1]),
                       "val": vm, "iterations": head.n_iter_.tolist()})
        if vm["auc"] > best_auc:
            selected, best_auc, best_c = head, vm["auc"], c
    threshold = validation_threshold(labels[val], selected.predict_proba(z[val])[:, 1])
    # Test scores are produced only after C and threshold have been fixed.
    scores = selected.predict_proba(z)[:, 1]
    logits = z @ selected.coef_[0] + selected.intercept_[0]
    np.testing.assert_allclose(scores, torch.sigmoid(torch.from_numpy(logits)).numpy(), atol=1e-12)
    result = {"task": "patient-level anisometropia", "encoder": encoder_name, "image_kind": kind,
              "seed": args.seed, "normalization": normalization, "normalization_mean": dataset.mean.flatten().tolist(),
              "normalization_std": dataset.std.flatten().tolist(), "split_counts": audit["split_counts"],
              "image_shape": ["B", 2, 3, size, size], "eye_feature_dim": dim, "pair_feature_dim": dim*2,
              "pair_fusion": "concatenate mean and absolute difference", "output_shape": ["B", 1],
              "encoder_parameters": sum(p.numel() for p in model.parameters()), "trainable_parameters": dim*2+1,
              "head": "L2-regularized logistic regression; balanced class weights; intercept included",
              "scaler_fit_split": "train", "selected_C": best_c, "C_selection": "validation AUC; ties choose smaller C",
              "validation_threshold": threshold, "threshold_selection": "validation balanced accuracy; ties closest to 0.5",
              "augmentation": False, "mask_input": False, "numerical_features": [], "trials": trials,
              "metrics_at_0p5": {s: evaluate(labels[split == s], scores[split == s]) for s in ("train", "val", "test")},
              "metrics_at_val_threshold": {s: evaluate(labels[split == s], scores[split == s], threshold) for s in ("train", "val", "test")},
              "weights": str(args.weights) if encoder_name != "google" else VIT_MODEL_NAME,
              "weights_sha256": sha256(args.weights) if encoder_name != "google" else None}
    np.savez_compressed(out / "features_and_head.npz", eyes=eyes, scaler_mean=scaler.mean_,
                        scaler_scale=scaler.scale_, coefficient=selected.coef_, intercept=selected.intercept_)
    save_json(out / "predictions.json", [{"record_id": r["record_id"], "patient_key": r["patient_key"],
        "split": r["split"], "label": r["label"], "probability": float(p), "prediction_0p5": int(p >= .5),
        "prediction_val_threshold": int(p >= threshold)} for r, p in zip(records, scores)])
    save_json(out / "results.json", result)
    print(f"{out.name}: " + json.dumps(result["metrics_at_0p5"]["test"]), flush=True)
    del model
    torch.cuda.empty_cache()
    return result


def memorization_check(audit, args, kind, root, device):
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    train = [r for r in audit["records"] if r["split"] == "train"]
    records = []
    for label in (0, 1):
        choices = [r for r in train if r["label"] == label]
        records.extend(choices[i] for i in rng.choice(len(choices), 8, replace=False))
    model = ReflectionAnisometropiaClassifier().to(device).eval()
    for layer in model.modules():
        if isinstance(layer, nn.Dropout):
            layer.p = 0.
    dataset = AuditedPairs(records, kind, 384, "checkpoint")
    with torch.no_grad():
        tokens = torch.cat([model.frozen_prefix(images.flatten(0, 1).to(device)).reshape(len(images), 2, 577, 768)
                            for images in DataLoader(dataset, batch_size=2)])
    truth = torch.tensor([r["label"] for r in records], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.)
    history = []
    for step in range(args.diagnostic_steps+1):
        model.eval()
        with torch.no_grad():
            logits = model.forward_from_tokens(tokens)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, truth).item()
            measured = evaluate(truth.cpu().numpy(), logits.sigmoid().cpu().numpy())
        history.append({"step": step, "loss": loss, "acc": measured["acc"], "auc": measured["auc"]})
        if step % 10 == 0:
            print(f"memorize {kind}: step={step} loss={loss:.4f} acc={measured['acc']:.3f}", flush=True)
        if (measured["acc"] == 1. and loss < .03) or step == args.diagnostic_steps:
            break
        model.train()
        optimizer.zero_grad()
        for start in range(0, len(records), 4):
            loss_batch = nn.functional.binary_cross_entropy_with_logits(
                model.forward_from_tokens(tokens[start:start+4]), truth[start:start+4])
            (loss_batch/4).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    result = {"kind": kind, "seed": args.seed, "normalization": "checkpoint", "records": [
        {k: r[k] for k in ("record_id", "patient_key", "split", "label")} for r in records],
        "passed": measured["acc"] == 1. and loss < .03, "final_metrics": measured,
        "final_loss": loss, "steps": step, "history": history,
        "settings": {"lr": 1e-3, "weight_decay": 0, "dropout": 0, "patients": 16,
                     "positive": 8, "train_only": True, "effective_batch_size": 16},
        "interpretation": "Training-only wiring/capacity diagnostic, not validation or generalization evidence"}
    save_json(root / f"memorization_{kind}.json", result)
    del model, tokens, optimizer
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--weights", type=Path, default=ROOT / "pretrained/retfoundgreen_statedict.pth")
    parser.add_argument("--output", type=Path, default=ROOT / "anisometropia_frozen_comparison_seed46")
    parser.add_argument("--diagnostic-steps", type=int, default=200)
    parser.add_argument("--skip-diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=False)
    audit, _ = prepare()
    for kind in ("raw", "binary"):
        old = json.loads((ROOT / f"anisometropia_{kind}_images_csvsplit_seed46/data_audit.json").read_text())
        if old["records"] != audit["records"]:
            raise ValueError("Patient cohort differs from prior image experiment")
    audit.update(preprocessing="verified fundus crop; aspect-preserving letterbox; raw bilinear, binary nearest; per-run size/normalization",
                 image_shape="see individual results", normalization_mean="see individual results",
                 normalization_std="see individual results")
    save_json(args.output / "data_audit.json", audit)
    save_json(args.output / "protocol.json", {"seed": args.seed, "C_grid": [.001, .01, .1, 1., 10.],
        "test_used_for_selection": False, "patient_splits": "unchanged", "torch": torch.__version__,
        "sklearn": sklearn.__version__, "python": platform.python_version(),
        "script_sha256": sha256(__file__), "weights_sha256": sha256(args.weights),
        "retfound_green_source": "https://github.com/justinengelmann/RETFound_Green",
        "preprocessing_note": "Retain audited crop/letterbox for comparison, not the official example's square distortion"})
    diagnostics = []
    if not args.skip_diagnostics:
        for kind in ("binary", "raw"):
            diagnostics.append(memorization_check(audit, args, kind, args.output, device))
    results = []
    for encoder, normalization in (("google", "legacy"), ("google", "checkpoint"), ("retfound-green", "checkpoint")):
        for kind in ("binary", "raw"):
            results.append(run_probe(audit, args, encoder, kind, normalization, args.output, device))
    save_json(args.output / "summary.json", {"diagnostics": diagnostics, "experiments": results})
    print(f"Complete: {args.output}", flush=True)


if __name__ == "__main__":
    main()
