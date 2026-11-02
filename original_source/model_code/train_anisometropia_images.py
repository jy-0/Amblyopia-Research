"""Patient-level anisometropia from paired binary reflection images only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from config import IMG_MEAN, IMG_STD
from model import MLPBlock
from model_reflection_image_pair import ReflectionImagePairClassifier
from train_anisometropia import metrics
from train_reflection_image import seed_everything

ROOT = Path(__file__).resolve().parent / "outputs"
IMAGE_BASE = ROOT / "reflection_image_only_global_seed46/data"


class ReflectionAnisometropiaClassifier(ReflectionImagePairClassifier):
    def __init__(self):
        super().__init__()
        self.pair = nn.Sequential(MLPBlock(64, 96, .3), nn.Linear(96, 48), nn.LayerNorm(48), nn.GELU())
        del self.laterality

    def forward_from_tokens(self, tokens, auxiliary=None):
        if auxiliary is not None:
            raise ValueError("Image-only experiment accepts no tabular features")
        b, eyes, length, width = tokens.shape
        if eyes != 2:
            raise ValueError("Expected two images per patient")
        x = self.block_forward(self.layers[-1], tokens.reshape(b*2, length, width))
        cls = self.image_encoder.vit.layernorm(x)[:, 0]
        r, l = self.projector(cls).reshape(b, 2, 32).unbind(1)
        mean, absolute = (r+l)/2, (r-l).abs()
        pair = self.pair(torch.cat([absolute, mean], 1))
        fused = self.fusion(torch.cat([pair, mean, absolute], 1))
        return self.severity(fused).squeeze(1)


class ImagePairs(Dataset):
    def __init__(self, pixels, cache_indices, mean=IMG_MEAN, std=IMG_STD):
        self.pixels = pixels
        self.indices = cache_indices
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        pair = torch.from_numpy(self.pixels[self.indices[index]].copy()).float().unsqueeze(1).repeat(1, 3, 1, 1)/255.
        return (pair-self.mean)/self.std


class RawImagePairs(Dataset):
    def __init__(self, records, mean=IMG_MEAN, std=IMG_STD):
        self.records = records
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        eyes = []
        for eye in self.records[index]["eyes"]:
            path = Path(eye["raw_path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != eye["raw_sha256"]:
                raise ValueError(f"Original image changed: {path}")
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB").crop(eye["crop_box"])
                resized = ImageOps.contain(image, (384, 384), Image.Resampling.BILINEAR)
                canvas = Image.new("RGB", (384, 384), 0)
                canvas.paste(resized, ((384-resized.width)//2, (384-resized.height)//2))
            pixels = np.asarray(canvas).copy()
            if pixels.std() < 1:
                raise ValueError(f"Blank or nearly constant raw image: {path}")
            eyes.append(torch.from_numpy(pixels).permute(2, 0, 1).float()/255.)
        return (torch.stack(eyes)-self.mean)/self.std


def prepare():
    previous = ROOT / "anisometropia_six_features_csvsplit_seed46/data_audit.json"
    clinical_audit = json.loads(previous.read_text())
    source = Path(clinical_audit["source"])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == clinical_audit["source_sha256"]
    frame = {r["record_id"]: r for r in csv.DictReader(source.open(encoding="utf-8-sig"))}
    cached_manifest = json.loads((IMAGE_BASE / "manifest.json").read_text())
    cached = {r["record_id"]: (i, r) for i, r in enumerate(cached_manifest["records"])}
    records, excluded, image_indices, hashes = [], [], [], {}
    for row in clinical_audit["retained_rows"]:
        if row["record_id"] not in cached:
            excluded.append({**row, "reason": "no corresponding binary reflection pair for this examination"})
            continue
        i, image_row = cached[row["record_id"]]
        current = frame[row["record_id"]]
        assert current["屈光参差"] == row["屈光参差"]
        assert current["split"] == row["split"] == image_row["split"]
        assert current["patient_key"] == row["patient_key"] == image_row["patient_key"]
        assert {Path(current[s+"_path"]).name for s in ("left", "right")} == {Path(e["raw_path"]).name for e in image_row["eyes"]}
        assert [e["verified_side"] for e in image_row["eyes"]] == ["R", "L"]
        for eye in image_row["eyes"]:
            for key in ("raw_sha256", "reflection_sha256"):
                old_split = hashes.setdefault((key, eye[key]), row["split"])
                if eye["nonzero_pixels_after_resize"] and old_split != row["split"]:
                    raise ValueError("Duplicate image across splits")
        records.append({**row, "label": int(row["屈光参差"] == "是"), "eyes": image_row["eyes"], "image_cache_index": i})
        image_indices.append(i)
    assert len({r["patient_key"] for r in records}) == len(records)
    counts = {s: {"patients": sum(r["split"] == s for r in records),
                  "positive": sum(r["split"] == s and r["label"] == 1 for r in records),
                  "negative": sum(r["split"] == s and r["label"] == 0 for r in records)} for s in ("train", "val", "test")}
    assert all(c["positive"] and c["negative"] for c in counts.values())
    audit = {"source": str(source), "source_sha256": clinical_audit["source_sha256"], "clinical_cohort_audit": str(previous),
             "image_cache_manifest": str(IMAGE_BASE / "manifest.json"), "split_counts": counts,
             "excluded_missing_image_pairs": excluded, "records": records, "numerical_model_inputs": [],
             "image_shape": ["B", 2, 3, 384, 384], "normalization_mean": IMG_MEAN, "normalization_std": IMG_STD,
             "preprocessing": "reuse audited binary masks: field crop, nearest aspect-preserving resize, centered padding, three replicated channels",
             "augmentation": False, "target": "屈光参差", "missing_label_as_negative": False}
    return audit, image_indices


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    scores, labels = [], []
    for tokens, truth in loader:
        scores.append(model.forward_from_tokens(tokens.to(device)).sigmoid().cpu())
        labels.append(truth)
    return torch.cat(labels).numpy(), torch.cat(scores).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image-kind", choices=["binary", "raw"], default="binary")
    parser.add_argument("--normalization", choices=["checkpoint", "legacy"], default="checkpoint")
    args = parser.parse_args()
    seed_everything(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out = ROOT / f"anisometropia_{args.image_kind}_images_{args.normalization}_csvsplit_seed{args.seed}"
    if (out / "results.json").exists():
        raise FileExistsError(f"Completed experiment already exists: {out}")
    out.mkdir(exist_ok=True)
    audit, image_indices = prepare()
    audit["image_kind"] = args.image_kind
    mean, std = ([.5]*3, [.5]*3) if args.normalization == "checkpoint" else (IMG_MEAN, IMG_STD)
    audit.update(normalization=args.normalization, normalization_mean=mean, normalization_std=std)
    if args.image_kind == "raw":
        previous_audit = json.loads((ROOT / f"anisometropia_binary_images_csvsplit_seed{args.seed}/data_audit.json").read_text())
        assert audit["records"] == previous_audit["records"]
        audit["preprocessing"] = "raw RGB, same audited field crop as binary masks, bilinear aspect-preserving resize to 384, centered black padding; normalization recorded separately"
        audit["cohort_policy"] = "identical examinations/labels/splits to binary image experiment; exclusions retained for paired comparison"
        image_dataset = RawImagePairs(audit["records"], mean, std)
    else:
        pixels = np.load(IMAGE_BASE / "images_uint8.npy", mmap_mode="r")
        image_dataset = ImagePairs(pixels, image_indices, mean, std)
    (out / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(audit["split_counts"]), flush=True)
    model = ReflectionAnisometropiaClassifier().to(device).eval()
    cached = []
    image_loader = DataLoader(image_dataset, batch_size=2)
    with torch.no_grad():
        for images in image_loader:
            b = len(images)
            cached.append(model.frozen_prefix(images.flatten(0, 1).to(device)).cpu().reshape(b, 2, 577, 768))
    tokens = torch.cat(cached)
    del cached
    torch.save(tokens[:2].clone(), out / "verification_tokens.pt")
    with torch.no_grad():
        probe = tokens[:2].to(device)
        torch.testing.assert_close(model.forward_from_tokens(probe), model.forward_from_tokens(probe.flip(1)), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(model(next(iter(image_loader)).to(device)), model.forward_from_tokens(probe), atol=1e-5, rtol=1e-5)
    indices = {s: [i for i, r in enumerate(audit["records"]) if r["split"] == s] for s in ("train", "val", "test")}
    labels = torch.tensor([r["label"] for r in audit["records"]], dtype=torch.float32)
    datasets = {s: TensorDataset(tokens[ix], labels[ix]) for s, ix in indices.items()}
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {s: DataLoader(ds, batch_size=2, shuffle=(s == "train"), generator=generator if s == "train" else None) for s, ds in datasets.items()}
    train = loaders["train"]
    c = audit["split_counts"]["train"]
    pos_weight = c["negative"]/c["positive"]
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=.5)
    best, wait, history = -1., 0, []
    for epoch in range(1, args.epochs+1):
        model.train()
        optimizer.zero_grad()
        total = 0.
        for step, (batch, truth) in enumerate(train):
            logits = model.forward_from_tokens(batch.to(device))
            loss = loss_fn(logits, truth.to(device))
            # Normalize even the final incomplete accumulation group by its actual size.
            window_start = (step//16)*32
            window_size = min(32, len(datasets["train"])-window_start)
            (loss*len(truth)/window_size).backward()
            total += loss.item()*len(truth)
            if (step+1) % 16 == 0 or step+1 == len(train):
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                optimizer.zero_grad()
        truth, score = predict(model, loaders["val"], device)
        vm = metrics(truth, score)
        scheduler.step(vm["auc"])
        history.append({"epoch": epoch, "train_loss": total/len(datasets["train"]), "val_metrics": vm})
        print(f"epoch={epoch} loss={history[-1]['train_loss']:.4f} val_auc={vm['auc']:.4f} val_f1={vm['f1']:.4f}", flush=True)
        if vm["auc"] > best:
            best, wait, best_epoch = vm["auc"], 0, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= 15:
            break
    model.load_state_dict(best_state)
    truth, score = predict(model, loaders["test"], device)
    tm = metrics(truth, score)
    vy, vs = predict(model, loaders["val"], device)
    with torch.no_grad():
        original = model.forward_from_tokens(probe)
        torch.testing.assert_close(original, model.forward_from_tokens(probe.flip(1)), rtol=1e-6, atol=1e-6)
    result = {"task": "patient-level anisometropia", "model": type(model).__name__, "seed": args.seed,
              "input": "two raw RGB fundus images only" if args.image_kind == "raw" else "two binary full-field reflection images only",
              "image_kind": args.image_kind, "image_shape": ["B", 2, 3, 384, 384], "output_shape": ["B"],
              "normalization": args.normalization, "normalization_mean": mean, "normalization_std": std,
              "numerical_features": [], "mask_input": False, "split_counts": audit["split_counts"], "best_epoch": best_epoch,
              "val_metrics": metrics(vy, vs), "test_metrics": tm, "history": history,
              "parameters": sum(p.numel() for p in model.parameters()), "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "training": {"loss": "BCEWithLogitsLoss", "pos_weight": pos_weight, "batch_size": 2, "accumulation": 16,
                           "effective_batch_size": 32, "lr": 1e-4, "max_epochs": args.epochs, "patience": 15, "selection": "validation AUC", "ranking_loss": False},
              "excluded_image_pairs": len(audit["excluded_missing_image_pairs"]), "verification_logits": original.cpu().tolist()}
    (out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    torch.save(best_state, out / "best_model.pt")
    predictions = [{"record_id": audit["records"][i]["record_id"], "patient_key": audit["records"][i]["patient_key"],
                    "label": int(truth[j]), "probability": float(score[j]), "prediction": int(score[j] >= .5)} for j, i in enumerate(indices["test"])]
    (out / "test_predictions.json").write_text(json.dumps(predictions, ensure_ascii=False, indent=2))
    print(json.dumps(tm, indent=2), flush=True)
    print(out, flush=True)


if __name__ == "__main__":
    main()
