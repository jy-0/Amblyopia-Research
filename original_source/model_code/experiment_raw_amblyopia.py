"""Change only the target of an existing frozen original-image experiment."""
import argparse
import copy
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from experiment_anisometropia_frozen import evaluate, pair_features, save_json, sha256
from experiment_raw_direct_concat import fit_classifier


ROOT = Path(__file__).resolve().parent / "outputs"
SPLITS = ("train", "val", "test")


def relabel_records(records, rows):
    by_id = {row["record_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate source examination IDs")
    if len({r["patient_key"] for r in records}) != len(records):
        raise ValueError("Repeated patient in baseline")
    if len({r["record_id"] for r in records}) != len(records):
        raise ValueError("Repeated examination in baseline")
    result = []
    for record in records:
        row = by_id[record["record_id"]]
        if row["patient_key"] != record["patient_key"]:
            raise ValueError("Source examination/patient mismatch")
        text = row["label_idx"].strip()
        if text not in ("0", "1"):
            raise ValueError("Unknown amblyopia label; cannot keep cohort unchanged")
        label = int(text)
        confirmations = [row.get(k, "").strip() for k in
                         ("满足单眼弱视标准", "是否满足单眼弱视标准")]
        if label == 0 and (row.get("弱视眼别", "").strip() or "是" in confirmations):
            raise ValueError("Negative amblyopia label conflicts with clinical fields")
        old_text = row["屈光参差"].strip()
        if old_text not in ("是", "否") or {"是": 1, "否": 0}[old_text] != record["label"]:
            raise ValueError("Baseline anisometropia label changed")
        item = copy.deepcopy(record)
        item["anisometropia_label"] = item["label"]
        item["anisometropia_label_text"] = item.pop("label_text", old_text)
        item["label"] = label
        item["label_text"] = "amblyopia" if label else "no amblyopia"
        item["label_source"] = "label_idx in the identical source examination"
        item["weak_eye_source"] = row.get("弱视眼别", "").strip()
        result.append(item)
    return result


def load_baseline(folder):
    audit = json.loads((folder / "data_audit.json").read_text())
    protocol = json.loads((folder / "protocol.json").read_text())
    checkpoint = folder / "retfound-green_raw_checkpoint"
    baseline = json.loads((checkpoint / "results.json").read_text())
    predictions = json.loads((checkpoint / "predictions.json").read_text())
    if baseline["task"] != "patient-level anisometropia" or baseline["pair_fusion"] != "concatenate mean and absolute difference":
        raise ValueError("Expected the original mean/absolute-difference anisometropia model")
    for path, digest in ((audit["source"], audit["source_sha256"]),
                         (audit["split_manifest"], audit["split_manifest_sha256"]),
                         (baseline["weights"], baseline["weights_sha256"])):
        if sha256(path) != digest:
            raise ValueError(f"Baseline input changed: {path}")
    if protocol["seed"] != baseline["seed"]:
        raise ValueError("Seed mismatch")
    records = audit["records"]
    if len(records) != len(predictions):
        raise ValueError("Prediction count mismatch")
    seen_images = set()
    for record, prediction in zip(records, predictions):
        if any(record[k] != prediction[k] for k in ("record_id", "patient_key", "split", "label")):
            raise ValueError("Baseline feature/prediction ordering mismatch")
        if [e.get("verified_side", e.get("side")) for e in record["eyes"]] != ["R", "L"]:
            raise ValueError("Expected right/left feature order")
        for eye in record["eyes"]:
            digest = eye["raw_sha256"]
            if digest in seen_images or sha256(eye["raw_path"]) != digest:
                raise ValueError("Duplicate or changed original image")
            seen_images.add(digest)
    cache = checkpoint / "features_and_head.npz"
    with np.load(cache, allow_pickle=False) as values:
        eyes = values["eyes"].copy()
        if eyes.shape != (len(records), 2, 384) or not np.isfinite(eyes).all():
            raise ValueError("Invalid frozen image features")
        x = pair_features(eyes).astype(np.float64)
        z = (x - values["scaler_mean"]) / values["scaler_scale"]
        scores = expit(z @ values["coefficient"][0] + values["intercept"][0])
        np.testing.assert_allclose(scores, [p["probability"] for p in predictions], rtol=0, atol=1e-12)
    return audit, protocol, baseline, predictions, eyes, cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=ROOT / "anisometropia_601_raw_frozen_seed42_only_seed")
    parser.add_argument("--output", type=Path, default=ROOT / "amblyopia_raw_frozen_seed42_same544")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    torch.set_num_threads(4)
    audit, protocol, baseline, old_predictions, eyes, cache = load_baseline(args.baseline)
    print("Validated original images, frozen features and baseline predictions", flush=True)
    with Path(audit["source"]).open(encoding="utf-8-sig", newline="") as handle:
        records = relabel_records(audit["records"], list(csv.DictReader(handle)))
    split = np.array([r["split"] for r in records])
    labels = np.array([r["label"] for r in records])
    counts = {}
    for name in SPLITS:
        c = Counter(labels[split == name].tolist())
        if not c[0] or not c[1]:
            raise ValueError(f"Both amblyopia classes required in {name}")
        counts[name] = {"patients": sum(c.values()), "positive": c[1], "negative": c[0]}
    x = pair_features(eyes).astype(np.float64)
    np.testing.assert_array_equal(x, pair_features(eyes[:, ::-1]).astype(np.float64))
    old_labels = np.array([r["anisometropia_label"] for r in records])
    reproduced = fit_classifier(x, old_labels, split, baseline["seed"])
    if reproduced["selected_C"] != baseline["selected_C"]:
        raise ValueError("Original C selection not reproduced")
    np.testing.assert_allclose(reproduced["validation_threshold"], baseline["validation_threshold"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(reproduced["probability"], [p["probability"] for p in old_predictions], rtol=0, atol=1e-12)
    print("Original anisometropia classifier reproduced exactly; fitting amblyopia head", flush=True)
    fitted = fit_classifier(x, labels, split, baseline["seed"])
    np.testing.assert_array_equal(fitted["scaler"].mean_, reproduced["scaler"].mean_)
    np.testing.assert_array_equal(fitted["scaler"].scale_, reproduced["scaler"].scale_)
    result = copy.deepcopy(baseline)
    for key in ("selected_C", "validation_threshold", "trials", "metrics_at_0p5", "metrics_at_val_threshold"):
        result[key] = fitted[key]
    result.update(task="patient-level amblyopia", split_counts=counts,
                  label_mapping={"0": "no amblyopia", "1": "amblyopia"},
                  label_column="label_idx", baseline_task="patient-level anisometropia",
                  baseline_results=str(args.baseline / "retfound-green_raw_checkpoint/results.json"),
                  frozen_feature_cache=str(cache), frozen_feature_cache_sha256=sha256(cache))
    updated_audit = copy.deepcopy(audit)
    updated_audit["records"], updated_audit["split_counts"] = records, counts
    updated_audit["task"] = "patient-level amblyopia"
    updated_audit["baseline_excluded_unknown_anisometropia_labels"] = updated_audit.pop("excluded_unknown_labels")
    updated_audit["excluded_unknown_amblyopia_labels"] = []
    updated_audit["cohort_policy"] = "Identical baseline 544 patients; do not add its 57 excluded patients despite the changed target"
    updated_protocol = copy.deepcopy(protocol)
    updated_protocol.update(
        task="patient-level amblyopia", only_change="replace patient-level anisometropia target by amblyopia",
        baseline_protocol=str(args.baseline / "protocol.json"),
        baseline_protocol_sha256=sha256(args.baseline / "protocol.json"),
        baseline_script_sha256=updated_protocol.pop("script_sha256"), script_sha256=sha256(__file__),
        fitting_helper_sha256=sha256(Path(__file__).with_name("experiment_raw_direct_concat.py")),
        scaler_and_head="fresh train-only fit; frozen per-eye image embeddings reused unchanged",
        unknown_labels="none in amblyopia labels of the unchanged baseline cohort; no additional patients admitted",
        device="cpu (cached frozen embeddings; no encoder update)",
        original_head_reproduced=True, identical_scaler=True, cohort_and_split_unchanged=True,
        class_weights="same balanced rule, recomputed from training amblyopia labels",
        hyperparameters="same C search and validation threshold rules; selected values may change with target",
        all_input_image_hashes_verified=True,
    )
    predictions = [{"record_id": r["record_id"], "patient_key": r["patient_key"],
                    "split": r["split"], "label": r["label"],
                    "anisometropia_label": r["anisometropia_label"], "probability": float(p),
                    "prediction_0p5": int(p >= .5),
                    "prediction_val_threshold": int(p >= fitted["validation_threshold"])}
                   for r, p in zip(records, fitted["probability"])]
    for name in SPLITS:
        chosen = [p for p in predictions if p["split"] == name]
        for key, threshold in (("metrics_at_0p5", .5), ("metrics_at_val_threshold", fitted["validation_threshold"])):
            measured = evaluate(np.array([p["label"] for p in chosen]),
                                np.array([p["probability"] for p in chosen]), threshold)
            if measured != result[key][name]:
                raise ValueError("Prediction/metric mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "data_audit.json", updated_audit)
    save_json(args.output / "protocol.json", updated_protocol)
    save_json(args.output / "results.json", result)
    save_json(args.output / "predictions.json", predictions)
    np.savez_compressed(args.output / "features_and_head.npz", eyes=eyes,
                        scaler_mean=fitted["scaler"].mean_, scaler_scale=fitted["scaler"].scale_,
                        coefficient=fitted["head"].coef_, intercept=fitted["head"].intercept_)
    for name in SPLITS:
        print(name, counts[name], result["metrics_at_0p5"][name], flush=True)
    print(f"Completed: {args.output}", flush=True)


if __name__ == "__main__":
    main()
