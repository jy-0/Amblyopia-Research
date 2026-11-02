"""Create a patient-stratified 400/100/101 split and audit reflection maps."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

from dataset_reflection_images import FULL_SUFFIX, REFLECTION_ROOT, image_index


ROOT = Path(__file__).resolve().parent / "outputs"
INVENTORY = ROOT / "reports/anisometropia_inventory_20261010"
OUTPUT = ROOT / "splits/anisometropia_601_seed46_400_100_101"
SOURCE = Path("/root/the_dataset 1.csv")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_split(records, seed=46):
    if len(records) != 601 or len({r["patient_key"] for r in records}) != 601:
        raise ValueError("Expected 601 unique patient keys")
    labels = np.array([r["label"] if r["label"] is not None else -1 for r in records])
    indices = np.arange(len(records))
    train, holdout = train_test_split(indices, train_size=400, random_state=seed, stratify=labels)
    val, test = train_test_split(holdout, train_size=100, test_size=101,
                                 random_state=seed, stratify=labels[holdout])
    assignment = {int(i): name for name, ix in (("train", train), ("val", val), ("test", test)) for i in ix}
    if len(assignment) != 601:
        raise ValueError("Split coverage error")
    return assignment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / f"splits/anisometropia_601_seed{args.seed}_400_100_101"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing split: {output}")
    previous = json.loads((INVENTORY / "summary.json").read_text())
    if sha256(SOURCE) != previous["source_sha256"]:
        raise ValueError("Source changed since inventory")
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        source_rows = {r["record_id"]: r for r in reader}
    available = [r for r in json.loads((INVENTORY / "record_details.json").read_text())
                 if r["complete_distinct_raw_pair"]]
    if len(available) != 601 or len({r["patient_key"] for r in available}) != 601:
        raise ValueError("Complete-image records no longer map one-to-one to 601 patients")
    people = {r["patient_key"]: r for r in json.loads((INVENTORY / "patient_details.json").read_text())}
    available.sort(key=lambda r: r["patient_key"])
    old_manifest = json.loads((ROOT / "reflection_image_only_global_seed46/data/manifest.json").read_text())
    verified = {r["record_id"]: r for r in old_manifest["records"]}
    mask_index = image_index(REFLECTION_ROOT, FULL_SUFFIX)
    file_cache = {}

    def reflection_info(path):
        path = Path(path)
        key = str(path)
        if key not in file_cache:
            try:
                with Image.open(path) as image:
                    image.load()
                    pixels = np.asarray(image.convert("L"))
                    frequencies = np.bincount(pixels.reshape(-1), minlength=256)
                    values = np.flatnonzero(frequencies).tolist()
                    file_cache[key] = {"path": key, "status": "ok" if set(values) <= {0, 1, 255} else "not_binary",
                        "size": list(image.size), "pixel_values": values,
                        "nonzero_pixels": int(np.count_nonzero(pixels)), "sha256": sha256(path)}
            except Exception as exc:
                file_cache[key] = {"path": key, "status": "unreadable", "error": str(exc)}
        return dict(file_cache[key])

    def find_reflection(raw_path, raw_size):
        paths = mask_index.get(Path(raw_path).stem.casefold(), [])
        if not paths:
            return {"status": "missing"}
        inspected = [reflection_info(p) for p in paths]
        valid = [r for r in inspected if r["status"] == "ok"]
        if not valid:
            return {"status": "invalid", "candidates": inspected}
        if len({r["sha256"] for r in valid}) > 1:
            return {"status": "ambiguous", "candidates": valid}
        result = valid[0]
        if result["size"] != raw_size:
            result["status"] = "size_mismatch"
        return result

    records, label_alignment_warnings = [], []
    for item in available:
        row = source_rows[item["record_id"]]
        label_text = row["屈光参差"].strip()
        # Do not transfer another visit's diagnosis to the selected examination.
        if (label_text or "unknown") != people[item["patient_key"]]["label"]:
            if label_text:
                raise ValueError(f"Conflicting explicit visit label: {item['record_id']}")
            label_alignment_warnings.append({"patient_key": item["patient_key"], "record_id": item["record_id"],
                "patient_summary_label": people[item["patient_key"]]["label"], "selected_visit_label": None,
                "policy": "retain unknown; do not borrow the other visit's label"})
        sides = {side: dict(item["eyes"][side]) for side in ("right", "left")}
        known = verified.get(item["record_id"])
        if known:
            if {e["raw_sha256"] for e in known["eyes"]} != {e["sha256"] for e in sides.values()}:
                raise ValueError("Prior eye audit refers to different images")
            sides = {side: next(e for e in sides.values() if e["sha256"] == old["raw_sha256"])
                     for side, old in zip(("right", "left"), known["eyes"])}
            if [e["verified_side"] for e in known["eyes"]] != ["R", "L"]:
                raise ValueError("Unexpected prior verified eye order")
        eyes = []
        for side in ("right", "left"):
            raw = sides[side]
            path = Path(raw["path"])
            if sha256(path) != raw["sha256"]:
                raise ValueError(f"Original image changed: {path}")
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image)
                image.load()
                raw_size = list(image.size)
            reflection = find_reflection(path, raw_size)
            if known:
                old_eye = next(e for e in known["eyes"] if e["raw_sha256"] == raw["sha256"])
                if reflection.get("sha256") != old_eye["reflection_sha256"]:
                    raise ValueError("Prior reflection map changed")
            eyes.append({"side": "R" if side == "right" else "L", "raw_path": str(path),
                         "raw_sha256": raw["sha256"], "raw_size": raw_size, "reflection": reflection})
        records.append({"record_id": item["record_id"], "patient_key": item["patient_key"], "name": row["name"],
                        "label": {"是": 1, "否": 0}.get(label_text), "label_text": label_text or "未知",
                        "previous_split": row["split"], "eye_order_status": "previously_verified" if known else "csv_not_reverified",
                        "eyes": eyes, "raw_pair_available": True,
                        "binary_pair_available": all(e["reflection"]["status"] == "ok" for e in eyes)})
        if len(records) % 100 == 0:
            print(f"Audited {len(records)}/601 patients", flush=True)
    assignment = make_split(records, seed=args.seed)
    if assignment != make_split(records, seed=args.seed):
        raise AssertionError("Seed does not reproduce the split")
    image_owners = defaultdict(set)
    split_owners = defaultdict(set)
    for i, record in enumerate(records):
        record["split"] = assignment[i]
        for eye in record["eyes"]:
            for kind, digest in (("raw", eye["raw_sha256"]), ("binary", eye["reflection"].get("sha256"))):
                if not digest or (kind == "binary" and not eye["reflection"].get("nonzero_pixels")):
                    continue
                image_owners[(kind, digest)].add(record["patient_key"])
                split_owners[(kind, digest)].add(record["split"])
    duplicate_owners = [{"kind": k[0], "sha256": k[1], "patients": sorted(v)}
                        for k, v in image_owners.items() if len(v) > 1]
    if any(len(s) > 1 for s in split_owners.values()):
        raise ValueError("Duplicate nonempty images across splits; grouped splitting required")

    def counts(items):
        c = Counter(r["label"] for r in items)
        known = c[0]+c[1]
        return {"patients": len(items), "positive": c[1], "negative": c[0], "unknown": c[None],
                "positive_fraction_among_known": c[1]/known if known else None,
                "raw_pairs": len(items), "binary_pairs": sum(r["binary_pair_available"] for r in items),
                "supervised_raw": known,
                "supervised_binary": sum(r["label"] is not None and r["binary_pair_available"] for r in items),
                "previously_verified_eye_pairs": sum(r["eye_order_status"] == "previously_verified" for r in items)}

    all_reflections = [e["reflection"] for r in records for e in r["eyes"]]
    summary = {"source": str(SOURCE), "source_sha256": previous["source_sha256"], "seed": args.seed,
               "split_sizes": {"train": 400, "val": 100, "test": 101},
               "policy": "patient-level; stratify positive/negative/unknown; extra patient assigned to test",
               "unknown_label_policy": "Label remains null; excluded from supervised loss and classification metrics until adjudicated",
               "label_alignment_warnings": label_alignment_warnings,
               "label_alignment": "Use selected image examination's label, not patient-wide aggregation across visits",
               "old_experiment_policy": "new partition; prior experiments preserved; no model trained on new split yet",
               "total": counts(records), "by_split": {s: counts([r for r in records if r["split"] == s]) for s in ("train", "val", "test")},
               "black_white_definition": "provided full-field binary reflection maps, not vessels or generated masks",
               "reflection_root": str(REFLECTION_ROOT), "reflection_eye_statuses": dict(Counter(r["status"] for r in all_reflections)),
               "empty_binary_maps": sum(r["status"] == "ok" and r["nonzero_pixels"] == 0 for r in all_reflections),
               "duplicate_image_owners": duplicate_owners, "cross_split_nonempty_image_duplicates": 0,
               "eye_side_caveat": "Reuse prior image-based eye verification where available; remaining records retain CSV assignment, not freshly verified",
               "csv_image_path_policy": "Preserve original image/feature columns; use *_raw_path_checked or image_pairs.json for resolved images and corrected eye order",
               "new_csv": str(output / "dataset_601.csv")}
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (output / "image_pairs.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))
    missing = [r for r in records if not r["binary_pair_available"]]
    (output / "missing_binary_pairs.json").write_text(json.dumps(missing, ensure_ascii=False, indent=2))
    with (output / "patient_split.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["姓名", "划分", "屈光参差", "双眼黑白图齐全"])
        writer.writeheader()
        writer.writerows({"姓名": r["patient_key"], "划分": r["split"], "屈光参差": r["label_text"],
                          "双眼黑白图齐全": "是" if r["binary_pair_available"] else "否"} for r in records)
    added = ["previous_csv_split", "anisometropia_label_known", "raw_pair_available_checked", "binary_pair_available_checked",
             "right_raw_path_checked", "left_raw_path_checked", "right_binary_path_checked", "left_binary_path_checked", "eye_order_status_checked"]
    with (output / "dataset_601.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields+added)
        writer.writeheader()
        for record in records:
            row = dict(source_rows[record["record_id"]])
            row.update(previous_csv_split=row["split"], split=record["split"], anisometropia_label_known=record["label"] is not None,
                       raw_pair_available_checked=True, binary_pair_available_checked=record["binary_pair_available"],
                       eye_order_status_checked=record["eye_order_status"])
            for side, eye in zip(("right", "left"), record["eyes"]):
                row[side+"_raw_path_checked"] = eye["raw_path"]
                row[side+"_binary_path_checked"] = eye["reflection"].get("path", "") if eye["reflection"]["status"] == "ok" else ""
            writer.writerow(row)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
