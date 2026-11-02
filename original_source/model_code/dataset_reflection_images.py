"""Paired binary reflection images on the fixed the_dataset.csv cohort."""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy import ndimage

ROOT = Path("/root/hospital")
REFLECTION_ROOT = ROOT / "dataset/reflective_regions_20260910"
FULL_SUFFIX = "_TWS_probabilities_Slice1_反光Mask_P0p50"
MACULA_SUFFIX = "_reflection_inside_fovea_roi"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def image_index(root, suffix=None):
    index = defaultdict(list)
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            continue
        if suffix is not None:
            if path.stem.endswith(suffix) and path.stat().st_size:
                index[path.stem[:-len(suffix)].casefold()].append(path)
        else:
            index[path.name.casefold()].append(path)
    return index


def read_marker(image):
    w, h = image.size
    corner = image.crop((int(w*.90), int(h*.85), w, int(h*.953))).convert("L")
    data = io.BytesIO()
    ImageOps.invert(corner).save(data, format="PNG")
    process = subprocess.run(
        ["tesseract", "stdin", "stdout", "--psm", "6", "-c", "tessedit_char_whitelist=RL"],
        input=data.getvalue(), capture_output=True, check=True, timeout=15,
        env={**os.environ, "OMP_THREAD_LIMIT": "1"},
    )
    letters = re.findall(r"[RL]", process.stdout.decode("utf-8"))
    return letters[0] if len(letters) == 1 else ""


def fundus_box(image):
    p = np.asarray(image).astype(np.int16)
    colored = (p.max(2)-p.min(2) > 12) & (p[:, :, 0] > 25)
    components, count = ndimage.label(colored)
    if not count:
        raise ValueError("No fundus field detected")
    areas = np.bincount(components.ravel())
    areas[0] = 0
    ys, xs = np.where(components == areas.argmax())
    pad = round(min(image.size)*.01)
    return [max(0, int(xs.min())-pad), max(0, int(ys.min())-pad),
            min(image.width, int(xs.max())+pad+1), min(image.height, int(ys.max())+pad+1)]


def prepare_eye(job):
    raw, reflection, disc, size = job
    with Image.open(raw) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        marker = read_marker(image)
        box = fundus_box(image)
        original_size = image.size
    disc_x = None
    if disc is not None:
        with Image.open(disc) as im:
            disc_pixels = np.asarray(im.convert("L")) > 0
        _, xs = np.where(disc_pixels)
        if len(xs):
            disc_x = float(xs.mean()/disc_pixels.shape[1])
    disc_side = "R" if disc_x is not None and disc_x > .55 else "L" if disc_x is not None and disc_x < .45 else ""
    if marker and disc_side and marker != disc_side:
        raise ValueError(f"L/R marker conflicts with disc position: {raw}")
    side = marker or disc_side
    if not side:
        raise ValueError(f"Cannot verify eye side: {raw}")
    with Image.open(reflection) as image:
        if image.size != original_size:
            raise ValueError(f"Reflection and original sizes differ: {reflection}")
        pixels = np.asarray(image.convert("L"))
        values = np.unique(pixels)
        if not set(values.tolist()) <= {0, 1, 255}:
            raise ValueError(f"Nonbinary reflection image: {reflection}: {values[:10]}")
        image = Image.fromarray((pixels > 0).astype(np.uint8)*255).crop(box)
        thumb = ImageOps.contain(image, (size, size), Image.Resampling.NEAREST)
        canvas = Image.new("L", (size, size), 0)
        canvas.paste(thumb, ((size-thumb.width)//2, (size-thumb.height)//2))
    return np.asarray(canvas).copy(), {
        "raw_path": str(raw), "reflection_path": str(reflection), "raw_sha256": sha256(raw),
        "reflection_sha256": sha256(reflection), "embedded_marker": marker,
        "disc_x": disc_x, "verified_side": side, "crop_box": box,
        "nonzero_pixels_after_resize": int(np.count_nonzero(canvas)),
    }


def prepare_dataset(csv_path, out, scope="global", size=384):
    out.mkdir(parents=True, exist_ok=True)
    cols = ["record_id", "patient_key", "name", "split", "right_path", "left_path", "label_idx", "弱视眼别",
            "has_image_pair", "has_macula_reflection", "has_global_reflection"]
    frame = pd.read_csv(csv_path, usecols=cols)
    selected = frame["split"].isin(["train", "val", "test"])
    for column in ("has_image_pair", "has_macula_reflection", "has_global_reflection"):
        selected &= frame[column].astype(str).str.lower().isin(["true", "1", "1.0"])
    frame = frame.loc[selected].copy()
    if frame.patient_key.duplicated().any():
        raise ValueError("The common cohort must contain one record per patient")
    raw_index = {
        "control": image_index(ROOT / "dataset/weak_amblyopia_20250829_raw/duizhaohuanzhe"),
        "weak": image_index(ROOT / "dataset/weak_amblyopia_20250829_raw/ruoshihuanzhe"),
    }
    mask_index = image_index(REFLECTION_ROOT, FULL_SUFFIX if scope == "global" else MACULA_SUFFIX)
    disc_index = defaultdict(list)
    for folder in (ROOT / "dataset/seg/duizhaohuanzhe/disc", ROOT / "dataset/seg/ruoshihuanzhe/disc"):
        for p in folder.glob("*.png"):
            disc_index[p.stem.casefold()].append(p)
    records, jobs = [], []
    for row in frame.to_dict("records"):
        if int(row["label_idx"]) == 0:
            labels, stratum = [0, 0], 0
        else:
            side = str(row["弱视眼别"]).strip().lower()
            if side in {"右", "右眼", "r", "right"}:
                labels, stratum = [1, 0], 3
            elif side in {"左", "左眼", "l", "left"}:
                labels, stratum = [0, 1], 2
            else:
                raise ValueError(f"Unresolved clinical eye label: {row['record_id']}")
        records.append({k: str(row[k]) for k in ("record_id", "patient_key", "name", "split")})
        records[-1].update(labels=labels, stratum=stratum)
        raw_group = "control" if stratum == 0 else "weak"
        for side in ("right", "left"):
            source = Path(str(row[f"{side}_path"]))
            raw_matches = [source] if source.is_file() else raw_index[raw_group][source.name.casefold()]
            reflections = mask_index[source.stem.casefold()]
            discs = disc_index[source.stem.casefold()]
            if len(raw_matches) != 1 or len(reflections) != 1 or len(discs) > 1:
                raise ValueError(f"Missing or ambiguous image pair: {row['record_id']}/{side}")
            jobs.append((raw_matches[0], reflections[0], discs[0] if discs else None, size))
    images = np.empty((len(records), 2, size, size), dtype=np.uint8)
    processed = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, result in enumerate(pool.map(prepare_eye, jobs)):
            processed.append(result)
            if (i+1) % 100 == 0:
                print(f"Prepared {i+1}/{len(jobs)} reflection images", flush=True)
    corrections = []
    hash_split = {}
    for i, record in enumerate(records):
        pair = processed[2*i:2*i+2]
        if {x[1]["verified_side"] for x in pair} != {"R", "L"}:
            raise ValueError(f"Images are not opposite eyes: {record['record_id']}")
        swapped = pair[0][1]["verified_side"] != "R"
        if swapped:
            pair.reverse()
            corrections.append({k: record[k] for k in ("record_id", "name", "split")})
        record["eye_paths_swapped"] = swapped
        record["eyes"] = [x[1] for x in pair]
        images[i] = np.stack([x[0] for x in pair])
        for eye in record["eyes"]:
            for key in ("raw_sha256", "reflection_sha256"):
                previous = hash_split.setdefault((key, eye[key]), record["split"])
                if previous != record["split"] and eye["nonzero_pixels_after_resize"]:
                    raise ValueError("Exact image duplicate across data splits")
    np.save(out / "images_uint8.npy", images)
    report = {
        "source_csv": str(csv_path), "source_sha256": sha256(csv_path), "scope": scope,
        "patients": len(records), "split_counts": dict(Counter(r["split"] for r in records)),
        "split_strata": {s: dict(Counter(str(r["stratum"]) for r in records if r["split"] == s)) for s in ("train", "val", "test")},
        "input_shape": ["B", 2, 3, size, size], "tabular_inputs": [], "missingness_mask_inputs": [],
        "preprocessing": "Raw field bbox removes black-border identifiers; same bbox applied to binary reflection image; nearest resize and letterbox preserve field; replicate 3 channels; normalize mean=std=0.5. Raw RGB is used only for bbox/side audit, never as model input.",
        "augmentation": False, "side_corrections": corrections,
        "empty_processed_images": int(sum(e["nonzero_pixels_after_resize"] == 0 for r in records for e in r["eyes"])),
        "side_marker_count": int(sum(bool(e["embedded_marker"]) for r in records for e in r["eyes"])),
    }
    (out / "manifest.json").write_text(json.dumps({"report": report, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return records, images, report
