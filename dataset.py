"""
Custom Dataset for multi-modal amblyopia data.
Loads 10 tabular features from xlsx and dual-eye images from eye_data folder.
Missing tabular values are encoded as 0 with a separate observed-value mask.
"""

import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import openpyxl

from config import (
    XLSX_PATH, EYE_DATA_DIR, IMG_SIZE, IMG_MEAN, IMG_STD, SEED,
    INPUT_COL_INDICES, OUTPUT_COL_INDICES, TABULAR_INPUT_DIM,
)

# ── Cached data (loaded once) ────────────────────────────────────────────
_cache = None  # {features, feature_mask, labels, name_to_images, valid_names}
_NUMERIC_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_MISSING_STRINGS = {"", "none", "nan", "null", "未找到", "-", "--", "/"}


def _is_missing(value):
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _MISSING_STRINGS
    return False


def _parse_numeric(value):
    """Return (parsed_value, observed_mask)."""
    if _is_missing(value):
        return 0.0, 0.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value), 1.0

    text = str(value).strip().replace("−", "-")
    match = _NUMERIC_RE.search(text)
    if not match:
        return 0.0, 0.0
    return float(match.group(0)), 1.0


def _parse_gender(value):
    """男 is encoded as 0, 女 as 1; missing/unknown is 0 with mask 0."""
    if _is_missing(value):
        return 0.0, 0.0
    text = str(value).strip()
    if text == "女":
        return 1.0, 1.0
    if text == "男":
        return 0.0, 1.0
    return 0.0, 0.0


def _read_spreadsheet_rows():
    """Read non-empty patient rows without materializing formatted blank rows."""
    wb = openpyxl.load_workbook(XLSX_PATH, read_only=True, data_only=True)
    ws = wb.active
    data = []
    blank_run = 0

    try:
        for row in ws.iter_rows(min_row=2, max_col=22, values_only=True):
            name = row[0] if row else None
            if _is_missing(name):
                if data:
                    blank_run += 1
                    if blank_run >= 100:
                        break
                continue

            blank_run = 0
            data.append(tuple(row))
    finally:
        wb.close()

    return data


def _load_all_data():
    """Load all data from xlsx and image directory. Cached globally."""
    global _cache
    if _cache is not None:
        return _cache

    # ── Read xlsx with openpyxl (streamed; sheet has many formatted blanks) ──
    data = _read_spreadsheet_rows()
    N = len(data)

    # ── Parse features (10 dims: gender, age, right 4, left 4) ───────
    feats = np.zeros((N, TABULAR_INPUT_DIM), dtype=np.float32)
    feature_mask = np.zeros_like(feats, dtype=np.float32)

    for i, row in enumerate(data):
        for j, col_idx in enumerate(INPUT_COL_INDICES):
            value = row[col_idx] if col_idx < len(row) else None
            if col_idx == 1:
                parsed, observed = _parse_gender(value)
            else:
                parsed, observed = _parse_numeric(value)
            feats[i, j] = parsed
            feature_mask[i, j] = observed

    # ── Parse labels (1 dim: amblyopia only) ────────────────────────
    # col 17=是否弱视
    labels = np.zeros((N, 1), dtype=np.float32)
    label_col = OUTPUT_COL_INDICES[0]
    for i, row in enumerate(data):
        value = row[label_col] if label_col < len(row) else None
        labels[i, 0] = 1.0 if str(value).strip() == "是" else 0.0

    # ── Build image index ──────────────────────────────────────────────
    all_files = sorted(os.listdir(EYE_DATA_DIR))
    name_to_files = {}
    for fname in all_files:
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        match = re.match(r"(.+)_(\d{17})\.\w+$", fname)
        if match:
            name = match.group(1)
        else:
            parts = fname.rsplit("_", 1)
            if len(parts) == 2:
                name = parts[0]
            else:
                continue
        name_to_files.setdefault(name, []).append(fname)

    name_to_images = {}
    for name, files in name_to_files.items():
        files.sort()
        if len(files) >= 2:
            name_to_images[name] = [
                os.path.join(EYE_DATA_DIR, files[0]),
                os.path.join(EYE_DATA_DIR, files[1]),
            ]

    # ── Gather valid sample indices (must have both eye images) ───────
    names = [str(r[0]) for r in data]
    valid_idx = [i for i, n in enumerate(names) if n in name_to_images]

    _cache = {
        "features": feats,
        "feature_mask": feature_mask,
        "labels": labels,
        "name_to_images": name_to_images,
        "names": names,
        "valid_idx": valid_idx,
    }
    print(f"[Data] {len(valid_idx)} valid samples "
          f"(弱视阳性={int(labels[valid_idx, 0].sum())} "
          f"阴性={len(valid_idx) - int(labels[valid_idx, 0].sum())})")
    return _cache


class AmblyopiaDataset(Dataset):
    """Multi-modal dataset.

    Each sample:
      - features:     (10,) float tensor
      - feature_mask: (10,) float tensor
      - images:       (2, 3, 224, 224) tensor (right_eye, left_eye)
      - labels:       (1,)  float tensor (是否弱视)
    """

    def __init__(self, split="train", val_split=0.15, test_split=0.15,
                 augment=False):
        self.split = split
        self.augment = augment

        data = _load_all_data()
        self.features = data["features"]
        self.feature_mask = data["feature_mask"]
        self.labels = data["labels"]
        self.name_to_images = data["name_to_images"]
        self.names = data["names"]
        self.valid_idx = np.array(data["valid_idx"])

        # ── Split ────────────────────────────────────────────────────
        np.random.seed(SEED)
        idx = self.valid_idx.copy()
        np.random.shuffle(idx)
        n = len(idx)
        n_test = int(n * test_split)
        n_val = int(n * val_split)

        if split == "train":
            self.indices = idx[:n - n_val - n_test]
        elif split == "val":
            self.indices = idx[n - n_val - n_test:n - n_test]
        elif split == "test":
            self.indices = idx[n - n_test:]
        else:
            raise ValueError(f"Unknown split: {split}")

        # ── Transforms ───────────────────────────────────────────────
        t_list = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
        if augment:
            t_list += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ]
        t_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ]
        self.transform = transforms.Compose(t_list)

        print(f"[{split}] {len(self.indices)} samples, "
              f"labels sum: {self.labels[self.indices].sum(axis=0).astype(int)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        name = self.names[real_idx]

        features = torch.from_numpy(self.features[real_idx])
        feature_mask = torch.from_numpy(self.feature_mask[real_idx])
        labels = torch.from_numpy(self.labels[real_idx])

        img_paths = self.name_to_images.get(name, [])
        if len(img_paths) >= 2:
            img_r = Image.open(img_paths[0]).convert("RGB")
            img_l = Image.open(img_paths[1]).convert("RGB")
        else:
            img_r = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
            img_l = Image.new("RGB", (IMG_SIZE, IMG_SIZE))

        img_r = self.transform(img_r)
        img_l = self.transform(img_l)
        images = torch.stack([img_r, img_l], dim=0)

        return {
            "features": features,
            "feature_mask": feature_mask,
            "images": images,
            "labels": labels,
            "name": name,
            "image_paths": img_paths,
        }


def get_dataloaders(batch_size=8, val_split=0.15, test_split=0.15):
    """Return train_dl, val_dl, test_dl."""
    train_ds = AmblyopiaDataset(split="train", val_split=val_split,
                                test_split=test_split, augment=True)
    val_ds = AmblyopiaDataset(split="val", val_split=val_split,
                              test_split=test_split, augment=False)
    test_ds = AmblyopiaDataset(split="test", val_split=val_split,
                               test_split=test_split, augment=False)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                         num_workers=2, pin_memory=True)

    return train_dl, val_dl, test_dl


if __name__ == "__main__":
    train_dl, val_dl, test_dl = get_dataloaders(batch_size=4)
    batch = next(iter(train_dl))
    print("Features:", batch["features"].shape)
    print("Mask:", batch["feature_mask"].shape)
    print("Images:", batch["images"].shape)
    print("Labels:", batch["labels"].shape)
    print("Labels sum:", batch["labels"].sum(dim=0))
    print("Names:", batch["name"])
    print("Image paths:", batch["image_paths"])
