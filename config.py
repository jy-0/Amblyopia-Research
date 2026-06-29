"""
Configuration for the multi-modal amblyopia classification model.
Based on: Multi-class Classification of Retinal Eye Diseases from Ophthalmoscopy
Images Using Transfer Learning-Based Vision Transformers.
Adapted for: multi-modal (tabular + dual-eye images) -> single-label classification
(是否弱视).
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = "/root/hospital"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
XLSX_PATH = os.path.join(DATASET_DIR, "data_analysis.xlsx")
EYE_DATA_DIR = os.path.join(DATASET_DIR, "eye_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "model_code", "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# ── Data ───────────────────────────────────────────────────────────────────
# Input columns from the spreadsheet:
# demographics + four right-eye values + four left-eye values.
INPUT_COLS = [
    "性别",      # 1  - gender (categorical: 男/女)
    "年龄",      # 2  - age (str like "3岁", parsed to int)
    "右球镜",    # 5
    "右柱镜",    # 6
    "右轴位",    # 7
    "右BCVA",    # 8
    "左球镜",    # 9
    "左柱镜",    # 10
    "左轴位",    # 11
    "左BCVA",    # 12
]

# Output label column (binary: 是/None -> 1/0)
OUTPUT_COLS = [
    "是否弱视",   # 17 - amblyopia
]

# Column indices in the spreadsheet (0-based)
INPUT_COL_INDICES = [1, 2, 5, 6, 7, 8, 9, 10, 11, 12]
OUTPUT_COL_INDICES = [17]
TABULAR_INPUT_DIM = len(INPUT_COL_INDICES)

# ── Image ──────────────────────────────────────────────────────────────────
IMG_SIZE = 224          # ViT expects 224x224
IMG_MEAN = [0.485, 0.456, 0.406]   # ImageNet stats
IMG_STD = [0.229, 0.224, 0.225]

# ── Model ──────────────────────────────────────────────────────────────────
VIT_MODEL_NAME = "google/vit-large-patch16-224"  # ViT-L/16 AugReg (aligned with paper ViT#5)
VIT_FEATURE_DIM = 1024  # hidden dim of ViT-L/16
TABULAR_HIDDEN_DIMS = [32, 64]    # MLP for 10 tabular features plus masks
TABULAR_OUTPUT_DIM = 16
FUSION_HIDDEN_DIMS = [256, 64]    # MLP after fusion (reduced to avoid overfitting)
NUM_CLASSES = 1         # single binary output (amblyopia only)
DROPOUT = 0.3

# ── Training ───────────────────────────────────────────────────────────────
BATCH_SIZE = 4            # reduced for ViT-L memory
NUM_EPOCHS = 50
LEARNING_RATE = 2e-5          # low LR for ViT fine-tuning
WEIGHT_DECAY = 1e-4
LR_WARMUP_EPOCHS = 3
LR_PATIENCE = 5               # reduce-on-plateau patience
EARLY_STOP_PATIENCE = 10

# Loss weight to handle class imbalance (computed from data)
# Will be set dynamically at runtime
POS_WEIGHT = None              # e.g. 9.0 for severe imbalance

# ── Validation split ──────────────────────────────────────────────────────
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42

# ── Device ─────────────────────────────────────────────────────────────────
import torch
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
