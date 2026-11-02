"""
Configuration for the multi-modal four-class amblyopia classifier.

Inputs:
  - age, gender
  - right/left sphere, cylinder, axis
  - right/left eye images

Output classes:
  0 normal, 1 bilateral amblyopia, 2 left amblyopia, 3 right amblyopia
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = "/root/hospital"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
XLSX_PATH = os.path.join(DATASET_DIR, "newest.xlsx")
EYE_DATA_DIR = os.path.join(DATASET_DIR, "eye_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "model_code", "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# ── Data ───────────────────────────────────────────────────────────────────
INPUT_COLS = [
    "性别",      # 1  - gender (categorical: 男/女)
    "年龄",      # 2  - age (str like "3岁", parsed to int)
    "右球镜",    # 4
    "右柱镜",    # 5
    "右轴位",    # 6
    "左球镜",    # 8
    "左柱镜",    # 9
    "左轴位",    # 10
]

OUTPUT_COLS = [
    "是否弱视",   # 16
    "弱视眼别",   # 17: 左/右/双/empty
]

# Column indices in the spreadsheet (0-based)
INPUT_COL_INDICES = [1, 2, 4, 5, 6, 8, 9, 10]
OUTPUT_COL_INDICES = [16, 17]
IMAGE_COL_INDICES = [19, 20]
TABULAR_INPUT_DIM = len(INPUT_COL_INDICES)

LABEL_NAMES = ["正常", "双眼弱视", "左眼弱视", "右眼弱视"]
NUM_CLASSES = len(LABEL_NAMES)

# ── Image ──────────────────────────────────────────────────────────────────
IMG_SIZE = 384
IMG_RESIZE = 448        # preserve aspect ratio, then center crop to 384
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

# ── Model ──────────────────────────────────────────────────────────────────
VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k"
VIT_FEATURE_DIM = 768
IMAGE_EMBED_DIM = 64
DEMO_EMBED_DIM = 24
EYE_DATA_EMBED_DIM = 32
EYE_FUSED_EMBED_DIM = 64
DIFF_EMBED_DIM = 32
FUSION_HIDDEN_DIMS = [128, 64]
DROPOUT = 0.3
IMAGE_BACKBONE_FREEZE_FRAC = 1.0
VIT_UNFREEZE_LAST_N_LAYERS = 1

# ── Training ───────────────────────────────────────────────────────────────
BATCH_SIZE = 2          # 384px ViT input uses much more memory
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
BACKBONE_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
GRAD_ACCUM_STEPS = 8    # effective batch = 16
LR_PATIENCE = 5
EARLY_STOP_PATIENCE = 10

USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0
USE_BALANCED_SAMPLER = True
USE_IMAGE_AUGMENTATION = False

# ── Validation split ──────────────────────────────────────────────────────
VAL_SPLIT = 0.1
TEST_SPLIT = 0.2
SEED = 46

# ── Device ─────────────────────────────────────────────────────────────────
import torch
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
