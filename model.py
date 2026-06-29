"""
Multi-modal model for amblyopia classification (aligned with paper ViT#5).

Architecture (based on the paper's ViT transfer-learning framework):
  1. ViT-L/16 AugReg encoder (shared weights) -> processes each eye independently
  2. Tabular MLP with missing-value masking -> encodes demographics + eye data
  3. Concatenation fusion -> combines image + tabular embeddings
  4. MLP classifier head -> 1 sigmoid output (binary amblyopia)

Reference:
  "Multi-class Classification of Retinal Eye Diseases from Ophthalmoscopy
   Images Using Transfer Learning-Based Vision Transformers"
  - ViT#5 (AugReg ViT-L/16_224, LR=2e-5, Adam)
"""

import torch
import torch.nn as nn
from transformers import ViTModel

from config import (
    VIT_MODEL_NAME, TABULAR_INPUT_DIM, TABULAR_HIDDEN_DIMS, TABULAR_OUTPUT_DIM,
    FUSION_HIDDEN_DIMS, NUM_CLASSES, DROPOUT, DEVICE,
)


class TabularEncoder(nn.Module):
    """MLP encoder for tabular features with missing-value awareness.

    Input:  (batch, input_dim) feature vector + (batch, input_dim) mask
    Output: (batch, tabular_output_dim)
    """

    def __init__(self, input_dim=TABULAR_INPUT_DIM, hidden_dims=None,
                 output_dim=16, dropout=0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64]

        layers = []
        prev_dim = input_dim + input_dim  # features + mask = 2 * input_dim

        for hdim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hdim),
                nn.BatchNorm1d(hdim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = hdim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, features, mask):
        # Concatenate features with their validity mask
        x = torch.cat([features, mask], dim=1)
        return self.net(x)


class ImageEncoder(nn.Module):
    """ViT-L/16 AugReg encoder for a single eye image.

    Uses a pre-trained ViT-Large from HuggingFace transformers, fine-tuned.
    Shared across right and left eyes.
    """

    def __init__(self, model_name=VIT_MODEL_NAME, freeze_backbone_frac=0.3):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)

        # Optionally freeze a fraction of ViT encoder layers for stability.
        encoder_layers = getattr(getattr(self.vit, "encoder", None), "layer", [])
        freeze_layers = int(len(encoder_layers) * freeze_backbone_frac)
        for layer in encoder_layers[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False

        self.feature_dim = self.vit.config.hidden_size

    def forward(self, x):
        """x: (batch, 3, H, W) single image"""
        outputs = self.vit(pixel_values=x)
        # Use the [CLS] token embedding as the image representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)
        return cls_embedding


class MultiModalAmblyopiaClassifier(nn.Module):
    """Full multi-modal classifier.

    Input:
      - images: (batch, 2, 3, 224, 224)  [right_eye, left_eye]
      - features: (batch, 10) demographics + right/left eye values
      - feature_mask: (batch, 10) 0/1 mask for observed values

    Output:
      - logits: (batch, 1)  raw logits for 是否弱视
    """

    def __init__(self):
        super().__init__()

        # Shared image encoder (processes each eye)
        self.image_encoder = ImageEncoder()

        # Tabular encoder
        self.tabular_encoder = TabularEncoder(
            input_dim=TABULAR_INPUT_DIM,
            hidden_dims=TABULAR_HIDDEN_DIMS,
            output_dim=TABULAR_OUTPUT_DIM,
            dropout=DROPOUT,
        )

        # Fusion + classifier
        fusion_input_dim = 2 * self.image_encoder.feature_dim + TABULAR_OUTPUT_DIM

        fusion_layers = []
        prev_dim = fusion_input_dim
        for hdim in FUSION_HIDDEN_DIMS:
            fusion_layers.extend([
                nn.Linear(prev_dim, hdim),
                nn.BatchNorm1d(hdim),
                nn.ReLU(inplace=True),
                nn.Dropout(DROPOUT),
            ])
            prev_dim = hdim
        fusion_layers.append(nn.Linear(prev_dim, NUM_CLASSES))

        self.classifier = nn.Sequential(*fusion_layers)

    def forward(self, images, features, feature_mask):
        # Encode right eye (index 0) and left eye (index 1)
        img_r = images[:, 0, :, :, :]  # (B, 3, H, W)
        img_l = images[:, 1, :, :, :]  # (B, 3, H, W)

        feat_r = self.image_encoder(img_r)
        feat_l = self.image_encoder(img_l)

        # Simple fusion of both eye embeddings
        img_features = torch.cat([feat_r, feat_l], dim=1)

        # Encode tabular features
        tab_features = self.tabular_encoder(features, feature_mask)

        # Concatenate everything
        fused = torch.cat([img_features, tab_features], dim=1)

        # Classifier
        logits = self.classifier(fused)

        return logits


def compute_pos_weight(dataloader):
    """Compute positive class weight for BCEWithLogitsLoss from training data."""
    all_labels = []
    for batch in dataloader:
        all_labels.append(batch["labels"])
    all_labels = torch.cat(all_labels, dim=0)  # (N, 1)

    pos_count = all_labels.sum()  # scalar
    neg_count = len(all_labels) - pos_count
    # pos_weight = neg / pos  (to balance the loss)
    pos_weight = neg_count / (pos_count + 1e-8)
    print(f"Computed pos_weight: {pos_weight.item():.2f}")
    return pos_weight


if __name__ == "__main__":
    # Smoke test
    model = MultiModalAmblyopiaClassifier()
    model = model.to(DEVICE)

    dummy_images = torch.randn(4, 2, 3, 224, 224).to(DEVICE)
    dummy_feats = torch.randn(4, TABULAR_INPUT_DIM).to(DEVICE)
    dummy_mask = torch.ones(4, TABULAR_INPUT_DIM).to(DEVICE)

    logits = model(dummy_images, dummy_feats, dummy_mask)
    print(f"Input images:  {dummy_images.shape}")
    print(f"Input feats:   {dummy_feats.shape}")
    print(f"Output logits: {logits.shape}")

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Percent trainable: {100 * trainable / total:.1f}%")
