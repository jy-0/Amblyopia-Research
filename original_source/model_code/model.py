"""
Multi-modal four-class amblyopia classifier.

The model uses age/gender once as a demographic branch, encodes each eye with a
shared image encoder and shared eye-data encoder, then adds explicit right-left
difference features for amblyopia laterality classification.
"""

import torch
import torch.nn as nn
from transformers import ViTModel

from config import (
    VIT_MODEL_NAME, IMAGE_EMBED_DIM, DEMO_EMBED_DIM, EYE_DATA_EMBED_DIM,
    EYE_FUSED_EMBED_DIM, DIFF_EMBED_DIM, FUSION_HIDDEN_DIMS, NUM_CLASSES,
    DROPOUT, DEVICE, IMAGE_BACKBONE_FREEZE_FRAC, VIT_UNFREEZE_LAST_N_LAYERS,
)


class MLPBlock(nn.Module):
    """Linear block using LayerNorm for small-batch stability."""

    def __init__(self, in_dim, out_dim, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ImageEncoder(nn.Module):
    """ViT encoder for one eye image, mostly frozen with optional final-layer tuning."""

    def __init__(self, model_name=VIT_MODEL_NAME,
                 freeze_backbone_frac=IMAGE_BACKBONE_FREEZE_FRAC):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name, local_files_only=True)

        for param in self.vit.parameters():
            param.requires_grad = False

        encoder_layers = getattr(getattr(self.vit, "encoder", None), "layer", None)
        if encoder_layers is None:
            encoder_layers = getattr(self.vit, "layers", [])

        if VIT_UNFREEZE_LAST_N_LAYERS > 0 and len(encoder_layers) > 0:
            for layer in encoder_layers[-VIT_UNFREEZE_LAST_N_LAYERS:]:
                for param in layer.parameters():
                    param.requires_grad = True
            if hasattr(self.vit, "layernorm"):
                for param in self.vit.layernorm.parameters():
                    param.requires_grad = True
        elif freeze_backbone_frac < 1.0 and len(encoder_layers) > 0:
            freeze_layers = int(len(encoder_layers) * freeze_backbone_frac)
            for layer in encoder_layers[freeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        self.backbone_frozen = not any(p.requires_grad for p in self.vit.parameters())
        self.feature_dim = self.vit.config.hidden_size

    def forward(self, x):
        if self.backbone_frozen:
            with torch.no_grad():
                outputs = self.vit(pixel_values=x, interpolate_pos_encoding=True)
        else:
            outputs = self.vit(pixel_values=x, interpolate_pos_encoding=True)
        return outputs.last_hidden_state[:, 0, :]


class DemoEncoder(nn.Module):
    """Encode age and gender once, avoiding repeated demographic use per eye."""

    def __init__(self):
        super().__init__()
        # gender, age plus their masks
        self.net = nn.Sequential(
            MLPBlock(4, 32, DROPOUT),
            nn.Linear(32, DEMO_EMBED_DIM),
            nn.LayerNorm(DEMO_EMBED_DIM),
        )

    def forward(self, features, mask):
        demo = features[:, [0, 1]]
        demo_mask = mask[:, [0, 1]]
        return self.net(torch.cat([demo, demo_mask], dim=1))


class EyeDataEncoder(nn.Module):
    """Shared encoder for one eye's sphere/cylinder/axis and mask."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            MLPBlock(6, 32, DROPOUT),
            nn.Linear(32, EYE_DATA_EMBED_DIM),
            nn.LayerNorm(EYE_DATA_EMBED_DIM),
        )

    def forward(self, eye_values, eye_mask):
        return self.net(torch.cat([eye_values, eye_mask], dim=1))


class DifferenceEncoder(nn.Module):
    """Encode explicit right-left numeric differences and their valid mask."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            MLPBlock(6, 32, DROPOUT),
            nn.Linear(32, DIFF_EMBED_DIM),
            nn.LayerNorm(DIFF_EMBED_DIM),
        )

    def forward(self, right_values, left_values, right_mask, left_mask):
        diff_mask = right_mask * left_mask
        signed_diff = torch.where(diff_mask > 0, right_values - left_values, torch.zeros_like(right_values))
        return self.net(torch.cat([signed_diff, diff_mask], dim=1))


class MultiModalAmblyopiaClassifier(nn.Module):
    """Four-class classifier: normal / bilateral / left / right amblyopia."""

    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder()

        self.image_projector = nn.Sequential(
            MLPBlock(self.image_encoder.feature_dim, 128, DROPOUT),
            nn.Linear(128, IMAGE_EMBED_DIM),
            nn.LayerNorm(IMAGE_EMBED_DIM),
        )
        self.demo_encoder = DemoEncoder()
        self.eye_data_encoder = EyeDataEncoder()
        self.eye_fuser = nn.Sequential(
            MLPBlock(IMAGE_EMBED_DIM + EYE_DATA_EMBED_DIM, 96, DROPOUT),
            nn.Linear(96, EYE_FUSED_EMBED_DIM),
            nn.LayerNorm(EYE_FUSED_EMBED_DIM),
        )
        self.diff_encoder = DifferenceEncoder()

        fusion_input_dim = (
            DEMO_EMBED_DIM
            + EYE_FUSED_EMBED_DIM
            + EYE_FUSED_EMBED_DIM
            + EYE_FUSED_EMBED_DIM
            + DIFF_EMBED_DIM
        )

        fusion_layers = []
        prev_dim = fusion_input_dim
        for hdim in FUSION_HIDDEN_DIMS:
            fusion_layers.append(MLPBlock(prev_dim, hdim, DROPOUT))
            prev_dim = hdim
        fusion_layers.append(nn.Linear(prev_dim, NUM_CLASSES))
        self.classifier = nn.Sequential(*fusion_layers)

    @staticmethod
    def _right_eye_data(features, mask):
        return features[:, [2, 3, 4]], mask[:, [2, 3, 4]]

    @staticmethod
    def _left_eye_data(features, mask):
        return features[:, [5, 6, 7]], mask[:, [5, 6, 7]]

    def forward(self, images, features, feature_mask):
        img_r = images[:, 0]
        img_l = images[:, 1]

        img_feat_r = self.image_projector(self.image_encoder(img_r))
        img_feat_l = self.image_projector(self.image_encoder(img_l))

        right_values, right_mask = self._right_eye_data(features, feature_mask)
        left_values, left_mask = self._left_eye_data(features, feature_mask)

        eye_data_r = self.eye_data_encoder(right_values, right_mask)
        eye_data_l = self.eye_data_encoder(left_values, left_mask)
        eye_r = self.eye_fuser(torch.cat([img_feat_r, eye_data_r], dim=1))
        eye_l = self.eye_fuser(torch.cat([img_feat_l, eye_data_l], dim=1))

        demo = self.demo_encoder(features, feature_mask)
        eye_abs_diff = torch.abs(eye_r - eye_l)
        numeric_diff = self.diff_encoder(right_values, left_values, right_mask, left_mask)

        fused = torch.cat([demo, eye_r, eye_l, eye_abs_diff, numeric_diff], dim=1)
        return self.classifier(fused)


def compute_class_weights(dataloader):
    """Compute inverse-frequency class weights from the underlying split."""
    dataset = dataloader.dataset
    if hasattr(dataset, "labels") and hasattr(dataset, "indices"):
        labels = torch.as_tensor(dataset.labels[dataset.indices], dtype=torch.long)
    else:
        labels = []
        for batch in dataloader:
            labels.append(batch["labels"])
        labels = torch.cat(labels, dim=0).long()

    counts = torch.bincount(labels, minlength=NUM_CLASSES).float()
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (NUM_CLASSES * counts)
    print(f"Computed class weights: {weights.tolist()}")
    return weights


if __name__ == "__main__":
    model = MultiModalAmblyopiaClassifier().to(DEVICE)

    dummy_images = torch.randn(4, 2, 3, 384, 384).to(DEVICE)
    dummy_feats = torch.randn(4, 8).to(DEVICE)
    dummy_mask = torch.ones(4, 8).to(DEVICE)

    logits = model(dummy_images, dummy_feats, dummy_mask)
    print(f"Input images:  {dummy_images.shape}")
    print(f"Input feats:   {dummy_feats.shape}")
    print(f"Output logits: {logits.shape}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    vit_trainable = sum(p.numel() for p in model.image_encoder.vit.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"ViT trainable:    {vit_trainable:,}")
