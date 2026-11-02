"""Image-only paired ViT with a shared eye encoder and signed pair differences."""
import torch
from torch import nn

from model import ImageEncoder, MLPBlock


class ReflectionImagePairClassifier(nn.Module):
    def __init__(self, dropout=0.3, auxiliary_dim=0):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.projector = nn.Sequential(MLPBlock(768, 128, dropout), nn.Linear(128, 32), nn.LayerNorm(32), nn.GELU())
        self.pair = nn.Sequential(MLPBlock(96, 96, dropout), nn.Linear(96, 48), nn.LayerNorm(48), nn.GELU())
        self.fusion = nn.Sequential(MLPBlock(112, 64, dropout), nn.Linear(64, 24), nn.LayerNorm(24), nn.GELU())
        self.severity = nn.Linear(24, 1)
        self.laterality = nn.Linear(24, 1)
        self.auxiliary_dim = auxiliary_dim
        if auxiliary_dim:
            self.auxiliary = nn.Sequential(MLPBlock(auxiliary_dim, 24, dropout), nn.Linear(24, 16), nn.LayerNorm(16), nn.GELU())
            self.fusion = nn.Sequential(MLPBlock(128, 64, dropout), nn.Linear(64, 24), nn.LayerNorm(24), nn.GELU())

    @property
    def layers(self):
        vit = self.image_encoder.vit
        return vit.encoder.layer if hasattr(vit, "encoder") else vit.layers

    @staticmethod
    def block_forward(block, x):
        output = block(x)
        return output[0] if isinstance(output, tuple) else output

    @torch.no_grad()
    def frozen_prefix(self, images):
        vit = self.image_encoder.vit
        vit.embeddings.eval()
        x = vit.embeddings(images, interpolate_pos_encoding=True)
        for block in self.layers[:-1]:
            block.eval()
            x = self.block_forward(block, x)
        return x

    def forward_from_tokens(self, tokens, auxiliary=None):
        b, eyes, length, width = tokens.shape
        if eyes != 2:
            raise ValueError("Expected one right/left image pair per patient")
        x = self.block_forward(self.layers[-1], tokens.reshape(b*2, length, width))
        cls = self.image_encoder.vit.layernorm(x)[:, 0]
        eye = self.projector(cls).reshape(b, 2, 32)
        r, l = eye[:, 0], eye[:, 1]
        mean, absolute, signed = (r+l)/2, (r-l).abs(), r-l
        pair = self.pair(torch.cat([signed, absolute, mean], dim=1))
        parts = [pair, mean, absolute]
        if self.auxiliary_dim:
            if auxiliary is None or auxiliary.shape != (b, self.auxiliary_dim):
                raise ValueError("Incorrect auxiliary input shape")
            parts.append(self.auxiliary(auxiliary))
        fused = self.fusion(torch.cat(parts, dim=1))
        base, side = self.severity(fused), self.laterality(fused)
        return torch.cat([base+side, base-side], dim=1)

    def forward(self, images, auxiliary=None):
        if images.ndim != 5 or images.shape[1:3] != (2, 3):
            raise ValueError("Expected images [B,2,3,H,W] only")
        b, _, _, h, w = images.shape
        tokens = self.frozen_prefix(images.reshape(b*2, 3, h, w))
        return self.forward_from_tokens(tokens.reshape(b, 2, *tokens.shape[1:]), auxiliary)
