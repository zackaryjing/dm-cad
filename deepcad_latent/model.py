from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34


def build_resnet_backbone(name: str) -> tuple[nn.Module, int]:
    if name == "resnet18":
        model = resnet18(weights=None)
        out_dim = 512
    elif name == "resnet34":
        model = resnet34(weights=None)
        out_dim = 512
    else:
        raise ValueError(f"Unsupported backbone: {name}")

    layers = list(model.children())[:-1]
    return nn.Sequential(*layers), out_dim


class GRUFusion(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, h_n = self.gru(x)
        del output
        h_n = h_n.transpose(0, 1).reshape(x.shape[0], -1)
        return self.proj(h_n)


class MultiViewLatentRegressor(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet18",
        n_views: int = 8,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.n_views = n_views
        self.backbone, feat_dim = build_resnet_backbone(backbone_name)
        self.fusion = GRUFusion(feat_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, n_views, channels, height, width = images.shape
        if n_views != self.n_views:
            raise ValueError(f"Expected {self.n_views} views but got {n_views}")

        features = self.backbone(images.view(batch_size * n_views, channels, height, width))
        features = features.flatten(1).view(batch_size, n_views, -1)
        fused = self.fusion(features)
        return self.head(fused)
