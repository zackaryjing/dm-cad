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

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, n_views, channels, height, width = images.shape
        if n_views != self.n_views:
            raise ValueError(f"Expected {self.n_views} views but got {n_views}")

        features = self.backbone(images.view(batch_size * n_views, channels, height, width))
        features = features.flatten(1).view(batch_size, n_views, -1)
        return self.fusion(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        fused = self.encode_images(images)
        return self.head(fused)


class TextResidualFusion(nn.Module):
    def __init__(self, text_dim: int = 768, latent_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        fusion_dim = latent_dim * 4
        self.gate = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Sigmoid(),
        )
        self.delta = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_img: torch.Tensor, text_emb: torch.Tensor, text_dropout_p: float = 0.0) -> torch.Tensor:
        z_txt = self.text_proj(text_emb)
        if self.training and text_dropout_p > 0.0:
            keep = (torch.rand(z_txt.shape[0], 1, device=z_txt.device) >= text_dropout_p).to(z_txt.dtype)
            z_txt = z_txt * keep
        fuse = torch.cat([z_img, z_txt, z_img - z_txt, z_img * z_txt], dim=-1)
        gate = self.gate(fuse)
        delta = self.delta(fuse)
        return torch.tanh(z_img + gate * delta)


class MultiModalLatentRegressor(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet18",
        n_views: int = 8,
        latent_dim: int = 256,
        image_hidden_dim: int = 512,
        text_dim: int = 768,
        freeze_backbone: bool = False,
        text_dropout_p: float = 0.3,
    ):
        super().__init__()
        self.image_model = MultiViewLatentRegressor(
            backbone_name=backbone_name,
            n_views=n_views,
            latent_dim=latent_dim,
            hidden_dim=image_hidden_dim,
            freeze_backbone=freeze_backbone,
        )
        self.fusion = TextResidualFusion(text_dim=text_dim, latent_dim=latent_dim, hidden_dim=image_hidden_dim)
        self.text_dropout_p = text_dropout_p

    def freeze_image_branch(self, freeze: bool = True):
        for param in self.image_model.parameters():
            param.requires_grad = not freeze

    def load_image_checkpoint(self, checkpoint_path: str | Path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt["model"]
        missing, unexpected = self.image_model.load_state_dict(state, strict=False)
        return {"missing_keys": missing, "unexpected_keys": unexpected}

    def forward(self, images: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        z_img = self.image_model(images)
        return self.fusion(z_img, text_emb, text_dropout_p=self.text_dropout_p)
