"""
视图编码器模块 - 实现单视图编码和多视图融合
基于设计文档 3.2 节
"""

import torch
import torch.nn as nn
from timm.models.vision_transformer import vit_base_patch16_224


class ViewEncoder(nn.Module):
    """单个视图的编码器

    使用预训练的 ViT 作为 backbone，默认冻结 backbone 参数
    """
    def __init__(self, embed_dim=512, pretrained=True, freeze_backbone=True):
        super().__init__()
        self.vit = vit_base_patch16_224(pretrained=pretrained)
        self.vit.head = nn.Identity()

        self.project = nn.Sequential(
            nn.Linear(768, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        """冻结 ViT backbone，仅保留投影层可训练。"""
        for param in self.vit.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """解冻 ViT backbone。"""
        for param in self.vit.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        Args:
            x: [batch, 3, 224, 224] - 单视图图像
        Returns:
            features: [batch, embed_dim] - 编码后的特征
        """
        features = self.vit(x)
        return self.project(features)


class MultiViewFusion(nn.Module):
    """多视图注意力池化模块

    使用 CLS Token 模式进行空间加权聚合:
    - 在 8 个视图特征前插入可学习 [CLS] 向量
    - [CLS] 通过自注意力主动吸纳 8 个角度中最重要的几何特征
    - 最终取 encoded[:, 0] 作为 z_img
    """
    def __init__(self, embed_dim=512, n_views=8, n_heads=8):
        super().__init__()
        self.n_views = n_views
        self.embed_dim = embed_dim

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.view_pos_embed = nn.Parameter(torch.randn(1, n_views, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.aggregate = nn.Linear(embed_dim, embed_dim)

    def forward(self, view_features):
        """
        Args:
            view_features: [batch, n_views, embed_dim] - 各视图特征
        Returns:
            fused: [batch, embed_dim] - 融合后的特征 (通过 [CLS] 聚合)
        """
        batch_size = view_features.shape[0]

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        view_features = view_features + self.view_pos_embed.expand(batch_size, -1, -1)
        transformer_input = torch.cat([cls_tokens, view_features], dim=1)
        encoded = self.transformer(transformer_input)
        fused = encoded[:, 0]
        return self.aggregate(fused)
