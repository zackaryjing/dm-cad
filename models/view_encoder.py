"""
视图编码器模块 - 实现单视图编码和多视图融合
基于设计文档 3.2 节
"""

import torch
import torch.nn as nn
from timm.models.vision_transformer import vit_base_patch16_224


class ViewEncoder(nn.Module):
    """单个视图的编码器

    使用预训练的 ViT 作为 backbone，冻结大部分参数
    """
    def __init__(self, embed_dim=512, pretrained=True):
        super().__init__()
        # 使用预训练 ViT，冻结大部分参数
        self.vit = vit_base_patch16_224(pretrained=pretrained)
        self.vit.head = nn.Identity()  # 移除分类头

        # 投影层：768 -> 512
        self.project = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU()
        )

    def forward(self, x):
        """
        Args:
            x: [batch, 3, 224, 224] - 单视图图像
        Returns:
            features: [batch, 512] - 编码后的特征
        """
        features = self.vit(x)  # [batch, 768]
        return self.project(features)  # [batch, 512]


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

        # 可学习的 [CLS] token - 用于聚合全局特征
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # 视图位置编码 (8 个固定视角)
        self.view_pos_embed = nn.Parameter(torch.randn(1, n_views, embed_dim))

        # Transformer encoder 进行视图间注意力
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 聚合层
        self.aggregate = nn.Linear(embed_dim, embed_dim)

    def forward(self, view_features):
        """
        Args:
            view_features: [batch, n_views, embed_dim] - 各视图特征
        Returns:
            fused: [batch, embed_dim] - 融合后的特征 (通过 [CLS] 聚合)
        """
        B = view_features.shape[0]

        # 扩展 [CLS] token 到 batch 大小
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [batch, 1, embed_dim]

        # 添加位置编码到视图特征
        view_features = view_features + self.view_pos_embed.expand(B, -1, -1)

        # 拼接 [CLS] + 8 个视图特征
        transformer_input = torch.cat([cls_tokens, view_features], dim=1)  # [batch, 9, embed_dim]

        # Transformer 编码 - [CLS] 通过自注意力吸收各视图信息
        encoded = self.transformer(transformer_input)  # [batch, 9, embed_dim]

        # 取 [CLS] 位置的特征作为融合结果 (encoded[:, 0])
        fused = encoded[:, 0]  # [batch, embed_dim]

        return self.aggregate(fused)
