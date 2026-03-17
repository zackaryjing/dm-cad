"""
模态融合模块 - 实现双模态特征融合
基于设计文档 3.4 节
"""

import torch
import torch.nn as nn


class ModalFusion(nn.Module):
    """双模态融合模块

    使用门控 (Gating) 机制实现"图像为主，文本为辅"的融合策略:
    - 计算权重 α = σ(Linear([z_img; z_txt]))
    - 融合特征 z_fused = α · z_img + (1 - α) · z_txt
    - 图像为主：给 z_img 设置较高的基础权重
    """
    def __init__(self, embed_dim=512, fusion_type='gating', img_bias=0.7):
        super().__init__()
        self.fusion_type = fusion_type
        self.img_bias = img_bias  # 图像基础权重偏置

        # 门控投影：拼接后投影到 1 维得到 α
        self.gate_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1)
        )

        # 可选的残差连接
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, z_img, z_txt):
        """
        Args:
            z_img: [batch, embed_dim] - 图像编码特征
            z_txt: [batch, embed_dim] - 文本编码特征
        Returns:
            z_fused: [batch, embed_dim] - 融合后的特征
        """
        # 拼接两个模态
        concat = torch.cat([z_img, z_txt], dim=-1)  # [batch, 2*embed_dim]

        # 计算门控权重 α (0 到 1 之间)
        alpha = torch.sigmoid(self.gate_proj(concat))  # [batch, 1]

        # 应用图像偏置 - 确保图像为主
        # alpha_adj = alpha * (1 - img_bias) + img_bias
        # 这样 α 的范围是 [img_bias, 1]，图像始终占主导
        alpha_adj = alpha * (1 - self.img_bias) + self.img_bias

        # 门控融合：z_fused = α · z_img + (1 - α) · z_txt
        z_fused = alpha_adj * z_img + (1 - alpha_adj) * z_txt

        # 残差连接 + 归一化
        z_fused = self.norm(z_fused + z_img)

        return z_fused
