"""
模态融合模块 - 实现双模态特征融合
基于设计文档 3.4 节
"""

import torch
import torch.nn as nn


class ModalFusion(nn.Module):
    """双模态融合模块"""

    def __init__(self, embed_dim=512, fusion_type='gating', img_bias=0.7):
        super().__init__()
        self.fusion_type = fusion_type
        self.img_bias = img_bias

        self.gate_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1)
        )
        self.concat_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, z_img, z_txt):
        """
        Args:
            z_img: [batch, embed_dim] - 图像编码特征
            z_txt: [batch, embed_dim] - 文本编码特征
        Returns:
            z_fused: [batch, embed_dim] - 融合后的特征
        """
        if self.fusion_type == 'concat':
            fused = self.concat_proj(torch.cat([z_img, z_txt], dim=-1))
            return self.norm(fused)

        if self.fusion_type == 'cross_attention':
            query = z_img.unsqueeze(1)
            key_value = torch.stack([z_img, z_txt], dim=1)
            attended, _ = self.cross_attention(query=query, key=key_value, value=key_value)
            return self.norm(attended.squeeze(1) + z_img)

        concat = torch.cat([z_img, z_txt], dim=-1)
        alpha = torch.sigmoid(self.gate_proj(concat))
        alpha_adj = alpha * (1 - self.img_bias) + self.img_bias
        z_fused = alpha_adj * z_img + (1 - alpha_adj) * z_txt
        return self.norm(z_fused + z_img)
