"""
模态融合模块 - 实现双模态特征融合
基于设计文档 3.4 节
"""

import torch
import torch.nn as nn


class ModalFusion(nn.Module):
    """双模态融合模块

    支持三种融合方式:
    - concat: 拼接后投影
    - cross_attention: 交叉注意力 (推荐)
    - gating: 门控机制
    """
    def __init__(self, embed_dim=512, fusion_type='cross_attention'):
        super().__init__()
        self.fusion_type = fusion_type

        if fusion_type == 'concat':
            self.fusion = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU()
            )
        elif fusion_type == 'cross_attention':
            # 文本作为 query，图像作为 key/value
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=8,
                batch_first=True
            )
            self.norm = nn.LayerNorm(embed_dim)
        elif fusion_type == 'gating':
            self.gate_proj = nn.Linear(embed_dim * 2, embed_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(self, z_img, z_txt):
        """
        Args:
            z_img: [batch, embed_dim] - 图像编码特征
            z_txt: [batch, embed_dim] - 文本编码特征
        Returns:
            z_fused: [batch, embed_dim] - 融合后的特征
        """
        if self.fusion_type == 'concat':
            fused = torch.cat([z_img, z_txt], dim=-1)
            return self.fusion(fused)

        elif self.fusion_type == 'cross_attention':
            # 添加 sequence 维度
            z_txt_q = z_txt.unsqueeze(1)  # [batch, 1, embed_dim]
            z_img_kv = z_img.unsqueeze(1)  # [batch, 1, embed_dim]

            attn_out, _ = self.cross_attn(
                query=z_txt_q,
                key=z_img_kv,
                value=z_img_kv
            )
            return self.norm(attn_out.squeeze(1))

        elif self.fusion_type == 'gating':
            concat = torch.cat([z_img, z_txt], dim=-1)
            gate = torch.sigmoid(self.gate_proj(concat))
            return gate * z_img + (1 - gate) * z_txt
