"""
双模态 CAD 生成器 - 完整模型
基于设计文档 3.6 节
"""

import torch
import torch.nn as nn
from .view_encoder import ViewEncoder, MultiViewFusion
from .text_encoder import TextEncoder
from .fusion import ModalFusion
from .cad_decoder import CADDecoder


class DualModalCADGenerator(nn.Module):
    """双模态 CAD 生成器

    输入：8 视角图像 + 文本描述
    输出：DeepCAD 格式的 CAD 命令序列
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = config or self._default_config()

        # 图像编码器
        self.view_encoder = ViewEncoder(embed_dim=512)
        self.multi_view_fusion = MultiViewFusion(embed_dim=512, n_views=8)

        # 文本编码器
        self.text_encoder = TextEncoder(embed_dim=512)

        # 模态融合
        self.modal_fusion = ModalFusion(embed_dim=512, fusion_type='cross_attention')

        # CAD 解码器
        self.cad_decoder = CADDecoder(embed_dim=512)

    def _default_config(self):
        return {
            'embed_dim': 512,
            'n_heads': 8,
            'n_layers': 6,
            'max_seq_len': 120,
            'n_views': 8
        }

    def forward(self, images, text_input_ids, text_attention_mask, tgt_cad_seq=None):
        """
        Args:
            images: [batch, 8, 3, 224, 224] - 8 个视图
            text_input_ids: [batch, seq_len] - 文本 token IDs
            text_attention_mask: [batch, seq_len] - 文本注意力掩码
            tgt_cad_seq: [batch, cad_seq_len, 20] - 目标 CAD 序列
        Returns:
            cmd_logits: [batch, seq_len, 4] - 命令预测
            param_pred: [batch, seq_len, 19] - 参数预测
        """
        B = images.shape[0]

        # 图像编码
        images_flat = images.view(B * 8, 3, 224, 224)
        view_features = self.view_encoder(images_flat)  # [B*8, 512]
        view_features = view_features.view(B, 8, 512)
        z_img = self.multi_view_fusion(view_features)  # [B, 512]

        # 文本编码
        z_txt = self.text_encoder(text_input_ids, text_attention_mask)  # [B, 512]

        # 模态融合
        z_fused = self.modal_fusion(z_img, z_txt)  # [B, 512]

        # CAD 序列生成
        cmd_logits, param_pred = self.cad_decoder(z_fused, tgt_cad_seq, training=self.training)

        return cmd_logits, param_pred

    def generate(self, images, text_input_ids, text_attention_mask, max_steps=120):
        """推理模式 - 生成 CAD 序列

        Args:
            images: [batch, 8, 3, 224, 224] - 8 个视图
            text_input_ids: [batch, seq_len] - 文本 token IDs
            text_attention_mask: [batch, seq_len] - 文本注意力掩码
            max_steps: 最大生成长度
        Returns:
            generated: 生成的命令序列
        """
        B = images.shape[0]

        # 编码
        images_flat = images.view(B * 8, 3, 224, 224)
        view_features = self.view_encoder(images_flat)
        view_features = view_features.view(B, 8, 512)
        z_img = self.multi_view_fusion(view_features)

        z_txt = self.text_encoder(text_input_ids, text_attention_mask)
        z_fused = self.modal_fusion(z_img, z_txt)

        # 生成
        return self.cad_decoder.generate(z_fused, max_steps)

    def encode_images(self, images):
        """仅图像编码"""
        B = images.shape[0]
        images_flat = images.view(B * 8, 3, 224, 224)
        view_features = self.view_encoder(images_flat)
        view_features = view_features.view(B, 8, 512)
        return self.multi_view_fusion(view_features)

    def encode_text(self, text_input_ids, text_attention_mask):
        """仅文本编码"""
        return self.text_encoder(text_input_ids, text_attention_mask)
