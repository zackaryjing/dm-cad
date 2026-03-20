"""
双模态 CAD 生成器 - 完整模型
基于设计文档 3.6 节
"""

import torch.nn as nn

from .cad_decoder import CADDecoder
from .fusion import ModalFusion
from .text_encoder import TextEncoder
from .view_encoder import MultiViewFusion, ViewEncoder


class DualModalCADGenerator(nn.Module):
    """双模态 CAD 生成器

    输入：8 视角图像 + 文本描述
    输出：DeepCAD 格式的 CAD 命令序列
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = self._default_config()
        if config:
            self.config.update(config)

        embed_dim = self.config['embed_dim']
        n_views = self.config['n_views']
        n_heads = self.config['n_heads']
        n_layers = self.config['n_layers']
        max_seq_len = self.config['max_seq_len']
        fusion_type = self.config.get('fusion_type', 'gating')
        start_token = self.config.get('start_token', 4)

        self.view_encoder = ViewEncoder(embed_dim=embed_dim)
        self.multi_view_fusion = MultiViewFusion(
            embed_dim=embed_dim,
            n_views=n_views,
            n_heads=n_heads
        )
        self.text_encoder = TextEncoder(embed_dim=embed_dim)
        self.modal_fusion = ModalFusion(embed_dim=embed_dim, fusion_type=fusion_type)
        self.cad_decoder = CADDecoder(
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            start_token=start_token
        )

    def _default_config(self):
        return {
            'embed_dim': 512,
            'n_heads': 8,
            'n_layers': 6,
            'max_seq_len': 120,
            'n_views': 8,
            'fusion_type': 'gating',
            'start_token': 4,
        }

    def forward(self, images, text_input_ids, text_attention_mask, tgt_cad_seq=None):
        """前向传播"""
        z_fused = self._encode_modalities(images, text_input_ids, text_attention_mask)
        return self.cad_decoder(z_fused, tgt_cad_seq, training=self.training)

    def generate(self, images, text_input_ids, text_attention_mask, max_steps=120):
        """推理模式 - 生成 CAD 序列"""
        z_fused = self._encode_modalities(images, text_input_ids, text_attention_mask)
        return self.cad_decoder.generate(z_fused, max_steps=max_steps)

    def encode_images(self, images):
        """仅图像编码"""
        batch_size, n_views, channels, height, width = images.shape
        embed_dim = self.config['embed_dim']
        images_flat = images.view(batch_size * n_views, channels, height, width)
        view_features = self.view_encoder(images_flat)
        view_features = view_features.view(batch_size, n_views, embed_dim)
        return self.multi_view_fusion(view_features)

    def encode_text(self, text_input_ids, text_attention_mask):
        """仅文本编码"""
        return self.text_encoder(text_input_ids, text_attention_mask)

    def _encode_modalities(self, images, text_input_ids, text_attention_mask):
        z_img = self.encode_images(images)
        z_txt = self.encode_text(text_input_ids, text_attention_mask)
        return self.modal_fusion(z_img, z_txt)
