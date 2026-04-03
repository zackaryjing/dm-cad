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
        freeze_vit = self.config.get('freeze_vit', True)
        pretrained_vit = self.config.get('pretrained_vit', True)
        visual_memory_mode = self.config.get('visual_memory_mode', 'view_tokens')
        use_global_fused_condition = self.config.get('use_global_fused_condition', True)
        decoder_condition_injection = self.config.get('decoder_condition_injection', 'film_residual')
        decoder_condition_hidden_dim = self.config.get('decoder_condition_hidden_dim', embed_dim)
        decoder_condition_scale = self.config.get('decoder_condition_scale', 1.0)

        self.view_encoder = ViewEncoder(
            embed_dim=embed_dim,
            pretrained=pretrained_vit,
            freeze_backbone=freeze_vit,
        )
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
            start_token=start_token,
            condition_injection=decoder_condition_injection,
            condition_hidden_dim=decoder_condition_hidden_dim,
            condition_scale=decoder_condition_scale,
        )
        self.visual_memory_mode = visual_memory_mode
        self.use_global_fused_condition = use_global_fused_condition

    def _default_config(self):
        return {
            'embed_dim': 512,
            'n_heads': 8,
            'n_layers': 6,
            'max_seq_len': 120,
            'n_views': 8,
            'fusion_type': 'gating',
            'start_token': 4,
            'freeze_vit': True,
            'pretrained_vit': True,
            'visual_memory_mode': 'view_tokens',
            'use_global_fused_condition': True,
            'decoder_condition_injection': 'film_residual',
            'decoder_condition_hidden_dim': 512,
            'decoder_condition_scale': 1.0,
        }

    def forward(self, images, text_input_ids, text_attention_mask, tgt_cad_seq=None):
        """前向传播"""
        visual_memory, global_condition = self._encode_modalities(images, text_input_ids, text_attention_mask)
        return self.cad_decoder(
            visual_memory,
            global_condition,
            tgt_cad_seq,
            training=self.training
        )

    def generate(self, images, text_input_ids, text_attention_mask, max_steps=120):
        """推理模式 - 生成 CAD 序列"""
        visual_memory, global_condition = self._encode_modalities(images, text_input_ids, text_attention_mask)
        return self.cad_decoder.generate(
            visual_memory,
            global_condition,
            max_steps=max_steps
        )

    def encode_images(self, images):
        """仅图像编码，返回全局图像特征。"""
        _, z_img_global = self.encode_image_memory(images)
        return z_img_global

    def encode_image_memory(self, images):
        """图像编码，返回视图级 memory token 和全局特征。"""
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
        z_img_tokens, z_img = self.encode_image_memory(images)
        z_txt = self.encode_text(text_input_ids, text_attention_mask)
        z_fused = self.modal_fusion(z_img, z_txt)

        if self.visual_memory_mode == 'global_only':
            visual_memory = z_img.unsqueeze(1)
        else:
            visual_memory = z_img_tokens

        global_condition = z_fused if self.use_global_fused_condition else None
        return visual_memory, global_condition
