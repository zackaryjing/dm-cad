"""
文本编码器模块 - 实现文本描述编码
基于设计文档 3.3 节
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    """文本编码器 (基于 BERT)

    使用预训练 BERT 编码文本描述，投影到 CAD latent 空间
    """
    def __init__(self, embed_dim=512, pretrained_bert='bert-base-uncased'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_bert)

        # 冻结 BERT 参数 (可选部分微调)
        for param in self.bert.parameters():
            param.requires_grad = False

        # 适配层：768 -> 512
        self.adapt = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [batch, seq_len] - BERT 输入 token IDs
            attention_mask: [batch, seq_len] - 注意力掩码
        Returns:
            z_txt: [batch, 512] - 文本编码特征
        """
        # BERT 编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # 使用 [CLS] token 作为句子表示
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch, 768]

        # 投影到 CAD latent 空间
        return self.adapt(cls_embedding)  # [batch, 512]

    def unfreeze_bert(self):
        """解冻 BERT 参数进行微调"""
        for param in self.bert.parameters():
            param.requires_grad = True
