"""
文本编码器模块 - 实现文本描述编码
基于设计文档 3.3 节
"""

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

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

        for param in self.bert.parameters():
            param.requires_grad = False

        hidden_size = self.bert.config.hidden_size
        self.adapt = nn.Sequential(
            nn.Linear(hidden_size, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [batch, seq_len] - BERT 输入 token IDs
            attention_mask: [batch, seq_len] - 注意力掩码
        Returns:
            z_txt: [batch, embed_dim] - 文本编码特征
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.adapt(cls_embedding)

    def unfreeze_bert(self):
        """解冻 BERT 参数进行微调"""
        for param in self.bert.parameters():
            param.requires_grad = True
