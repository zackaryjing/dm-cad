"""
CAD 序列解码器模块 - 实现 CAD 命令序列生成
基于设计文档 3.5 节和 DeepCAD 架构

优化：
- 每层 TransformerDecoder 都进行 Memory Cross-Attention，增强长序列记忆
"""

import torch
import torch.nn as nn


class CADDecoder(nn.Module):
    """基于 DeepCAD 的 CAD 序列解码器

    将融合特征解码为 CAD 命令序列 (sketch + extrusion 操作)
    每层都进行 Memory Cross-Attention，确保层层感知条件信息

    命令类型（适配 DeepCAD 原始数据）:
    - 0: Line (线段)
    - 1: Arc (圆弧)
    - 2: Circle (圆)
    - 3: EOS (序列结束)
    - 4: SOL (实体开始)
    - 5: Ext (拉伸)
    """
    def __init__(self, embed_dim=512, n_layers=6, n_heads=8, max_seq_len=120):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        # CAD 命令嵌入 (6 种类型：Line, Arc, Circle, EOS, SOL, Ext)
        self.n_cmd_types = 6
        self.cmd_embed = nn.Embedding(num_embeddings=self.n_cmd_types, embedding_dim=embed_dim)

        # 参数嵌入 (19 维参数向量)
        self.n_params = 19
        self.param_embed = nn.Linear(self.n_params, embed_dim)

        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len + 1, embed_dim))

        # Transformer Decoder - 每层都会使用 memory 进行交叉注意力
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # 输出头
        self.cmd_head = nn.Linear(embed_dim, self.n_cmd_types)
        self.param_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_params)
        )

    def forward(self, z_fused, tgt_seq=None, training=True):
        """
        Args:
            z_fused: [batch, embed_dim] - 融合后的条件向量
            tgt_seq: 目标 CAD 序列 (training 时使用) [batch, seq_len, 20]
        Returns:
            cmd_logits: [batch, seq_len, 4] - 命令类型预测
            param_pred: [batch, seq_len, 20] - 参数预测
        """
        B = z_fused.shape[0]

        # 准备 decoder 输入
        if training and tgt_seq is not None:
            # Teacher forcing - 使用完整目标序列
            tgt_embed = self._embed_sequence(tgt_seq)
            seq_len = tgt_embed.shape[1]
        else:
            # 自回归生成 - 从 START token 开始
            tgt_embed = self.cmd_embed(torch.zeros(B, 1, dtype=torch.long).to(z_fused.device))
            seq_len = 1

        # 添加位置编码
        tgt_embed = tgt_embed + self.pos_embed[:, :seq_len, :]

        # memory = 条件向量 (每层都会进行交叉注意力)
        memory = z_fused.unsqueeze(1)  # [batch, 1, embed_dim]

        # Transformer 解码 - 每一层都使用 memory 进行交叉注意力
        output = self.transformer_decoder(tgt_embed, memory=memory)

        # 输出预测
        cmd_logits = self.cmd_head(output)
        param_pred = self.param_head(output)

        return cmd_logits, param_pred

    def _embed_sequence(self, seq):
        """嵌入 CAD 序列
        Args:
            seq: [batch, seq_len, 20] - CAD 序列 (第 0 维是命令类型，后 19 维是参数)
                 命令类型：0=Line, 1=Arc, 2=Circle, 3=EOS, 4=SOL, 5=Ext
        """
        cmd_types = seq[:, :, 0].long().clamp(0, self.n_cmd_types - 1)  # 限制命令类型在有效范围内
        params = seq[:, :, 1:]  # 取后 19 维参数

        cmd_emb = self.cmd_embed(cmd_types)
        param_emb = self.param_embed(params)

        return cmd_emb + param_emb

        cmd_emb = self.cmd_embed(cmd_types)
        param_emb = self.param_embed(params)

        return cmd_emb + param_emb

    def generate(self, z_fused, max_steps=120):
        """自回归生成 CAD 序列

        Args:
            z_fused: [batch, embed_dim] - 融合条件向量
            max_steps: 最大生成长度
        Returns:
            generated: 生成的命令序列列表
        """
        B = z_fused.shape[0]
        device = z_fused.device

        generated = []
        current_input = self.cmd_embed(torch.zeros(B, 1, dtype=torch.long).to(device))

        # 跟踪每个样本的生成状态
        ended = torch.zeros(B, dtype=torch.bool, device=device)  # 标记每个样本是否结束

        for step in range(max_steps):
            current_input = current_input + self.pos_embed[:, step:step+1, :]
            memory = z_fused.unsqueeze(1)  # [batch, 1, embed_dim]

            # Transformer 解码 - 每一层都使用 memory 进行交叉注意力
            output = self.transformer_decoder(current_input, memory=memory)

            cmd_logits = self.cmd_head(output[:, -1:, :])
            param_pred = self.param_head(output[:, -1:, :])

            cmd_pred = torch.argmax(cmd_logits, dim=-1)  # [B, 1]

            generated.append((cmd_pred, param_pred))

            # 检查是否所有样本都结束
            current_ended = (cmd_pred.squeeze(-1) == 3)  # END token
            ended = ended | current_ended
            if ended.all():
                break

            # 准备下一步输入 (仅对未结束的样本更新)
            next_embed = self.cmd_embed(cmd_pred) + self.param_embed(param_pred)
            current_input = next_embed

        return generated
