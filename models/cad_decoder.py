"""
CAD 序列解码器模块 - 实现 CAD 命令序列生成
基于设计文档 3.5 节和 DeepCAD 架构

优化：
- 训练时使用标准 teacher forcing（右移一位）
- 解码时使用 causal mask，避免看到未来 token
- 推理时基于完整前缀做自回归生成
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
    def __init__(self, embed_dim=512, n_layers=6, n_heads=8, max_seq_len=120,
                 start_token=4):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.start_token = start_token

        self.n_cmd_types = 6
        self.cmd_embed = nn.Embedding(num_embeddings=self.n_cmd_types, embedding_dim=embed_dim)

        self.n_params = 19
        self.param_embed = nn.Linear(self.n_params, embed_dim)

        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len + 1, embed_dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

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
            tgt_seq: [batch, seq_len, 20] - 目标 CAD 序列
        Returns:
            cmd_logits: [batch, seq_len, 6] - 命令类型预测
            param_pred: [batch, seq_len, 19] - 参数预测
        """
        batch_size = z_fused.shape[0]

        if tgt_seq is not None:
            decoder_input = self._shift_right(tgt_seq)
        else:
            decoder_input = self._build_start_sequence(batch_size, z_fused.device)

        seq_len = decoder_input.shape[1]
        tgt_embed = self._embed_sequence(decoder_input)
        tgt_embed = tgt_embed + self.pos_embed[:, :seq_len, :]
        causal_mask = self._build_causal_mask(seq_len, z_fused.device)

        memory = z_fused.unsqueeze(1)
        output = self.transformer_decoder(tgt_embed, memory=memory, tgt_mask=causal_mask)

        cmd_logits = self.cmd_head(output)
        param_pred = self.param_head(output)
        return cmd_logits, param_pred

    def _embed_sequence(self, seq):
        """嵌入 CAD 序列"""
        cmd_types = seq[:, :, 0].long().clamp(0, self.n_cmd_types - 1)
        params = seq[:, :, 1:]

        cmd_emb = self.cmd_embed(cmd_types)
        param_emb = self.param_embed(params)
        return cmd_emb + param_emb

    def _build_start_sequence(self, batch_size, device):
        start_seq = torch.zeros(batch_size, 1, self.n_params + 1, device=device)
        start_seq[:, 0, 0] = self.start_token
        return start_seq

    def _shift_right(self, tgt_seq):
        start_seq = self._build_start_sequence(tgt_seq.shape[0], tgt_seq.device)
        return torch.cat([start_seq, tgt_seq[:, :-1, :]], dim=1)

    def _build_causal_mask(self, seq_len, device):
        return torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device),
            diagonal=1
        )

    def generate(self, z_fused, max_steps=120):
        """自回归生成 CAD 序列

        Returns:
            cmd_tensor: [batch, steps] - 生成的命令类型
            param_tensor: [batch, steps, 19] - 生成的参数
        """
        batch_size = z_fused.shape[0]
        device = z_fused.device

        generated_cmds = []
        generated_params = []
        decoder_seq = self._build_start_sequence(batch_size, device)
        ended = torch.zeros(batch_size, dtype=torch.bool, device=device)
        memory = z_fused.unsqueeze(1)

        for _ in range(max_steps):
            seq_len = decoder_seq.shape[1]
            tgt_embed = self._embed_sequence(decoder_seq)
            tgt_embed = tgt_embed + self.pos_embed[:, :seq_len, :]
            causal_mask = self._build_causal_mask(seq_len, device)

            output = self.transformer_decoder(tgt_embed, memory=memory, tgt_mask=causal_mask)
            last_hidden = output[:, -1:, :]

            cmd_logits = self.cmd_head(last_hidden)
            param_pred = self.param_head(last_hidden)
            cmd_pred = torch.argmax(cmd_logits, dim=-1)

            if generated_cmds:
                prev_cmd = generated_cmds[-1]
                prev_param = generated_params[-1]
                cmd_pred = torch.where(ended.unsqueeze(-1), prev_cmd, cmd_pred)
                param_pred = torch.where(ended.unsqueeze(-1).unsqueeze(-1), prev_param, param_pred)

            generated_cmds.append(cmd_pred)
            generated_params.append(param_pred)

            ended = ended | (cmd_pred.squeeze(-1) == 3)
            next_token = torch.cat([cmd_pred.float().unsqueeze(-1), param_pred], dim=-1)
            decoder_seq = torch.cat([decoder_seq, next_token], dim=1)

            if ended.all():
                break

        if generated_cmds:
            cmd_tensor = torch.cat(generated_cmds, dim=1)
            param_tensor = torch.cat(generated_params, dim=1)
        else:
            cmd_tensor = torch.empty(batch_size, 0, dtype=torch.long, device=device)
            param_tensor = torch.empty(batch_size, 0, self.n_params, device=device)

        return cmd_tensor, param_tensor
