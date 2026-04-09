"""
CAD 序列解码器模块 - 实现 CAD 命令序列生成
基于设计文档 3.5 节和 DeepCAD 架构

改进：
- 训练时使用标准 teacher forcing（右移一位）
- 解码时使用 causal mask，避免看到未来 token
- 推理时基于完整前缀做自回归生成
- Cross-Attention 读取视图级 memory token，而非单个全局向量
- 每层 Decoder 通过全局条件向量做残差式条件注入
"""

import torch
import torch.nn as nn


class ConditionedDecoderLayer(nn.Module):
    """带多级条件注入的 Decoder Layer。"""

    def __init__(self, embed_dim, n_heads, condition_hidden_dim=512,
                 condition_injection='film_residual', condition_scale=1.0, dropout=0.1):
        super().__init__()
        self.condition_injection = condition_injection
        self.condition_scale = float(condition_scale)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        if condition_injection == 'film_residual':
            self.condition_proj = nn.Sequential(
                nn.Linear(embed_dim, condition_hidden_dim),
                nn.GELU(),
                nn.Linear(condition_hidden_dim, embed_dim * 2),
            )
        else:
            self.condition_proj = None

    def forward(self, hidden_states, memory, cond=None, causal_mask=None):
        self_attn_output, _ = self.self_attn(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            attn_mask=causal_mask,
            need_weights=False
        )
        hidden_states = self.norm1(hidden_states + self.dropout(self_attn_output))

        cross_attn_output, _ = self.cross_attn(
            query=hidden_states,
            key=memory,
            value=memory,
            need_weights=False
        )
        hidden_states = self.norm2(hidden_states + self.dropout(cross_attn_output))

        ffn_output = self.ffn(hidden_states)
        hidden_states = self.norm3(hidden_states + self.dropout(ffn_output))
        return self._inject_condition(hidden_states, cond)

    def _inject_condition(self, hidden_states, cond):
        if cond is None or self.condition_proj is None or self.condition_scale == 0.0:
            return hidden_states

        gamma_beta = self.condition_proj(cond)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        conditioned = gamma * hidden_states + beta
        return hidden_states + self.condition_scale * conditioned


class ConditionedTransformerDecoder(nn.Module):
    """由多个带条件注入的 Decoder Layer 堆叠而成。"""

    def __init__(self, embed_dim, n_layers, n_heads, condition_hidden_dim=512,
                 condition_injection='film_residual', condition_scale=1.0, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            ConditionedDecoderLayer(
                embed_dim=embed_dim,
                n_heads=n_heads,
                condition_hidden_dim=condition_hidden_dim,
                condition_injection=condition_injection,
                condition_scale=condition_scale,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

    def forward(self, hidden_states, memory, cond=None, causal_mask=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, memory, cond=cond, causal_mask=causal_mask)
        return hidden_states


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
                 start_token=4, condition_injection='film_residual',
                 condition_hidden_dim=512, condition_scale=1.0,
                 n_param_bins=256):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.start_token = start_token

        self.n_cmd_types = 6
        self.cmd_embed = nn.Embedding(num_embeddings=self.n_cmd_types, embedding_dim=embed_dim)

        self.n_params = 19
        self.n_param_bins = n_param_bins
        self.param_embed = nn.Linear(self.n_params, embed_dim)

        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len + 1, embed_dim))
        self.transformer_decoder = ConditionedTransformerDecoder(
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            condition_hidden_dim=condition_hidden_dim,
            condition_injection=condition_injection,
            condition_scale=condition_scale,
            dropout=0.1,
        )

        self.cmd_head = nn.Linear(embed_dim, self.n_cmd_types)
        self.param_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_params * self.n_param_bins)
        )

    def forward(self, visual_memory, global_condition=None, tgt_seq=None, training=True):
        """
        Args:
            visual_memory: [batch, memory_len, embed_dim] - 视觉条件 memory
            global_condition: [batch, embed_dim] or None - 每层条件注入使用的全局向量
            tgt_seq: [batch, seq_len, 20] - 目标 CAD 序列
        Returns:
            cmd_logits: [batch, seq_len, 6] - 命令类型预测
            param_logits: [batch, seq_len, 19, 256] - 参数分类 logits
        """
        batch_size = visual_memory.shape[0]

        if tgt_seq is not None:
            decoder_input = self._shift_right(tgt_seq)
        else:
            decoder_input = self._build_start_sequence(batch_size, visual_memory.device)

        seq_len = decoder_input.shape[1]
        tgt_embed = self._embed_sequence(decoder_input)
        tgt_embed = tgt_embed + self.pos_embed[:, :seq_len, :]
        causal_mask = self._build_causal_mask(seq_len, visual_memory.device)

        output = self.transformer_decoder(
            tgt_embed,
            memory=visual_memory,
            cond=global_condition,
            causal_mask=causal_mask
        )

        cmd_logits = self.cmd_head(output)
        param_logits = self.param_head(output).view(batch_size, seq_len, self.n_params, self.n_param_bins)
        return cmd_logits, param_logits

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
            torch.ones((seq_len, seq_len), device=device, dtype=torch.bool),
            diagonal=1
        )

    def generate(self, visual_memory, global_condition=None, max_steps=120):
        """自回归生成 CAD 序列

        Returns:
            cmd_tensor: [batch, steps] - 生成的命令类型
            param_tensor: [batch, steps, 19] - 生成的离散参数值
        """
        batch_size = visual_memory.shape[0]
        device = visual_memory.device

        generated_cmds = []
        generated_params = []
        decoder_seq = self._build_start_sequence(batch_size, device)
        ended = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_steps):
            seq_len = decoder_seq.shape[1]
            tgt_embed = self._embed_sequence(decoder_seq)
            tgt_embed = tgt_embed + self.pos_embed[:, :seq_len, :]
            causal_mask = self._build_causal_mask(seq_len, device)

            output = self.transformer_decoder(
                tgt_embed,
                memory=visual_memory,
                cond=global_condition,
                causal_mask=causal_mask
            )
            last_hidden = output[:, -1:, :]

            cmd_logits = self.cmd_head(last_hidden)
            param_logits = self.param_head(last_hidden).view(batch_size, 1, self.n_params, self.n_param_bins)
            cmd_pred = torch.argmax(cmd_logits, dim=-1)
            param_pred = torch.argmax(param_logits, dim=-1)

            if generated_cmds:
                prev_cmd = generated_cmds[-1]
                prev_param = generated_params[-1]
                cmd_pred = torch.where(ended.unsqueeze(-1), prev_cmd, cmd_pred)
                param_pred = torch.where(ended.unsqueeze(-1).unsqueeze(-1), prev_param, param_pred)

            generated_cmds.append(cmd_pred)
            generated_params.append(param_pred)

            ended = ended | (cmd_pred.squeeze(-1) == 3)
            next_token = torch.cat([cmd_pred.float().unsqueeze(-1), param_pred.float()], dim=-1)
            decoder_seq = torch.cat([decoder_seq, next_token], dim=1)

            if ended.all():
                break

        if generated_cmds:
            cmd_tensor = torch.cat(generated_cmds, dim=1)
            param_tensor = torch.cat(generated_params, dim=1)
        else:
            cmd_tensor = torch.empty(batch_size, 0, dtype=torch.long, device=device)
            param_tensor = torch.empty(batch_size, 0, self.n_params, dtype=torch.long, device=device)

        return cmd_tensor, param_tensor
