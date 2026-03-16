# 双模态 CAD 生成网络设计方案

## 基于多视图图像与文本描述的参数化 CAD 序列生成

**作者**: AI Assistant  
**日期**: 2026-03-10  
**目标**: 毕业设计级别的技术方案

---

## 1. 研究背景与动机

### 1.1 问题定义

本设计旨在构建一个**双模态条件 CAD 生成网络**，输入为：
- **图像模态**: 8 视角线框 + 光照风格渲染图（从立方体顶点向原点观察）
- **文本模态**: 自然语言描述（从抽象到参数化级别）

输出为：**DeepCAD 格式的 CAD 命令序列**（sketch + extrusion 操作序列）

### 1.2 核心参考文献

| 论文 | 会议/年份 | 引用 | 代码开源 | 核心贡献 |
|------|----------|------|----------|----------|
| DeepCAD | ICCV 2021 | 354+ | ✅ [GitHub](https://github.com/rundiwu/DeepCAD) | 首个 CAD 序列生成 Transformer |
| Text2CAD | NeurIPS 2024 Spotlight | 84+ | ✅ [GitHub](https://github.com/SadilKhan/Text2CAD) | 文本到 CAD 序列，多级别标注 |
| Auto Reverse Eng. | GCPR 2023 | 8+ | ❌ | 多视图图像到 CAD |
| CAD-MLLM | arXiv 2024 | 66+ | ❌ | 多模态统一 CAD 生成 |

---

## 2. 相关工作深度分析

### 2.1 DeepCAD: 基础架构

#### 2.1.1 CAD 序列表示

DeepCAD 将 CAD 模型表示为**命令序列**，每个命令包含：

```
命令序列 = [CMD_START, sketch_1, extrude_1, sketch_2, extrude_2, ..., CMD_END]
```

**Sketch 命令结构** (19 维向量):
```
[cmd_type=0, face_type, has_arc, x1, y1, x2, y2, ..., radius, extrude_dir_x, extrude_dir_y, extrude_dir_z, extrude_dist]
```

**Extrusion 命令结构** (10 维向量):
```
[cmd_type=1, profile_id, extrude_op, extrude_dist, ...]
```

#### 2.1.2 网络架构

```
┌─────────────────────────────────────────────────────────────┐
│                    DeepCAD Autoencoder                       │
├─────────────────────────────────────────────────────────────┤
│  Input: CAD Sequence (vectorized)                           │
│         ↓                                                    │
│  ┌──────────────────┐                                       │
│  │ Command Embedding │ (learned embedding table)            │
│  │ + Positional Enc  │                                       │
│  └──────────────────┘                                       │
│         ↓                                                    │
│  ┌──────────────────────────────────────┐                   │
│  │    Transformer Encoder (6 layers)    │ ← Latent z        │
│  │    d_model=512, n_head=8             │                   │
│  └──────────────────────────────────────┘                   │
│         ↓                                                    │
│  ┌──────────────────────────────────────┐                   │
│  │    Transformer Decoder (6 layers)    │                   │
│  │    d_model=512, n_head=8             │                   │
│  └──────────────────────────────────────┘                   │
│         ↓                                                    │
│  ┌──────────────────┐                                       │
│  │ Command Predictor │ (MLP → softmax over cmd types)       │
│  │ Parameter Predictor│ (MLP → continuous params)           │
│  └──────────────────┘                                       │
└─────────────────────────────────────────────────────────────┘
```

**关键设计点**:
- 命令类型预测：分类问题（sketch vs extrusion）
- 参数预测：回归问题（坐标、半径、距离等）
- 两阶段训练：先训练 Autoencoder，再训练 Latent GAN

#### 2.1.3 损失函数

```python
L_total = L_cmd + λ_param * L_param + λ_kl * L_KL

L_cmd = CrossEntropy(cmd_pred, cmd_gt)
L_param = SmoothL1(param_pred, param_gt)  # 仅对有效命令计算
L_KL = KL(q(z|x) || p(z))  # VAE 正则化
```

### 2.2 Text2CAD: 文本条件扩展

#### 2.2.1 架构设计

Text2CAD 在 DeepCAD 基础上增加了**文本条件机制**:

```
┌─────────────────────────────────────────────────────────────┐
│                    Text2CAD Architecture                     │
├─────────────────────────────────────────────────────────────┤
│  Text Input: "A cylinder with a hole through the center"    │
│         ↓                                                    │
│  ┌──────────────────┐                                       │
│  │  BERT Encoder    │ (pretrained, frozen or fine-tuned)   │
│  │  + Adapt Layer   │ (project to d_model=512)             │
│  └──────────────────┘                                       │
│         ↓ T_adapt (text embedding)                          │
│                                                            │
│  CAD Sequence Input                                        │
│         ↓                                                    │
│  ┌──────────────────┐                                       │
│  │ CAD Embedding    │                                       │
│  └──────────────────┘                                       │
│         ↓                                                    │
│  ┌──────────────────────────────────────┐                   │
│  │  Transformer Decoder (conditioned)   │ ← T_adapt as memory│
│  │  (cross-attention with text)         │                   │
│  └──────────────────────────────────────┘                   │
│         ↓                                                    │
│  Command + Parameter Prediction                              │
└─────────────────────────────────────────────────────────────┘
```

#### 2.2.2 文本编码细节

```python
# Text2CAD 文本处理流程
text_input = "draw a circle at origin with radius 10 and extrude 5 units"

# 1. BERT 编码
bert_output = BERT(text_input)  # [batch, seq_len, 768]
text_pool = bert_output[:, 0, :]  # [CLS] token, [batch, 768]

# 2. 适配层投影到 CAD latent 空间
adapt_layer = Linear(768, 512)
T_adapt = adapt_layer(text_pool)  # [batch, 512]

# 3. 作为 decoder 的 memory 输入
cad_output = TransformerDecoder(
    cad_embed, 
    memory=T_adapt.unsqueeze(1)  # [batch, 1, 512]
)
```

#### 2.2.3 数据标注策略

Text2CAD 使用**两阶段标注 pipeline**:

1. **Stage 1 (VLM)**: LLaVA-NeXT 生成形状描述
2. **Stage 2 (LLM)**: Mixtral-8x7B 生成多级别文本提示
   - Abstract: "a box with a hole"
   - Beginner: "draw a rectangle and extrude it"
   - Intermediate: "draw 20x30 rectangle, extrude 10mm"
   - Expert: "sketch: rect(0,0,20,30), extrude(d=10,dir=z)"

### 2.3 Automatic Reverse Engineering: 多视图图像编码

#### 2.3.1 架构概述

```
┌─────────────────────────────────────────────────────────────┐
│           Auto Reverse Engineering Architecture              │
├─────────────────────────────────────────────────────────────┤
│  Input: 8 Multi-view Images (224x224)                       │
│         ↓                                                    │
│  ┌──────────────────┐                                       │
│  │ CNN Encoder      │ (ResNet-18/34, shared weights)        │
│  │ (per-view)       │ output: [batch, 8, 512]               │
│  └──────────────────┘                                       │
│         ↓                                                    │
│  ┌──────────────────┐                                       │
│  │ Multi-View Pool  │ (max/attention pooling across views)  │
│  │                  │ output: [batch, 512]                  │
│  └──────────────────┘                                       │
│         ↓                                                    │
│  ┌──────────────────┐                                       │
│  │ Projection Layer │ (512 → 512)                           │
│  └──────────────────┘                                       │
│         ↓ z_img                                             │
│  ┌──────────────────────────────────────┐                   │
│  │  Transformer Decoder                 │                   │
│  │  (conditioned on z_img)              │                   │
│  └──────────────────────────────────────┘                   │
│         ↓                                                    │
│  CAD Sequence Output                                         │
└─────────────────────────────────────────────────────────────┘
```

#### 2.3.2 多视图池化

```python
# 方案 1: Max Pooling
z_img = torch.max(view_features, dim=1)[0]  # [batch, 512]

# 方案 2: Attention Pooling (推荐)
attention_weights = softmax(Linear(view_features))  # [batch, 8, 1]
z_img = sum(attention_weights * view_features, dim=1)  # [batch, 512]

# 方案 3: Transformer Pooling
view_tokens = view_features + positional_encoding(8)
z_img = TransformerEncoder(view_tokens)[:, 0, :]  # [CLS] token
```

### 2.4 CAD-MLLM: 多模态统一框架

#### 2.4.1 核心思想

CAD-MLLM 提出**统一的多模态 CAD 生成框架**:

```
┌─────────────────────────────────────────────────────────────┐
│                    CAD-MLLM Architecture                     │
├─────────────────────────────────────────────────────────────┤
│  Modality Encoders (frozen):                                │
│  - Text: LLM tokenizer + embedding                          │
│  - Image: ViT / ResNet (pretrained)                         │
│  - Point Cloud: PointNet++ (pretrained)                     │
│                                                            │
│         ↓ (project to common space)                         │
│                                                            │
│  ┌──────────────────────────────────────┐                   │
│  │     LLM (LLaMA / Mistral)            │                   │
│  │     + LoRA Fine-tuning               │                   │
│  └──────────────────────────────────────┘                   │
│         ↓                                                    │
│  CAD Command Sequence (tokenized)                           │
└─────────────────────────────────────────────────────────────┘
```

#### 2.4.2 关键设计

- **特征对齐**: 各模态 encoder 输出投影到 LLM 的 embedding 空间
- **LoRA 微调**: 仅微调 LLM 的低秩适配器，保持主干冻结
- **Omni-CAD 数据集**: 450K 实例，包含文本 + 多视图图像 + 点云+CAD 序列

---

## 3. Proposed 网络架构设计

### 3.1 整体架构

基于上述分析，我们提出**Dual-Modal CAD Generator (DM-CAD)**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DM-CAD Architecture                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐    ┌─────────────────┐                         │
│  │  Image Branch   │    │  Text Branch    │                         │
│  │  (8 views)      │    │  (description)  │                         │
│  │                 │    │                 │                         │
│  │  ┌───────────┐  │    │  ┌───────────┐  │                         │
│  │  │ ViT-Base  │  │    │  │ BERT-Base │  │                         │
│  │  │ (frozen)  │  │    │  │ (frozen)  │  │                         │
│  │  └───────────┘  │    │  └───────────┘  │                         │
│  │       ↓         │    │       ↓         │                         │
│  │  ┌───────────┐  │    │  ┌───────────┐  │                         │
│  │  │ View      │  │    │  │ Adapt     │  │                         │
│  │  │ Encoder   │  │    │  │ Layer     │  │                         │
│  │  │ (shared)  │  │    │  │ 768→512   │  │                         │
│  │  └───────────┘  │    │  └───────────┘  │                         │
│  │       ↓         │    │                 │                         │
│  │  ┌───────────┐  │    │                 │                         │
│  │  │ Multi-View│  │    │                 │                         │
│  │  │ Attention │  │    │                 │                         │
│  │  │ Pooling   │  │    │                 │                         │
│  │  └───────────┘  │    │                 │                         │
│  │       ↓         │    │                 │                         │
│  │  z_img [512]    │    │  z_txt [512]    │                         │
│  └─────────────────┘    └─────────────────┘                         │
│           ↓                       ↓                                  │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              Modality Fusion Layer                            │   │
│  │                                                               │   │
│  │  Option A: Concat + Project                                   │   │
│  │    z_fused = Linear([z_img; z_txt]) → [512]                  │   │
│  │                                                               │   │
│  │  Option B: Cross-Attention Fusion (推荐）                     │   │
│  │    z_fused = CrossAttn(query=z_txt, key=z_img, value=z_img)  │   │
│  │                                                               │   │
│  │  Option C: Gating Mechanism                                   │   │
│  │    gate = sigmoid(Linear([z_img; z_txt]))                    │   │
│  │    z_fused = gate * z_img + (1-gate) * z_txt                 │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              ↓                                       │
│                       z_fused [512]                                  │
│                              ↓                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              DeepCAD-style Decoder                            │   │
│  │                                                               │   │
│  │  ┌─────────────────────────────────────────────────────────┐ │   │
│  │  │  Transformer Decoder (6 layers)                         │ │   │
│  │  │  d_model=512, n_head=8, d_ff=2048                       │ │   │
│  │  │  memory = z_fused.unsqueeze(1)                          │ │   │
│  │  └─────────────────────────────────────────────────────────┘ │   │
│  │                              ↓                                │   │
│  │  ┌─────────────────┐  ┌─────────────────┐                    │   │
│  │  │ Command Head    │  │ Parameter Head  │                    │   │
│  │  │ (Linear+Softmax)│  │ (MLP+ReLU)      │                    │   │
│  │  │ 3 classes       │  │ 19 dim output   │                    │   │
│  │  └─────────────────┘  └─────────────────┘                    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 图像编码器设计

#### 3.2.1 视图编码

```python
class ViewEncoder(nn.Module):
    """单个视图的编码器"""
    def __init__(self, embed_dim=512):
        super().__init__()
        # 使用预训练 ViT，冻结大部分参数
        self.vit = vit_base_patch16_224(pretrained=True)
        self.vit.head = nn.Identity()  # 移除分类头
        
        # 投影层
        self.project = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU()
        )
    
    def forward(self, x):
        # x: [batch, 3, 224, 224]
        features = self.vit(x)  # [batch, 768]
        return self.project(features)  # [batch, 512]
```

#### 3.2.2 多视图融合

```python
class MultiViewFusion(nn.Module):
    """多视图注意力池化"""
    def __init__(self, embed_dim=512, n_views=8, n_heads=8):
        super().__init__()
        self.view_pos_embed = nn.Parameter(torch.randn(1, n_views, embed_dim))
        
        # 使用 Transformer encoder 进行视图间注意力
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 聚合为单个向量
        self.aggregate = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, view_features):
        # view_features: [batch, n_views, embed_dim]
        B, N, D = view_features.shape
        
        # 添加位置编码
        view_features = view_features + self.view_pos_embed.expand(B, -1, -1)
        
        # Transformer 编码
        encoded = self.transformer(view_features)  # [batch, n_views, embed_dim]
        
        # 全局平均池化 (或使用 [CLS] token)
        fused = encoded.mean(dim=1)  # [batch, embed_dim]
        
        return self.aggregate(fused)
```

### 3.3 文本编码器设计

```python
class TextEncoder(nn.Module):
    """文本编码器 (基于 BERT)"""
    def __init__(self, embed_dim=512, pretrained_bert='bert-base-uncased'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_bert)
        
        # 冻结 BERT 参数 (可选部分微调)
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # 适配层
        self.adapt = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, input_ids, attention_mask):
        # BERT 编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 使用 [CLS] token 作为句子表示
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch, 768]
        
        # 投影到 CAD latent 空间
        return self.adapt(cls_embedding)  # [batch, 512]
```

### 3.4 模态融合设计

```python
class ModalFusion(nn.Module):
    """双模态融合模块"""
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
    
    def forward(self, z_img, z_txt):
        # z_img, z_txt: [batch, embed_dim]
        
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
```

### 3.5 CAD 序列解码器

```python
class CADDecoder(nn.Module):
    """基于 DeepCAD 的 CAD 序列解码器"""
    def __init__(self, embed_dim=512, n_layers=6, n_heads=8, max_seq_len=20):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        
        # CAD 命令嵌入
        self.cmd_embed = nn.Embedding(num_embeddings=3, embedding_dim=embed_dim)
        self.param_embed = nn.Linear(19, embed_dim)
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len + 1, embed_dim))
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # 输出头
        self.cmd_head = nn.Linear(embed_dim, 3)  # START, SKETCH, EXTRUDE, END
        self.param_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 19)
        )
    
    def forward(self, z_fused, tgt_seq=None, training=True):
        """
        z_fused: [batch, embed_dim] - 融合后的条件向量
        tgt_seq: 目标 CAD 序列 (training 时使用)
        """
        B = z_fused.shape[0]
        
        # 准备 decoder 输入
        if training and tgt_seq is not None:
            # Teacher forcing
            tgt_embed = self._embed_sequence(tgt_seq)
        else:
            # 自回归生成
            tgt_embed = self.cmd_embed(torch.zeros(B, 1, dtype=torch.long).to(z_fused.device))
        
        # 添加位置编码
        tgt_embed = tgt_embed + self.pos_embed[:, :tgt_embed.shape[1], :]
        
        # memory = 条件向量
        memory = z_fused.unsqueeze(1)  # [batch, 1, embed_dim]
        
        # Transformer 解码
        output = self.transformer_decoder(tgt_embed, memory=memory)
        
        # 输出预测
        cmd_logits = self.cmd_head(output)
        param_pred = self.param_head(output)
        
        return cmd_logits, param_pred
    
    def _embed_sequence(self, seq):
        # seq: [batch, seq_len, 19+1] (19 params + 1 cmd_type)
        cmd_types = seq[:, :, 0].long()
        params = seq[:, :, 1:]
        
        cmd_emb = self.cmd_embed(cmd_types)
        param_emb = self.param_embed(params)
        
        return cmd_emb + param_emb
    
    def generate(self, z_fused, max_steps=20):
        """自回归生成 CAD 序列"""
        B = z_fused.shape[0]
        device = z_fused.device
        
        generated = []
        current_input = self.cmd_embed(torch.zeros(B, 1, dtype=torch.long).to(device))
        
        for step in range(max_steps):
            current_input = current_input + self.pos_embed[:, step:step+1, :]
            memory = z_fused.unsqueeze(1)
            
            output = self.transformer_decoder(current_input, memory=memory)
            
            cmd_logits = self.cmd_head(output[:, -1:, :])
            param_pred = self.param_head(output[:, -1:, :])
            
            cmd_pred = torch.argmax(cmd_logits, dim=-1)
            
            generated.append((cmd_pred, param_pred))
            
            if cmd_pred.item() == 3:  # END token
                break
            
            # 准备下一步输入
            next_embed = self.cmd_embed(cmd_pred) + self.param_embed(param_pred)
            current_input = next_embed
        
        return generated
```

### 3.6 完整模型

```python
class DualModalCADGenerator(nn.Module):
    """双模态 CAD 生成器"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 图像编码器
        self.view_encoder = ViewEncoder(embed_dim=512)
        self.multi_view_fusion = MultiViewFusion(embed_dim=512, n_views=8)
        
        # 文本编码器
        self.text_encoder = TextEncoder(embed_dim=512)
        
        # 模态融合
        self.modal_fusion = ModalFusion(embed_dim=512, fusion_type='cross_attention')
        
        # CAD 解码器
        self.cad_decoder = CADDecoder(embed_dim=512)
    
    def forward(self, images, text_input_ids, text_attention_mask, tgt_cad_seq=None):
        """
        images: [batch, 8, 3, 224, 224] - 8 个视图
        text_input_ids: [batch, seq_len]
        text_attention_mask: [batch, seq_len]
        tgt_cad_seq: [batch, cad_seq_len, 20] - 目标 CAD 序列
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
    
    def generate(self, images, text_input_ids, text_attention_mask, max_steps=20):
        """推理模式"""
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
```

---

## 4. 训练方案

### 4.1 损失函数

```python
class CADLoss(nn.Module):
    def __init__(self, cmd_weight=1.0, param_weight=0.5, kl_weight=0.001):
        super().__init__()
        self.cmd_weight = cmd_weight
        self.param_weight = param_weight
        self.kl_weight = kl_weight
        
        self.cmd_criterion = nn.CrossEntropyLoss()
        self.param_criterion = nn.SmoothL1Loss()
    
    def forward(self, cmd_logits, param_pred, cmd_gt, param_gt, valid_mask):
        """
        cmd_logits: [batch, seq_len, n_commands]
        param_pred: [batch, seq_len, n_params]
        cmd_gt: [batch, seq_len] - 真实命令类型
        param_gt: [batch, seq_len, n_params] - 真实参数
        valid_mask: [batch, seq_len] - 有效位置掩码
        """
        # 命令损失
        cmd_loss = self.cmd_criterion(
            cmd_logits.view(-1, cmd_logits.shape[-1]),
            cmd_gt.view(-1)
        )
        
        # 参数损失 (仅对有效命令计算)
        param_loss = self.param_criterion(
            param_pred[valid_mask],
            param_gt[valid_mask]
        )
        
        # 总损失
        total_loss = self.cmd_weight * cmd_loss + self.param_weight * param_loss
        
        return total_loss, {'cmd_loss': cmd_loss, 'param_loss': param_loss}
```

### 4.2 训练策略

```python
# 分阶段训练策略
training_stages = [
    {
        'name': 'Stage 1: Image Encoder Warmup',
        'epochs': 10,
        'trainable': ['multi_view_fusion', 'modal_fusion', 'cad_decoder'],
        'frozen': ['view_encoder.vit', 'text_encoder.bert'],
        'lr': 1e-4
    },
    {
        'name': 'Stage 2: Full Model Training',
        'epochs': 50,
        'trainable': 'all',
        'lr': 5e-5
    },
    {
        'name': 'Stage 3: Fine-tuning with LoRA',
        'epochs': 20,
        'trainable': ['lora_adapters'],
        'frozen': 'all',
        'lr': 1e-5
    }
]
```

### 4.3 数据增强

```python
class CADDataAugmentation:
    """CAD 训练数据增强"""
    def __init__(self):
        self.image_transforms = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def augment_views(self, views):
        # views: [8, 3, 224, 224]
        augmented = []
        for view in views:
            view_pil = transforms.ToPILImage()(view)
            view_aug = self.image_transforms(view_pil)
            augmented.append(transforms.ToTensor()(view_aug))
        return torch.stack(augmented)
    
    def augment_text(self, text):
        # 文本增强：同义词替换、随机删除等
        # 可使用 nlpaug 库
        pass
```

---

## 5. 数据集方案

### 5.1 数据来源

| 数据集 | 规模 | 内容 | 获取方式 |
|--------|------|------|----------|
| DeepCAD | 178K | CAD 序列 | [下载](http://www.cs.columbia.edu/cg/deepcad/data.tar) |
| Text2CAD Annotations | 660K | 文本标注 | [HuggingFace](https://huggingface.co/datasets/SadilKhan/Text2CAD) |
| Omni-CAD | 450K | 多模态数据 | 需联系作者 |

### 5.2 自渲染数据生成

由于需要**线框 + 光照风格的 8 视角图**，建议自建渲染 pipeline:

```python
class CADRenderer:
    """CAD 模型多视图渲染器"""
    def __init__(self, img_size=224):
        self.img_size = img_size
        # 8 个视角：立方体顶点向原点看
        self.camera_positions = [
            (1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, 1),
            (1, 1, -1), (-1, 1, -1), (1, -1, -1), (-1, -1, -1)
        ]
    
    def render_wireframe(self, cad_model, view_idx):
        # 使用 OpenCASCADE 或 trimesh 渲染线框图
        pass
    
    def render_shaded(self, cad_model, view_idx):
        # 使用 Blender 或 PyTorch3D 渲染光照图
        pass
    
    def render_pair(self, cad_model):
        """渲染一对线框 + 光照图"""
        views = []
        for i in range(8):
            wireframe = self.render_wireframe(cad_model, i)
            shaded = self.render_shaded(cad_model, i)
            # 拼接为双通道或分别处理
            views.append(torch.cat([wireframe, shaded], dim=0))
        return torch.stack(views)  # [8, 6, 224, 224]
```

### 5.3 数据格式

```python
# 训练样本格式
training_sample = {
    'uid': 'cad_00001',
    'images': torch.FloatTensor(8, 6, 224, 224),  # 8 视图，每视图 6 通道 (线框 3+光照 3)
    'text': 'A rectangular box with a cylindrical hole through the center',
    'text_input_ids': torch.LongTensor([101, 2345, ...]),
    'text_attention_mask': torch.LongTensor([1, 1, 1, ...]),
    'cad_seq': torch.FloatTensor(seq_len, 20),  # CAD 命令序列
    'cad_valid_mask': torch.BoolTensor(seq_len)  # 有效序列掩码
}
```

---

## 6. 评估指标

### 6.1 序列级指标

```python
def evaluate_sequence_accuracy(pred_cmds, gt_cmds):
    """命令类型准确率"""
    correct = (pred_cmds == gt_cmds).sum()
    total = gt_cmds.numel()
    return correct / total

def evaluate_parameter_accuracy(pred_params, gt_params, threshold=0.1):
    """参数准确率 (相对误差<threshold 视为正确)"""
    rel_error = torch.abs(pred_params - gt_params) / (torch.abs(gt_params) + 1e-8)
    correct = (rel_error < threshold).float()
    return correct.mean()
```

### 6.2 几何级指标

```python
def evaluate_chamfer_distance(pred_pc, gt_pc):
    """Chamfer 距离 (需要先将 CAD 序列转换为点云)"""
    # 使用 DeepCAD 的 evaluation 脚本
    pass

def evaluate_invalidity_ratio(cad_sequences):
    """无效序列比例"""
    # 检查 CAD 序列的几何有效性
    pass
```

### 6.3 推荐评估组合

| 指标 | 权重 | 说明 |
|------|------|------|
| Command Accuracy | 30% | 命令类型预测准确率 |
| Parameter Accuracy | 30% | 参数预测准确率 |
| Chamfer Distance | 25% | 几何形状相似度 |
| Invalidity Ratio | 15% | 生成有效性 |

---

## 7. 实现路线图

### 7.1 阶段划分

```
Week 1-2: 环境搭建与数据准备
├── 配置 PyTorch 环境
├── 下载 DeepCAD 数据集
├── 实现 CAD 渲染 pipeline
└── 数据预处理与增强

Week 3-4: 基础模型实现
├── 实现 DeepCAD 复现
├── 验证 Autoencoder 性能
└── 建立评估 pipeline

Week 5-6: 单模态扩展
├── 实现图像编码器
├── 实现文本编码器
└── 分别训练单模态条件生成

Week 7-8: 双模态融合
├── 实现融合模块
├── 联合训练
└── 超参数调优

Week 9-10: 实验与评估
├── 消融实验
├── 对比实验
└── 结果分析

Week 11-12: 论文撰写
├── 整理实验结果
├── 撰写毕业论文
└── 准备答辩
```

### 7.2 计算资源需求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| GPU | RTX 3070 (8GB) | RTX 4090 (24GB) |
| 内存 | 32GB | 64GB |
| 存储 | 100GB SSD | 500GB NVMe |
| 训练时间 | ~3 天 | ~12 小时 |

---

## 8. 可行性分析

### 8.1 技术可行性 ✅

- **DeepCAD 代码开源**: 完整实现可参考
- **Text2CAD 代码开源**: 文本条件机制可直接借鉴
- **预训练模型可用**: ViT、BERT 等均有成熟实现
- **PyTorch 生态完善**: Transformer、注意力机制均为标准组件

### 8.2 数据可行性 ⚠️

- **DeepCAD 数据集**: 公开可用，178K 样本充足
- **文本标注**: Text2CAD 已提供 660K 标注
- **多视图图像**: 需自建渲染 pipeline (工作量可控)

### 8.3 复杂度评估

| 模块 | 复杂度 | 备注 |
|------|--------|------|
| 图像编码器 | 中 | ViT 冻结，仅训练投影层 |
| 文本编码器 | 低 | BERT 冻结，仅训练适配层 |
| 融合模块 | 低 | 标准注意力机制 |
| CAD 解码器 | 中 | DeepCAD 已有实现 |
| 数据渲染 | 中 | 一次性工作 |

**总体评估**: 适合本科/硕士毕业设计，工作量适中，技术难度可控

---

## 9. 潜在风险与应对

### 9.1 风险识别

| 风险 | 可能性 | 影响 | 应对措施 |
|------|--------|------|----------|
| 渲染 pipeline 开发困难 | 中 | 高 | 使用现成工具 (trimesh/Blender) |
| 训练不收敛 | 低 | 高 | 分阶段训练，先冻结 encoder |
| 显存不足 | 中 | 中 | 梯度累积，减小 batch size |
| 生成质量差 | 中 | 高 | 增加训练数据，调优超参数 |

### 9.2 降级方案

如果双模态效果不佳:
1. 回退到单模态 (仅文本或仅图像)
2. 简化融合机制 (concat 代替 cross-attention)
3. 减少视图数量 (8→4)

---

## 10. 总结

### 10.1 核心创新点

1. **双模态条件 CAD 生成**: 首次结合多视图图像与文本描述
2. **线框 + 光照双通道输入**: 同时利用几何轮廓与外观信息
3. **轻量级融合设计**: 基于 cross-attention 的高效融合

### 10.2 预期贡献

- 开源代码实现
- 多视图 CAD 渲染数据集
- 消融实验与分析

### 10.3 参考实现清单

```
项目结构:
dual_modal_cad/
├── models/
│   ├── view_encoder.py      # 视图编码器
│   ├── text_encoder.py      # 文本编码器
│   ├── fusion.py            # 融合模块
│   ├── cad_decoder.py       # CAD 解码器
│   └── dual_modal_cad.py    # 完整模型
├── data/
│   ├── dataset.py           # 数据集类
│   ├── renderer.py          # 渲染 pipeline
│   └── augment.py           # 数据增强
├── train/
│   ├── train.py             # 训练脚本
│   ├── loss.py              # 损失函数
│   └── config.yaml          # 配置文件
├── eval/
│   ├── evaluate.py          # 评估脚本
│   └── metrics.py           # 评估指标
└── utils/
    ├── visualize.py         # 可视化工具
    └── export_step.py       # STEP 导出
```

---

## 附录 A: 关键超参数

```yaml
# config.yaml
model:
  embed_dim: 512
  n_heads: 8
  n_layers: 6
  max_seq_len: 20
  n_views: 8

training:
  batch_size: 32
  lr: 5e-5
  epochs: 80
  warmup_epochs: 5
  gradient_clip: 1.0
  
loss:
  cmd_weight: 1.0
  param_weight: 0.5
  
data:
  img_size: 224
  cad_vec_dir: ./data/cad_vec
  text_anno_dir: ./data/text_annotations
```

---

## 附录 B: 核心论文引用

```bibtex
@InProceedings{Wu_2021_ICCV,
    author    = {Wu, Rundi and Xiao, Chang and Zheng, Changxi},
    title     = {DeepCAD: A Deep Generative Network for Computer-Aided Design Models},
    booktitle = {ICCV},
    year      = {2021}
}

@inproceedings{khan2024text2cad,
    title = {Text2CAD: Generating Sequential CAD Designs from Text Prompts},
    author = {Khan, Mohammad Sadil et al.},
    booktitle = {NeurIPS},
    year = {2024}
}

@inproceedings{homann2023auto,
    title = {Automatic Reverse Engineering: Creating CAD Models from Multi-View Images},
    author = {Homann, Hanno et al.},
    booktitle = {GCPR},
    year = {2023}
}

@article{xu2024cadmllm,
    title = {CAD-MLLM: Unifying Multimodality-Conditioned CAD Generation With MLLM},
    author = {Xu, Jingwei et al.},
    journal = {arXiv preprint arXiv:2411.04954},
    year = {2024}
}
```

---

**文档版本**: 1.0  
**最后更新**: 2026-03-10
