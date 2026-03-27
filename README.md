# Dual-Modal CAD Generator (DM-CAD)

基于多视图图像与文本描述的参数化 CAD 序列生成网络

## 项目结构

```
dmcad/
├── models/
│   ├── __init__.py
│   ├── view_encoder.py      # 视图编码器 (ViT + 多视图融合)
│   ├── text_encoder.py      # 文本编码器 (BERT)
│   ├── fusion.py            # 双模态融合模块
│   ├── cad_decoder.py       # CAD 序列解码器 (6 命令类型，19 参数)
│   └── dual_modal_cad.py    # 完整模型
├── data/
│   ├── __init__.py
│   ├── dataset.py           # 数据集类 (DeepCAD 格式)
│   ├── renderer.py          # CAD 渲染 pipeline
│   └── augment.py           # 数据增强
├── train/
│   ├── __init__.py
│   ├── train.py             # 训练脚本
│   ├── loss.py              # 损失函数 (6 命令类型掩码)
│   └── config.yaml          # 配置文件
├── eval/
│   ├── __init__.py
│   ├── evaluate.py          # 评估脚本
│   └── metrics.py           # 评估指标
├── utils/
│   ├── __init__.py
│   ├── visualize.py         # 可视化工具
│   └── export_step.py       # STEP 导出
├── train_main.py            # 训练入口
├── eval_main.py             # 评估入口
├── infer.py                 # 推理入口
├── requirements.txt         # 依赖列表
└── README.md                # 本文件
```

## 安装

```bash
# 创建 conda 环境
conda activate /home/jing/allprojects/pythonenvironment/dmcad

# 依赖已在 environment.yml 中定义
```

## 训练

```bash
# 使用默认配置训练 (data_root 从 config.yaml 读取)
python train_main.py --config train/config.yaml

# 从检查点恢复训练
python train_main.py --config train/config.yaml --resume checkpoints/epoch_10.pth
```

**配置说明** (`train/config.yaml`):
- `data.data_root`: 数据集根目录 (默认：`datasets/dataset_v1`)
- `data.train_ids_file`: 训练集 ids 文件 (**相对于 data_root**)
- `data.test_ids_file`: 测试集 ids 文件 (**相对于 data_root**)

## 评估

```bash
# 在测试集上评估
python eval_main.py --checkpoint checkpoints/best.pth
```

## 推理

```bash
# 方式 1：使用 8 视图图像和文本描述生成 CAD 序列
python infer.py     --checkpoint checkpoints/best.pth     --images view_00.png view_01.png ... view_07.png     --text "A rectangular box with a cylindrical hole through the center"

# 方式 2：直接从配置里的 test_ids 中选一个测试样本跑推理
python infer.py     --checkpoint checkpoints/best.pth     --config train/config_5k.yaml     --split test     --sample-index 0     --device cuda
```

## 模型架构

```
┌─────────────────────────────────────────────────────────────┐
│                    DM-CAD Architecture                       │
├─────────────────────────────────────────────────────────────┤
│  Image Branch (8 views)    Text Branch                      │
│  ┌───────────┐             ┌───────────┐                    │
│  │ ViT-Base  │             │ BERT-Base │                    │
│  └───────────┘             └───────────┘                    │
│       ↓                         ↓                           │
│  Multi-View Fusion        Adapt Layer                       │
│       ↓                         ↓                           │
│     z_img [512]            z_txt [512]                      │
│              ↓               ↓                              │
│         Modal Fusion (Gating)                      │
│                    ↓                                        │
│              z_fused [512]                                  │
│                    ↓                                        │
│         CAD Decoder (Transformer)                           │
│                    ↓                                        │
│    Command Head    Parameter Head                           │
│    (6 classes)     (19 dim)                                 │
└─────────────────────────────────────────────────────────────┘
```

## 数据集格式

使用 DeepCAD 原始格式，数据组织如下：

```
datasets/dataset_v1/
├── train_ids_5k.txt         # 训练样本 ID 列表
├── test_ids_5k.txt          # 测试样本 ID 列表
├── cad_desc/
│   └── <group_id>.json      # 文本描述 (按 group_id 组织)
├── cad_img/
│   └── <group_id>/
│       └── <sample_id>/
│           ├── <sample_id>_000.png
│           ├── <sample_id>_001.png
│           └── ... (8 个视图)
└── cad_vec/
    └── <group_id>/
        └── <sample_id>.h5   # CAD 向量序列
```

## 关键设计说明

### 命令类型 (6 种，DeepCAD 格式)

| ID | 名称 | 说明 | 有效参数维度 |
|----|------|------|-------------|
| 0 | Line | 线段 | x, y (终点坐标) |
| 1 | Arc | 圆弧 | x, y, alpha, f |
| 2 | Circle | 圆 | x, y, r |
| 3 | EOS | 序列结束 | 无 |
| 4 | SOL | 实体开始 | 无 |
| 5 | Ext | 拉伸 | 11 维挤压参数 |

### CAD 序列格式

- **Shape**: `[seq_len, 20]`
- **结构**: `[cmd_type, param_0, param_1, ..., param_18]`
  - `cmd_type`: 命令类型 (0-5)
  - `param_*`: 19 维参数

### 张量维度说明

| 模块 | 输入 | 输出 |
|------|------|------|
| ViewEncoder | `[B, 3, 224, 224]` | `[B, 512]` |
| MultiViewFusion | `[B, 8, 512]` | `[B, 512]` |
| TextEncoder | `[B, seq_len]` | `[B, 512]` |
| ModalFusion | `[B, 512], [B, 512]` | `[B, 512]` |
| CADDecoder | `[B, 512], [B, T, 20]` | `[B, T, 6]`, `[B, T, 19]` |

### 损失函数

```
L_total = cmd_weight * L_cmd + param_weight * L_param

L_cmd = CrossEntropy(cmd_logits, cmd_gt)  # 6 类
L_param = SmoothL1(param_pred, param_gt)  # 19 维，仅有效位置计算
```

## 评估指标

| 指标 | 说明 |
|------|------|
| Command Accuracy | 命令类型预测准确率 (6 类) |
| Parameter Accuracy | 参数预测准确率 (相对误差<0.1) |
| Chamfer Distance | 几何形状相似度 (点云) |
| Invalidity Ratio | 生成无效序列比例 |

## 常见问题

### 数据路径问题
- ids 文件路径**相对于 `data_root`**，在 `train/config.yaml` 中配置
- 训练时不需传递 `--data_dir`，直接从配置文件读取

### 命令类型不匹配
- 模型使用 **6 种命令类型** (DeepCAD 原始格式)
- 早期设计使用 4 种命令类型 (START/SKETCH/EXTRUDE/END)，已过时

### 维度不匹配
- CAD 序列：20 维 = 1 命令类型 + 19 参数
- 参数预测输出：**19 维** (不是 20 维)
- `cad_valid_mask` 过滤无效位置 (cmd_type < 0)

## 参考文献

1. DeepCAD: A Deep Generative Network for Computer-Aided Design Models (ICCV 2021)
2. Text2CAD: Generating Sequential CAD Designs from Text Prompts (NeurIPS 2024)
3. Automatic Reverse Engineering: Creating CAD Models from Multi-View Images (GCPR 2023)
4. CAD-MLLM: Unifying Multimodality-Conditioned CAD Generation With MLLM (arXiv 2024)

## License

MIT License
