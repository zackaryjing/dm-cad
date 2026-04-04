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
# 使用默认配置训练 (会自动创建独立的 run 目录)
python train_main.py --config train/config.yaml

# 从检查点恢复训练 (继续写入该 checkpoint 所在的 run 目录)
python train_main.py --config train/config.yaml --resume runs/dmcad/<run_name>/checkpoints/epoch_10.pth

# 从旧 checkpoint 初始化参数，但创建新的 run 目录继续实验
# 这种模式只加载模型权重；optimizer、scheduler、epoch、best_val_loss 都会按新训练重置
python train_main.py --config train/config.yaml --resume runs/dmcad/<run_name>/checkpoints/epoch_10.pth --no-resume-in-place
```

`--resume` 的两种模式：
- 默认行为：原地续训。新的 checkpoint、TensorBoard events、`config.resolved.yaml` 继续写入该 checkpoint 所在的 run 目录。
- `--resume --no-resume-in-place`：新建 run。会像一次新的训练任务那样创建独立目录，但模型参数从给定 checkpoint 初始化。

`--resume --no-resume-in-place` 时会保留/重置的内容：
- 保留：`model_state_dict`
- 重置：`optimizer_state_dict`
- 重置：`scheduler_state_dict`
- 重置：`epoch` 计数，从 0 开始新的 run
- 重置：`best_val_loss`

这适合你修改了学习率、batch size、scheduler、训练轮数等配置，但仍想把旧 checkpoint 作为参数初始化起点的场景。

**配置说明** (`train/config.yaml`):
- `data.data_root`: 数据集根目录 (默认：`datasets/dataset_v1`)
- `data.train_ids_file`: 训练集 ids 文件 (**相对于 data_root**)
- `data.test_ids_file`: 测试集 ids 文件 (**相对于 data_root**)
- `training.progress_total_epochs`: 可选；仅用于计算 loss curriculum 的训练进度分母。默认等于 `training.num_epochs`
- `training.precision`: 训练精度模式，支持 `fp32` / `fp16` / `bf16`
- `data.backend`: 数据后端，`files` 表示散文件读取，`lmdb` 表示从 LMDB 读取，默认 `files`
- `data.lmdb_path`: LMDB 路径；相对路径相对于 `data_root`
- `data.pin_memory`: 是否启用 DataLoader pin memory，默认 `true`
- `data.persistent_workers`: 是否在 epoch 间保留 worker 进程，默认 `false`
- `data.prefetch_factor`: 每个 worker 预取的 batch 数，默认 `1`
- `data.max_prefetch_gb`: DataLoader 允许的预取图像内存预算上限（GiB）；会据此自动下调有效 `num_workers`

## 评估

```bash
# 在测试集上评估
# 默认会把 metrics 保存到该 checkpoint 所在 run 目录下的 eval/ 子目录
python eval_main.py --checkpoint checkpoints/best.pth

# 相对 --output/--metrics_output 也会默认解析到该 run 的 eval/ 子目录
python eval_main.py     --checkpoint checkpoints/best.pth     --output generated.pt     --metrics_output metrics.yaml
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

### LMDB 格式

项目支持将散文件数据集预打包为单个 LMDB 数据库，以减少大量小文件随机读取造成的 I/O 抖动。

推荐构建命令：

```bash
python -m data.build_lmdb \
    --data-root datasets/dataset_v0 \
    --output datasets/dataset_v0/cad_data.lmdb \
    --ids-files train_ids.txt test_ids.txt
```

启用 LMDB 时，在配置文件中添加：

```yaml
data:
  backend: lmdb
  lmdb_path: cad_data.lmdb
```

LMDB 中每个 sample 使用 `group_id/sample_name` 作为 key，value 包含：
- 8 视图图像的原始 PNG bytes
- 文本描述
- `float32 [seq_len, 20]` 的 CAD 序列

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

### DataLoader 内存控制
- `pin_memory`: 让 CPU 到 GPU 的拷贝更快，但会增加主机侧 pinned memory 占用
- `persistent_workers`: 若为 `true`，worker 不会在每个 epoch 结束后退出；吞吐更稳，但长期占用更多内存与进程资源
- `prefetch_factor`: 每个 worker 在后台提前准备多少个 batch；值越大，吞吐潜力越高，但更容易堆积大量 batch 内存
- `max_prefetch_gb`: 预取图像张量的估算内存预算上限；DataLoader 会按 `batch_size × 8 × 3 × H × W × 4 bytes` 估算单 batch 图像内存，并自动限制有效 `num_workers`
- 对于大 batch 训练，建议保持 `prefetch_factor=1`，再根据机器内存逐步增加 `max_prefetch_gb`

### Loss Curriculum 进度
- `training.progress_total_epochs` 只影响 loss 中的 curriculum progress 计算。
- 当前它只用于参数损失权重随训练进度逐步增强这一逻辑。
- 若不设置，默认使用 `training.num_epochs`，行为与普通训练一致。
- 这个字段的主要用途是做短程调试或从中间 checkpoint 恢复时，仍然复现原始长训练的 loss curriculum 节奏。

例如：

```yaml
training:
  num_epochs: 20
  progress_total_epochs: 200
```

含义是：
- 实际只训练到第 20 个 epoch
- 但 loss curriculum 会按“总训练长度 200 个 epoch”来计算当前 progress

注意：
- `progress_total_epochs` **不会**改变学习率调度器的总 epoch
- `progress_total_epochs` **不会**改变 checkpoint 保存频率
- `progress_total_epochs` **不会**改变训练循环实际跑多少个 epoch

### 训练精度模式

- `training.precision: fp32`：最稳，但速度最慢
- `training.precision: fp16`：速度快，但更容易出现数值不稳定
- `training.precision: bf16`：在支持 BF16 的 CUDA 设备上通常兼顾速度和稳定性；当前默认推荐这个模式
- 它当前只影响 loss curriculum，不影响 optimizer / scheduler 的其它行为

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
