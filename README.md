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
│   ├── cad_decoder.py       # CAD 序列解码器
│   └── dual_modal_cad.py    # 完整模型
├── data/
│   ├── __init__.py
│   ├── dataset.py           # 数据集类
│   ├── renderer.py          # CAD 渲染 pipeline
│   └── augment.py           # 数据增强
├── train/
│   ├── __init__.py
│   ├── train.py             # 训练脚本
│   ├── loss.py              # 损失函数
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
conda create -n dmcad python=3.10
conda activate dmcad

# 安装依赖
pip install -r requirements.txt
```

## 训练

```bash
# 使用默认配置训练
python train_main.py --config train/config.yaml --data_dir ./data

# 从检查点恢复训练
python train_main.py --config train/config.yaml --data_dir ./data --resume checkpoints/epoch_10.pth
```

## 评估

```bash
# 在测试集上评估
python eval_main.py --checkpoint checkpoints/best.pth --data_dir ./data/test
```

## 推理

```bash
# 使用 8 视图图像和文本描述生成 CAD 序列
python infer.py \
    --checkpoint checkpoints/best.pth \
    --images view_00.png view_01.png ... view_07.png \
    --text "A rectangular box with a cylindrical hole through the center"
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
│         Modal Fusion (Cross-Attention)                      │
│                    ↓                                        │
│              z_fused [512]                                  │
│                    ↓                                        │
│         CAD Decoder (Transformer)                           │
│                    ↓                                        │
│    Command Head    Parameter Head                           │
│    (4 classes)     (19 dim)                                 │
└─────────────────────────────────────────────────────────────┘
```

## 数据集格式

训练数据应组织为以下格式：

```
data/
├── train.json             # 训练数据索引
├── val.json               # 验证数据索引
├── test.json              # 测试数据索引
├── images/
│   └── <uid>/
│       ├── view_00.png
│       ├── view_01.png
│       └── ...
└── cad_seq/
    ├── <uid>.pt
    └── ...
```

## 评估指标

| 指标 | 说明 |
|------|------|
| Command Accuracy | 命令类型预测准确率 |
| Parameter Accuracy | 参数预测准确率 (相对误差<0.1) |
| Chamfer Distance | 几何形状相似度 |
| Invalidity Ratio | 生成无效序列比例 |

## 参考文献

1. DeepCAD: A Deep Generative Network for Computer-Aided Design Models (ICCV 2021)
2. Text2CAD: Generating Sequential CAD Designs from Text Prompts (NeurIPS 2024)
3. Automatic Reverse Engineering: Creating CAD Models from Multi-View Images (GCPR 2023)
4. CAD-MLLM: Unifying Multimodality-Conditioned CAD Generation With MLLM (arXiv 2024)

## License

MIT License
