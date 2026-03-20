# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (conda recommended)
conda activate /home/jing/allprojects/pythonenvironment/dmcad

# Training (config.yaml controls data_root and ids files)
python train_main.py --config train/config.yaml

# Resume from checkpoint
python train_main.py --config train/config.yaml --resume checkpoints/epoch_10.pth

# Evaluation
python eval_main.py --checkpoint checkpoints/best.pth

# Inference (requires 8 view images)
python infer.py --checkpoint checkpoints/best.pth --images view_00.png ... view_07.png --text "description"
```

## Architecture Overview

**DM-CAD** is a dual-modal CAD generator that takes 8-view images + text description and outputs parametric CAD command sequences (DeepCAD format).

### Model Components (`models/`)

```
DualModalCADGenerator
├── ViewEncoder (ViT-Base, frozen) → [B*8, 512]
├── MultiViewFusion (Transformer pooling) → [B, 512]
├── TextEncoder (BERT-Base, frozen) → [B, 512]
├── ModalFusion (cross-attention) → [B, 512]
└── CADDecoder (Transformer) → cmd_logits + param_pred
```

**Input/Output shapes:**
- Images: `[batch, 8, 3, 224, 224]`
- Text: `[batch, seq_len]` (tokenized by BERT)
- CAD sequence: `[batch, seq_len, 20]` (1 cmd_type + 19 params per step)
- Output: cmd_logits `[batch, seq_len, 6]`, param_pred `[batch, seq_len, 19]`

### Data Pipeline (`data/`)

- `CADDataset` loads triples: (8-view images, text description, CAD vector)
- Data format expected:
  ```
  datasets/dataset_v1/
  ├── train_ids_5k.txt / test_ids_5k.txt  # sample IDs (relative to data_root)
  ├── cad_desc/<group_id>.json            # text descriptions ({id, text caption})
  ├── cad_img/<group_id>/<sample_id>/<sample_id>_{000-007}.png
  └── cad_vec/<group_id>/<sample_id>.h5
  ```
- CAD vectors loaded from h5 files (`vec` dataset, shape [seq_len, 17], padded to 20)
- **IDS FILE PATH**: Relative to `data_root` (configured in `train/config.yaml`)

### Training (`train/`)

- Loss: CrossEntropy (cmd, 6 classes) + SmoothL1 (params, 19 dim), weighted sum
- Optimizer: AdamW with CosineAnnealingLR
- Gradient clipping enabled (default: 1.0)
- TensorBoard logging to `runs/dmcad/`

### Configuration

**CRITICAL**: `train/config.yaml` controls:
- `data.data_root`: Path to dataset directory (default: `datasets/dataset_v1`)
- `data.train_ids_file`: Training IDs file (relative to data_root)
- `data.test_ids_file`: Test IDs file (relative to data_root)

Key model defaults:
- embed_dim: 512, n_heads: 8, n_layers: 6
- batch_size: 4, epochs: 80, lr: 5e-5
- 8 views, max_seq_len: 120
- **6 command types** (DeepCAD format), **19 parameters**

## Key Design References

The implementation follows `dual_modal_cad_design.md` which contains:
- Detailed architecture diagrams
- Loss function formulations
- Training strategy (3-stage: warmup → full → LoRA fine-tune)
- Evaluation metrics (Command Acc, Parameter Acc, Chamfer Distance)

## Development Notes

### Command Types (DeepCAD Format)
The model uses **6 command types** (NOT 4!):
- 0: Line (线段)
- 1: Arc (圆弧)
- 2: Circle (圆)
- 3: EOS (序列结束)
- 4: SOL (实体开始)
- 5: Ext (拉伸)

### CAD Sequence Format
- Shape: `[seq_len, 20]` = 1 (cmd_type) + 19 (params)
- cmd_type is at index 0, params at indices 1-19
- Invalid positions use cmd_type < 0 (e.g., -1), filtered by `cad_valid_mask`

### Loss Computation
- `cmd_gt` must be clamped to `[0, 5]` to avoid CrossEntropyLoss assertion errors
- `param_pred` and `param_gt` are 19-dimensional
- `CMD_PARAM_MASK` in `train/loss.py` defines valid param dims per command type

### Model Architecture Notes
- ViT and BERT backbones are frozen by default; only adaptation layers are trainable
- `CADDecoder.generate()` uses autoregressive decoding with batch support
- `cad_valid_mask` masks out invalid positions during loss computation

### Data Loading Chain
```
train_main.py
  └── reads config.yaml → data_root, train_ids_file, test_ids_file
       └── build_dataloader(data_root=data_root, ids_file=ids_file)
            └── CADDataset._load_data_list(ids_file)
                 └── if ids_file is relative: os.path.join(data_root, ids_file)
```
