# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (conda recommended)
conda create -n deepcad38 python=3.8
conda activate deepcad38
pip install -r requirements.txt

# Dataset partition (generate train/test split)
python data/partition_dataset.py --data_dir datasets/dataset_v0 --train_ratio 0.8 --seed 42 --workers 8

# Training
python train_main.py --config train/config.yaml --data_dir datasets/dataset_v0

# Resume from checkpoint
python train_main.py --config train/config.yaml --data_dir datasets/dataset_v0 --resume checkpoints/epoch_10.pth

# Evaluation
python eval_main.py --checkpoint checkpoints/best.pth --data_dir datasets/dataset_v0 --split test

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

### Data Pipeline (`data/`)

- `CADDataset` loads triples: (8-view images, text description, CAD vector)
- Data format expected:
  ```
  datasets/dataset_v0/
  ├── train_ids.txt / test_ids.txt  # sample IDs for train/test split
  ├── cad_desc/<group_id>.json      # text descriptions ({id, text caption})
  ├── cad_img/<group_id>/<sample_id>/{sample_id}_{000-007}.png
  └── cad_vec/<group_id>/<sample_id>.h5
  ```
- CAD vectors are loaded from h5 files (`vec` dataset, shape [seq_len, 17])
- CAD sequences use DeepCAD format: START → [SKETCH, EXTRUDE]* → END
- Data partition script: `python data/partition_dataset.py --data_dir datasets/dataset_v0`

### Training (`train/`)

- Loss: CrossEntropy (cmd) + SmoothL1 (params), weighted sum
- Optimizer: AdamW with CosineAnnealingLR
- Gradient clipping enabled (default: 1.0)
- TensorBoard logging to `runs/dmcad/`

### Configuration

All hyperparameters in `train/config.yaml`. Key defaults:
- embed_dim: 512, n_heads: 8, n_layers: 6
- batch_size: 32, epochs: 80, lr: 5e-5
- 8 views, max_seq_len: 120

## Key Design References

The implementation follows `dual_modal_cad_design.md` which contains:
- Detailed architecture diagrams
- Loss function formulations
- Training strategy (3-stage: warmup → full → LoRA fine-tune)
- Evaluation metrics (Command Acc, Parameter Acc, Chamfer Distance)

## Development Notes

- ViT and BERT backbones are frozen by default; only adaptation layers are trainable
- The `generate()` method in `dual_modal_cad.py` uses autoregressive decoding
- Invalid CAD sequences are masked out during loss computation via `cad_valid_mask`
