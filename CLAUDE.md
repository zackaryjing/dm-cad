# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (conda recommended)
conda create -n dmcad python=3.10
conda activate dmcad
pip install -r requirements.txt

# Training
python train_main.py --config train/config.yaml --data_dir ./data

# Resume from checkpoint
python train_main.py --config train/config.yaml --data_dir ./data --resume checkpoints/epoch_10.pth

# Evaluation
python eval_main.py --checkpoint checkpoints/best.pth --data_dir ./data/test

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

- `CADDataset` loads triples: (8-view images, text, CAD sequence)
- Data format expected:
  ```
  data/
  ├── train.json / val.json / test.json  # lists of {uid, text, ...}
  ├── images/<uid>/view_{00-07}.png
  └── cad_seq/<uid>.pt
  ```
- CAD sequences use DeepCAD format: START → [SKETCH, EXTRUDE]* → END

### Training (`train/`)

- Loss: CrossEntropy (cmd) + SmoothL1 (params), weighted sum
- Optimizer: AdamW with CosineAnnealingLR
- Gradient clipping enabled (default: 1.0)
- TensorBoard logging to `runs/dmcad/`

### Configuration

All hyperparameters in `train/config.yaml`. Key defaults:
- embed_dim: 512, n_heads: 8, n_layers: 6
- batch_size: 32, epochs: 80, lr: 5e-5
- 8 views, max_seq_len: 20

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
