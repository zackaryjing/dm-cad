#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/jing/allprojects/pythonenvironment/dmcad/bin/python}"
DEVICE="${DEVICE:-cuda}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-32}"
TEST_SAMPLES="${TEST_SAMPLES:-8}"
RUN_NAME="${RUN_NAME:-regression_smoke}"
TMP_DIR="${TMPDIR:-/tmp}/dmcad_${RUN_NAME}"
mkdir -p "$TMP_DIR"

TRAIN_IDS="$TMP_DIR/train_ids.txt"
TEST_IDS="$TMP_DIR/test_ids.txt"
CONFIG_PATH="$TMP_DIR/config.yaml"
CHECKPOINT_PATH="$ROOT_DIR/runs/dmcad/${RUN_NAME}/checkpoints/best.pth"

head -n "$TRAIN_SAMPLES" "$ROOT_DIR/datasets/dataset_v1/train_ids_5k.txt" > "$TRAIN_IDS"
head -n "$TEST_SAMPLES" "$ROOT_DIR/datasets/dataset_v1/test_ids_5k.txt" > "$TEST_IDS"

cat > "$CONFIG_PATH" <<EOF2
model:
  embed_dim: 512
  n_heads: 8
  n_layers: 6
  max_seq_len: 120
  n_views: 8
  fusion_type: gating
  start_token: 4

training:
  batch_size: 2
  num_epochs: 1
  lr: 5.0e-05
  weight_decay: 0.01
  warmup_epochs: 0
  gradient_clip: 1.0
  num_workers: 0

loss:
  cmd_weight: 1.0
  param_weight: 0.5

optimizer:
  type: AdamW
  lr: 5.0e-05
  weight_decay: 0.01

scheduler:
  type: CosineAnnealingLR
  T_max: 1
  eta_min: 1.0e-06

data:
  img_size: 224
  text_max_len: 64
  data_root: datasets/dataset_v1
  train_ids_file: $TRAIN_IDS
  test_ids_file: $TEST_IDS

log:
  log_dir: ./runs/dmcad/$RUN_NAME
  print_freq: 1
  val_freq: 1
EOF2

cd "$ROOT_DIR"
"$PYTHON_BIN" train_main.py --config "$CONFIG_PATH" --device "$DEVICE"
"$PYTHON_BIN" eval_main.py --checkpoint "$CHECKPOINT_PATH" --config "$CONFIG_PATH" --device "$DEVICE"
