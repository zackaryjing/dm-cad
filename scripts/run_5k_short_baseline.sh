#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/jing/allprojects/pythonenvironment/dmcad/bin/python}"
DEVICE="${DEVICE:-cuda}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/train/config_5k_short.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$ROOT_DIR/runs/dmcad/short_5k_baseline/checkpoints/best.pth}"

cd "$ROOT_DIR"
"$PYTHON_BIN" train_main.py --config "$CONFIG_PATH" --device "$DEVICE"
"$PYTHON_BIN" eval_main.py --checkpoint "$CHECKPOINT_PATH" --config "$CONFIG_PATH" --device "$DEVICE"
