# CODEX Session Notes

## Scope
This file summarizes the issues found and fixes made during the Codex session on 2026-03-20.

## Major Issues Found

### 1. Decoder training and generation were inconsistent
Files: `models/cad_decoder.py`, `models/dual_modal_cad.py`

Problems:
- Training used the full target sequence directly instead of standard teacher forcing.
- No causal mask was applied in the decoder, so tokens could attend to future tokens.
- Autoregressive generation only fed the most recent token, not the full prefix.
- Validation in `eval()` mode produced only a single decoding step even when a full target sequence was provided.

Fixes:
- Added right-shifted teacher forcing input.
- Added decoder causal mask.
- Reworked generation to decode from the full generated prefix.
- Changed decoder forward logic so `tgt_seq` always triggers full-sequence decoding, regardless of `model.training`.

### 2. Loss computation incorrectly included padding / invalid samples
Files: `train/loss.py`, `data/dataset.py`

Problems:
- Command cross-entropy was computed on all positions, including padding.
- Missing CAD vector files returned a zero-filled fake sequence and were treated as valid data.
- Padding used command `0`, which biased the model toward `Line`.
- CUDA training exposed a device mismatch because the command-parameter mask buffer stayed on CPU.

Fixes:
- Switched command loss to `reduction='none'` and masked it with `valid_mask`.
- Kept parameter loss masked by both `valid_mask` and command-specific parameter masks.
- Missing CAD data now returns an empty sequence plus an empty valid mask.
- Padding now uses command `-1` and `valid_mask=False`.
- Moved command-parameter mask to the target device before indexing.

### 3. Evaluation pipeline was invalid
Files: `eval/evaluate.py`, `eval/metrics.py`, `eval_main.py`

Problems:
- Evaluation called the model without `tgt_seq`, so it produced one decoding step, then compared that to full ground truth.
- Generated sequences of different lengths were concatenated across batches, causing runtime errors.
- `generate_and_save()` used `batch['uids']`, but the dataloader returns `sample_ids`.
- Parameter accuracy did not respect command-specific valid parameter dimensions.

Fixes:
- Evaluation now uses `model.generate()` and pads/truncates per batch to GT length.
- Metric accumulation is done batch-by-batch instead of concatenating incompatible time dimensions.
- `generate_and_save()` now uses `sample_ids` and writes per-sample command lists.
- Parameter accuracy now uses the same command-specific mask logic as training.
- `eval_main.py` now supports `--max_batches` and also reads bounded validation settings from config.

### 4. Config wiring was largely ineffective
Files: `models/dual_modal_cad.py`, `models/view_encoder.py`, `models/text_encoder.py`, `models/fusion.py`

Problems:
- Model config values were accepted but mostly ignored.
- `embed_dim`, `n_heads`, `n_layers`, `max_seq_len`, and `fusion_type` did not fully propagate.

Fixes:
- Wired config values into the full model construction path.
- Updated image and text projection layers to respect `embed_dim`.
- Added actual behavior branches for `fusion_type` (`gating`, `concat`, `cross_attention`).
- Centralized modality encoding in `DualModalCADGenerator`.

### 5. Inference script used the wrong command protocol
File: `infer.py`

Problems:
- Inference still printed old 4-command labels instead of the 6-command DeepCAD protocol.
- It also assumed the old generation return structure.

Fixes:
- Updated inference to use the corrected `generate()` API.
- Switched displayed labels to `Line`, `Arc`, `Circle`, `EOS`, `SOL`, `Ext`.
- Added a hard check for exactly 8 views.

### 6. Short-baseline support was missing
Files: `train/train.py`, `eval/evaluate.py`, `eval_main.py`, `train/config_5k_short.yaml`

Problems:
- A full 5k epoch on the local WSL + RTX 3050 Ti setup was too slow to be practical.
- There was no built-in way to run a bounded short baseline on the full dataset split.

Fixes:
- Added optional `training.max_train_batches` and `training.max_val_batches`.
- Added bounded evaluation support via `eval_main.py --max_batches` and matching evaluator support.
- Added `train/config_5k_short.yaml` as a reusable short-baseline config.

## New Files Added

### `scripts/run_regression_smoke.sh`
Purpose:
- Recreates the minimal smoke regression using a small temporary subset from `dataset_v1`.
- Runs both `train_main.py` and `eval_main.py`.

Defaults:
- `TRAIN_SAMPLES=32`
- `TEST_SAMPLES=8`
- `DEVICE=cuda`
- `PYTHON_BIN=/home/jing/allprojects/pythonenvironment/dmcad/bin/python`

### `scripts/run_5k_short_baseline.sh`
Purpose:
- Runs the bounded short-baseline config on the full `dataset_v1` split.
- Uses `train/config_5k_short.yaml` and then evaluates the produced checkpoint.

### `train/config_5k_short.yaml`
Purpose:
- Reusable short-baseline config for the full 5k split.

Current values at the end of this session:
- `batch_size: 4`
- `num_epochs: 1`
- `max_train_batches: 100` was attempted during development, then shortened locally due WSL speed constraints; verify before server runs.
- `max_val_batches` support is available.

Note:
- Before running on a real server, review the final values in this file and set the desired bounded schedule explicitly.

## Validation Performed

### Static checks
Used:
- `/home/jing/allprojects/pythonenvironment/dmcad/bin/python -m py_compile ...`

Result:
- Passed for the updated training, evaluation, inference, and model files.

### Unit / smoke checks
Verified locally with small in-process tests:
- Decoder training output shapes.
- Decoder generation output contract.
- Loss masking behavior.
- Metrics return schema.
- Eval-mode teacher forcing shape handling.
- CUDA loss path after device-fix.

### End-to-end smoke regression on CPU
Dataset:
- `dataset_v1` temporary subset: 32 train / 8 test

Result:
- Training completed.
- Validation completed.
- Best checkpoint saved.
- Standalone evaluation completed.

Observed metrics:
- Train loss: `59.5889`
- Val loss: `60.2253`
- Standalone eval: `cmd_accuracy=0.4182`, `param_accuracy=0.0000`, `combined_score=0.2091`

### End-to-end smoke regression on GPU
Environment:
- WSL
- GPU detected by PyTorch: `NVIDIA GeForce RTX 3050 Ti Laptop GPU`

Result:
- Training completed.
- Validation completed.
- Best checkpoint saved.
- Standalone evaluation completed.

Observed metrics:
- Train loss: `60.7102`
- Val loss: `60.2660`
- Standalone eval: `cmd_accuracy=0.4182`, `param_accuracy=0.0703`, `combined_score=0.2443`

## Attempted Full 5k Short Baseline

Status:
- Partially attempted on the local WSL + 3050 Ti environment.
- A full 5k epoch was too slow locally.
- Bounded short-baseline support was added so the same workflow can be rerun on a stronger server.

Practical guidance:
- On server hardware, rerun `scripts/run_5k_short_baseline.sh`.
- Review and adjust `train/config_5k_short.yaml` first.
- If throughput is acceptable, increase `max_train_batches`, `max_val_batches`, or `num_epochs`.

## Local Runtime Cleanup
At the end of this session:
- Leftover local training processes were stopped.
- `nvidia-smi` showed no remaining GPU compute processes.

## Files Changed In This Session
- `data/dataset.py`
- `eval/evaluate.py`
- `eval/metrics.py`
- `eval_main.py`
- `infer.py`
- `models/cad_decoder.py`
- `models/dual_modal_cad.py`
- `models/fusion.py`
- `models/text_encoder.py`
- `models/view_encoder.py`
- `train/loss.py`
- `train/train.py`
- `train_main.py`
- `train/config_5k_short.yaml`
- `scripts/run_regression_smoke.sh`
- `scripts/run_5k_short_baseline.sh`

## Recommended Next Steps
- Run `scripts/run_5k_short_baseline.sh` on a server GPU.
- Decide on the final bounded schedule for `train/config_5k_short.yaml`.
- If this repository needs repeatable CI coverage, convert the current smoke checks into automated tests.
- Consider adding AMP / mixed precision and profiler-based optimization if training speed remains a bottleneck.


## Session 2026-03-24

### Completed Fix
Files: `runtime_device.py`, `train/train.py`

Commit:
- `4ecddb8` - `Fix CUDA device mapping for single-GPU training`

Problem:
- `CUDA_VISIBLE_DEVICES` can remap physical GPUs into local visible indices, but the trainer still trusted `output_device` without validating it against the visible range.
- If only one CUDA device is visible, the code should not attempt DataParallel setup at all.

Fixes:
- Added helper logic to count configured visible devices.
- Validated `output_device` against `torch.cuda.device_count()` and fall back to `cuda:0` if out of range.
- Disabled `DataParallel` when the config exposes only one visible device.
- Reused the resolved runtime device index when constructing `nn.DataParallel`.

Why this matters:
- It removes a common single-GPU misconfiguration path.
- It makes GPU selection consistent with `CUDA_VISIBLE_DEVICES` remapping.
- It reduces the chance of parallel wrapper issues after changing config device visibility.


## Environment Memory
- Local test environment (WSL): `/home/jing/allprojects/pythonenvironment/dmcad`
- Local Python interpreter: `/home/jing/allprojects/pythonenvironment/dmcad/bin/python`
- Remote SSH environment name: `dmcad`
- Future convention: if the user says `本地`, use the WSL environment above; if the user says `远端`, activate the remote `dmcad` conda environment before running tests.


## Session 2026-03-24 AMP And ViT Freeze

### Completed Work
Files:
- `train/train.py`
- `models/view_encoder.py`
- `models/dual_modal_cad.py`
- `train/config.yaml`
- `train/config_5k.yaml`
- `train/config_20k.yaml`
- `train/config_full.yaml`
- `train/config_5k_short.yaml`

Changes:
- Added configurable training precision support with `training.precision`.
- Enabled AMP in both training and validation.


## Session 2026-04-01 LMDB Follow-Up And Loader Memory Safety

### Context
Previous commit:
- `0d45c6e` - `bug: try to add lmdb but caused problem, HAVEN'T FIX`

This follow-up addresses the memory blow-up risk introduced after switching full training to LMDB-backed loading.

### Root Cause Analysis
Files:
- `data/dataset.py`
- `train_main.py`
- `train/config_full.yaml`

Findings:
- The critical failure mode was not LMDB `map_size` directly consuming RAM.
- The actual risk came from combining:
  - very large `batch_size` (`512`)
  - many dataloader workers (`48`)
  - worker-side prefetching
  - `pin_memory=True`
  - faster backend throughput after moving from scattered files to LMDB
- For the full config, image tensors alone are about `2.30 GiB` per batch:
  - `512 × 8 × 3 × 224 × 224 × 4 bytes`
- With `48` workers and `prefetch_factor=2`, the loader could theoretically accumulate around `96` prefetched batches.
- Image tensors alone at that scale are about `220 GiB`, before adding:
  - pinned-memory staging
  - PNG decode intermediates
  - tokenizer outputs
  - CAD tensors
  - Python object overhead
  - page cache and general process memory
- This makes host memory usage in the `300 GiB+` range plausible on a `504 GiB` machine.

### Fixes Applied
Files:
- `data/dataset.py`
- `data/build_lmdb.py`
- `train_main.py`
- `eval_main.py`
- `train/config.yaml`
- `train/config_5k.yaml`
- `train/config_5k_short.yaml`
- `train/config_20k.yaml`
- `train/config_full.yaml`
- `README.md`

Changes:
- Added an explicit prefetch memory budget control: `data.max_prefetch_gb`.
- Changed loader defaults to safer values:
  - `prefetch_factor: 1`
  - `persistent_workers: false`
- Added automatic worker downscaling based on estimated prefetched image memory.
- Added startup logging for:
  - requested worker count
  - effective worker count
  - estimated image memory per batch
  - estimated prefetched batch count
  - estimated prefetched image memory
  - configured prefetch cap
- Added matching config fields to all train configs with scale-dependent defaults.
- Documented LMDB and loader-memory controls in `README.md`.
- Lowered the LMDB build script default `map_size` from `512 GB` to `64 GB` to reduce confusion and keep defaults closer to the actual dataset footprint.

### Design Notes
- `max_prefetch_gb` has higher priority than the requested `num_workers`.
- The loader computes an `effective_num_workers` before constructing `DataLoader`.
- Workers are not created in a partially-starved state. The code reduces worker count so that each remaining worker can still prefetch full batches normally.
- The current estimator controls image-tensor memory only. Real total memory usage will be higher because it excludes non-image allocations.

### Resource Understanding For This Machine
- This server has enough RAM to tolerate a meaningful prefetch window, but not an unbounded one.
- A `32 GiB` prefetched-image budget is aggressive enough to use the machine while still leaving headroom for:
  - the model process
  - pinned memory
  - operating system cache
  - remote management and shell responsiveness
- The bottleneck after moving to LMDB is no longer just raw disk I/O. CPU-side decode, tokenization, and dataloader queue sizing become first-order concerns.

### Validation Status
- Static validation only:
  - `python -m compileall data train_main.py eval_main.py`
- This fix has **not** been runtime-verified yet on the target server after the memory-guard changes.
- Before long full runs, start with a short launch and confirm the startup log matches the expected effective worker count and prefetch memory estimate.
- Added `GradScaler` integration and AMP-safe gradient clipping.
- Added default ViT backbone freezing in `ViewEncoder`.
- Propagated `model.freeze_vit` and `model.pretrained_vit` through the main model config.
- Updated training configs to enable AMP and freeze ViT by default.

### Local WSL Smoke Test
Environment:
- Python: `/home/jing/allprojects/pythonenvironment/dmcad/bin/python`
- GPU visibility: `CUDA_VISIBLE_DEVICES=0`
- Config: `train/config_5k_short.yaml`
- Effective settings: single GPU, `batch_size=4`, `num_workers=2`, `precision=bf16`, `freeze_vit=true`

Observed result:
- Training completed for the bounded 1-epoch short run.
- Validation completed.
- Checkpoint saved to `./runs/dmcad/short_5k_baseline/checkpoints/best.pth`.
- Final log line: `Epoch 0: train_loss=58.6708, val_loss=56.5759, time=24.2s`

Conclusion:
- The local single-GPU training path is working with AMP enabled.
- The default-frozen ViT change did not break the training or validation flow.
- No CUDA OOM or invalid device ordinal error appeared in this smoke test.
