# TODO

## Highest Priority

### 1. Stabilize single-GPU training first
Reason:
- The current default config still points to multi-GPU DP settings, which is risky for day-to-day debugging and easy to misconfigure.
- For a graduation-project codebase, a stable single-GPU baseline is more valuable than an early complex parallel setup.

Suggested actions:
- Make the default training config single-GPU safe.
- Keep DP/DDP in separate config files instead of the main default config.

### 2. Add AMP mixed precision
Status: done on 2026-03-24.
Reason:
- The main crash started from CUDA OOM in the ViT image branch.
- AMP is the lowest-complexity way to reduce memory use on RTX 4090 while keeping the project understandable.

Suggested actions:
- Add `torch.cuda.amp.autocast` and `GradScaler` to training.
- Keep it configurable with a simple boolean switch in config.

Completion notes:
- Added configurable mixed-precision support to the training configs.
- Enabled mixed precision in both training and validation paths.
- Verified with a local WSL smoke run on `gpu0`, `batch_size=4`, `num_workers=2`.

### 3. Freeze ViT by default
Status: done on 2026-03-24.
Reason:
- The code comment says the pretrained ViT is mostly frozen, but the implementation does not actually freeze it.
- Full ViT training is expensive and unnecessary for a first working baseline.

Suggested actions:
- Freeze ViT backbone parameters by default.
- Only train the projection layer, fusion module, and CAD decoder at first.

Completion notes:
- Added `model.freeze_vit: true` to the training configs.
- `ViewEncoder` now freezes the ViT backbone by default while keeping the projection layer trainable.

## Medium Priority

### 4. Add minimal DDP instead of expanding DataParallel
Reason:
- DDP is more correct and scalable than DP.
- But the project is for a graduation thesis, so the implementation should stay small and easy to explain.

Suggested actions:
- Add a simple `single | dp | ddp` strategy setting.
- Use `torchrun`, `LOCAL_RANK`, `DistributedSampler`, and rank-0-only logging/checkpointing.
- Avoid introducing heavyweight training frameworks.

### 5. Make Hugging Face model/tokenizer loading robust offline
Reason:
- Training currently depends on online downloads from Hugging Face mirrors.
- The observed timeout logs show this path is fragile and can block experiments.

Suggested actions:
- Support local cache paths and `local_files_only` fallback.
- Document how to pre-download model weights and tokenizers.

### 6. Improve training failure handling around CUDA errors
Reason:
- After OOM, later operations may report delayed CUDA errors such as `invalid device ordinal`, which makes debugging confusing.
- Better failure reporting will reduce time lost when experiments crash.

Suggested actions:
- Catch `torch.cuda.OutOfMemoryError` in the train loop and print a clear message.
- Suggest reducing batch size and restarting the process after CUDA context corruption.
- Optionally add a debug note for `CUDA_LAUNCH_BLOCKING=1`.

## Lower Priority But Important

### 7. Tighten evaluation validity
Reason:
- Some evaluation items are still placeholders and do not fully support strong experimental claims.
- This matters for thesis credibility more than for raw training throughput.

Suggested actions:
- Make invalidity checks reflect real CAD sequence rules.
- Only report metrics that are actually implemented and trustworthy.
- Align README claims with current evaluator behavior.

### 8. Pin risky dependency versions more carefully
Reason:
- The `timm -> wandb -> pydantic` import chain already showed environment fragility.
- Reproducibility matters for later thesis experiments.

Suggested actions:
- Freeze a known-good environment version set.
- Document one recommended environment as the official experiment setup.
