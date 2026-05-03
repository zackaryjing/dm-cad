# DM-CAD Rescue Branch

This branch no longer contains the old end-to-end autoregressive CAD generator.
It is now focused on a lower-risk pipeline built on top of a frozen DeepCAD latent space.

## Current Route

- `images -> latent z -> DeepCAD decoder`
- DeepCAD autoencoder stays frozen
- current working baseline: image-only
- current extension under implementation: image + text

Core code lives in:

- `deepcad_latent/`
- `scripts/train_image_to_latent.py`
- `scripts/evaluate_image_to_cad.py`
- `scripts/web_image_to_cad.py`
- `scripts/precompute_deepcad_latents.py`
- `scripts/precompute_text_embeddings.py`
- `scripts/train_image_text_to_latent.py`

## Data Assets

Important dataset ids and latent roots are summarized in:

- `runs/deepcad_latent/EXPERIMENT_SNAPSHOT_2026-04-16.md`
- `runs/deepcad_latent/experiment_manifest.json`

## Typical Workflow

1. Prepare ids

```bash
python scripts/prepare_rescue_ids.py --length-thresholds 60
```

2. Precompute DeepCAD latents

```bash
python scripts/precompute_deepcad_latents.py ...
```

3. Train image-only model

```bash
torchrun --nproc_per_node=2 scripts/train_image_to_latent.py ...
```

4. Evaluate direct / nearest / blend

```bash
python scripts/evaluate_image_to_cad.py ...
```

5. Launch qualitative web inspector

```bash
python scripts/web_image_to_cad.py ...
```

## Current Presentation Recommendation

- baseline: `direct`
- main improved method: `blend(alpha=0.5)`
- analysis / upper-bound: `nearest`
