#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepcad_latent import DeepCADAdapter
from deepcad_latent.data import load_raw_cad_vec

DEEP_CAD_ROOT = Path("/root/projects/CAD-MLLM/3rd_party/DeepCAD")
if str(DEEP_CAD_ROOT) not in sys.path:
    sys.path.insert(0, str(DEEP_CAD_ROOT))

from cadlib.macro import ALL_COMMANDS, CMD_ARGS_MASK, EOS_IDX, SOL_IDX


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze local structure of the pretrained DeepCAD latent space.")
    parser.add_argument(
        "--ids",
        type=Path,
        default=Path("datasets/rescue_deepcad_latent/full_v0_len60_excluding_rescue_test/train_ids.txt"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("datasets/dataset_v0"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--top-k-families", type=int, default=8)
    parser.add_argument("--min-family-size", type=int, default=64)
    parser.add_argument("--max-family-samples", type=int, default=512)
    parser.add_argument("--pair-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-every", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=Path("runs/deepcad_latent/latent_structure_analysis.json"))
    return parser.parse_args()


def trim_eos(cad_vec: np.ndarray) -> np.ndarray:
    if cad_vec.ndim == 1:
        cad_vec = cad_vec.reshape(-1, 17)
    eos_positions = np.where(cad_vec[:, 0] == EOS_IDX)[0]
    if len(eos_positions) > 0:
        return cad_vec[: int(eos_positions[0])]
    valid = np.where(cad_vec[:, 0] >= 0)[0]
    if len(valid) == 0:
        return cad_vec[:0]
    return cad_vec[: int(valid[-1]) + 1]


def family_signature(cad_vec: np.ndarray) -> tuple[int, ...]:
    trimmed = trim_eos(cad_vec)
    cmds = [int(cmd) for cmd in trimmed[:, 0].tolist() if int(cmd) != SOL_IDX]
    return tuple(cmds)


def family_name(signature: tuple[int, ...]) -> str:
    return " -> ".join(ALL_COMMANDS[idx] for idx in signature)


def parameter_vector(cad_vec: np.ndarray) -> np.ndarray:
    trimmed = trim_eos(cad_vec)
    values = []
    for row in trimmed:
        cmd = int(row[0])
        if cmd == SOL_IDX or cmd == EOS_IDX:
            continue
        mask = CMD_ARGS_MASK[cmd].astype(bool)
        values.extend(int(x) for x in row[1:][mask].tolist())
    return np.asarray(values, dtype=np.float32)


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def pairwise_distance_correlation(z: np.ndarray, params: np.ndarray, pair_samples: int, rng: np.random.Generator) -> float | None:
    n = len(z)
    if n < 2:
        return None
    total_pairs = n * (n - 1) // 2
    sample_count = min(pair_samples, total_pairs)
    idx_i = rng.integers(0, n, size=sample_count)
    idx_j = rng.integers(0, n, size=sample_count)
    mask = idx_i != idx_j
    idx_i = idx_i[mask]
    idx_j = idx_j[mask]
    if len(idx_i) == 0:
        return None
    z_dist = np.linalg.norm(z[idx_i] - z[idx_j], axis=1)
    p_dist = np.linalg.norm(params[idx_i] - params[idx_j], axis=1)
    if np.std(z_dist) < 1e-12 or np.std(p_dist) < 1e-12:
        return None
    return float(np.corrcoef(z_dist, p_dist)[0, 1])


def linear_regression_r2(z: np.ndarray, params: np.ndarray, rng: np.random.Generator) -> dict[str, float | int | None]:
    n = len(z)
    if n < 10:
        return {"r2": None, "train_size": n, "test_size": 0}
    perm = rng.permutation(n)
    split = max(1, int(round(0.8 * n)))
    train_idx = perm[:split]
    test_idx = perm[split:]
    if len(test_idx) == 0:
        return {"r2": None, "train_size": len(train_idx), "test_size": 0}

    x_train = np.concatenate([z[train_idx], np.ones((len(train_idx), 1), dtype=np.float32)], axis=1)
    x_test = np.concatenate([z[test_idx], np.ones((len(test_idx), 1), dtype=np.float32)], axis=1)
    y_train = params[train_idx]
    y_test = params[test_idx]

    coef, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)
    y_pred = x_test @ coef
    denom = float(np.sum((y_test - y_test.mean(axis=0, keepdims=True)) ** 2))
    if denom < 1e-12:
        return {"r2": None, "train_size": len(train_idx), "test_size": len(test_idx)}
    num = float(np.sum((y_test - y_pred) ** 2))
    r2 = 1.0 - num / denom
    return {"r2": float(r2), "train_size": len(train_idx), "test_size": len(test_idx)}


def pca_summary(z: np.ndarray) -> dict[str, list[float] | None]:
    if len(z) < 2:
        return {"explained_variance_ratio": None}
    centered = z - z.mean(axis=0, keepdims=True)
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    variances = singular_values ** 2
    total = float(variances.sum())
    if total < 1e-12:
        return {"explained_variance_ratio": None}
    ratios = (variances / total).tolist()
    return {"explained_variance_ratio": [float(x) for x in ratios[:5]]}


def normalize_params(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = params.mean(axis=0, keepdims=True)
    std = params.std(axis=0, keepdims=True)
    keep = (std[0] > 1e-6)
    if not np.any(keep):
        return params[:, :0], keep
    norm = (params[:, keep] - mean[:, keep]) / std[:, keep]
    return norm.astype(np.float32), keep


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    ids = read_ids(args.ids)

    family_to_ids: dict[tuple[int, ...], list[str]] = defaultdict(list)
    param_dim_by_family: dict[tuple[int, ...], int] = {}

    print(f"Scanning {len(ids)} samples from {args.ids} to build topology families...")
    for idx, sample_id in enumerate(ids, start=1):
        cad_vec = load_raw_cad_vec(args.data_root, sample_id)
        signature = family_signature(cad_vec)
        if not signature:
            continue
        family_to_ids[signature].append(sample_id)
        if signature not in param_dim_by_family:
            param_dim_by_family[signature] = int(parameter_vector(cad_vec).shape[0])
        if idx % args.report_every == 0:
            print(f"  scanned {idx}/{len(ids)} samples | families so far: {len(family_to_ids)}")

    family_counts = Counter({sig: len(v) for sig, v in family_to_ids.items()})
    selected_families = [sig for sig, count in family_counts.most_common(args.top_k_families * 3) if count >= args.min_family_size]
    selected_families = selected_families[: args.top_k_families]

    print(f"Selected {len(selected_families)} families for analysis:")
    for signature in selected_families:
        print(f"  {family_name(signature)} | count={family_counts[signature]} | raw_param_dim={param_dim_by_family[signature]}")

    adapter = DeepCADAdapter(device=args.device)
    print(f"DeepCAD adapter initialized on {args.device}. Starting latent analysis...")

    analyses = []
    for family_idx, signature in enumerate(selected_families, start=1):
        member_ids = family_to_ids[signature]
        if len(member_ids) > args.max_family_samples:
            sampled_ids = sorted(rng.choice(member_ids, size=args.max_family_samples, replace=False).tolist())
        else:
            sampled_ids = list(member_ids)

        print(
            f"[{family_idx}/{len(selected_families)}] {family_name(signature)} | "
            f"total={len(member_ids)} analyzed={len(sampled_ids)}"
        )

        cad_batch = [load_raw_cad_vec(args.data_root, sample_id) for sample_id in sampled_ids]
        z_parts = []
        for start in range(0, len(cad_batch), args.batch_size):
            chunk = cad_batch[start : start + args.batch_size]
            z_parts.append(adapter.encode(chunk).detach().cpu().numpy())
            end = min(start + args.batch_size, len(cad_batch))
            print(f"    encoded {end}/{len(cad_batch)} samples")
        z = np.concatenate(z_parts, axis=0).astype(np.float32)

        params = np.stack([parameter_vector(cad) for cad in cad_batch], axis=0).astype(np.float32)
        params_norm, keep_mask = normalize_params(params)
        dist_corr = pairwise_distance_correlation(z, params_norm, args.pair_samples, rng) if params_norm.shape[1] > 0 else None
        linreg = linear_regression_r2(z, params_norm, rng) if params_norm.shape[1] > 0 else {"r2": None, "train_size": len(z), "test_size": 0}

        analyses.append(
            {
                "family_signature": list(signature),
                "family_name": family_name(signature),
                "count_total": len(member_ids),
                "count_analyzed": len(sampled_ids),
                "param_dim_raw": int(params.shape[1]),
                "param_dim_variable": int(params_norm.shape[1]),
                "examples": sampled_ids[:5],
                "latent_pca": pca_summary(z),
                "latent_param_distance_corr": dist_corr,
                "linear_regression": linreg,
                "latent_norm_mean": float(np.linalg.norm(z, axis=1).mean()),
                "latent_norm_std": float(np.linalg.norm(z, axis=1).std()),
            }
        )

    result = {
        "ids": str(args.ids),
        "num_ids_scanned": len(ids),
        "num_nonempty_families": len(family_counts),
        "top_families_by_count": [
            {
                "family_signature": list(sig),
                "family_name": family_name(sig),
                "count": int(count),
                "param_dim_raw": int(param_dim_by_family[sig]),
            }
            for sig, count in family_counts.most_common(20)
        ],
        "analyzed_families": analyses,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved latent structure analysis to {args.output}")


if __name__ == "__main__":
    main()
