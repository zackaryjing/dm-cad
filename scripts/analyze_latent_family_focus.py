#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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


DEFAULT_FAMILIES = [
    "Circle -> Ext",
    "Circle -> Circle -> Ext",
    "Line -> Line -> Line -> Line -> Ext",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Focused latent-family analysis for interpretable CAD topology families.")
    parser.add_argument(
        "--ids",
        type=Path,
        default=Path("datasets/rescue_deepcad_latent/full_v0_len60_excluding_rescue_test/train_ids.txt"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("datasets/dataset_v0"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-family-samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--families", type=str, nargs="*", default=DEFAULT_FAMILIES)
    parser.add_argument("--report-every", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=Path("runs/deepcad_latent/latent_family_focus.json"))
    return parser.parse_args()


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


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
    return tuple(int(cmd) for cmd in trimmed[:, 0].tolist() if int(cmd) != SOL_IDX)


def family_name(signature: tuple[int, ...]) -> str:
    return " -> ".join(ALL_COMMANDS[idx] for idx in signature)


def named_parameter_vector(cad_vec: np.ndarray) -> tuple[list[str], np.ndarray]:
    trimmed = trim_eos(cad_vec)
    names: list[str] = []
    values: list[float] = []
    counts = defaultdict(int)

    for row in trimmed:
        cmd = int(row[0])
        if cmd in (SOL_IDX, EOS_IDX):
            continue
        counts[cmd] += 1
        occ = counts[cmd]
        params = row[1:]
        mask = CMD_ARGS_MASK[cmd].astype(bool)

        if ALL_COMMANDS[cmd] == "Line":
            local = ["end_x", "end_y"]
        elif ALL_COMMANDS[cmd] == "Arc":
            local = ["end_x", "end_y", "sweep", "clock_sign"]
        elif ALL_COMMANDS[cmd] == "Circle":
            local = ["center_x", "center_y", "radius"]
        elif ALL_COMMANDS[cmd] == "Ext":
            local = [
                "plane_theta",
                "plane_phi",
                "plane_gamma",
                "sketch_pos_x",
                "sketch_pos_y",
                "sketch_pos_z",
                "sketch_size",
                "extent_one",
                "extent_two",
                "operation",
                "extent_type",
            ]
        else:
            local = [f"arg{i}" for i, use in enumerate(mask) if use]

        used_values = params[mask]
        if len(local) != len(used_values):
            local = [f"arg{i}" for i in range(len(used_values))]

        prefix = f"{ALL_COMMANDS[cmd].lower()}{occ}"
        for name, value in zip(local, used_values):
            names.append(f"{prefix}_{name}")
            values.append(float(value))

    return names, np.asarray(values, dtype=np.float32)


def fit_linear_single(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> float | None:
    xtr = np.concatenate([x_train, np.ones((len(x_train), 1), dtype=np.float32)], axis=1)
    xte = np.concatenate([x_test, np.ones((len(x_test), 1), dtype=np.float32)], axis=1)
    coef, *_ = np.linalg.lstsq(xtr, y_train, rcond=None)
    pred = xte @ coef
    denom = float(np.sum((y_test - y_test.mean()) ** 2))
    if denom < 1e-12:
        return None
    num = float(np.sum((y_test - pred) ** 2))
    return float(1.0 - num / denom)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    ids = read_ids(args.ids)

    target_names = set(args.families)
    family_to_ids: dict[str, list[str]] = defaultdict(list)

    print(f"Scanning {len(ids)} samples to collect target families...")
    for idx, sample_id in enumerate(ids, start=1):
        cad_vec = load_raw_cad_vec(args.data_root, sample_id)
        name = family_name(family_signature(cad_vec))
        if name in target_names:
            family_to_ids[name].append(sample_id)
        if idx % args.report_every == 0:
            print(f"  scanned {idx}/{len(ids)}")

    adapter = DeepCADAdapter(device=args.device)
    results = []

    for family in args.families:
        member_ids = family_to_ids.get(family, [])
        if not member_ids:
            results.append({"family_name": family, "error": "not_found"})
            continue

        if len(member_ids) > args.max_family_samples:
            sampled_ids = sorted(rng.choice(member_ids, size=args.max_family_samples, replace=False).tolist())
        else:
            sampled_ids = list(member_ids)

        print(f"Analyzing {family} | total={len(member_ids)} analyzed={len(sampled_ids)}")
        cad_batch = [load_raw_cad_vec(args.data_root, sample_id) for sample_id in sampled_ids]
        name_list, _ = named_parameter_vector(cad_batch[0])
        params = np.stack([named_parameter_vector(cad)[1] for cad in cad_batch], axis=0).astype(np.float32)

        z_parts = []
        for start in range(0, len(cad_batch), args.batch_size):
            chunk = cad_batch[start : start + args.batch_size]
            z_parts.append(adapter.encode(chunk).detach().cpu().numpy())
            print(f"  encoded {min(start + args.batch_size, len(cad_batch))}/{len(cad_batch)}")
        z = np.concatenate(z_parts, axis=0).astype(np.float32)

        centered = z - z.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        pc_axes = vh[:3]
        pc_scores = centered @ pc_axes.T

        param_std = params.std(axis=0)
        variable_idx = np.where(param_std > 1e-6)[0]

        perm = rng.permutation(len(z))
        split = max(1, int(round(0.8 * len(z))))
        train_idx = perm[:split]
        test_idx = perm[split:]

        param_reports = []
        for idx_param in variable_idx.tolist():
            values = params[:, idx_param]
            correlations = {}
            for pc_i in range(pc_scores.shape[1]):
                score = pc_scores[:, pc_i]
                if np.std(score) < 1e-12 or np.std(values) < 1e-12:
                    corr = None
                else:
                    corr = float(np.corrcoef(score, values)[0, 1])
                correlations[f"pc{pc_i+1}_corr"] = corr

            r2 = None
            if len(test_idx) > 0:
                r2 = fit_linear_single(z[train_idx], values[train_idx], z[test_idx], values[test_idx])

            param_reports.append(
                {
                    "name": name_list[idx_param],
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "linear_r2_from_latent": r2,
                    **correlations,
                }
            )

        param_reports.sort(
            key=lambda item: (
                -1.0 if item["linear_r2_from_latent"] is None else -float(item["linear_r2_from_latent"]),
                item["name"],
            )
        )

        results.append(
            {
                "family_name": family,
                "count_total": len(member_ids),
                "count_analyzed": len(sampled_ids),
                "latent_pca_explained_variance_ratio": (
                    (np.var(pc_scores, axis=0) / max(np.var(centered, axis=0).sum(), 1e-12)).tolist()
                ),
                "parameters_ranked_by_linear_r2": param_reports,
            }
        )

    out = {
        "ids": str(args.ids),
        "families": args.families,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"saved focused latent family analysis to {args.output}")


if __name__ == "__main__":
    main()
