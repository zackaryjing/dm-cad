#!/usr/bin/env python3
"""Prepare overlap-based dataset ID subsets for the DeepCAD-latent rescue route."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import h5py


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare rescue dataset ID files")
    parser.add_argument(
        "--v0-root",
        type=Path,
        default=Path("/root/projects/dm-cad2/datasets/dataset_v0"),
        help="Path to dataset_v0 root",
    )
    parser.add_argument(
        "--deepcad-root",
        type=Path,
        default=Path("/root/projects/CAD-MLLM/3rd_party/DeepCAD/datasets/data"),
        help="Path to DeepCAD dataset root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/projects/dm-cad2/datasets/rescue_deepcad_latent"),
        help="Output directory for generated ID files and metadata",
    )
    parser.add_argument(
        "--length-thresholds",
        type=int,
        nargs="*",
        default=[],
        help="Optional sequence-length thresholds for extra conservative subsets",
    )
    return parser.parse_args()


def normalize_v0_id(sample_id: str) -> str:
    group_id, sample_name = sample_id.split("/")[:2]
    base_name = sample_name.rsplit("_", 1)[0] if "_" in sample_name else sample_name
    return f"{group_id}/{base_name}"


def read_lines(path: Path) -> list[str]:
    with path.open("r") as f:
        return [line.strip() for line in f if line.strip()]


def load_v0_ids(v0_root: Path) -> dict[str, list[str]]:
    return {
        "train": read_lines(v0_root / "train_ids.txt"),
        "test": read_lines(v0_root / "test_ids.txt"),
    }


def load_deepcad_split(deepcad_root: Path) -> dict[str, list[str]]:
    with (deepcad_root / "train_val_test_split.json").open("r") as f:
        return json.load(f)


def load_sequence_length(v0_root: Path, sample_id: str) -> int:
    group_id, sample_name = sample_id.split("/")[:2]
    path = v0_root / "cad_vec" / group_id / f"{sample_name}.h5"
    with h5py.File(path, "r") as f:
        key = next(iter(f.keys()))
        cad_vec = f[key][:]
    if cad_vec.ndim == 1:
        cad_vec = cad_vec.reshape(-1, 17)
    return int(cad_vec.shape[0])


def write_ids(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for sample_id in ids:
            f.write(sample_id + "\n")


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    v0_ids = load_v0_ids(args.v0_root)
    deepcad_split = load_deepcad_split(args.deepcad_root)
    deep_phase_by_base = {}
    for phase, ids in deepcad_split.items():
        for sample_id in ids:
            deep_phase_by_base[sample_id] = phase

    need_lengths = bool(args.length_thresholds)
    # Build per-sample metadata once so every subset is reproducible.
    records = []
    variants_by_base = defaultdict(list)
    base_counter_by_v0_split = {"train": Counter(), "test": Counter()}
    for split_name, ids in v0_ids.items():
        for sample_id in ids:
            base_id = normalize_v0_id(sample_id)
            seq_len = load_sequence_length(args.v0_root, sample_id) if need_lengths else None
            phase = deep_phase_by_base.get(base_id)
            record = {
                "sample_id": sample_id,
                "base_id": base_id,
                "v0_split": split_name,
                "deepcad_phase": phase,
                "is_overlap": phase is not None,
                "seq_len": seq_len,
            }
            records.append(record)
            variants_by_base[base_id].append(record)
            base_counter_by_v0_split[split_name][base_id] += 1

    # Order variants deterministically so "first" is reproducible.
    for variants in variants_by_base.values():
        variants.sort(key=lambda item: item["sample_id"])

    # Prepare versions.
    versions = {}

    versions["full_v0"] = {
        "train": sorted(v0_ids["train"]),
        "test": sorted(v0_ids["test"]),
    }

    overlap_records = [record for record in records if record["is_overlap"]]
    versions["overlap_deep_all"] = {
        phase: sorted(
            record["sample_id"] for record in overlap_records if record["deepcad_phase"] == phase
        )
        for phase in ["train", "validation", "test"]
    }

    first_overlap_records = []
    for base_id, variants in variants_by_base.items():
        first = next((item for item in variants if item["is_overlap"]), None)
        if first is not None:
            first_overlap_records.append(first)

    versions["overlap_deep_first"] = {
        phase: sorted(
            record["sample_id"] for record in first_overlap_records if record["deepcad_phase"] == phase
        )
        for phase in ["train", "validation", "test"]
    }

    for threshold in sorted(set(args.length_thresholds)):
        key_all = f"overlap_deep_all_len{threshold}"
        versions[key_all] = {
            phase: sorted(
                record["sample_id"]
                for record in overlap_records
                if record["deepcad_phase"] == phase and record["seq_len"] <= threshold
            )
            for phase in ["train", "validation", "test"]
        }

        key_first = f"overlap_deep_first_len{threshold}"
        versions[key_first] = {
            phase: sorted(
                record["sample_id"]
                for record in first_overlap_records
                if record["deepcad_phase"] == phase and record["seq_len"] <= threshold
            )
            for phase in ["train", "validation", "test"]
        }

    # Persist ids.
    for version_name, split_map in versions.items():
        version_dir = args.output_root / version_name
        for split_name, ids in split_map.items():
            write_ids(version_dir / f"{split_name}_ids.txt", ids)

    # Persist metadata summary for later decisions.
    train_bases = set(base_counter_by_v0_split["train"])
    test_bases = set(base_counter_by_v0_split["test"])
    overlap_bases = {base_id for base_id, variants in variants_by_base.items() if any(v["is_overlap"] for v in variants)}
    summary = {
        "v0_total_samples": len(records),
        "v0_total_bases": len(variants_by_base),
        "v0_train_bases": len(train_bases),
        "v0_test_bases": len(test_bases),
        "v0_base_leakage_count": len(train_bases & test_bases),
        "deepcad_total_bases": len(deep_phase_by_base),
        "overlap_base_count": len(overlap_bases),
        "overlap_ratio_vs_v0_bases": len(overlap_bases) / max(len(variants_by_base), 1),
        "overlap_ratio_vs_deepcad_bases": len(overlap_bases) / max(len(deep_phase_by_base), 1),
        "variants_per_base": {
            "min": min(len(items) for items in variants_by_base.values()),
            "p50": sorted(len(items) for items in variants_by_base.values())[len(variants_by_base) // 2],
            "max": max(len(items) for items in variants_by_base.values()),
        },
        "versions": {
            version_name: {split_name: len(ids) for split_name, ids in split_map.items()}
            for version_name, split_map in versions.items()
        },
    }
    if need_lengths:
        length_counter = Counter(record["seq_len"] for record in records)
        overlap_length_counter = Counter(record["seq_len"] for record in overlap_records)
        summary["length_counts_all"] = dict(sorted(length_counter.items()))
        summary["length_counts_overlap"] = dict(sorted(overlap_length_counter.items()))
    with (args.output_root / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"Wrote rescue ID files under {args.output_root}")
    print(json.dumps(summary["versions"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
