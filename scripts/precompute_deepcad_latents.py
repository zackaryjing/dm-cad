#!/usr/bin/env python3
"""Precompute DeepCAD latent codes for a given ID subset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepcad_latent import DeepCADAdapter


class CadVecDataset(Dataset):
    def __init__(self, ids_file: Path, data_root: Path):
        with ids_file.open("r") as f:
            self.sample_ids = [line.strip() for line in f if line.strip()]
        self.data_root = data_root

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int):
        sample_id = self.sample_ids[index]
        group_id, sample_name = sample_id.split("/")[:2]
        path = self.data_root / "cad_vec" / group_id / f"{sample_name}.h5"
        with h5py.File(path, "r") as f:
            key = next(iter(f.keys()))
            cad_vec = f[key][:]
        if cad_vec.ndim == 1:
            cad_vec = cad_vec.reshape(-1, 17)
        return sample_id, cad_vec.astype(np.int64, copy=False)


def collate_batch(batch):
    sample_ids = [item[0] for item in batch]
    cad_vecs = [item[1] for item in batch]
    return sample_ids, cad_vecs


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute DeepCAD latent codes")
    parser.add_argument("--ids", type=Path, required=True, help="Path to *_ids.txt")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/root/projects/dm-cad2/datasets/dataset_v0"),
        help="Path to dataset_v0 root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write shard .pt files",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Encoding batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--shard-size", type=int, default=50000, help="Samples per output shard")
    parser.add_argument("--device", type=str, default="cuda", help="Device for DeepCAD encoder")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    adapter = DeepCADAdapter(device=args.device)
    dataset = CadVecDataset(args.ids, args.data_root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=collate_batch,
        pin_memory=args.device.startswith("cuda"),
    )

    shard_sample_ids: list[str] = []
    shard_zs: list[torch.Tensor] = []
    shard_index = 0
    total = 0
    skipped = 0

    def flush():
        nonlocal shard_sample_ids, shard_zs, shard_index
        if not shard_sample_ids:
            return
        output_path = args.output_dir / f"shard_{shard_index:05d}.pt"
        payload = {
            "sample_ids": shard_sample_ids,
            "z": torch.cat(shard_zs, dim=0).cpu().to(torch.float16),
            "source_ids_file": str(args.ids),
        }
        torch.save(payload, output_path)
        print(f"saved {output_path} ({len(shard_sample_ids)} samples)")
        shard_sample_ids = []
        shard_zs = []
        shard_index += 1

    for sample_ids, cad_vecs in loader:
        kept_ids = []
        kept_vecs = []
        for sample_id, cad_vec in zip(sample_ids, cad_vecs):
            if cad_vec.shape[0] > adapter.max_total_len:
                skipped += 1
                continue
            kept_ids.append(sample_id)
            kept_vecs.append(cad_vec)
        if not kept_ids:
            continue
        z = adapter.encode(kept_vecs)
        shard_sample_ids.extend(kept_ids)
        shard_zs.append(z.detach())
        total += len(kept_ids)
        if len(shard_sample_ids) >= args.shard_size:
            flush()

    flush()
    print(f"done: {total} samples from {args.ids} (skipped {skipped} over-length samples; checkpoint max_total_len={adapter.max_total_len})")


if __name__ == "__main__":
    main()
