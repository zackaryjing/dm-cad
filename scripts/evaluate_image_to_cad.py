#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepcad_latent import DeepCADAdapter
from deepcad_latent.data import ImageOnlyDataset, collate_image_only, load_raw_cad_vec
from deepcad_latent.model import MultiViewLatentRegressor


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate image-to-CAD model on sequence-level metrics")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/dataset_v0"))
    parser.add_argument("--lmdb-path", type=str, default="cad_data.lmdb")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--n-views", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means evaluate all")
    parser.add_argument("--check-solid-validity", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to save detailed json")
    return parser.parse_args()


def maybe_build_solid_checker(enabled: bool):
    if not enabled:
        return None
    deepcad_root = Path("/root/projects/CAD-MLLM/3rd_party/DeepCAD")
    if str(deepcad_root) not in sys.path:
        sys.path.insert(0, str(deepcad_root))
    try:
        from OCC.Core.BRepCheck import BRepCheck_Analyzer
        from cadlib.visualize import vec2CADsolid
    except Exception as exc:
        print(f"warning: solid validity disabled because OCC/DeepCAD import failed: {exc}")
        return None

    def checker(cad_vec: np.ndarray) -> bool:
        try:
            shape = vec2CADsolid(cad_vec.astype(np.float64))
            analyzer = BRepCheck_Analyzer(shape)
            return bool(analyzer.IsValid())
        except Exception:
            return False

    return checker


def trim_eos_exclusive(cad_vec: np.ndarray) -> np.ndarray:
    if cad_vec.ndim == 1:
        cad_vec = cad_vec.reshape(-1, 17)
    eos_positions = np.where(cad_vec[:, 0] == 3)[0]
    if len(eos_positions) > 0:
        return cad_vec[: int(eos_positions[0])]
    valid = np.where(cad_vec[:, 0] >= 0)[0]
    if len(valid) == 0:
        return cad_vec[:0]
    return cad_vec[: int(valid[-1]) + 1]


def pad_for_compare(pred: np.ndarray, gt: np.ndarray, fill: int = -999) -> tuple[np.ndarray, np.ndarray]:
    max_len = max(len(pred), len(gt))
    pred_pad = np.full((max_len, pred.shape[1]), fill, dtype=np.int64)
    gt_pad = np.full((max_len, gt.shape[1]), fill, dtype=np.int64)
    if len(pred):
        pred_pad[: len(pred)] = pred
    if len(gt):
        gt_pad[: len(gt)] = gt
    return pred_pad, gt_pad


def length_bucket(gt_len: int) -> str:
    if gt_len <= 6:
        return "len_01_06"
    if gt_len <= 12:
        return "len_07_12"
    if gt_len <= 24:
        return "len_13_24"
    return "len_25_plus"


def sample_metrics(pred: np.ndarray, gt: np.ndarray, solid_checker=None) -> dict[str, object]:
    pred = trim_eos_exclusive(pred)
    gt = trim_eos_exclusive(gt)
    pred_pad, gt_pad = pad_for_compare(pred, gt)

    cmd_equal = pred_pad[:, 0] == gt_pad[:, 0]
    token_equal = np.all(pred_pad == gt_pad, axis=1)

    mismatch = np.where(~token_equal)[0]
    first_div = int(mismatch[0]) if len(mismatch) > 0 else -1

    metrics = {
        "pred_len": int(len(pred)),
        "gt_len": int(len(gt)),
        "len_abs_error": int(abs(len(pred) - len(gt))),
        "cmd_token_acc": float(cmd_equal.mean()) if len(cmd_equal) else 1.0,
        "token_exact_acc": float(token_equal.mean()) if len(token_equal) else 1.0,
        "sequence_cmd_exact": bool(cmd_equal.all()),
        "sequence_exact": bool(token_equal.all()),
        "first_divergence_step": first_div,
        "bucket": length_bucket(int(len(gt))),
    }
    if solid_checker is not None:
        metrics["pred_solid_valid"] = bool(solid_checker(pred)) if len(pred) > 0 else False
        metrics["gt_solid_valid"] = bool(solid_checker(gt)) if len(gt) > 0 else False
    return metrics


def aggregate(records: list[dict]) -> dict[str, object]:
    total = max(len(records), 1)

    def mean(key: str) -> float:
        vals = [float(r[key]) for r in records]
        return float(sum(vals) / max(len(vals), 1))

    summary = {
        "num_samples": len(records),
        "cmd_token_acc": mean("cmd_token_acc"),
        "token_exact_acc": mean("token_exact_acc"),
        "sequence_cmd_exact_rate": mean("sequence_cmd_exact"),
        "sequence_exact_rate": mean("sequence_exact"),
        "mean_pred_len": mean("pred_len"),
        "mean_gt_len": mean("gt_len"),
        "mean_len_abs_error": mean("len_abs_error"),
        "first_divergence_missing_rate": float(sum(r["first_divergence_step"] == -1 for r in records) / total),
    }

    if records and "pred_solid_valid" in records[0]:
        summary["pred_solid_valid_rate"] = mean("pred_solid_valid")
        summary["gt_solid_valid_rate"] = mean("gt_solid_valid")

    buckets = defaultdict(list)
    for record in records:
        buckets[record["bucket"]].append(record)

    summary["by_bucket"] = {}
    for bucket_name, bucket_records in sorted(buckets.items()):
        summary["by_bucket"][bucket_name] = {
            "count": len(bucket_records),
            "cmd_token_acc": float(sum(float(r["cmd_token_acc"]) for r in bucket_records) / len(bucket_records)),
            "token_exact_acc": float(sum(float(r["token_exact_acc"]) for r in bucket_records) / len(bucket_records)),
            "sequence_exact_rate": float(sum(float(r["sequence_exact"]) for r in bucket_records) / len(bucket_records)),
            "mean_len_abs_error": float(sum(float(r["len_abs_error"]) for r in bucket_records) / len(bucket_records)),
        }
        if "pred_solid_valid" in bucket_records[0]:
            summary["by_bucket"][bucket_name]["pred_solid_valid_rate"] = float(
                sum(float(r["pred_solid_valid"]) for r in bucket_records) / len(bucket_records)
            )

    worst = sorted(
        records,
        key=lambda r: (-int(not r["sequence_exact"]), -float(r["len_abs_error"]), -float(1.0 - r["cmd_token_acc"])),
    )[:20]
    summary["worst_examples"] = [
        {
            "sample_id": r["sample_id"],
            "bucket": r["bucket"],
            "pred_len": r["pred_len"],
            "gt_len": r["gt_len"],
            "len_abs_error": r["len_abs_error"],
            "cmd_token_acc": r["cmd_token_acc"],
            "token_exact_acc": r["token_exact_acc"],
            "sequence_exact": r["sequence_exact"],
            "first_divergence_step": r["first_divergence_step"],
            **({"pred_solid_valid": r["pred_solid_valid"]} if "pred_solid_valid" in r else {}),
        }
        for r in worst
    ]
    return summary


def main():
    args = parse_args()
    device = torch.device(args.device)

    dataset = ImageOnlyDataset(
        ids_file=args.ids,
        data_root=args.data_root,
        lmdb_path=args.lmdb_path,
        img_size=args.img_size,
        n_views=args.n_views,
    )
    if args.max_samples > 0:
        dataset = Subset(dataset, list(range(min(args.max_samples, len(dataset)))))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_image_only,
        pin_memory=args.device.startswith("cuda"),
    )

    model = MultiViewLatentRegressor(
        backbone_name=args.backbone,
        n_views=args.n_views,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    adapter = DeepCADAdapter(device=args.device)
    solid_checker = maybe_build_solid_checker(args.check_solid_validity)

    records: list[dict] = []
    seen = 0
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        sample_ids = batch["sample_ids"]
        with torch.no_grad():
            pred_z = model(images).cpu()
        pred_cad_batch = adapter.decode(pred_z)

        for sample_id, pred_cad in zip(sample_ids, pred_cad_batch):
            gt_cad = load_raw_cad_vec(args.data_root, sample_id)
            metrics = sample_metrics(pred_cad, gt_cad, solid_checker=solid_checker)
            metrics["sample_id"] = sample_id
            records.append(metrics)
            seen += 1
            if seen % 200 == 0:
                print(
                    f"processed {seen} samples | "
                    f"seq_exact={sum(float(r['sequence_exact']) for r in records)/len(records):.4f} "
                    f"cmd_acc={sum(float(r['cmd_token_acc']) for r in records)/len(records):.4f}"
                )

    summary = aggregate(records)
    result = {
        "checkpoint": str(args.checkpoint),
        "ids": str(args.ids),
        "num_samples": len(records),
        "summary": summary,
    }
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"saved detailed report to {args.output}")


if __name__ == "__main__":
    main()
