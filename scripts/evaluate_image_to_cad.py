#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepcad_latent import DeepCADAdapter, ImageToCadPipeline
from deepcad_latent.pipeline import _is_multimodal_state_dict
from deepcad_latent.data import (
    ImageOnlyDataset,
    ImageTextOnlyDataset,
    collate_image_only,
    collate_image_text_only,
    load_raw_cad_vec,
)

DEEP_CAD_ROOT = Path("/root/projects/CAD-MLLM/3rd_party/DeepCAD")
if str(DEEP_CAD_ROOT) not in sys.path:
    sys.path.insert(0, str(DEEP_CAD_ROOT))

from cadlib.macro import ARC_IDX, CMD_ARGS_MASK, EOS_IDX, EXT_IDX, SOL_IDX

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*face_normals didn't match triangles.*")
logging.getLogger("trimesh").setLevel(logging.ERROR)


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
    parser.add_argument("--text-root", type=Path, default=None, help="Required when evaluating a multimodal checkpoint")
    parser.add_argument("--retrieval-latent-root", type=Path, default=None)
    parser.add_argument("--retrieval-mode", type=str, default="direct", choices=["direct", "nearest", "blend"])
    parser.add_argument("--retrieval-metric", type=str, default="cosine", choices=["cosine", "l2"])
    parser.add_argument("--retrieval-topk", type=int, default=1)
    parser.add_argument("--blend-alpha", type=float, default=0.5)
    parser.add_argument("--param-tol", type=int, default=3, help="Tolerance for ACC_param, aligned with DeepCAD")
    parser.add_argument("--compute-chamfer", action="store_true")
    parser.add_argument("--num-cd-points", type=int, default=2000)
    parser.add_argument(
        "--compare-target",
        type=str,
        default="true",
        choices=["true", "decoded-latent"],
        help="Compare CAD_pred against CAD_true or against CAD_z_true decoded from GT latent.",
    )
    parser.add_argument(
        "--report-ae-ceiling",
        action="store_true",
        help="Also report CAD_z_true vs CAD_true to estimate the DeepCAD autoencoder ceiling.",
    )
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


def maybe_build_chamfer_evaluator(enabled: bool, num_points: int):
    if not enabled:
        return None
    deepcad_root = Path("/root/projects/CAD-MLLM/3rd_party/DeepCAD")
    if str(deepcad_root) not in sys.path:
        sys.path.insert(0, str(deepcad_root))
    try:
        from scipy.spatial import cKDTree as KDTree
        from cadlib.visualize import CADsolid2pc, vec2CADsolid
    except Exception as exc:
        print(f"warning: chamfer evaluation disabled because imports failed: {exc}")
        return None

    def chamfer_dist(gt_points: np.ndarray, pred_points: np.ndarray) -> float:
        pred_tree = KDTree(pred_points)
        gt_to_pred, _ = pred_tree.query(gt_points)
        gt_to_pred = float(np.mean(np.square(gt_to_pred)))

        gt_tree = KDTree(gt_points)
        pred_to_gt, _ = gt_tree.query(pred_points)
        pred_to_gt = float(np.mean(np.square(pred_to_gt)))
        return gt_to_pred + pred_to_gt

    def evaluator(pred_cad_vec: np.ndarray, gt_cad_vec: np.ndarray, sample_id: str) -> float:
        try:
            sink = io.StringIO()
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred_shape = vec2CADsolid(pred_cad_vec.astype(np.float64))
                gt_shape = vec2CADsolid(gt_cad_vec.astype(np.float64))
                pred_pc = CADsolid2pc(pred_shape, num_points, name=f"pred_{sample_id.replace('/', '_')}")
                gt_pc = CADsolid2pc(gt_shape, num_points, name=f"gt_{sample_id.replace('/', '_')}")
            return float(chamfer_dist(gt_pc, pred_pc))
        except Exception:
            return -1.0

    return evaluator


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


def sample_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    sample_id: str,
    solid_checker=None,
    chamfer_evaluator=None,
    param_tol: int = 3,
) -> dict[str, object]:
    pred = trim_eos_exclusive(pred)
    gt = trim_eos_exclusive(gt)
    pred_pad, gt_pad = pad_for_compare(pred, gt)

    cmd_equal = pred_pad[:, 0] == gt_pad[:, 0]
    token_equal = np.all(pred_pad == gt_pad, axis=1)

    mismatch = np.where(~token_equal)[0]
    first_div = int(mismatch[0]) if len(mismatch) > 0 else -1

    gt_len = int(len(gt))
    cmd_correct_count = 0
    param_correct_count = 0
    param_total_count = 0

    for j in range(gt_len):
        gt_cmd = int(gt[j, 0])
        pred_cmd = int(pred[j, 0]) if j < len(pred) else -999
        if pred_cmd == gt_cmd:
            cmd_correct_count += 1

        if gt_cmd in [SOL_IDX, EOS_IDX]:
            continue

        if pred_cmd != gt_cmd or j >= len(pred):
            continue

        gt_param = gt[j, 1:]
        pred_param = pred[j, 1:]
        tol_acc = (np.abs(pred_param - gt_param) <= param_tol).astype(np.int64)

        # Align with DeepCAD official ACC_param evaluation.
        if gt_cmd == EXT_IDX:
            tol_acc[-2:] = (pred_param[-2:] == gt_param[-2:]).astype(np.int64)
        elif gt_cmd == ARC_IDX:
            tol_acc[3] = int(pred_param[3] == gt_param[3])

        valid_mask = CMD_ARGS_MASK[gt_cmd].astype(bool)
        valid_param_acc = tol_acc[valid_mask]
        param_correct_count += int(valid_param_acc.sum())
        param_total_count += int(valid_param_acc.size)

    metrics = {
        "pred_len": int(len(pred)),
        "gt_len": gt_len,
        "len_abs_error": int(abs(len(pred) - len(gt))),
        "cmd_token_acc": float(cmd_equal.mean()) if len(cmd_equal) else 1.0,
        "token_exact_acc": float(token_equal.mean()) if len(token_equal) else 1.0,
        "sequence_cmd_exact": bool(cmd_equal.all()),
        "sequence_exact": bool(token_equal.all()),
        "first_divergence_step": first_div,
        "bucket": length_bucket(int(len(gt))),
        "acc_cmd": float(cmd_correct_count / max(gt_len, 1)),
        "cmd_correct_count": cmd_correct_count,
        "cmd_total_count": gt_len,
        "param_correct_count": param_correct_count,
        "param_total_count": param_total_count,
        "acc_param": float(param_correct_count / param_total_count) if param_total_count > 0 else 1.0,
    }
    if solid_checker is not None:
        metrics["pred_solid_valid"] = bool(solid_checker(pred)) if len(pred) > 0 else False
        metrics["gt_solid_valid"] = bool(solid_checker(gt)) if len(gt) > 0 else False
    if chamfer_evaluator is not None:
        metrics["cd"] = float(chamfer_evaluator(pred, gt, sample_id))
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
        "acc_cmd": float(sum(int(r["cmd_correct_count"]) for r in records) / max(sum(int(r["cmd_total_count"]) for r in records), 1)),
        "acc_param": float(sum(int(r["param_correct_count"]) for r in records) / max(sum(int(r["param_total_count"]) for r in records), 1)),
    }

    if records and "pred_solid_valid" in records[0]:
        summary["pred_solid_valid_rate"] = mean("pred_solid_valid")
        summary["gt_solid_valid_rate"] = mean("gt_solid_valid")
        summary["invalidity_ratio"] = float(1.0 - summary["pred_solid_valid_rate"])

    if records and "cd" in records[0]:
        valid_cds = [float(r["cd"]) for r in records if float(r["cd"]) >= 0.0]
        summary["cd_valid_count"] = len(valid_cds)
        summary["cd_invalid_count"] = len(records) - len(valid_cds)
        summary["cd_mean"] = float(np.mean(valid_cds)) if valid_cds else None
        summary["cd_median"] = float(np.median(valid_cds)) if valid_cds else None

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
            "acc_cmd": float(
                sum(int(r["cmd_correct_count"]) for r in bucket_records)
                / max(sum(int(r["cmd_total_count"]) for r in bucket_records), 1)
            ),
            "acc_param": float(
                sum(int(r["param_correct_count"]) for r in bucket_records)
                / max(sum(int(r["param_total_count"]) for r in bucket_records), 1)
            ),
        }
        if "pred_solid_valid" in bucket_records[0]:
            summary["by_bucket"][bucket_name]["pred_solid_valid_rate"] = float(
                sum(float(r["pred_solid_valid"]) for r in bucket_records) / len(bucket_records)
            )
            summary["by_bucket"][bucket_name]["invalidity_ratio"] = float(
                1.0 - summary["by_bucket"][bucket_name]["pred_solid_valid_rate"]
            )
        if "cd" in bucket_records[0]:
            bucket_valid_cds = [float(r["cd"]) for r in bucket_records if float(r["cd"]) >= 0.0]
            summary["by_bucket"][bucket_name]["cd_mean"] = (
                float(np.mean(bucket_valid_cds)) if bucket_valid_cds else None
            )
            summary["by_bucket"][bucket_name]["cd_median"] = (
                float(np.median(bucket_valid_cds)) if bucket_valid_cds else None
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
            **({"cd": r["cd"]} if "cd" in r else {}),
            **({"pred_solid_valid": r["pred_solid_valid"]} if "pred_solid_valid" in r else {}),
        }
        for r in worst
    ]
    return summary


def main():
    args = parse_args()
    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["model"]
    is_multimodal = _is_multimodal_state_dict(state_dict)

    if is_multimodal:
        if args.text_root is None:
            raise ValueError("--text-root is required when evaluating a multimodal checkpoint.")
        dataset = ImageTextOnlyDataset(
            ids_file=args.ids,
            text_root=args.text_root,
            data_root=args.data_root,
            lmdb_path=args.lmdb_path,
            img_size=args.img_size,
            n_views=args.n_views,
        )
        collate_fn = collate_image_text_only
    else:
        dataset = ImageOnlyDataset(
            ids_file=args.ids,
            data_root=args.data_root,
            lmdb_path=args.lmdb_path,
            img_size=args.img_size,
            n_views=args.n_views,
        )
        collate_fn = collate_image_only

    if args.max_samples > 0:
        dataset = Subset(dataset, list(range(min(args.max_samples, len(dataset)))))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=args.device.startswith("cuda"),
    )

    pipeline = ImageToCadPipeline(
        checkpoint_path=args.checkpoint,
        device=args.device,
        backbone=args.backbone,
        n_views=args.n_views,
        freeze_backbone=args.freeze_backbone,
        retrieval_latent_root=args.retrieval_latent_root,
        retrieval_metric=args.retrieval_metric,
    )
    adapter = pipeline.adapter
    solid_checker = maybe_build_solid_checker(args.check_solid_validity)
    chamfer_evaluator = maybe_build_chamfer_evaluator(args.compute_chamfer, args.num_cd_points)

    records: list[dict] = []
    ae_ceiling_records: list[dict] = []
    seen = 0
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        sample_ids = batch["sample_ids"]
        text_emb = batch.get("text_emb")
        if text_emb is not None:
            text_emb = text_emb.to(device, non_blocking=True)
        pred_z = pipeline.predict_latent(images, text_emb=text_emb)
        resolved = pipeline.resolve_latent(
            pred_z,
            mode=args.retrieval_mode,
            topk=args.retrieval_topk,
            blend_alpha=args.blend_alpha,
        )
        pred_cad_batch = adapter.decode(resolved["final_z"])
        retrieval = resolved["retrieval"]

        for index, (sample_id, pred_cad) in enumerate(zip(sample_ids, pred_cad_batch)):
            gt_cad = load_raw_cad_vec(args.data_root, sample_id)
            target_cad = gt_cad
            gt_decoded_cad = None
            if args.compare_target == "decoded-latent" or args.report_ae_ceiling:
                z_true = adapter.encode([gt_cad]).detach().cpu()
                gt_decoded_cad = adapter.decode(z_true)[0]
            if args.compare_target == "decoded-latent":
                target_cad = gt_decoded_cad
            metrics = sample_metrics(
                pred_cad,
                target_cad,
                sample_id=sample_id,
                solid_checker=solid_checker,
                chamfer_evaluator=chamfer_evaluator,
                param_tol=args.param_tol,
            )
            metrics["sample_id"] = sample_id
            metrics["retrieval_mode"] = args.retrieval_mode
            metrics["compare_target"] = args.compare_target
            if gt_decoded_cad is not None:
                metrics["gt_true_len"] = int(len(trim_eos_exclusive(gt_cad)))
                metrics["gt_decoded_len"] = int(len(trim_eos_exclusive(gt_decoded_cad)))
            if retrieval is not None:
                metrics["retrieved_sample_id"] = retrieval["sample_ids"][index][0]
                metrics["retrieval_score"] = float(retrieval["scores"][index][0])
            records.append(metrics)
            if args.report_ae_ceiling:
                ceiling_metrics = sample_metrics(
                    gt_decoded_cad,
                    gt_cad,
                    sample_id=sample_id,
                    solid_checker=solid_checker,
                    chamfer_evaluator=chamfer_evaluator,
                    param_tol=args.param_tol,
                )
                ceiling_metrics["sample_id"] = sample_id
                ceiling_metrics["compare_target"] = "ae_ceiling"
                ae_ceiling_records.append(ceiling_metrics)
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
        "is_multimodal": is_multimodal,
        "text_root": str(args.text_root) if args.text_root is not None else None,
        "retrieval_mode": args.retrieval_mode,
        "compare_target": args.compare_target,
        "param_tolerance": int(args.param_tol),
        "num_samples": len(records),
        "summary": summary,
        "paper_metrics": {
            "acc_cmd": summary["acc_cmd"],
            "acc_param": summary["acc_param"],
            "invalidity_ratio": summary.get("invalidity_ratio"),
            "cd_mean": summary.get("cd_mean"),
            "cd_median": summary.get("cd_median"),
        },
    }
    if ae_ceiling_records:
        ae_summary = aggregate(ae_ceiling_records)
        result["ae_ceiling_summary"] = ae_summary
        result["ae_ceiling_paper_metrics"] = {
            "acc_cmd": ae_summary["acc_cmd"],
            "acc_param": ae_summary["acc_param"],
            "invalidity_ratio": ae_summary.get("invalidity_ratio"),
            "cd_mean": ae_summary.get("cd_mean"),
            "cd_median": ae_summary.get("cd_median"),
        }
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    if "ae_ceiling_summary" in result:
        print("AE ceiling:")
        print(json.dumps(result["ae_ceiling_summary"], indent=2, ensure_ascii=False))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"saved detailed report to {args.output}")


if __name__ == "__main__":
    main()
