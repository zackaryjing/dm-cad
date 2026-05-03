#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepcad_latent import ImageToCadPipeline
from deepcad_latent.data import ImageLatentDataset, collate_image_latent


CMD_NAMES = {
    0: "Line",
    1: "Arc",
    2: "Circle",
    3: "EOS",
    4: "SOL",
    5: "Ext",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Infer CAD from images via predicted DeepCAD latent")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/dataset_v0"))
    parser.add_argument("--lmdb-path", type=str, default="cad_data.lmdb")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--n-views", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--retrieval-latent-root", type=Path, default=None)
    parser.add_argument("--retrieval-mode", type=str, default="direct", choices=["direct", "nearest", "blend"])
    parser.add_argument("--retrieval-metric", type=str, default="cosine", choices=["cosine", "l2"])
    parser.add_argument("--retrieval-topk", type=int, default=1)
    parser.add_argument("--blend-alpha", type=float, default=0.5)
    return parser.parse_args()


def format_sequence(cad_vec) -> list[dict]:
    rows = []
    for step, token in enumerate(cad_vec):
        cmd = int(token[0])
        params = token[1:6].tolist()
        rows.append(
            {
                "step": step,
                "cmd_id": cmd,
                "cmd_name": CMD_NAMES.get(cmd, f"UNK_{cmd}"),
                "params_head": params,
            }
        )
        if cmd == 3:
            break
    return rows


def main():
    args = parse_args()
    dataset = ImageLatentDataset(
        ids_file=args.ids,
        latent_root=args.latent_root,
        data_root=args.data_root,
        lmdb_path=args.lmdb_path,
        img_size=args.img_size,
        n_views=args.n_views,
    )
    dataset = Subset(dataset, list(range(min(args.max_samples, len(dataset)))))
    loader = DataLoader(
        dataset,
        batch_size=min(args.max_samples, len(dataset)),
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_image_latent,
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

    batch = next(iter(loader))
    images = batch["images"]
    gt_z = batch["z"]
    pred_z = pipeline.predict_latent(images)
    resolved = pipeline.resolve_latent(
        pred_z,
        mode=args.retrieval_mode,
        topk=args.retrieval_topk,
        blend_alpha=args.blend_alpha,
    )
    final_z = resolved["final_z"]

    pred_cad = pipeline.decode_latent(final_z)
    gt_cad = pipeline.decode_latent(gt_z)

    results = []
    retrieval = resolved["retrieval"]
    for index, (sample_id, pred_seq, gt_seq) in enumerate(zip(batch["sample_ids"], pred_cad, gt_cad)):
        row = {
            "sample_id": sample_id,
            "pred_sequence": format_sequence(pred_seq),
            "gt_sequence_from_latent": format_sequence(gt_seq),
            "retrieval_mode": args.retrieval_mode,
        }
        if retrieval is not None:
            row["retrieved_sample_ids"] = retrieval["sample_ids"][index]
            row["retrieval_scores"] = retrieval["scores"][index].tolist()
        results.append(row)

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
