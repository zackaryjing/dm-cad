#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepcad_latent import ImageToCadPipeline
from deepcad_latent.data import ImageLatentDataset, collate_image_latent, load_raw_cad_vec

from report_material.generate_thesis_qualitative import (
    DATA_ROOT,
    GRU_CKPT,
    IDS_FILE,
    OUT_ROOT,
    TEXT_ROOT,
    TRANSFORMER_CKPT,
    is_valid_cad,
    load_text_caption,
    make_panel,
    sequence_metrics,
)

LATENT_ROOT = REPO_ROOT / "datasets" / "rescue_deepcad_latent" / "latents" / "overlap_deep_first_len60_trainplusval_fp16" / "test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find stronger GRU-vs-Transformer qualitative examples.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means scan the full test split.")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--render-rows", type=int, default=4)
    parser.add_argument("--render-size", type=int, default=360)
    parser.add_argument("--require-gru-valid", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "report_material" / "figures" / "qualitative_candidates",
    )
    return parser.parse_args()


def latent_scores(z_pred: torch.Tensor, z_gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mse = torch.mean((z_pred - z_gt) ** 2, dim=1)
    cosine = torch.nn.functional.cosine_similarity(z_pred, z_gt, dim=1)
    return mse.cpu(), cosine.cpu()


def sequence_score(m_gru: dict, m_trans: dict, valid_gru: bool, valid_trans: bool) -> float:
    score = 0.0
    score += 8.0 * (float(m_trans["sequence_exact"]) - float(m_gru["sequence_exact"]))
    score += 4.0 * (m_trans["cmd_token_acc"] - m_gru["cmd_token_acc"])
    score += 3.0 * (m_trans["token_exact_acc"] - m_gru["token_exact_acc"])
    score += 0.6 * (m_gru["len_abs_error"] - m_trans["len_abs_error"])
    score += 2.0 * (float(valid_trans) - float(valid_gru))
    return float(score)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    dataset = ImageLatentDataset(ids_file=IDS_FILE, latent_root=LATENT_ROOT, data_root=DATA_ROOT)
    if args.max_samples > 0:
        ids = list(range(min(args.max_samples, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, ids)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_image_latent,
        pin_memory=args.device.startswith("cuda"),
    )

    pipe_gru = ImageToCadPipeline(GRU_CKPT, device=args.device, backbone="resnet18", n_views=8)
    pipe_trans = ImageToCadPipeline(TRANSFORMER_CKPT, device=args.device, backbone="resnet18", n_views=8)

    latent_candidates: list[dict] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch["images"].to(args.device, non_blocking=True)
            z_gt = batch["z"].to(args.device, non_blocking=True)
            sample_ids = batch["sample_ids"]

            z_gru = pipe_gru.predict_latent(images).to(args.device)
            z_trans = pipe_trans.predict_latent(images).to(args.device)
            mse_gru, cos_gru = latent_scores(z_gru, z_gt)
            mse_trans, cos_trans = latent_scores(z_trans, z_gt)

            for i, sid in enumerate(sample_ids):
                improvement = float(mse_gru[i] - mse_trans[i])
                cosine_gain = float(cos_trans[i] - cos_gru[i])
                score = improvement + 0.1 * cosine_gain
                if improvement <= 0:
                    continue
                latent_candidates.append(
                    {
                        "sample_id": sid,
                        "score": score,
                        "latent_mse_gru": float(mse_gru[i]),
                        "latent_mse_transformer": float(mse_trans[i]),
                        "latent_cos_gru": float(cos_gru[i]),
                        "latent_cos_transformer": float(cos_trans[i]),
                        "z_gru": z_gru[i].detach().cpu(),
                        "z_trans": z_trans[i].detach().cpu(),
                        "caption": load_text_caption(sid),
                    }
                )

            print(f"scanned batch {batch_idx + 1}/{len(loader)}, latent_candidates={len(latent_candidates)}", flush=True)

    ranked_latent = sorted(latent_candidates, key=lambda x: x["score"], reverse=True)
    decode_pool = ranked_latent[: max(args.top_k * 6, args.render_rows * 8)]
    z_gru_pool = torch.stack([item["z_gru"] for item in decode_pool])
    z_trans_pool = torch.stack([item["z_trans"] for item in decode_pool])
    cad_gru_pool = pipe_gru.decode_latent(z_gru_pool)
    cad_trans_pool = pipe_trans.decode_latent(z_trans_pool)

    decoded_candidates: list[dict] = []
    for item, cad_gru, cad_trans in zip(decode_pool, cad_gru_pool, cad_trans_pool):
        gt = load_raw_cad_vec(DATA_ROOT, item["sample_id"])
        m_gru = sequence_metrics(cad_gru, gt)
        m_trans = sequence_metrics(cad_trans, gt)
        valid_gru = is_valid_cad(cad_gru)
        valid_trans = is_valid_cad(cad_trans)
        seq_score = sequence_score(m_gru, m_trans, valid_gru, valid_trans)
        if args.require_gru_valid and not valid_gru:
            continue
        if not valid_trans:
            continue
        if seq_score <= 0:
            continue
        if m_trans["cmd_token_acc"] < m_gru["cmd_token_acc"] + 0.08 and not (
            m_trans["sequence_exact"] and not m_gru["sequence_exact"]
        ):
            continue
        item = dict(item)
        item.update(
            {
                "score": float(item["score"] + 0.5 * seq_score),
                "sequence_score": seq_score,
                "gru_metrics": m_gru,
                "trans_metrics": m_trans,
                "gru_valid": valid_gru,
                "trans_valid": valid_trans,
                "cad_gru": cad_gru,
                "cad_trans": cad_trans,
                "gt": gt,
            }
        )
        decoded_candidates.append(item)

    top = sorted(decoded_candidates, key=lambda x: x["score"], reverse=True)[: args.top_k]

    metadata = []
    for item in top:
        metadata.append(
            {
                "sample_id": item["sample_id"],
                "score": item["score"],
                "caption": item["caption"],
                "latent_mse_gru": item["latent_mse_gru"],
                "latent_mse_transformer": item["latent_mse_transformer"],
                "latent_cos_gru": item["latent_cos_gru"],
                "latent_cos_transformer": item["latent_cos_transformer"],
                "sequence_score": item["sequence_score"],
                "gru_metrics": item["gru_metrics"],
                "transformer_metrics": item["trans_metrics"],
                "gru_valid": item["gru_valid"],
                "transformer_valid": item["trans_valid"],
            }
        )

    meta_path = args.output_dir / "frontend_contrast_candidates.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    render_rows = top[: args.render_rows]
    if render_rows:
        panel = make_panel(render_rows, "frontend", args.render_size)
        suffix = "_both_valid" if args.require_gru_valid else ""
        panel_path = args.output_dir / f"frontend_contrast_top{suffix}.png"
        panel.save(panel_path)
        print(f"wrote {panel_path}")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
