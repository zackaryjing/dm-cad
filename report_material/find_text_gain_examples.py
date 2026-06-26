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
from deepcad_latent.data import ImageTextLatentDataset, collate_image_text_latent, load_raw_cad_vec

from report_material.generate_thesis_qualitative import (
    DATA_ROOT,
    IDS_FILE,
    OUT_ROOT,
    TEXT_ROOT,
    TRANSFORMER_CKPT,
    TRANSFORMER_TEXT_CKPT,
    is_valid_cad,
    load_text_caption,
    make_panel,
    sequence_metrics,
)

LATENT_ROOT = (
    REPO_ROOT
    / "datasets"
    / "rescue_deepcad_latent"
    / "latents"
    / "overlap_deep_first_len60_trainplusval_fp16"
    / "test"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find stronger image-only vs image+text qualitative examples.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means scan the full test split.")
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--decode-pool-multiplier", type=int, default=8)
    parser.add_argument("--render-rows", type=int, default=8)
    parser.add_argument("--render-size", type=int, default=320)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "report_material" / "figures" / "text_gain_candidates",
    )
    return parser.parse_args()


def latent_scores(z_pred: torch.Tensor, z_gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mse = torch.mean((z_pred - z_gt) ** 2, dim=1)
    cosine = torch.nn.functional.cosine_similarity(z_pred, z_gt, dim=1)
    return mse.cpu(), cosine.cpu()


def sequence_gain_score(m_img: dict, m_text: dict, valid_img: bool, valid_text: bool) -> float:
    score = 0.0
    score += 8.0 * (float(m_text["sequence_exact"]) - float(m_img["sequence_exact"]))
    score += 4.0 * (m_text["cmd_token_acc"] - m_img["cmd_token_acc"])
    score += 3.0 * (m_text["token_exact_acc"] - m_img["token_exact_acc"])
    score += 0.8 * (m_img["len_abs_error"] - m_text["len_abs_error"])
    score += 1.5 * (float(valid_text) - float(valid_img))
    return float(score)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    dataset = ImageTextLatentDataset(
        ids_file=IDS_FILE,
        latent_root=LATENT_ROOT,
        text_root=TEXT_ROOT,
        data_root=DATA_ROOT,
    )
    if args.max_samples > 0:
        dataset = Subset(dataset, list(range(min(args.max_samples, len(dataset)))))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_image_text_latent,
        pin_memory=args.device.startswith("cuda"),
    )

    pipe_img = ImageToCadPipeline(TRANSFORMER_CKPT, device=args.device, backbone="resnet18", n_views=8)
    pipe_text = ImageToCadPipeline(TRANSFORMER_TEXT_CKPT, device=args.device, backbone="resnet18", n_views=8)

    latent_candidates: list[dict] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch["images"].to(args.device, non_blocking=True)
            text_emb = batch["text_emb"].to(args.device, non_blocking=True)
            z_gt = batch["z"].to(args.device, non_blocking=True)
            sample_ids = batch["sample_ids"]

            z_img = pipe_img.predict_latent(images).to(args.device)
            z_text = pipe_text.predict_latent(images, text_emb=text_emb).to(args.device)
            mse_img, cos_img = latent_scores(z_img, z_gt)
            mse_text, cos_text = latent_scores(z_text, z_gt)

            for i, sid in enumerate(sample_ids):
                mse_gain = float(mse_img[i] - mse_text[i])
                cos_gain = float(cos_text[i] - cos_img[i])
                if mse_gain <= 0 and cos_gain <= 0:
                    continue
                score = mse_gain + 0.15 * cos_gain
                latent_candidates.append(
                    {
                        "sample_id": sid,
                        "score": float(score),
                        "latent_mse_image": float(mse_img[i]),
                        "latent_mse_text": float(mse_text[i]),
                        "latent_cos_image": float(cos_img[i]),
                        "latent_cos_text": float(cos_text[i]),
                        "z_image": z_img[i].detach().cpu(),
                        "z_text": z_text[i].detach().cpu(),
                        "caption": load_text_caption(sid),
                    }
                )

            print(
                f"scanned batch {batch_idx + 1}/{len(loader)}, latent_candidates={len(latent_candidates)}",
                flush=True,
            )

    ranked_latent = sorted(latent_candidates, key=lambda x: x["score"], reverse=True)
    decode_pool = ranked_latent[: max(args.top_k * args.decode_pool_multiplier, args.render_rows * 8)]
    if not decode_pool:
        raise RuntimeError("No latent candidates found.")

    z_img_pool = torch.stack([item["z_image"] for item in decode_pool])
    z_text_pool = torch.stack([item["z_text"] for item in decode_pool])
    cad_img_pool = pipe_img.decode_latent(z_img_pool)
    cad_text_pool = pipe_text.decode_latent(z_text_pool)

    decoded_candidates: list[dict] = []
    for item, cad_img, cad_text in zip(decode_pool, cad_img_pool, cad_text_pool):
        gt = load_raw_cad_vec(DATA_ROOT, item["sample_id"])
        m_img = sequence_metrics(cad_img, gt)
        m_text = sequence_metrics(cad_text, gt)
        valid_img = is_valid_cad(cad_img)
        valid_text = is_valid_cad(cad_text)

        # The figure should show that image-only already generates content, then text improves it.
        if not valid_img or not valid_text:
            continue

        seq_gain = sequence_gain_score(m_img, m_text, valid_img, valid_text)
        if seq_gain <= 0:
            continue

        cmd_gain = m_text["cmd_token_acc"] - m_img["cmd_token_acc"]
        token_gain = m_text["token_exact_acc"] - m_img["token_exact_acc"]
        len_gain = m_img["len_abs_error"] - m_text["len_abs_error"]
        if cmd_gain < 0.08 and token_gain < 0.08 and len_gain <= 0 and not (
            m_text["sequence_exact"] and not m_img["sequence_exact"]
        ):
            continue

        item = dict(item)
        item.update(
            {
                "score": float(item["score"] + 0.6 * seq_gain),
                "sequence_gain_score": seq_gain,
                "image_metrics": m_img,
                "trans_metrics": m_img,
                "text_metrics": m_text,
                "image_valid": valid_img,
                "text_valid": valid_text,
                "cad_trans": cad_img,
                "cad_text": cad_text,
                "gt": gt,
            }
        )
        decoded_candidates.append(item)

    top = sorted(decoded_candidates, key=lambda x: x["score"], reverse=True)[: args.top_k]
    if not top:
        raise RuntimeError("No decoded candidates survived filtering. Try increasing --top-k or relaxing thresholds.")

    metadata = []
    for item in top:
        metadata.append(
            {
                "sample_id": item["sample_id"],
                "score": item["score"],
                "sequence_gain_score": item["sequence_gain_score"],
                "caption": item["caption"],
                "latent_mse_image": item["latent_mse_image"],
                "latent_mse_text": item["latent_mse_text"],
                "latent_cos_image": item["latent_cos_image"],
                "latent_cos_text": item["latent_cos_text"],
                "image_metrics": item["image_metrics"],
                "text_metrics": item["text_metrics"],
                "image_valid": item["image_valid"],
                "text_valid": item["text_valid"],
            }
        )

    meta_path = args.output_dir / "text_gain_candidates.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    render_rows = top[: args.render_rows]
    panel = make_panel(render_rows, "text_gain", args.render_size)
    panel_path = args.output_dir / "text_gain_top.png"
    panel.save(panel_path)

    print(f"wrote {panel_path}")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
