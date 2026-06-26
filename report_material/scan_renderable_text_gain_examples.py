#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepcad_latent import ImageToCadPipeline
from deepcad_latent.data import ImageTextOnlyDataset, collate_image_text_only, load_ids, load_raw_cad_vec

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan dispersed test IDs for renderable text-gain examples.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--render-size", type=int, default=300)
    parser.add_argument("--target-count", type=int, default=10)
    parser.add_argument("--max-scan", type=int, default=1024)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--stride", type=int, default=7)
    parser.add_argument("--exclude-json", nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, default=None)
    return parser.parse_args()


def load_excluded(paths: list[str]) -> set[str]:
    excluded: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for item in data:
            if isinstance(item, dict) and "sample_id" in item:
                excluded.add(item["sample_id"])
    return excluded


def rendered_pngs_exist(sample_id: str, mode: str = "text_gain") -> bool:
    sample_dir = OUT_ROOT / mode / sample_id.replace("/", "_")
    return all((sample_dir / f"{key}.png").exists() for key in ("image", "text", "gt"))


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.metadata_output or args.output.with_suffix(".json")

    all_ids = load_ids(IDS_FILE)
    excluded = load_excluded(args.exclude_json)
    sampled_ids = [sid for sid in all_ids[args.offset :: args.stride] if sid not in excluded][: args.max_scan]
    if not sampled_ids:
        raise RuntimeError("No sampled IDs after exclusion.")

    dataset = ImageTextOnlyDataset(ids_file=IDS_FILE, text_root=TEXT_ROOT, data_root=DATA_ROOT)
    index_by_id = {sid: idx for idx, sid in enumerate(dataset.sample_ids)}
    indices = [index_by_id[sid] for sid in sampled_ids if sid in index_by_id]
    dataset = Subset(dataset, indices)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_image_text_only,
        pin_memory=args.device.startswith("cuda"),
    )

    pipe_img = ImageToCadPipeline(TRANSFORMER_CKPT, device=args.device, backbone="resnet18", n_views=8)
    pipe_text = ImageToCadPipeline(TRANSFORMER_TEXT_CKPT, device=args.device, backbone="resnet18", n_views=8)

    accepted: list[dict] = []
    scanned = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            images = batch["images"].to(args.device, non_blocking=True)
            text_emb = batch["text_emb"].to(args.device, non_blocking=True)
            sample_ids = batch["sample_ids"]
            pred_img = pipe_img.decode_latent(pipe_img.predict_latent(images))
            pred_text = pipe_text.decode_latent(pipe_text.predict_latent(images, text_emb=text_emb))

            for sid, cad_img, cad_text in zip(sample_ids, pred_img, pred_text):
                scanned += 1
                gt = load_raw_cad_vec(DATA_ROOT, sid)
                m_img = sequence_metrics(cad_img, gt)
                m_text = sequence_metrics(cad_text, gt)
                cmd_gain = m_text["cmd_token_acc"] - m_img["cmd_token_acc"]
                token_gain = m_text["token_exact_acc"] - m_img["token_exact_acc"]
                len_gain = m_img["len_abs_error"] - m_text["len_abs_error"]
                if cmd_gain < 0.12 and token_gain < 0.08 and len_gain <= 2 and not (
                    m_text["sequence_exact"] and not m_img["sequence_exact"]
                ):
                    continue
                if not (is_valid_cad(cad_img) and is_valid_cad(cad_text) and is_valid_cad(gt)):
                    continue

                row = {
                    "sample_id": sid,
                    "caption": load_text_caption(sid),
                    "trans_metrics": m_img,
                    "text_metrics": m_text,
                    "image_valid": True,
                    "text_valid": True,
                    "cad_trans": cad_img,
                    "cad_text": cad_text,
                    "gt": gt,
                }
                make_panel([row], "text_gain", args.render_size)
                if not rendered_pngs_exist(sid):
                    continue

                accepted.append(row)
                print(
                    f"accept {len(accepted)}/{args.target_count} {sid}: "
                    f"cmd {m_img['cmd_token_acc']:.3f}->{m_text['cmd_token_acc']:.3f}, "
                    f"len {m_img['len_abs_error']}->{m_text['len_abs_error']}",
                    flush=True,
                )
                if len(accepted) >= args.target_count:
                    break
            print(f"batch {batch_idx}/{len(loader)}, scanned={scanned}, accepted={len(accepted)}", flush=True)
            if len(accepted) >= args.target_count:
                break

    if not accepted:
        raise RuntimeError("No renderable examples collected.")

    panel = make_panel(accepted, "text_gain", args.render_size)
    panel.save(args.output)
    metadata = [
        {
            "sample_id": item["sample_id"],
            "caption": item["caption"],
            "image_metrics": item["trans_metrics"],
            "text_metrics": item["text_metrics"],
        }
        for item in accepted
    ]
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"wrote {args.output}")
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    for key in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(key, None)
    main()
