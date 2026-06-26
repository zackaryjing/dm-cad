#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepcad_latent import ImageToCadPipeline
from deepcad_latent.data import load_embedding_shards, load_raw_cad_vec
from scripts.web_image_to_cad import build_image_tensor, load_image_paths

from report_material.generate_thesis_qualitative import (
    DATA_ROOT,
    TEXT_ROOT,
    TRANSFORMER_CKPT,
    TRANSFORMER_TEXT_CKPT,
    is_valid_cad,
    load_text_caption,
    make_panel,
    sequence_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Image-only vs Image+Text panel for selected sample IDs.")
    parser.add_argument("--sample-ids", nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render-size", type=int, default=320)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipe_img = ImageToCadPipeline(TRANSFORMER_CKPT, device=args.device, backbone="resnet18", n_views=8)
    pipe_text = ImageToCadPipeline(TRANSFORMER_TEXT_CKPT, device=args.device, backbone="resnet18", n_views=8)
    text_by_id = load_embedding_shards(TEXT_ROOT, tensor_key="text_emb")

    rows = []
    with torch.no_grad():
        for sid in args.sample_ids:
            image_paths = load_image_paths(DATA_ROOT, sid, 8)
            images = build_image_tensor(image_paths, 224)
            text_emb = text_by_id[sid].unsqueeze(0)
            cad_img = pipe_img.decode_latent(pipe_img.predict_latent(images))[0]
            cad_text = pipe_text.decode_latent(pipe_text.predict_latent(images, text_emb=text_emb))[0]
            gt = load_raw_cad_vec(DATA_ROOT, sid)
            rows.append(
                {
                    "sample_id": sid,
                    "caption": load_text_caption(sid),
                    "trans_metrics": sequence_metrics(cad_img, gt),
                    "text_metrics": sequence_metrics(cad_text, gt),
                    "image_valid": is_valid_cad(cad_img),
                    "text_valid": is_valid_cad(cad_text),
                    "cad_trans": cad_img,
                    "cad_text": cad_text,
                    "gt": gt,
                }
            )
            print(
                sid,
                "image_valid",
                rows[-1]["image_valid"],
                "text_valid",
                rows[-1]["text_valid"],
                "cmd",
                f"{rows[-1]['trans_metrics']['cmd_token_acc']:.3f}->{rows[-1]['text_metrics']['cmd_token_acc']:.3f}",
                "seq",
                f"{int(rows[-1]['trans_metrics']['sequence_exact'])}->{int(rows[-1]['text_metrics']['sequence_exact'])}",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel = make_panel(rows, "text_gain", args.render_size)
    panel.save(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
