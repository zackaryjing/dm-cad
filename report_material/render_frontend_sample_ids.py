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
from deepcad_latent.data import load_raw_cad_vec
from scripts.web_image_to_cad import build_image_tensor, load_image_paths

from report_material.generate_thesis_qualitative import (
    DATA_ROOT,
    GRU_CKPT,
    TRANSFORMER_CKPT,
    is_valid_cad,
    load_text_caption,
    make_panel,
    sequence_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render GRU-vs-Transformer panel for selected sample IDs.")
    parser.add_argument("--sample-ids", nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render-size", type=int, default=320)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipe_gru = ImageToCadPipeline(GRU_CKPT, device=args.device, backbone="resnet18", n_views=8)
    pipe_trans = ImageToCadPipeline(TRANSFORMER_CKPT, device=args.device, backbone="resnet18", n_views=8)

    rows = []
    with torch.no_grad():
        for sid in args.sample_ids:
            image_paths = load_image_paths(DATA_ROOT, sid, 8)
            images = build_image_tensor(image_paths, 224)
            cad_gru = pipe_gru.decode_latent(pipe_gru.predict_latent(images))[0]
            cad_trans = pipe_trans.decode_latent(pipe_trans.predict_latent(images))[0]
            gt = load_raw_cad_vec(DATA_ROOT, sid)
            rows.append(
                {
                    "sample_id": sid,
                    "caption": load_text_caption(sid),
                    "gru_metrics": sequence_metrics(cad_gru, gt),
                    "trans_metrics": sequence_metrics(cad_trans, gt),
                    "gru_valid": is_valid_cad(cad_gru),
                    "trans_valid": is_valid_cad(cad_trans),
                    "cad_gru": cad_gru,
                    "cad_trans": cad_trans,
                    "gt": gt,
                }
            )
            print(
                sid,
                "gru_valid",
                rows[-1]["gru_valid"],
                "trans_valid",
                rows[-1]["trans_valid"],
                "cmd",
                f"{rows[-1]['gru_metrics']['cmd_token_acc']:.3f}->{rows[-1]['trans_metrics']['cmd_token_acc']:.3f}",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel = make_panel(rows, "frontend", args.render_size)
    panel.save(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
