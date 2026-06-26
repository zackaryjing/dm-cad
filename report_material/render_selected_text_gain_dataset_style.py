#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEEP_CAD_ROOT = Path("/root/projects/CAD-MLLM/3rd_party/DeepCAD")
if str(DEEP_CAD_ROOT) not in sys.path:
    sys.path.insert(0, str(DEEP_CAD_ROOT))

from OCC.Extend.DataExchange import write_ply_file
from cadlib.visualize import vec2CADsolid

from deepcad_latent import ImageToCadPipeline
from deepcad_latent.data import build_image_transform, load_embedding_shards, load_raw_cad_vec

from report_material.generate_thesis_qualitative import (
    DATA_ROOT,
    TEXT_ROOT,
    TRANSFORMER_CKPT,
    TRANSFORMER_TEXT_CKPT,
    FONT_PATH,
    load_text_caption,
    sequence_metrics,
    stitch_multiview,
)

BLENDER_SCRIPT = REPO_ROOT / "report_material" / "render_dataset_style_single_blender.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render selected text-gain examples with original dataset style.")
    parser.add_argument("--sample-ids", nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render-size", type=int, default=320)
    parser.add_argument("--view-index", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_image_tensor(sample_id: str, img_size: int = 224, n_views: int = 8) -> torch.Tensor:
    group_id, sample_name = sample_id.split("/")[:2]
    image_dir = DATA_ROOT / "cad_img" / group_id / sample_name
    transform = build_image_transform(img_size)
    images = []
    for idx in range(n_views):
        path = image_dir / f"{sample_name}_{idx:03d}.png"
        images.append(transform(Image.open(path).convert("RGB")))
    return torch.stack(images, dim=0).unsqueeze(0)


def export_ply(cad_vec: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    shape = vec2CADsolid(cad_vec.astype(np.float64))
    write_ply_file(shape, str(path))
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"PLY export failed: {path}")


def render_ply(ply_path: Path, png_path: Path, size: int, view_index: int) -> None:
    png_path.unlink(missing_ok=True)
    blender_cmd = [
        "blender",
        "-b",
        "-P",
        str(BLENDER_SCRIPT),
        "--",
        "--input",
        str(ply_path),
        "--output",
        str(png_path),
        "--size",
        str(size),
        "--view-index",
        str(view_index),
    ]
    xvfb = shutil.which("xvfb-run")
    cmd = ["xvfb-run", "-a", *blender_cmd] if xvfb else blender_cmd
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0 or not png_path.exists():
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        tail = detail[-1] if detail else "no blender output"
        raise RuntimeError(f"Blender render failed for {ply_path}: {tail}")


def make_panel(rows: list[dict], render_size: int, output: Path) -> None:
    font_title = ImageFont.truetype(str(FONT_PATH), 24) if FONT_PATH.exists() else ImageFont.load_default()
    font_label = ImageFont.truetype(str(FONT_PATH), 20) if FONT_PATH.exists() else ImageFont.load_default()
    font_text = ImageFont.truetype(str(FONT_PATH), 18) if FONT_PATH.exists() else ImageFont.load_default()
    cell_gap = 18
    left_w = 520
    right_w = render_size
    row_h = max(300, render_size + 110)
    total_w = left_w + 3 * right_w + 5 * cell_gap
    total_h = 60 + len(rows) * row_h + 40
    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)
    x_positions = [
        cell_gap,
        left_w + 2 * cell_gap,
        left_w + right_w + 3 * cell_gap,
        left_w + 2 * right_w + 4 * cell_gap,
    ]
    for x, label in zip(x_positions, ["Input Views", "Image-only", "Image+Text", "GT"]):
        draw.text((x, 16), label, fill="black", font=font_title)

    for row_idx, item in enumerate(rows):
        top = 60 + row_idx * row_h
        stitched = stitch_multiview(item["sample_id"]).resize((left_w, int(left_w * 0.5)))
        canvas.paste(stitched, (cell_gap, top + 24))
        for col_idx, key in enumerate(["image", "text", "gt"]):
            png = Image.open(item["pngs"][key]).convert("RGB")
            canvas.paste(png, (x_positions[col_idx + 1], top + 24))
        caption = textwrap.fill(item["caption"].strip() or "(no caption)", width=40)
        draw.text((cell_gap, top + 24 + stitched.height + 10), f"{item['sample_id']}\n{caption}", fill="black", font=font_text)
        metric_line = (
            f"Image-only cmd={item['image_metrics']['cmd_token_acc']:.3f}, seq={int(item['image_metrics']['sequence_exact'])}; "
            f"Image+Text cmd={item['text_metrics']['cmd_token_acc']:.3f}, seq={int(item['text_metrics']['sequence_exact'])}"
        )
        draw.text((left_w + 2 * cell_gap, top + render_size + 34), metric_line, fill="black", font=font_label)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    args = parse_args()
    for key in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(key, None)
    pipe_img = ImageToCadPipeline(TRANSFORMER_CKPT, device=args.device, backbone="resnet18", n_views=8)
    pipe_text = ImageToCadPipeline(TRANSFORMER_TEXT_CKPT, device=args.device, backbone="resnet18", n_views=8)
    text_by_id = load_embedding_shards(TEXT_ROOT, tensor_key="text_emb")

    render_root = REPO_ROOT / "report_material" / "figures" / "qualitative_dataset_style" / "text_gain"
    rows = []
    with torch.no_grad():
        for sid in args.sample_ids:
            images = load_image_tensor(sid).to(args.device)
            text_emb = text_by_id[sid].unsqueeze(0).to(args.device)
            cad_img = pipe_img.decode_latent(pipe_img.predict_latent(images))[0]
            cad_text = pipe_text.decode_latent(pipe_text.predict_latent(images, text_emb=text_emb))[0]
            gt = load_raw_cad_vec(DATA_ROOT, sid)
            sample_dir = render_root / sid.replace("/", "_")
            pngs = {}
            for key, cad_vec in {"image": cad_img, "text": cad_text, "gt": gt}.items():
                ply_path = sample_dir / f"{key}.ply"
                png_path = sample_dir / f"{key}.png"
                export_ply(cad_vec, ply_path)
                render_ply(ply_path, png_path, args.render_size, args.view_index)
                pngs[key] = png_path
            row = {
                "sample_id": sid,
                "caption": load_text_caption(sid),
                "image_metrics": sequence_metrics(cad_img, gt),
                "text_metrics": sequence_metrics(cad_text, gt),
                "pngs": pngs,
            }
            print(
                sid,
                f"cmd {row['image_metrics']['cmd_token_acc']:.3f}->{row['text_metrics']['cmd_token_acc']:.3f}",
                f"seq {int(row['image_metrics']['sequence_exact'])}->{int(row['text_metrics']['sequence_exact'])}",
                flush=True,
            )
            rows.append(row)
    make_panel(rows, args.render_size, args.output)
    meta = [
        {
            "sample_id": row["sample_id"],
            "caption": row["caption"],
            "image_metrics": row["image_metrics"],
            "text_metrics": row["text_metrics"],
        }
        for row in rows
    ]
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
