#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepcad_latent import ImageToCadPipeline
from deepcad_latent.data import build_image_transform, load_embedding_shards, load_raw_cad_vec

from report_material.generate_thesis_qualitative import (
    DATA_ROOT,
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
    parser = argparse.ArgumentParser(
        description="Collect Image-only vs Image+Text examples whose STL export and Blender render all succeed."
    )
    parser.add_argument("--candidates-json", nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render-size", type=int, default=300)
    parser.add_argument("--target-count", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, default=None)
    return parser.parse_args()


def load_candidate_ids(paths: list[str], max_candidates: int) -> list[str]:
    seen: set[str] = set()
    sample_ids: list[str] = []
    for raw_path in paths:
        data = json.loads(Path(raw_path).read_text())
        for item in data:
            sid = item["sample_id"]
            if sid in seen:
                continue
            seen.add(sid)
            sample_ids.append(sid)
            if len(sample_ids) >= max_candidates:
                return sample_ids
    return sample_ids


def load_image_tensor(sample_id: str, img_size: int = 224, n_views: int = 8) -> torch.Tensor:
    group_id, sample_name = sample_id.split("/")[:2]
    image_dir = DATA_ROOT / "cad_img" / group_id / sample_name
    transform = build_image_transform(img_size)
    images = []
    for idx in range(n_views):
        path = image_dir / f"{sample_name}_{idx:03d}.png"
        img = Image.open(path).convert("RGB")
        images.append(transform(img))
    return torch.stack(images, dim=0).unsqueeze(0)


def rendered_pngs_exist(sample_id: str, mode: str = "text_gain") -> bool:
    sample_dir = OUT_ROOT / mode / sample_id.replace("/", "_")
    return all((sample_dir / f"{key}.png").exists() for key in ("image", "text", "gt"))


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.metadata_output or args.output.with_suffix(".json")

    candidate_ids = load_candidate_ids(args.candidates_json, args.max_candidates)
    if not candidate_ids:
        raise RuntimeError("No candidates found.")

    pipe_img = ImageToCadPipeline(TRANSFORMER_CKPT, device=args.device, backbone="resnet18", n_views=8)
    pipe_text = ImageToCadPipeline(TRANSFORMER_TEXT_CKPT, device=args.device, backbone="resnet18", n_views=8)
    text_by_id = load_embedding_shards(TEXT_ROOT, tensor_key="text_emb")

    accepted: list[dict] = []
    with torch.no_grad():
        for idx, sid in enumerate(candidate_ids, start=1):
            if sid not in text_by_id:
                print(f"[{idx}/{len(candidate_ids)}] skip {sid}: no text embedding", flush=True)
                continue

            images = load_image_tensor(sid).to(args.device)
            text_emb = text_by_id[sid].unsqueeze(0).to(args.device)
            cad_img = pipe_img.decode_latent(pipe_img.predict_latent(images))[0]
            cad_text = pipe_text.decode_latent(pipe_text.predict_latent(images, text_emb=text_emb))[0]
            gt = load_raw_cad_vec(DATA_ROOT, sid)
            m_img = sequence_metrics(cad_img, gt)
            m_text = sequence_metrics(cad_text, gt)

            if not (is_valid_cad(cad_img) and is_valid_cad(cad_text) and is_valid_cad(gt)):
                print(f"[{idx}/{len(candidate_ids)}] skip {sid}: invalid CAD", flush=True)
                continue
            if m_text["cmd_token_acc"] <= m_img["cmd_token_acc"] and not (
                m_text["sequence_exact"] and not m_img["sequence_exact"]
            ):
                print(f"[{idx}/{len(candidate_ids)}] skip {sid}: no sequence gain", flush=True)
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
            # Render one row first; make_panel deletes stale PNGs before rendering.
            make_panel([row], "text_gain", args.render_size)
            if not rendered_pngs_exist(sid):
                print(f"[{idx}/{len(candidate_ids)}] skip {sid}: render failed", flush=True)
                continue

            accepted.append(row)
            print(
                f"[{idx}/{len(candidate_ids)}] accept {sid}: "
                f"cmd {m_img['cmd_token_acc']:.3f}->{m_text['cmd_token_acc']:.3f}, "
                f"seq {int(m_img['sequence_exact'])}->{int(m_text['sequence_exact'])}",
                flush=True,
            )
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
    # Avoid gradio/httpx proxy side effects in environments where SOCKS support is unavailable.
    for key in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(key, None)
    main()
