#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepcad_latent import ImageToCadPipeline
from deepcad_latent.data import ImageTextOnlyDataset, collate_image_text_only, load_raw_cad_vec

DEEP_CAD_ROOT = Path("/root/projects/CAD-MLLM/3rd_party/DeepCAD")
if str(DEEP_CAD_ROOT) not in sys.path:
    sys.path.insert(0, str(DEEP_CAD_ROOT))

from OCC.Extend.DataExchange import write_stl_file
from cadlib.visualize import vec2CADsolid

DATA_ROOT = REPO_ROOT / "datasets" / "dataset_v0"
TEXT_ROOT = REPO_ROOT / "datasets" / "rescue_deepcad_latent" / "text_emb" / "full_v0_len60_excluding_rescue_test_bert_base_uncased" / "test"
IDS_FILE = REPO_ROOT / "datasets" / "rescue_deepcad_latent" / "full_v0_len60_excluding_rescue_test" / "test_ids.txt"

RUNS_ROOT = REPO_ROOT / "runs" / "deepcad_latent"
GRU_CKPT = RUNS_ROOT / "resnet18_gru_fullv0len60_v1_ddp2" / "best.pt"
TRANSFORMER_CKPT = RUNS_ROOT / "resnet18_transformer_v1_ddp2" / "best.pt"
TRANSFORMER_TEXT_CKPT = RUNS_ROOT / "resnet18_transformer_text_v1_ddp2" / "best.pt"

RETRIEVAL_ROOT = REPO_ROOT / "datasets" / "rescue_deepcad_latent" / "latents" / "full_v0_len60_excluding_rescue_test_fp16" / "train"
OUT_ROOT = REPO_ROOT / "report_material" / "figures" / "qualitative"
BLENDER_SCRIPT = REPO_ROOT / "report_material" / "render_stl_preview_blender.py"
THESIS_FIG_ROOT = Path("/root/projects/web_projects/latex_workspace/papers/xjtu_bachelor_2024/figures/generated")
FONT_PATH = Path("/root/projects/web_projects/latex_workspace/papers/xjtu_bachelor_2024/SimSun.ttf")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate thesis qualitative comparison figures.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-search-samples", type=int, default=768)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--render-size", type=int, default=360)
    return parser.parse_args()


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


def sequence_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | bool | int]:
    pred = trim_eos_exclusive(pred)
    gt = trim_eos_exclusive(gt)
    pred_pad, gt_pad = pad_for_compare(pred, gt)
    cmd_equal = pred_pad[:, 0] == gt_pad[:, 0]
    token_equal = np.all(pred_pad == gt_pad, axis=1)
    return {
        "cmd_token_acc": float(cmd_equal.mean()) if len(cmd_equal) else 1.0,
        "token_exact_acc": float(token_equal.mean()) if len(token_equal) else 1.0,
        "sequence_exact": bool(token_equal.all()),
        "pred_len": int(len(pred)),
        "gt_len": int(len(gt)),
        "len_abs_error": int(abs(len(pred) - len(gt))),
    }


def is_valid_cad(cad_vec: np.ndarray) -> bool:
    try:
        vec2CADsolid(cad_vec.astype(np.float64))
        return True
    except Exception:
        return False


def load_text_caption(sample_id: str) -> str:
    group_id = sample_id.split("/")[0]
    desc_path = DATA_ROOT / "cad_desc" / f"{group_id}.json"
    items = json.loads(desc_path.read_text())
    for item in items:
        if item.get("id") == sample_id:
            return item.get("text caption", "")
    return ""


def export_stl(cad_vec: np.ndarray, path: Path) -> tuple[bool, str]:
    try:
        shape = vec2CADsolid(cad_vec.astype(np.float64))
        path.parent.mkdir(parents=True, exist_ok=True)
        write_stl_file(shape, str(path))
        if not path.exists() or path.stat().st_size == 0:
            return False, "export_empty_stl"
        return True, "ok"
    except Exception as exc:
        return False, f"export_failed:{type(exc).__name__}:{exc}"


def render_preview(mesh_path: Path, output_path: Path, size: int) -> tuple[bool, str]:
    blender_cmd = [
        "blender",
        "-b",
        "-P",
        str(BLENDER_SCRIPT),
        "--",
        "--input",
        str(mesh_path),
        "--output",
        str(output_path),
        "--size",
        str(size),
    ]
    xvfb = shutil.which("xvfb-run")
    cmd = ["xvfb-run", "-a", *blender_cmd] if xvfb else blender_cmd
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        tail = detail[-1] if detail else "no blender output"
        ok, fallback_status = render_preview_software(mesh_path, output_path, size)
        if ok:
            return True, "ok"
        return False, f"blender_failed:{proc.returncode}:{tail}; fallback:{fallback_status}"
    if not output_path.exists() or output_path.stat().st_size == 0:
        ok, fallback_status = render_preview_software(mesh_path, output_path, size)
        if ok:
            return True, "ok"
        return False, f"blender_no_output; fallback:{fallback_status}"
    return True, "ok"


def render_preview_software(mesh_path: Path, output_path: Path, size: int) -> tuple[bool, str]:
    """Small software fallback for headless environments without GLX/EGL."""
    try:
        import trimesh

        mesh = trimesh.load_mesh(str(mesh_path), force="mesh")
        if mesh.is_empty or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return False, "software_empty_mesh"

        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
        vertices = vertices - center
        scale = np.max(np.ptp(vertices, axis=0))
        if not np.isfinite(scale) or scale <= 1e-9:
            return False, "software_bad_scale"
        vertices = vertices / scale

        yaw = math.radians(-38)
        pitch = math.radians(26)
        rz = np.array(
            [
                [math.cos(yaw), -math.sin(yaw), 0.0],
                [math.sin(yaw), math.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rx = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, math.cos(pitch), -math.sin(pitch)],
                [0.0, math.sin(pitch), math.cos(pitch)],
            ]
        )
        projected = vertices @ rz.T @ rx.T
        xy = projected[:, :2]
        xy[:, 1] *= -1.0
        xy = xy * (size * 0.62) + size * 0.5

        face_vertices = projected[faces]
        normals = np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0])
        norm = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(normals, np.maximum(norm, 1e-9))
        light = np.array([0.25, -0.35, 0.9])
        light = light / np.linalg.norm(light)
        intensity = np.clip(normals @ light, 0.0, 1.0)
        depth = face_vertices[:, :, 2].mean(axis=1)

        img = Image.new("RGB", (size, size), (232, 232, 232))
        draw = ImageDraw.Draw(img)
        for face_idx in np.argsort(depth):
            poly = [tuple(pt) for pt in xy[faces[face_idx]]]
            gray = int(178 + 58 * float(intensity[face_idx]))
            draw.polygon(poly, fill=(gray, gray, gray))

        # Draw only real mesh boundary / crease edges, not every triangulation edge.
        edge_to_faces: dict[tuple[int, int], list[int]] = {}
        for face_idx, face in enumerate(faces):
            for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                edge = (int(min(a, b)), int(max(a, b)))
                edge_to_faces.setdefault(edge, []).append(face_idx)

        crease_cos = math.cos(math.radians(24))
        for edge, adjacent in edge_to_faces.items():
            should_draw = len(adjacent) == 1
            if len(adjacent) == 2:
                n0, n1 = normals[adjacent[0]], normals[adjacent[1]]
                should_draw = float(np.dot(n0, n1)) < crease_cos
            if not should_draw:
                continue
            p0, p1 = xy[list(edge)]
            draw.line((tuple(p0), tuple(p1)), fill=(36, 36, 36), width=max(1, size // 180))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        return True, "ok"
    except Exception as exc:
        return False, f"software_failed:{type(exc).__name__}:{exc}"


def stitch_multiview(sample_id: str) -> Image.Image:
    group_id, sample_name = sample_id.split("/")[:2]
    img_dir = DATA_ROOT / "cad_img" / group_id / sample_name
    images = [Image.open(img_dir / f"{sample_name}_{i:03d}.png").convert("RGB") for i in range(8)]
    images = [img.crop((16, 16, img.width - 16, img.height - 16)) for img in images]
    w, h = images[0].size
    canvas = Image.new("RGB", (4 * w, 2 * h), "white")
    for idx, img in enumerate(images):
        r, c = divmod(idx, 4)
        canvas.paste(img, (c * w, r * h))
    return canvas


def choose_samples(args) -> tuple[list[dict], list[dict]]:
    dataset = ImageTextOnlyDataset(
        ids_file=IDS_FILE,
        text_root=TEXT_ROOT,
        data_root=DATA_ROOT,
    )
    if args.max_search_samples > 0:
        dataset = Subset(dataset, list(range(min(args.max_search_samples, len(dataset)))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_image_text_only,
        pin_memory=args.device.startswith("cuda"),
    )

    pipe_gru = ImageToCadPipeline(GRU_CKPT, device=args.device, backbone="resnet18", n_views=8)
    pipe_trans = ImageToCadPipeline(TRANSFORMER_CKPT, device=args.device, backbone="resnet18", n_views=8)
    pipe_text = ImageToCadPipeline(
        TRANSFORMER_TEXT_CKPT,
        device=args.device,
        backbone="resnet18",
        n_views=8,
        retrieval_latent_root=RETRIEVAL_ROOT,
    )

    text_candidates = []
    frontend_candidates = []

    for batch in loader:
        images = batch["images"].to(args.device, non_blocking=True)
        text_emb = batch["text_emb"].to(args.device, non_blocking=True)
        sample_ids = batch["sample_ids"]

        pred_gru = pipe_gru.decode_latent(pipe_gru.predict_latent(images))
        pred_trans = pipe_trans.decode_latent(pipe_trans.predict_latent(images))
        pred_text = pipe_text.decode_latent(pipe_text.predict_latent(images, text_emb=text_emb))

        for sid, cad_gru, cad_trans, cad_text in zip(sample_ids, pred_gru, pred_trans, pred_text):
            gt = load_raw_cad_vec(DATA_ROOT, sid)
            m_gru = sequence_metrics(cad_gru, gt)
            m_trans = sequence_metrics(cad_trans, gt)
            m_text = sequence_metrics(cad_text, gt)

            score_text = (
                5.0 * (float(m_text["sequence_exact"]) - float(m_trans["sequence_exact"]))
                + 2.5 * (m_text["cmd_token_acc"] - m_trans["cmd_token_acc"])
                + 1.5 * (m_text["token_exact_acc"] - m_trans["token_exact_acc"])
                + 0.3 * (m_trans["len_abs_error"] - m_text["len_abs_error"])
            )
            score_frontend = (
                5.0 * (float(m_trans["sequence_exact"]) - float(m_gru["sequence_exact"]))
                + 2.5 * (m_trans["cmd_token_acc"] - m_gru["cmd_token_acc"])
                + 1.5 * (m_trans["token_exact_acc"] - m_gru["token_exact_acc"])
                + 0.3 * (m_gru["len_abs_error"] - m_trans["len_abs_error"])
            )

            text_candidates.append(
                {
                    "sample_id": sid,
                    "score": score_text,
                    "caption": load_text_caption(sid),
                    "gru_metrics": m_gru,
                    "trans_metrics": m_trans,
                    "text_metrics": m_text,
                    "cad_gru": cad_gru,
                    "cad_trans": cad_trans,
                    "cad_text": cad_text,
                    "gt": gt,
                }
            )
            frontend_candidates.append(
                {
                    "sample_id": sid,
                    "score": score_frontend,
                    "caption": load_text_caption(sid),
                    "gru_metrics": m_gru,
                    "trans_metrics": m_trans,
                    "text_metrics": m_text,
                    "cad_gru": cad_gru,
                    "cad_trans": cad_trans,
                    "cad_text": cad_text,
                    "gt": gt,
                }
            )

    def shortlist(candidates: list[dict], key: str) -> list[dict]:
        ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
        picked = []
        seen = set()
        for item in ranked:
            if item["score"] <= 0:
                continue
            sid = item["sample_id"]
            if sid in seen:
                continue
            if key == "text":
                better = item["text_metrics"]["cmd_token_acc"] > item["trans_metrics"]["cmd_token_acc"] or (
                    item["text_metrics"]["sequence_exact"] and not item["trans_metrics"]["sequence_exact"]
                )
                valid = is_valid_cad(item["cad_text"]) and is_valid_cad(item["cad_trans"])
            else:
                better = item["trans_metrics"]["cmd_token_acc"] > item["gru_metrics"]["cmd_token_acc"] or (
                    item["trans_metrics"]["sequence_exact"] and not item["gru_metrics"]["sequence_exact"]
                )
                valid = is_valid_cad(item["cad_trans"]) and is_valid_cad(item["cad_gru"])
            if better and valid:
                picked.append(item)
                seen.add(sid)
            if len(picked) >= args.num_samples:
                break
        return picked

    return shortlist(text_candidates, "text"), shortlist(frontend_candidates, "frontend")


def make_panel(rows: list[dict], mode: str, render_size: int) -> Image.Image:
    font_title = ImageFont.truetype(str(FONT_PATH), 24) if FONT_PATH.exists() else ImageFont.load_default()
    font_label = ImageFont.truetype(str(FONT_PATH), 20) if FONT_PATH.exists() else ImageFont.load_default()
    font_text = ImageFont.truetype(str(FONT_PATH), 18) if FONT_PATH.exists() else ImageFont.load_default()

    cell_gap = 18
    left_w = 520
    right_w = render_size
    row_h = max(300, render_size + 110)
    cols = ["Input Views", "GRU" if mode == "frontend" else "Image-only", "Transformer" if mode == "frontend" else "Image+Text", "GT"]
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
    for x, label in zip(x_positions, cols):
        draw.text((x, 16), label, fill="black", font=font_title)

    for row_idx, item in enumerate(rows):
        top = 60 + row_idx * row_h
        sid = item["sample_id"]
        stitched = stitch_multiview(sid).resize((left_w, int(left_w * 0.5)))
        canvas.paste(stitched, (cell_gap, top + 24))

        sample_dir = OUT_ROOT / mode / sid.replace("/", "_")
        sample_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        if mode == "frontend":
            variants = {"gru": item["cad_gru"], "transformer": item["cad_trans"], "gt": item["gt"]}
        else:
            variants = {"image": item["cad_trans"], "text": item["cad_text"], "gt": item["gt"]}

        order = list(variants.keys())
        render_statuses = {}
        for key in order:
            stl_path = sample_dir / f"{key}.stl"
            png_path = sample_dir / f"{key}.png"
            stl_path.unlink(missing_ok=True)
            png_path.unlink(missing_ok=True)
            ok, status = export_stl(variants[key], stl_path)
            if ok:
                ok, status = render_preview(stl_path, png_path, render_size)
            render_statuses[key] = status
            if status != "ok":
                print(f"{sid} {key} render status: {status}", file=sys.stderr, flush=True)
            paths[key] = png_path

        keys_for_cols = ["gru", "transformer", "gt"] if mode == "frontend" else ["image", "text", "gt"]
        for col_idx, key in enumerate(keys_for_cols):
            if paths[key].exists():
                png = Image.open(paths[key]).convert("RGB")
            else:
                png = Image.new("RGB", (render_size, render_size), "white")
                fail_draw = ImageDraw.Draw(png)
                fail_draw.rectangle((1, 1, render_size - 2, render_size - 2), outline="black", width=2)
                status = render_statuses.get(key, "render_failed")
                fail_draw.text((18, render_size // 2 - 22), "Render failed", fill="black", font=font_label)
                fail_draw.text((18, render_size // 2 + 8), status[:28], fill="black", font=font_text)
            canvas.paste(png, (x_positions[col_idx + 1], top + 24))

        caption = item["caption"].strip() or "(no caption)"
        wrapped = textwrap.fill(caption, width=40)
        draw.text((cell_gap, top + 24 + stitched.height + 10), f"{sid}\n{wrapped}", fill="black", font=font_text)

        if mode == "frontend":
            metric_line = (
                f"GRU cmd={item['gru_metrics']['cmd_token_acc']:.3f}, seq={int(item['gru_metrics']['sequence_exact'])}; "
                f"Transformer cmd={item['trans_metrics']['cmd_token_acc']:.3f}, seq={int(item['trans_metrics']['sequence_exact'])}"
            )
        else:
            metric_line = (
                f"Image-only cmd={item['trans_metrics']['cmd_token_acc']:.3f}, seq={int(item['trans_metrics']['sequence_exact'])}; "
                f"Image+Text cmd={item['text_metrics']['cmd_token_acc']:.3f}, seq={int(item['text_metrics']['sequence_exact'])}"
            )
        draw.text((left_w + 2 * cell_gap, top + render_size + 34), metric_line, fill="black", font=font_label)

    return canvas


def main():
    args = parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    THESIS_FIG_ROOT.mkdir(parents=True, exist_ok=True)

    text_rows, frontend_rows = choose_samples(args)
    if not text_rows or not frontend_rows:
        raise RuntimeError("Failed to find enough qualitative examples. Try increasing --max-search-samples.")

    fig_text = make_panel(text_rows, "text_gain", args.render_size)
    fig_frontend = make_panel(frontend_rows, "frontend", args.render_size)

    out_text = OUT_ROOT / "thesis_text_gain_examples.png"
    out_front = OUT_ROOT / "thesis_frontend_examples.png"
    fig_text.save(out_text)
    fig_frontend.save(out_front)

    (THESIS_FIG_ROOT / out_text.name).write_bytes(out_text.read_bytes())
    (THESIS_FIG_ROOT / out_front.name).write_bytes(out_front.read_bytes())

    meta = {
        "text_gain_samples": [item["sample_id"] for item in text_rows],
        "frontend_samples": [item["sample_id"] for item in frontend_rows],
    }
    (OUT_ROOT / "thesis_qualitative_samples.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
