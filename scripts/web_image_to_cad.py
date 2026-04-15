#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import gradio as gr
import h5py
import numpy as np
import torch
from OCC.Extend.DataExchange import write_stl_file
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepcad_latent import DeepCADAdapter
from deepcad_latent.model import MultiViewLatentRegressor

DEEP_CAD_ROOT = Path("/root/projects/CAD-MLLM/3rd_party/DeepCAD")
if str(DEEP_CAD_ROOT) not in sys.path:
    sys.path.insert(0, str(DEEP_CAD_ROOT))

from cadlib.visualize import vec2CADsolid


def parse_args():
    parser = argparse.ArgumentParser(description="Web app for image-to-CAD qualitative inspection")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/deepcad_latent/resnet18_gru_v1_ddp/best.pt"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("datasets/dataset_v0"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--n-views", type=int, default=8)
    parser.add_argument("--server-port", type=int, default=7864)
    parser.add_argument("--work-dir", type=Path, default=Path("runs/deepcad_latent/web_cache"))
    return parser.parse_args()


def load_cad_vec(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        key = next(iter(f.keys()))
        cad_vec = f[key][:]
    if cad_vec.ndim == 1:
        cad_vec = cad_vec.reshape(-1, 17)
    return cad_vec.astype(np.int64, copy=False)


def load_image_paths(data_root: Path, sample_id: str, n_views: int) -> list[Path]:
    group_id, sample_name = sample_id.split("/")[:2]
    img_dir = data_root / "cad_img" / group_id / sample_name
    return [img_dir / f"{sample_name}_{i:03d}.png" for i in range(n_views)]


def build_image_tensor(image_paths: list[Path], img_size: int) -> torch.Tensor:
    image_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    images = []
    for path in image_paths:
        if path.exists():
            img = Image.open(path).convert("RGB")
            img = image_transform(img)
        else:
            img = torch.ones(3, img_size, img_size)
        images.append(img)
    return torch.stack(images).unsqueeze(0)


def export_cad_to_stl(cad_vec: np.ndarray, stl_path: Path) -> tuple[bool, str]:
    try:
        shape = vec2CADsolid(cad_vec.astype(np.float64))
        stl_path.parent.mkdir(parents=True, exist_ok=True)
        write_stl_file(shape, str(stl_path))
        return True, ""
    except Exception as e:
        return False, str(e)


def format_sequence(cad_vec: np.ndarray, max_steps: int = 20) -> str:
    cmd_names = {0: "Line", 1: "Arc", 2: "Circle", 3: "EOS", 4: "SOL", 5: "Ext"}
    lines = []
    for step, token in enumerate(cad_vec[:max_steps]):
        cmd = int(token[0])
        name = cmd_names.get(cmd, f"UNK_{cmd}")
        params = " ".join(str(int(x)) for x in token[1:6])
        lines.append(f"{step:02d}: {name:<6} [{params}]")
        if cmd == 3:
            break
    return "\n".join(lines)


class ImageToCadInspector:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.model = MultiViewLatentRegressor(
            backbone_name=args.backbone,
            n_views=args.n_views,
            freeze_backbone=args.freeze_backbone,
        ).to(self.device)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.adapter = DeepCADAdapter(device=args.device)
        self.work_dir = args.work_dir

    def infer_sample(self, sample_id: str):
        sample_id = sample_id.strip()
        if not sample_id:
            raise gr.Error("sample_id 不能为空。")

        try:
            group_id, sample_name = sample_id.split("/")[:2]
        except ValueError as exc:
            raise gr.Error("sample_id 格式应为 0000/00000093_00001") from exc

        image_paths = load_image_paths(self.args.data_root, sample_id, self.args.n_views)
        missing_images = [str(path) for path in image_paths if not path.exists()]
        if missing_images:
            raise gr.Error(f"缺少图像文件，例如: {missing_images[0]}")

        gt_h5_path = self.args.data_root / "cad_vec" / group_id / f"{sample_name}.h5"
        if not gt_h5_path.exists():
            raise gr.Error(f"GT CAD 不存在: {gt_h5_path}")

        images = build_image_tensor(image_paths, self.args.img_size).to(self.device)
        with torch.no_grad():
            pred_z = self.model(images).cpu()
        pred_cad = self.adapter.decode(pred_z)[0]
        gt_cad = load_cad_vec(gt_h5_path)

        pred_stl = self.work_dir / "pred" / group_id / f"{sample_name}.stl"
        gt_stl = self.work_dir / "gt" / group_id / f"{sample_name}.stl"

        ok_pred, err_pred = export_cad_to_stl(pred_cad, pred_stl)
        ok_gt, err_gt = export_cad_to_stl(gt_cad, gt_stl)

        pred_model = str(pred_stl) if ok_pred else None
        gt_model = str(gt_stl) if ok_gt else None
        pred_status = "ok" if ok_pred else f"export failed: {err_pred}"
        gt_status = "ok" if ok_gt else f"export failed: {err_gt}"

        meta = [
            f"sample_id: `{sample_id}`",
            f"pred_len: **{len(pred_cad)}**",
            f"gt_len: **{len(gt_cad)}**",
            f"pred_export: **{pred_status}**",
            f"gt_export: **{gt_status}**",
        ]
        return (
            [str(path) for path in image_paths],
            pred_model,
            gt_model,
            "\n".join(meta),
            format_sequence(pred_cad),
            format_sequence(gt_cad),
        )


def build_demo(inspector: ImageToCadInspector):
    with gr.Blocks(title="Image-to-CAD Inspector") as demo:
        gr.Markdown(
            """
            # Image-to-CAD Inspector
            输入 `sample_id`，查看 8 视图、预测 CAD 和 GT CAD。
            """
        )
        with gr.Row():
            sample_id = gr.Textbox(label="sample_id", value="0000/00000093_00001")
            run_btn = gr.Button("Run", variant="primary")
        meta = gr.Markdown()
        gallery = gr.Gallery(label="Input Views", columns=4, rows=2, height=420)
        with gr.Row():
            pred_model = gr.Model3D(label="Predicted CAD", clear_color=[0.95, 0.95, 0.95, 1.0])
            gt_model = gr.Model3D(label="Ground Truth CAD", clear_color=[0.95, 0.95, 0.95, 1.0])
        with gr.Row():
            pred_text = gr.Textbox(label="Predicted Sequence", lines=20)
            gt_text = gr.Textbox(label="Ground Truth Sequence", lines=20)

        run_btn.click(
            fn=inspector.infer_sample,
            inputs=[sample_id],
            outputs=[gallery, pred_model, gt_model, meta, pred_text, gt_text],
        )
    return demo


def main():
    args = parse_args()
    inspector = ImageToCadInspector(args)
    demo = build_demo(inspector)
    demo.launch(server_name="0.0.0.0", server_port=args.server_port, share=False, show_api=False)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
        raise
