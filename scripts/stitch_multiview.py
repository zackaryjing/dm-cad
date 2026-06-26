"""Stitch 8 multi-view images into a 2x4 grid for thesis figures."""

import os
from PIL import Image

DATA_ROOT = "/root/projects/CAD-MLLM/datasets/Omni-CAD"
GROUP_ID = "0000"
SAMPLE_NAME = "00000168_00001"
N_VIEWS = 8
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "thesis")


def load_views():
    img_dir = os.path.join(DATA_ROOT, "cad_img", GROUP_ID, SAMPLE_NAME)
    images = []
    for i in range(N_VIEWS):
        path = os.path.join(img_dir, f"{SAMPLE_NAME}_{i:03d}.png")
        images.append(Image.open(path).convert("RGB"))
    return images


def crop_inner(img, ratio=0.1):
    """Crop ratio from each edge (e.g. 0.1 crops 10% from left, right, top, bottom)."""
    w, h = img.size
    left = int(w * ratio)
    upper = int(h * ratio)
    right = w - left
    lower = h - upper
    return img.crop((left, upper, right, lower))


def stitch(images, cols=4):
    rows = len(images) // cols
    w, h = images[0].size
    canvas = Image.new("RGB", (cols * w, rows * h))
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        canvas.paste(img, (c * w, r * h))
    return canvas


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images = load_views()

    # Version 1: normal tight stitch
    canvas_full = stitch(images)
    path_full = os.path.join(OUTPUT_DIR, f"{SAMPLE_NAME}_multiview.png")
    canvas_full.save(path_full)
    print(f"Saved {path_full} ({canvas_full.size[0]}x{canvas_full.size[1]})")

    # Version 2: inner crop 10% then tight stitch
    images_cropped = [crop_inner(img, 0.07) for img in images]
    canvas_crop = stitch(images_cropped)
    path_crop = os.path.join(OUTPUT_DIR, f"{SAMPLE_NAME}_multiview_crop7.png")
    canvas_crop.save(path_crop)
    print(f"Saved {path_crop} ({canvas_crop.size[0]}x{canvas_crop.size[1]})")
