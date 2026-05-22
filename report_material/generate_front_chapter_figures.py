#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


REPO_ROOT = Path(__file__).resolve().parents[1]
THESIS_FIG_ROOT = (
    Path("/root/projects/web_projects/latex_workspace/papers/xjtu_bachelor_2024")
    / "figures"
    / "generated"
)


def ensure_dir() -> None:
    THESIS_FIG_ROOT.mkdir(parents=True, exist_ok=True)


def add_box(ax, x, y, w, h, text, fc="#f5f7fb", ec="#4b5563", fontsize=11, weight="normal"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.4,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        wrap=True,
    )


def add_arrow(ax, start, end, color="#4b5563", style="-|>", lw=1.5, mutation_scale=14):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
    )
    ax.add_patch(arrow)


def finalize(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def fig_research_route() -> None:
    fig, ax = plt.subplots(figsize=(12, 3.8))
    add_box(ax, 0.03, 0.3, 0.16, 0.38, "Omni-CAD\nJSON + Text", fc="#e8f1ff", ec="#2563eb", weight="bold")
    add_box(ax, 0.23, 0.3, 0.18, 0.38, "DeepCAD-compatible\nCAD sequence / latent\nrepresentation", fc="#f3e8ff", ec="#7c3aed")
    add_box(ax, 0.45, 0.3, 0.16, 0.38, "Image-only baseline\nResNet18 + GRU", fc="#fff7ed", ec="#ea580c")
    add_box(ax, 0.65, 0.3, 0.16, 0.38, "Multimodal extension\nText residual fusion", fc="#ecfdf5", ec="#059669")
    add_box(ax, 0.84, 0.3, 0.13, 0.38, "Transformer\nfront-end +\nanalysis", fc="#eef2ff", ec="#4338ca")

    add_arrow(ax, (0.19, 0.49), (0.23, 0.49), color="#2563eb")
    add_arrow(ax, (0.41, 0.49), (0.45, 0.49), color="#7c3aed")
    add_arrow(ax, (0.61, 0.49), (0.65, 0.49), color="#ea580c")
    add_arrow(ax, (0.81, 0.49), (0.84, 0.49), color="#059669")

    ax.text(0.5, 0.82, "Research Route of This Thesis", ha="center", va="center", fontsize=15, weight="bold")
    ax.text(
        0.5,
        0.14,
        "Data organization → latent-space formulation → image-only exploration → multimodal enhancement → structure improvement and analysis",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#374151",
    )
    finalize(ax)
    fig.tight_layout()
    fig.savefig(THESIS_FIG_ROOT / "thesis_research_route.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def fig_dataset_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(12.6, 4.2))
    add_box(ax, 0.03, 0.56, 0.17, 0.22, "Omni-CAD JSON", fc="#e8f1ff", ec="#2563eb", weight="bold")
    add_box(ax, 0.03, 0.18, 0.17, 0.18, "Text descriptions", fc="#ecfdf5", ec="#059669")

    add_box(ax, 0.27, 0.66, 0.18, 0.16, "CAD vectorization\ncad_vec", fc="#f3e8ff", ec="#7c3aed")
    add_box(ax, 0.27, 0.44, 0.18, 0.16, "Geometry export\nSTEP / PLY", fc="#fff7ed", ec="#ea580c")
    add_box(ax, 0.27, 0.20, 0.18, 0.16, "Blender multi-view\nrendering", fc="#fee2e2", ec="#dc2626")

    add_box(ax, 0.54, 0.66, 0.16, 0.16, "cad_vec", fc="#f3e8ff", ec="#7c3aed")
    add_box(ax, 0.54, 0.43, 0.16, 0.16, "cad_ply", fc="#fff7ed", ec="#ea580c")
    add_box(ax, 0.54, 0.20, 0.16, 0.16, "8-view cad_img", fc="#fee2e2", ec="#dc2626")

    add_box(ax, 0.77, 0.44, 0.19, 0.24, "Modality intersection\ntext ∩ image ∩ vector\n+\nquality control", fc="#f9fafb", ec="#4b5563", weight="bold")
    add_box(ax, 0.77, 0.14, 0.19, 0.16, "dataset_v0", fc="#eef2ff", ec="#4338ca", weight="bold")

    add_arrow(ax, (0.20, 0.67), (0.27, 0.74), color="#2563eb")
    add_arrow(ax, (0.20, 0.67), (0.27, 0.52), color="#2563eb")
    add_arrow(ax, (0.20, 0.27), (0.77, 0.52), color="#059669")
    add_arrow(ax, (0.20, 0.67), (0.27, 0.28), color="#2563eb")
    add_arrow(ax, (0.45, 0.74), (0.54, 0.74), color="#7c3aed")
    add_arrow(ax, (0.45, 0.52), (0.54, 0.51), color="#ea580c")
    add_arrow(ax, (0.45, 0.28), (0.54, 0.28), color="#dc2626")
    add_arrow(ax, (0.70, 0.74), (0.77, 0.58), color="#7c3aed")
    add_arrow(ax, (0.70, 0.51), (0.77, 0.56), color="#ea580c")
    add_arrow(ax, (0.70, 0.28), (0.77, 0.52), color="#dc2626")
    add_arrow(ax, (0.865, 0.44), (0.865, 0.30), color="#4338ca")

    ax.text(0.5, 0.92, "Construction Pipeline of the Multimodal CAD Dataset", ha="center", va="center", fontsize=15, weight="bold")
    finalize(ax)
    fig.tight_layout()
    fig.savefig(THESIS_FIG_ROOT / "thesis_dataset_pipeline.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def fig_dataset_subsets() -> None:
    fig, ax = plt.subplots(figsize=(12.4, 4.4))
    add_box(ax, 0.05, 0.37, 0.16, 0.24, "dataset_v0", fc="#eef2ff", ec="#4338ca", weight="bold")
    add_box(ax, 0.30, 0.62, 0.18, 0.18, "overlap_deep_all", fc="#f3e8ff", ec="#7c3aed")
    add_box(ax, 0.30, 0.36, 0.18, 0.18, "overlap_deep_first", fc="#f3e8ff", ec="#7c3aed")
    add_box(ax, 0.30, 0.10, 0.18, 0.18, "full_v0_len60", fc="#fff7ed", ec="#ea580c")
    add_box(ax, 0.58, 0.10, 0.23, 0.18, "full_v0_len60\nexcluding_rescue_test", fc="#fee2e2", ec="#dc2626", weight="bold")
    add_box(ax, 0.58, 0.36, 0.23, 0.18, "overlap_deep_first\nlen60 / train+val / test", fc="#ecfdf5", ec="#059669")
    add_box(ax, 0.84, 0.36, 0.12, 0.18, "ARE-\nreference\nsubset", fc="#f9fafb", ec="#4b5563", weight="bold")

    add_arrow(ax, (0.21, 0.50), (0.30, 0.71), color="#4338ca")
    add_arrow(ax, (0.21, 0.49), (0.30, 0.45), color="#4338ca")
    add_arrow(ax, (0.21, 0.43), (0.30, 0.19), color="#4338ca")
    add_arrow(ax, (0.48, 0.19), (0.58, 0.19), color="#ea580c")
    add_arrow(ax, (0.48, 0.45), (0.58, 0.45), color="#059669")
    add_arrow(ax, (0.81, 0.45), (0.84, 0.45), color="#4b5563")

    ax.text(0.5, 0.92, "Relationship Among the Main Training and Evaluation Subsets", ha="center", va="center", fontsize=15, weight="bold")
    ax.text(0.39, 0.86, "DeepCAD overlap alignment", ha="center", va="center", fontsize=10, color="#5b21b6")
    ax.text(0.40, 0.02, "Length filtering", ha="center", va="bottom", fontsize=10, color="#9a3412")
    ax.text(0.695, 0.02, "Leakage removal", ha="center", va="bottom", fontsize=10, color="#b91c1c")
    finalize(ax)
    fig.tight_layout()
    fig.savefig(THESIS_FIG_ROOT / "thesis_dataset_subsets.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def fig_task_definition() -> None:
    fig, ax = plt.subplots(figsize=(12, 4.0))
    add_box(ax, 0.05, 0.60, 0.18, 0.20, "Image-only input\n8 rendered views", fc="#e8f1ff", ec="#2563eb", weight="bold")
    add_box(ax, 0.05, 0.22, 0.18, 0.20, "Image + text input\n8 views + description", fc="#ecfdf5", ec="#059669", weight="bold")
    add_box(ax, 0.35, 0.41, 0.20, 0.22, "Latent prediction\n$f_{img}(I)$ or\n$f_{mm}(I, T)$", fc="#f3e8ff", ec="#7c3aed", weight="bold")
    add_box(ax, 0.67, 0.41, 0.16, 0.22, "Frozen DeepCAD\ndecoder", fc="#fff7ed", ec="#ea580c", weight="bold")
    add_box(ax, 0.86, 0.41, 0.11, 0.22, "CAD\nsequence /\nsolid", fc="#eef2ff", ec="#4338ca", weight="bold")

    add_arrow(ax, (0.23, 0.70), (0.35, 0.56), color="#2563eb")
    add_arrow(ax, (0.23, 0.32), (0.35, 0.48), color="#059669")
    add_arrow(ax, (0.55, 0.52), (0.67, 0.52), color="#7c3aed")
    add_arrow(ax, (0.83, 0.52), (0.86, 0.52), color="#ea580c")

    add_box(ax, 0.37, 0.08, 0.46, 0.14, "Inference modes: direct / nearest / blend", fc="#f9fafb", ec="#6b7280", fontsize=10.5)
    ax.text(0.5, 0.90, "Problem Formulation of This Thesis", ha="center", va="center", fontsize=15, weight="bold")
    finalize(ax)
    fig.tight_layout()
    fig.savefig(THESIS_FIG_ROOT / "thesis_task_definition.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dir()
    fig_research_route()
    fig_dataset_pipeline()
    fig_dataset_subsets()
    fig_task_definition()


if __name__ == "__main__":
    main()
