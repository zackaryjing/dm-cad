#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "runs" / "deepcad_latent"
OUT_ROOT = REPO_ROOT / "report_material"
FIG_ROOT = OUT_ROOT / "figures"
TABLE_ROOT = OUT_ROOT / "tables"
DATA_ROOT = OUT_ROOT / "data"


def ensure_dirs() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def load_summary(path: Path):
    return load_json(path)["summary"]


def normalize_history(history: list[dict]) -> list[dict]:
    by_epoch: OrderedDict[int, dict] = OrderedDict()
    for item in history:
        by_epoch[int(item["epoch"])] = item
    return list(by_epoch.values())


def load_history(path: Path) -> list[dict]:
    return normalize_history(load_json(path))


def fmt(v: float, digits: int = 4) -> str:
    return f"{v:.{digits}f}"


def write_markdown_table(path: Path, header: list[str], rows: list[list[str]]) -> None:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    path.write_text("\n".join(lines) + "\n")


def make_frontend_curves(histories: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    style = {
        "gru": ("GRU", "#1f77b4"),
        "gru_attn": ("GRU+Attn", "#ff7f0e"),
        "transformer": ("Transformer", "#2ca02c"),
    }
    for key, history in histories.items():
        label, color = style[key]
        epochs = [item["epoch"] for item in history]
        axes[0].plot(epochs, [item["test_mse"] for item in history], label=label, color=color, linewidth=2)
        axes[1].plot(
            epochs,
            [item["test_cosine"] for item in history],
            label=label,
            color=color,
            linewidth=2,
        )
    axes[0].set_title("Image-only Test MSE")
    axes[1].set_title("Image-only Test Cosine")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_ROOT / "thesis_frontend_training_curves.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_multimodal_curves(histories: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    style = {
        "image_only_transformer": ("Image-only Transformer", "#1f77b4"),
        "multimodal_transformer_v1": ("Transformer + Text", "#d62728"),
    }
    for key, history in histories.items():
        label, color = style[key]
        epochs = [item["epoch"] for item in history]
        axes[0].plot(epochs, [item["test_mse"] for item in history], label=label, color=color, linewidth=2)
        axes[1].plot(
            epochs,
            [item["test_cosine"] for item in history],
            label=label,
            color=color,
            linewidth=2,
        )
    axes[0].set_title("Transformer Backbone: Test MSE")
    axes[1].set_title("Transformer Backbone: Test Cosine")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_ROOT / "thesis_multimodal_training_curves.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_method_bar(rows: list[dict]) -> None:
    metrics = [
        ("cmd_token_acc", "Cmd Token Acc"),
        ("sequence_exact_rate", "Sequence Exact Rate"),
        ("pred_solid_valid_rate", "Solid Valid Rate"),
    ]
    labels = [row["label"] for row in rows]
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    for idx, (metric, title) in enumerate(metrics):
        vals = [row[metric] for row in rows]
        ax.bar(x + (idx - 1) * width, vals, width=width, label=title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Representative Method Comparison on the Held-out Test Set")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_ROOT / "thesis_method_comparison.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()

    hist_gru = load_history(RUNS_ROOT / "resnet18_gru_fullv0len60_v1_ddp2/history.json")
    hist_gru_attn = load_history(RUNS_ROOT / "resnet18_gruattn_v1_ddp2/history.json")
    hist_transformer = load_history(RUNS_ROOT / "resnet18_transformer_v1_ddp2/history.json")
    hist_transformer_text = load_history(RUNS_ROOT / "resnet18_transformer_text_v1_ddp2/history.json")

    eval_gru_direct = load_summary(RUNS_ROOT / "resnet18_gru_fullv0len60_v1_ddp2/eval_test_best_solid.json")
    eval_gru_text_direct = load_summary(
        RUNS_ROOT / "resnet18_gru_text_fullv0len60_frozenimg_v1_ddp2/eval_test_best_paper_metrics_cd.json"
    )
    eval_transformer_direct = load_summary(
        RUNS_ROOT / "resnet18_transformer_v1_ddp2/eval_test_best_paper_metrics_cd.json"
    )
    eval_transformer_text_direct = load_summary(
        RUNS_ROOT / "resnet18_transformer_text_v1_ddp2/eval_test_best_paper_metrics_cd.json"
    )
    eval_transformer_text_blend = load_summary(
        RUNS_ROOT / "resnet18_transformer_text_v1_ddp2/eval_test_best_blend_a05_paper_metrics_cd.json"
    )
    eval_are_ref_direct = load_summary(
        RUNS_ROOT / "resnet18_transformer_text_v1_ddp2/eval_are_reference_direct_paper_metrics_cd.json"
    )
    eval_are_ref_blend = load_summary(
        RUNS_ROOT / "resnet18_transformer_text_v1_ddp2/eval_are_reference_blend_a05_paper_metrics_cd.json"
    )

    payload = {
        "histories": {
            "gru": hist_gru,
            "gru_attn": hist_gru_attn,
            "transformer": hist_transformer,
            "transformer_text_v1": hist_transformer_text,
        },
        "summaries": {
            "gru_direct": eval_gru_direct,
            "gru_text_direct": eval_gru_text_direct,
            "transformer_direct": eval_transformer_direct,
            "transformer_text_direct": eval_transformer_text_direct,
            "transformer_text_blend": eval_transformer_text_blend,
            "are_reference_direct": eval_are_ref_direct,
            "are_reference_blend": eval_are_ref_blend,
        },
    }
    with (DATA_ROOT / "thesis_material_payloads_2026-05-15.json").open("w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    make_frontend_curves(
        {
            "gru": hist_gru,
            "gru_attn": hist_gru_attn,
            "transformer": hist_transformer,
        }
    )
    make_multimodal_curves(
        {
            "image_only_transformer": hist_transformer,
            "multimodal_transformer_v1": hist_transformer_text,
        }
    )
    make_method_bar(
        [
            {
                "label": "GRU Img",
                "cmd_token_acc": eval_gru_direct["cmd_token_acc"],
                "sequence_exact_rate": eval_gru_direct["sequence_exact_rate"],
                "pred_solid_valid_rate": eval_gru_direct["pred_solid_valid_rate"],
            },
            {
                "label": "GRU+Text",
                "cmd_token_acc": eval_gru_text_direct["cmd_token_acc"],
                "sequence_exact_rate": eval_gru_text_direct["sequence_exact_rate"],
                "pred_solid_valid_rate": eval_gru_text_direct["pred_solid_valid_rate"],
            },
            {
                "label": "Trans Img",
                "cmd_token_acc": eval_transformer_direct["cmd_token_acc"],
                "sequence_exact_rate": eval_transformer_direct["sequence_exact_rate"],
                "pred_solid_valid_rate": eval_transformer_direct["pred_solid_valid_rate"],
            },
            {
                "label": "Trans+Text",
                "cmd_token_acc": eval_transformer_text_direct["cmd_token_acc"],
                "sequence_exact_rate": eval_transformer_text_direct["sequence_exact_rate"],
                "pred_solid_valid_rate": eval_transformer_text_direct["pred_solid_valid_rate"],
            },
        ]
    )

    write_markdown_table(
        TABLE_ROOT / "thesis_frontend_ablation_2026-05-15.md",
        ["方法", "Cmd Token Acc", "Token Exact Acc", "Sequence Exact Rate", "Solid Valid Rate", "Mean Len Error"],
        [
            [
                "GRU image-only",
                fmt(eval_gru_direct["cmd_token_acc"]),
                fmt(eval_gru_direct["token_exact_acc"]),
                fmt(eval_gru_direct["sequence_exact_rate"]),
                fmt(eval_gru_direct["pred_solid_valid_rate"]),
                fmt(eval_gru_direct["mean_len_abs_error"]),
            ],
            [
                "GRU + text (frozen)",
                fmt(eval_gru_text_direct["cmd_token_acc"]),
                fmt(eval_gru_text_direct["token_exact_acc"]),
                fmt(eval_gru_text_direct["sequence_exact_rate"]),
                fmt(eval_gru_text_direct["pred_solid_valid_rate"]),
                fmt(eval_gru_text_direct["mean_len_abs_error"]),
            ],
            [
                "Transformer image-only",
                fmt(eval_transformer_direct["cmd_token_acc"]),
                fmt(eval_transformer_direct["token_exact_acc"]),
                fmt(eval_transformer_direct["sequence_exact_rate"]),
                fmt(eval_transformer_direct["pred_solid_valid_rate"]),
                fmt(eval_transformer_direct["mean_len_abs_error"]),
            ],
            [
                "Transformer + text",
                fmt(eval_transformer_text_direct["cmd_token_acc"]),
                fmt(eval_transformer_text_direct["token_exact_acc"]),
                fmt(eval_transformer_text_direct["sequence_exact_rate"]),
                fmt(eval_transformer_text_direct["pred_solid_valid_rate"]),
                fmt(eval_transformer_text_direct["mean_len_abs_error"]),
            ],
        ],
    )

    write_markdown_table(
        TABLE_ROOT / "thesis_final_metrics_2026-05-15.md",
        ["方法", "ACC_cmd", "ACC_param", "Invalidity", "CD Mean", "CD Median"],
        [
            [
                "Transformer image-only direct",
                fmt(eval_transformer_direct["acc_cmd"]),
                fmt(eval_transformer_direct["acc_param"]),
                fmt(eval_transformer_direct["invalidity_ratio"]),
                fmt(eval_transformer_direct["cd_mean"]),
                fmt(eval_transformer_direct["cd_median"]),
            ],
            [
                "Transformer + text direct",
                fmt(eval_transformer_text_direct["acc_cmd"]),
                fmt(eval_transformer_text_direct["acc_param"]),
                fmt(eval_transformer_text_direct["invalidity_ratio"]),
                fmt(eval_transformer_text_direct["cd_mean"]),
                fmt(eval_transformer_text_direct["cd_median"]),
            ],
            [
                "Transformer + text blend(0.5)",
                fmt(eval_transformer_text_blend["acc_cmd"]),
                fmt(eval_transformer_text_blend["acc_param"]),
                fmt(eval_transformer_text_blend["invalidity_ratio"]),
                fmt(eval_transformer_text_blend["cd_mean"]),
                fmt(eval_transformer_text_blend["cd_median"]),
            ],
        ],
    )

    write_markdown_table(
        TABLE_ROOT / "thesis_are_reference_2026-05-15.md",
        ["方法", "ACC_cmd", "ACC_param", "Invalidity", "CD Median"],
        [
            [
                "Transformer + text direct",
                fmt(eval_are_ref_direct["acc_cmd"]),
                fmt(eval_are_ref_direct["acc_param"]),
                fmt(eval_are_ref_direct["invalidity_ratio"]),
                fmt(eval_are_ref_direct["cd_median"]),
            ],
            [
                "Transformer + text blend(0.5)",
                fmt(eval_are_ref_blend["acc_cmd"]),
                fmt(eval_are_ref_blend["acc_param"]),
                fmt(eval_are_ref_blend["invalidity_ratio"]),
                fmt(eval_are_ref_blend["cd_median"]),
            ],
        ],
    )


if __name__ == "__main__":
    main()
