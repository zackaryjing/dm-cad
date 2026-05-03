#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "runs"
OUT_ROOT = REPO_ROOT / "report_material"
FIG_ROOT = OUT_ROOT / "figures"
TABLE_ROOT = OUT_ROOT / "tables"
DATA_ROOT = OUT_ROOT / "data"


def ensure_dirs():
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)


def load_json(path: str | Path):
    with Path(path).open("r") as f:
        return json.load(f)


def load_history(path: str | Path):
    return load_json(path)


def load_eval_summary(path: str | Path):
    return load_json(path)["summary"]


def extract_old_framework_scalars(run_dir: Path):
    scalar_map: dict[str, list[tuple[int, float]]] = {}
    for event_file in sorted(run_dir.glob("events.out.tfevents.*")):
        ea = EventAccumulator(str(event_file))
        ea.Reload()
        for tag in ea.Tags()["scalars"]:
            scalar_map.setdefault(tag, [])
            scalar_map[tag].extend((event.step, event.value) for event in ea.Scalars(tag))

    deduped: dict[str, list[dict[str, float]]] = {}
    for tag, values in scalar_map.items():
        last_by_step = {}
        for step, value in values:
            last_by_step[step] = value
        deduped[tag] = [{"step": int(step), "value": float(last_by_step[step])} for step in sorted(last_by_step)]
    return deduped


def best_entry(series: list[dict[str, float]], mode: str):
    if not series:
        return None
    if mode == "min":
        return min(series, key=lambda x: x["value"])
    return max(series, key=lambda x: x["value"])


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def make_old_framework_figure(old_scalars: dict[str, list[dict[str, float]]]):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plots = [
        ("val/loss", "Validation Loss"),
        ("val/cmd_token_acc", "Validation Cmd Token Acc"),
        ("val/param_token_acc", "Validation Param Token Acc"),
        ("val/sequence_exact_acc", "Validation Sequence Exact Acc"),
    ]
    for ax, (tag, title) in zip(axes.flat, plots):
        series = old_scalars.get(tag, [])
        xs = [item["step"] for item in series]
        ys = [item["value"] for item in series]
        ax.plot(xs, ys, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Global Step")
        ax.grid(alpha=0.3)
    fig.suptitle("Old Framework Training Signals", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_ROOT / "old_framework_training_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_new_route_training_figure(histories: dict[str, list[dict[str, float]]]):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = {
        "image_only_fullv0": "Image-only",
        "text_unfreeze": "Image+Text (unfreeze)",
        "text_frozen": "Image+Text (frozen)",
    }
    colors = {
        "image_only_fullv0": "#1f77b4",
        "text_unfreeze": "#ff7f0e",
        "text_frozen": "#2ca02c",
    }
    for key, history in histories.items():
        epochs = [item["epoch"] for item in history]
        axes[0].plot(epochs, [item["test_mse"] for item in history], label=labels[key], color=colors[key], linewidth=2)
        axes[1].plot(
            epochs,
            [item["test_cosine"] for item in history],
            label=labels[key],
            color=colors[key],
            linewidth=2,
        )
    axes[0].set_title("New Route Test MSE")
    axes[1].set_title("New Route Test Cosine")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_ROOT / "new_route_training_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_main_comparison_figure(method_rows: list[dict]):
    metrics = ["token_exact_acc", "sequence_exact_rate", "pred_solid_valid_rate"]
    titles = ["Token Exact Acc", "Sequence Exact Rate", "Solid Valid Rate"]
    labels = [row["label"] for row in method_rows]
    x = np.arange(len(labels))
    width = 0.22

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        values = [row[metric] for row in method_rows]
        ax.bar(x + (idx - 1) * width, values, width=width, label=title)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Core Method Comparison on Held-out Test Set")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_ROOT / "main_method_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_bucket_solid_figure(bucket_rows: dict[str, dict[str, float]]):
    buckets = ["len_01_06", "len_07_12", "len_13_24", "len_25_plus"]
    methods = list(bucket_rows.keys())
    labels = {
        "image_only_direct": "Image-only Direct",
        "text_direct_frozen": "Text Direct (Frozen)",
        "text_blend_frozen": "Text Blend 0.5 (Frozen)",
    }
    x = np.arange(len(buckets))
    width = 0.24

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for idx, method in enumerate(methods):
        vals = [bucket_rows[method][bucket] for bucket in buckets]
        ax.bar(x + (idx - 1) * width, vals, width=width, label=labels.get(method, method))
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Solid Valid Rate")
    ax.set_title("Solid Valid Rate by Sequence-Length Bucket")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_ROOT / "bucket_solid_validity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_markdown_table(path: Path, header: list[str], rows: list[list[str]]):
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    path.write_text("\n".join(lines) + "\n")


def main():
    ensure_dirs()

    old_run_dir = RUNS_ROOT / "dmcad" / "config_full_arch_iter_v1_20260411_122510"
    old_scalars = extract_old_framework_scalars(old_run_dir)
    save_json(DATA_ROOT / "old_framework_scalars.json", old_scalars)

    old_best = {
        "val_loss_best": best_entry(old_scalars.get("val/loss", []), "min"),
        "val_cmd_token_acc_best": best_entry(old_scalars.get("val/cmd_token_acc", []), "max"),
        "val_param_token_acc_best": best_entry(old_scalars.get("val/param_token_acc", []), "max"),
        "val_token_exact_acc_best": best_entry(old_scalars.get("val/token_exact_acc", []), "max"),
        "val_sequence_exact_acc_best": best_entry(old_scalars.get("val/sequence_exact_acc", []), "max"),
    }
    save_json(DATA_ROOT / "old_framework_best_scalars.json", old_best)

    hist_image = load_history(RUNS_ROOT / "deepcad_latent/resnet18_gru_fullv0len60_v1_ddp2/history.json")
    hist_text_unfreeze = load_history(RUNS_ROOT / "deepcad_latent/resnet18_gru_text_fullv0len60_v1_ddp2/history.json")
    hist_text_frozen = load_history(RUNS_ROOT / "deepcad_latent/resnet18_gru_text_fullv0len60_frozenimg_v1_ddp2/history.json")
    save_json(
        DATA_ROOT / "new_route_histories.json",
        {
            "image_only_fullv0": hist_image,
            "text_unfreeze": hist_text_unfreeze,
            "text_frozen": hist_text_frozen,
        },
    )

    eval_image_direct = load_eval_summary(RUNS_ROOT / "deepcad_latent/resnet18_gru_fullv0len60_v1_ddp2/eval_test_best_solid.json")
    eval_text_direct_frozen = load_eval_summary(
        RUNS_ROOT / "deepcad_latent/resnet18_gru_text_fullv0len60_frozenimg_v1_ddp2/eval_test_best_solid.json"
    )
    eval_text_blend_frozen = load_eval_summary(
        RUNS_ROOT / "deepcad_latent/resnet18_gru_text_fullv0len60_frozenimg_v1_ddp2/eval_test_best_blend_a05_solid.json"
    )
    eval_text_direct_prev = load_eval_summary(
        RUNS_ROOT / "deepcad_latent/resnet18_gru_text_fullv0len60_v1_ddp2/eval_test_best_solid_v2.json"
    )

    comparison_payload = {
        "image_only_direct": eval_image_direct,
        "text_direct_unfreeze": eval_text_direct_prev,
        "text_direct_frozen": eval_text_direct_frozen,
        "text_blend_frozen": eval_text_blend_frozen,
    }
    save_json(DATA_ROOT / "new_route_eval_summaries.json", comparison_payload)

    make_old_framework_figure(old_scalars)
    make_new_route_training_figure(
        {
            "image_only_fullv0": hist_image,
            "text_unfreeze": hist_text_unfreeze,
            "text_frozen": hist_text_frozen,
        }
    )
    make_main_comparison_figure(
        [
            {
                "label": "Image-only Direct",
                "token_exact_acc": eval_image_direct["token_exact_acc"],
                "sequence_exact_rate": eval_image_direct["sequence_exact_rate"],
                "pred_solid_valid_rate": eval_image_direct["pred_solid_valid_rate"],
            },
            {
                "label": "Text Direct (Frozen)",
                "token_exact_acc": eval_text_direct_frozen["token_exact_acc"],
                "sequence_exact_rate": eval_text_direct_frozen["sequence_exact_rate"],
                "pred_solid_valid_rate": eval_text_direct_frozen["pred_solid_valid_rate"],
            },
            {
                "label": "Text Blend 0.5 (Frozen)",
                "token_exact_acc": eval_text_blend_frozen["token_exact_acc"],
                "sequence_exact_rate": eval_text_blend_frozen["sequence_exact_rate"],
                "pred_solid_valid_rate": eval_text_blend_frozen["pred_solid_valid_rate"],
            },
        ]
    )
    make_bucket_solid_figure(
        {
            "image_only_direct": {
                bucket: eval_image_direct["by_bucket"][bucket]["pred_solid_valid_rate"]
                for bucket in ["len_01_06", "len_07_12", "len_13_24", "len_25_plus"]
            },
            "text_direct_frozen": {
                bucket: eval_text_direct_frozen["by_bucket"][bucket]["pred_solid_valid_rate"]
                for bucket in ["len_01_06", "len_07_12", "len_13_24", "len_25_plus"]
            },
            "text_blend_frozen": {
                bucket: eval_text_blend_frozen["by_bucket"][bucket]["pred_solid_valid_rate"]
                for bucket in ["len_01_06", "len_07_12", "len_13_24", "len_25_plus"]
            },
        }
    )

    write_markdown_table(
        TABLE_ROOT / "dataset_summary.md",
        ["数据集版本", "训练样本数", "测试样本数", "说明"],
        [
            ["旧框架训练集", "约 355,254", "71,088", "基于 CAD-MLLM/Omni-CAD 组织的自回归训练集"],
            ["新框架 overlap 子集", "117,497", "5,592", "DeepCAD overlap, seq_len<=60, train+val 合并"],
            ["新框架扩展全量子集", "318,886", "5,592", "full_v0_len60_excluding_rescue_test"],
        ],
    )

    write_markdown_table(
        TABLE_ROOT / "method_comparison.md",
        ["方法", "Cmd Token Acc", "Token Exact Acc", "Sequence Exact Rate", "Solid Valid Rate", "Mean Len Abs Error"],
        [
            [
                "Image-only Direct",
                f"{eval_image_direct['cmd_token_acc']:.4f}",
                f"{eval_image_direct['token_exact_acc']:.4f}",
                f"{eval_image_direct['sequence_exact_rate']:.4f}",
                f"{eval_image_direct['pred_solid_valid_rate']:.4f}",
                f"{eval_image_direct['mean_len_abs_error']:.4f}",
            ],
            [
                "Image+Text Direct (unfreeze)",
                f"{eval_text_direct_prev['cmd_token_acc']:.4f}",
                f"{eval_text_direct_prev['token_exact_acc']:.4f}",
                f"{eval_text_direct_prev['sequence_exact_rate']:.4f}",
                f"{eval_text_direct_prev['pred_solid_valid_rate']:.4f}",
                f"{eval_text_direct_prev['mean_len_abs_error']:.4f}",
            ],
            [
                "Image+Text Direct (frozen image)",
                f"{eval_text_direct_frozen['cmd_token_acc']:.4f}",
                f"{eval_text_direct_frozen['token_exact_acc']:.4f}",
                f"{eval_text_direct_frozen['sequence_exact_rate']:.4f}",
                f"{eval_text_direct_frozen['pred_solid_valid_rate']:.4f}",
                f"{eval_text_direct_frozen['mean_len_abs_error']:.4f}",
            ],
            [
                "Image+Text Blend 0.5 (frozen image)",
                f"{eval_text_blend_frozen['cmd_token_acc']:.4f}",
                f"{eval_text_blend_frozen['token_exact_acc']:.4f}",
                f"{eval_text_blend_frozen['sequence_exact_rate']:.4f}",
                f"{eval_text_blend_frozen['pred_solid_valid_rate']:.4f}",
                f"{eval_text_blend_frozen['mean_len_abs_error']:.4f}",
            ],
        ],
    )

    bucket_rows = []
    for bucket in ["len_01_06", "len_07_12", "len_13_24", "len_25_plus"]:
        bucket_rows.append(
            [
                bucket,
                f"{eval_image_direct['by_bucket'][bucket]['pred_solid_valid_rate']:.4f}",
                f"{eval_text_direct_frozen['by_bucket'][bucket]['pred_solid_valid_rate']:.4f}",
                f"{eval_text_blend_frozen['by_bucket'][bucket]['pred_solid_valid_rate']:.4f}",
            ]
        )
    write_markdown_table(
        TABLE_ROOT / "bucket_solid_validity.md",
        ["长度分桶", "Image-only Direct", "Image+Text Direct (Frozen)", "Image+Text Blend 0.5 (Frozen)"],
        bucket_rows,
    )

    old_summary_rows = []
    for label, key in [
        ("Best Val Loss", "val_loss_best"),
        ("Best Val Cmd Token Acc", "val_cmd_token_acc_best"),
        ("Best Val Param Token Acc", "val_param_token_acc_best"),
        ("Best Val Token Exact Acc", "val_token_exact_acc_best"),
        ("Best Val Sequence Exact Acc", "val_sequence_exact_acc_best"),
    ]:
        item = old_best[key]
        old_summary_rows.append([label, str(item["step"]), f"{item['value']:.4f}"])
    write_markdown_table(TABLE_ROOT / "old_framework_best_scalars.md", ["旧框架指标", "Step", "Value"], old_summary_rows)

    readme = f"""# Report Material Index

本目录用于存放项目设计报告可直接复用的图表、表格和摘要。

## 目录结构

- `figures/`
  - `old_framework_training_curves.png`：旧框架训练/验证关键曲线
  - `new_route_training_curves.png`：新框架三组训练策略的测试曲线
  - `main_method_comparison.png`：主方法横向对比图
  - `bucket_solid_validity.png`：不同序列长度分桶下的实体有效率
- `tables/`
  - `dataset_summary.md`
  - `method_comparison.md`
  - `bucket_solid_validity.md`
  - `old_framework_best_scalars.md`
- `data/`
  - `old_framework_scalars.json`
  - `old_framework_best_scalars.json`
  - `new_route_histories.json`
  - `new_route_eval_summaries.json`

## 可直接写进报告的关键结论

1. 旧框架在 teacher-forced 训练曲线上出现了看似乐观的指标，但自由生成阶段稳定性不足，促使框架切换。
2. 新框架在冻结 DeepCAD latent 空间后，可以稳定输出可用 CAD，并且随着训练集从 `117,497` 扩大到 `318,886`，实体有效率进一步提升。
3. 双模态确实有效：
   - `Image-only Direct` 的实体有效率为 `{eval_image_direct['pred_solid_valid_rate']:.4f}`
   - `Image+Text Direct (Frozen)` 提升到 `{eval_text_direct_frozen['pred_solid_valid_rate']:.4f}`
4. 当前最好结果为 `Image+Text Blend 0.5 (Frozen)`：
   - `Token Exact Acc = {eval_text_blend_frozen['token_exact_acc']:.4f}`
   - `Sequence Exact Rate = {eval_text_blend_frozen['sequence_exact_rate']:.4f}`
   - `Solid Valid Rate = {eval_text_blend_frozen['pred_solid_valid_rate']:.4f}`
5. 长序列仍然是主要难点，但双模态和 blend 在中短序列以及中等长度样本上均显著改善。

## 备注

- 生成脚本：`report_material/generate_materials.py`
- 若后续有新实验结果，可重新运行该脚本覆盖更新图表。
"""
    (OUT_ROOT / "README.md").write_text(readme)


if __name__ == "__main__":
    main()
