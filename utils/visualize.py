"""
可视化工具 - CAD 序列和训练曲线可视化
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_cad_sequence(cad_sequence, save_path=None):
    """可视化 CAD 命令序列

    Args:
        cad_sequence: CAD 命令序列 (命令类型 + 参数)
        save_path: 保存路径
    Returns:
        fig: matplotlib 图形
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 解析命令序列
    cmd_types = []
    param_values = []

    for cmd, params in cad_sequence:
        if isinstance(cmd, torch.Tensor):
            cmd = cmd.item()
        cmd_types.append(cmd)

        if isinstance(params, torch.Tensor):
            params = params.cpu().numpy()
        param_values.append(params)

    cmd_types = np.array(cmd_types)

    # 上图：命令类型序列
    ax1 = axes[0]
    cmd_names = ['START', 'SKETCH', 'EXTRUDE', 'END']
    ax1.bar(range(len(cmd_types)), cmd_types, tick_label=cmd_names[:len(cmd_types)])
    ax1.set_xlabel('Sequence Position')
    ax1.set_ylabel('Command Type')
    ax1.set_title('CAD Command Sequence')
    ax1.set_ylim(-0.5, 3.5)
    ax1.set_yticks(range(4))
    ax1.set_yticklabels(cmd_names)

    # 下图：参数值
    ax2 = axes[1]
    param_array = np.array(param_values)
    im = ax2.imshow(param_array.T, aspect='auto', cmap='viridis')
    ax2.set_xlabel('Command Index')
    ax2.set_ylabel('Parameter Index')
    ax2.set_title('Command Parameters')
    plt.colorbar(im, ax=ax2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    return fig


def plot_training_curves(log_file, save_path=None):
    """绘制训练曲线

    Args:
        log_file: 训练日志文件 (TensorBoard events 或 JSON)
        save_path: 保存路径
    Returns:
        fig: matplotlib 图形
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 这里应该读取 TensorBoard 或 JSON 日志
    # 示例代码
    train_losses = []  # 从日志读取
    val_losses = []
    epochs = []

    ax1 = axes[0]
    ax1.plot(epochs, train_losses, label='Train Loss')
    ax1.plot(epochs, val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    ax2 = axes[1]
    # 准确率曲线
    cmd_acc = []
    param_acc = []
    ax2.plot(epochs, cmd_acc, label='Command Accuracy')
    ax2.plot(epochs, param_acc, label='Parameter Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Evaluation Metrics')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    return fig


def visualize_attention(attention_weights, save_path=None):
    """可视化注意力权重

    Args:
        attention_weights: 注意力权重矩阵
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(attention_weights, cmap='viridis', aspect='auto')
    ax.set_xlabel('View Index')
    ax.set_ylabel('Query Position')
    ax.set_title('Multi-View Attention Weights')
    plt.colorbar(im, ax=ax)

    if save_path:
        plt.savefig(save_path, dpi=150)

    return fig


def visualize_multi_view_images(images, titles=None, save_path=None):
    """可视化 8 视图图像

    Args:
        images: [8, 3, H, W] 或 [8, H, W, 3] 图像
        titles: 每个视图的标题
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(8):
        img = images[i]
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
            img = np.transpose(img, (1, 2, 0))  # CHW -> HWC

        # 归一化到 0-1
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        axes[i].imshow(img)
        if titles:
            axes[i].set_title(titles[i])
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    return fig
