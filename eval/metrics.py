"""
评估指标模块 - 实现 CAD 生成评估指标
基于设计文档 6 节
"""

import torch
import numpy as np


def evaluate_sequence_accuracy(pred_cmds, gt_cmds, valid_mask=None):
    """命令类型准确率

    Args:
        pred_cmds: [batch, seq_len] - 预测命令类型
        gt_cmds: [batch, seq_len] - 真实命令类型
        valid_mask: [batch, seq_len] - 有效位置掩码
    Returns:
        accuracy: 准确率
    """
    if valid_mask is not None:
        correct = (pred_cmds == gt_cmds)[valid_mask].sum()
        total = valid_mask.sum()
    else:
        correct = (pred_cmds == gt_cmds).sum()
        total = gt_cmds.numel()

    if total == 0:
        return torch.tensor(0.0)
    return correct.float() / total


def evaluate_parameter_accuracy(pred_params, gt_params, valid_mask=None, threshold=0.1):
    """参数准确率 (相对误差 < threshold 视为正确)

    Args:
        pred_params: [batch, seq_len, n_params] - 预测参数
        gt_params: [batch, seq_len, n_params] - 真实参数
        valid_mask: [batch, seq_len] - 有效位置掩码
        threshold: 相对误差阈值
    Returns:
        accuracy: 参数准确率
    """
    rel_error = torch.abs(pred_params - gt_params) / (torch.abs(gt_params) + 1e-8)

    if valid_mask is not None:
        # 扩展 mask 到参数维度
        mask_3d = valid_mask.unsqueeze(-1).expand_as(rel_error)
        correct = (rel_error < threshold)[mask_3d].float()
    else:
        correct = (rel_error < threshold).float()

    return correct.mean()


def evaluate_chamfer_distance(pred_pc, gt_pc):
    """Chamfer 距离 (需要先将 CAD 序列转换为点云)

    Args:
        pred_pc: [batch, N, 3] - 预测点云
        gt_pc: [batch, N, 3] - 真实点云
    Returns:
        chamfer_dist: Chamfer 距离
    """
    # 计算 pred 到 gt 的最近邻距离
    dist_pred_to_gt = _chamfer_distance_one_side(pred_pc, gt_pc)
    dist_gt_to_pred = _chamfer_distance_one_side(gt_pc, pred_pc)

    chamfer_dist = dist_pred_to_gt + dist_gt_to_pred
    return chamfer_dist.mean()


def _chamfer_distance_one_side(pc1, pc2):
    """计算单向 Chamfer 距离"""
    # pc1: [batch, N, 3], pc2: [batch, M, 3]
    batch_size = pc1.shape[0]

    # 计算点对之间的距离
    pc1_expand = pc1.unsqueeze(2)  # [batch, N, 1, 3]
    pc2_expand = pc2.unsqueeze(1)  # [batch, 1, M, 3]

    dist = torch.norm(pc1_expand - pc2_expand, dim=-1)  # [batch, N, M]

    # 最近邻距离
    min_dist = dist.min(dim=2)[0]  # [batch, N]

    return min_dist.mean(dim=1)  # [batch]


def evaluate_invalidity_ratio(cad_sequences):
    """无效序列比例

    Args:
        cad_sequences: CAD 序列列表
    Returns:
        invalid_ratio: 无效序列比例
    """
    # TODO: 检查 CAD 序列的几何有效性
    # 例如：检查 sketch 是否闭合，extrusion 是否有效等
    invalid_count = 0
    total_count = len(cad_sequences)

    for seq in cad_sequences:
        if not _is_valid_cad_sequence(seq):
            invalid_count += 1

    return invalid_count / total_count


def _is_valid_cad_sequence(seq):
    """检查 CAD 序列是否有效"""
    # TODO: 实现 CAD 序列有效性检查
    # 基本检查:
    # 1. 序列必须以 START 开始，END 结束
    # 2. sketch 和 extrusion 必须交替出现
    # 3. 参数必须在合理范围内
    return True


class CADMetrics:
    """CAD 生成综合评估指标"""

    def __init__(self, param_threshold=0.1):
        self.param_threshold = param_threshold

    def compute_all_metrics(self, pred_cmds, param_pred, gt_cmds, gt_params,
                            valid_mask=None):
        """计算所有指标

        Args:
            pred_cmds: 预测命令类型
            param_pred: 预测参数
            gt_cmds: 真实命令类型
            gt_params: 真实参数
            valid_mask: 有效掩码
        Returns:
            metrics: 指标字典
        """
        cmd_acc = evaluate_sequence_accuracy(pred_cmds, gt_cmds, valid_mask)
        param_acc = evaluate_parameter_accuracy(
            param_pred, gt_params, valid_mask, self.param_threshold
        )

        return {
            'cmd_accuracy': cmd_acc.item(),
            'param_accuracy': param_acc.item(),
            'combined_score': 0.5 * cmd_acc.item() + 0.5 * param_acc.item()
        }
