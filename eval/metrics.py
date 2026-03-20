"""
评估指标模块 - 实现 CAD 生成评估指标
基于设计文档 6 节
"""

import torch


CMD_PARAM_MASK = torch.tensor([
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
], dtype=torch.bool)


def evaluate_sequence_accuracy(pred_cmds, gt_cmds, valid_mask=None):
    """命令类型准确率"""
    if valid_mask is not None:
        valid_mask = valid_mask.bool()
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred_cmds.device)
        correct = (pred_cmds == gt_cmds)[valid_mask].float()
    else:
        correct = (pred_cmds == gt_cmds).float().reshape(-1)

    if correct.numel() == 0:
        return torch.tensor(0.0, device=pred_cmds.device)
    return correct.mean()


def evaluate_parameter_accuracy(pred_params, gt_params, gt_cmds, valid_mask=None, threshold=0.1):
    """参数准确率 (绝对误差 < threshold 视为正确，仅统计命令有效维度)"""
    abs_error = torch.abs(pred_params - gt_params)
    correct = abs_error < threshold

    cmd_mask = CMD_PARAM_MASK.to(pred_params.device)[gt_cmds.clamp(min=0, max=len(CMD_PARAM_MASK) - 1)]
    if valid_mask is not None:
        valid_mask = valid_mask.bool()
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred_params.device)
        combined_mask = valid_mask.unsqueeze(-1) & cmd_mask
    else:
        combined_mask = cmd_mask

    if not combined_mask.any():
        return torch.tensor(0.0, device=pred_params.device)
    return correct[combined_mask].float().mean()


def evaluate_chamfer_distance(pred_pc, gt_pc):
    """Chamfer 距离 (需要先将 CAD 序列转换为点云)"""
    dist_pred_to_gt = _chamfer_distance_one_side(pred_pc, gt_pc)
    dist_gt_to_pred = _chamfer_distance_one_side(gt_pc, pred_pc)
    return (dist_pred_to_gt + dist_gt_to_pred).mean()


def _chamfer_distance_one_side(pc1, pc2):
    """计算单向 Chamfer 距离"""
    pc1_expand = pc1.unsqueeze(2)
    pc2_expand = pc2.unsqueeze(1)
    dist = torch.norm(pc1_expand - pc2_expand, dim=-1)
    min_dist = dist.min(dim=2)[0]
    return min_dist.mean(dim=1)


def evaluate_invalidity_ratio(cad_sequences):
    """无效序列比例"""
    invalid_count = 0
    total_count = len(cad_sequences)
    if total_count == 0:
        return 0.0

    for seq in cad_sequences:
        if not _is_valid_cad_sequence(seq):
            invalid_count += 1

    return invalid_count / total_count


def _is_valid_cad_sequence(seq):
    """检查 CAD 序列是否有效"""
    if not seq:
        return False
    return True


class CADMetrics:
    """CAD 生成综合评估指标"""

    def __init__(self, param_threshold=0.1):
        self.param_threshold = param_threshold

    def compute_all_metrics(self, pred_cmds, param_pred, gt_cmds, gt_params,
                            valid_mask=None):
        cmd_acc = evaluate_sequence_accuracy(pred_cmds, gt_cmds, valid_mask)
        param_acc = evaluate_parameter_accuracy(
            param_pred, gt_params, gt_cmds, valid_mask, self.param_threshold
        )

        return {
            'cmd_accuracy': cmd_acc.item(),
            'param_accuracy': param_acc.item(),
            'combined_score': 0.5 * cmd_acc.item() + 0.5 * param_acc.item()
        }

    def empty_metrics(self):
        return {
            'cmd_accuracy': 0.0,
            'param_accuracy': 0.0,
            'combined_score': 0.0,
        }
