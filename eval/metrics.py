"""
评估指标模块 - 离散 exact-match 风格的 CAD 评估
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


def compute_exact_match_metrics(pred_cmds, pred_params, gt_cmds, gt_params, valid_mask=None):
    cmd_mask_table = CMD_PARAM_MASK.to(pred_cmds.device)
    gt_cmds = gt_cmds.clamp(min=0, max=cmd_mask_table.shape[0] - 1)

    if valid_mask is None:
        valid_mask = torch.ones_like(gt_cmds, dtype=torch.bool)
    else:
        valid_mask = valid_mask.bool()

    if not valid_mask.any():
        return CADMetrics().empty_metrics()

    cmd_correct = (pred_cmds == gt_cmds) & valid_mask
    cmd_token_acc = cmd_correct[valid_mask].float().mean()

    param_mask = cmd_mask_table[gt_cmds]
    combined_mask = valid_mask.unsqueeze(-1) & param_mask
    if combined_mask.any():
        param_correct = (pred_params == gt_params) & combined_mask
        param_token_acc = param_correct[combined_mask].float().mean()
    else:
        param_token_acc = torch.tensor(0.0, device=pred_cmds.device)

    token_param_correct = (~combined_mask | (pred_params == gt_params)).all(dim=-1)
    token_exact = cmd_correct & token_param_correct
    token_exact_acc = token_exact[valid_mask].float().mean()

    sequence_cmd_exact_acc = (cmd_correct | ~valid_mask).all(dim=-1).float().mean()
    sequence_exact_acc = (token_exact | ~valid_mask).all(dim=-1).float().mean()

    return {
        'cmd_token_acc': cmd_token_acc.item(),
        'param_token_acc': param_token_acc.item(),
        'token_exact_acc': token_exact_acc.item(),
        'sequence_cmd_exact_acc': sequence_cmd_exact_acc.item(),
        'sequence_exact_acc': sequence_exact_acc.item(),
    }


class CADMetrics:
    """CAD 生成综合评估指标。"""

    def compute_all_metrics(self, pred_cmds, pred_params, gt_cmds, gt_params, valid_mask=None):
        return compute_exact_match_metrics(pred_cmds, pred_params, gt_cmds, gt_params, valid_mask)

    def empty_metrics(self):
        return {
            'cmd_token_acc': 0.0,
            'param_token_acc': 0.0,
            'token_exact_acc': 0.0,
            'sequence_cmd_exact_acc': 0.0,
            'sequence_exact_acc': 0.0,
        }
