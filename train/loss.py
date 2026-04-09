"""
损失函数模块 - 统一离散监督的 CAD 序列损失

设计原则：
- 命令和参数都视为离散分类任务
- EOS 仅作为普通命令类别处理，不再单独建模
- 不使用参数回归、尺度压缩或 curriculum
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


CMD_PARAM_MASK = torch.tensor([
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
], dtype=torch.bool)


class CADLoss(nn.Module):
    """统一离散监督的 CAD 损失。"""

    def __init__(
        self,
        cmd_weight=0.5,
        param_weight=0.5,
        use_cmd_mask=True,
        label_smoothing=0.05,
        class_weights=None,
        n_param_bins=256,
    ):
        super().__init__()
        self.cmd_weight = float(cmd_weight)
        self.param_weight = float(param_weight)
        self.use_cmd_mask = bool(use_cmd_mask)
        self.label_smoothing = float(label_smoothing)
        self.n_param_bins = int(n_param_bins)

        self.register_buffer('cmd_param_mask', CMD_PARAM_MASK)
        default_class_weights = torch.tensor([1.0, 1.1, 1.0, 1.5, 1.25, 1.15], dtype=torch.float32)
        if class_weights is None:
            class_weights = default_class_weights
        else:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer('class_weights', class_weights)

    def forward(self, cmd_logits, param_logits, cmd_gt, param_gt, valid_mask, progress=None):
        del progress
        valid_mask = valid_mask.bool()
        has_valid_tokens = bool(valid_mask.any())

        cmd_gt = cmd_gt.clamp(min=0, max=cmd_logits.shape[-1] - 1)
        safe_cmd_logits = torch.nan_to_num(cmd_logits, nan=0.0, posinf=20.0, neginf=-20.0)
        safe_param_logits = torch.nan_to_num(param_logits, nan=0.0, posinf=20.0, neginf=-20.0)
        param_gt = param_gt.clamp(min=0, max=self.n_param_bins - 1).long()

        flat_cmd_loss = F.cross_entropy(
            safe_cmd_logits.reshape(-1, safe_cmd_logits.shape[-1]),
            cmd_gt.reshape(-1),
            weight=self.class_weights.to(cmd_gt.device),
            reduction='none',
            label_smoothing=self.label_smoothing,
        )
        if has_valid_tokens:
            cmd_loss = flat_cmd_loss[valid_mask.reshape(-1)].mean()
        else:
            cmd_loss = flat_cmd_loss.new_zeros(())

        if has_valid_tokens:
            if self.use_cmd_mask:
                cmd_mask = self.cmd_param_mask.to(cmd_gt.device)[cmd_gt]
            else:
                cmd_mask = torch.ones_like(param_gt, dtype=torch.bool)
            combined_mask = valid_mask.unsqueeze(-1) & cmd_mask
            param_loss = self._compute_param_loss(safe_param_logits, param_gt, combined_mask)
        else:
            param_loss = safe_param_logits.new_zeros(())

        total_loss = self.cmd_weight * cmd_loss + self.param_weight * param_loss
        total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=100.0, neginf=0.0)
        loss_dict = {
            'total_loss': total_loss.detach(),
            'cmd_loss': cmd_loss.detach(),
            'param_loss': param_loss.detach(),
        }
        return total_loss, loss_dict

    def _compute_param_loss(self, param_logits, param_gt, combined_mask):
        flat_logits = param_logits.reshape(-1, param_logits.shape[-1])
        flat_targets = param_gt.reshape(-1)
        flat_loss = F.cross_entropy(
            flat_logits,
            flat_targets,
            reduction='none',
            label_smoothing=0.0,
        )
        flat_mask = combined_mask.reshape(-1)
        if flat_mask.any():
            return flat_loss[flat_mask].mean()
        return flat_loss.new_zeros(())
