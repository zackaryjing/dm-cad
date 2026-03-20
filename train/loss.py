"""
损失函数模块 - 实现 CAD 序列生成损失
基于设计文档 4.1 节

改进：根据命令类型，只对有效参数维度计算 Loss
"""

import torch
import torch.nn as nn


CMD_PARAM_MASK = torch.tensor([
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
], dtype=torch.bool)


class CADLoss(nn.Module):
    """CAD 序列生成损失函数"""

    def __init__(self, cmd_weight=1.0, param_weight=0.5, use_cmd_mask=True):
        super().__init__()
        self.cmd_weight = cmd_weight
        self.param_weight = param_weight
        self.use_cmd_mask = use_cmd_mask

        self.cmd_criterion = nn.CrossEntropyLoss(reduction='none')
        self.param_criterion = nn.SmoothL1Loss(reduction='none')
        self.register_buffer('cmd_param_mask', CMD_PARAM_MASK)

    def forward(self, cmd_logits, param_pred, cmd_gt, param_gt, valid_mask):
        """
        Args:
            cmd_logits: [batch, seq_len, n_commands] - 命令预测 logits
            param_pred: [batch, seq_len, n_params] - 参数预测
            cmd_gt: [batch, seq_len] - 真实命令类型
            param_gt: [batch, seq_len, n_params] - 真实参数
            valid_mask: [batch, seq_len] - 有效位置掩码
        Returns:
            total_loss: 总损失
            loss_dict: 各分项损失字典
        """
        valid_mask = valid_mask.bool()
        has_valid_tokens = bool(valid_mask.any())

        cmd_gt_clamped = cmd_gt.clamp(min=0, max=cmd_logits.shape[-1] - 1)
        flat_cmd_loss = self.cmd_criterion(
            cmd_logits.reshape(-1, cmd_logits.shape[-1]),
            cmd_gt_clamped.reshape(-1)
        )
        flat_valid_mask = valid_mask.reshape(-1)
        if has_valid_tokens:
            cmd_loss = flat_cmd_loss[flat_valid_mask].mean()
        else:
            cmd_loss = flat_cmd_loss.new_zeros(())

        if has_valid_tokens:
            if self.use_cmd_mask:
                cmd_param_mask = self.cmd_param_mask.to(cmd_gt_clamped.device)
                cmd_mask = cmd_param_mask[cmd_gt_clamped]
                combined_mask = valid_mask.unsqueeze(-1) & cmd_mask
            else:
                combined_mask = valid_mask.unsqueeze(-1).expand_as(param_pred)

            param_loss_all = self.param_criterion(param_pred, param_gt)
            if combined_mask.any():
                param_loss = param_loss_all[combined_mask].mean()
            else:
                param_loss = param_loss_all.new_zeros(())
        else:
            param_loss = param_pred.new_zeros(())

        total_loss = self.cmd_weight * cmd_loss + self.param_weight * param_loss
        loss_dict = {
            'total_loss': total_loss.detach(),
            'cmd_loss': cmd_loss.detach(),
            'param_loss': param_loss.detach()
        }
        return total_loss, loss_dict


class WeightedCADLoss(CADLoss):
    """加权 CAD 损失 - 对不同命令类型使用不同权重"""

    def __init__(self, cmd_weight=1.0, param_weight=0.5,
                 sketch_weight=1.0, extrude_weight=1.5):
        super().__init__(cmd_weight, param_weight)
        self.sketch_weight = sketch_weight
        self.extrude_weight = extrude_weight

    def forward(self, cmd_logits, param_pred, cmd_gt, param_gt, valid_mask):
        total_loss, loss_dict = super().forward(
            cmd_logits, param_pred, cmd_gt, param_gt, valid_mask
        )
        return total_loss, loss_dict
