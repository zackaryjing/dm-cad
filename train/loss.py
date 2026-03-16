"""
损失函数模块 - 实现 CAD 序列生成损失
基于设计文档 4.1 节
"""

import torch
import torch.nn as nn


class CADLoss(nn.Module):
    """CAD 序列生成损失函数

    包含命令类型损失和参数预测损失
    """
    def __init__(self, cmd_weight=1.0, param_weight=0.5):
        """
        Args:
            cmd_weight: 命令损失权重
            param_weight: 参数损失权重
        """
        super().__init__()
        self.cmd_weight = cmd_weight
        self.param_weight = param_weight

        self.cmd_criterion = nn.CrossEntropyLoss()
        self.param_criterion = nn.SmoothL1Loss()

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
        # 命令损失
        cmd_loss = self.cmd_criterion(
            cmd_logits.view(-1, cmd_logits.shape[-1]),
            cmd_gt.view(-1)
        )

        # 参数损失 (仅对有效命令计算)
        if valid_mask.sum() > 0:
            param_loss = self.param_criterion(
                param_pred[valid_mask],
                param_gt[valid_mask]
            )
        else:
            param_loss = torch.tensor(0.0, device=param_pred.device)

        # 总损失
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
        # 基础损失
        total_loss, loss_dict = super().forward(
            cmd_logits, param_pred, cmd_gt, param_gt, valid_mask
        )

        # 可以添加命令类型特定的加权逻辑
        # 例如：extrude 命令比 sketch 命令更重要

        return total_loss, loss_dict
