"""
损失函数模块 - 实现 CAD 序列生成损失
基于设计文档 4.1 节

改进：根据命令类型，只对"有效参数维度"计算 Loss
"""

import torch
import torch.nn as nn


# 命令类型定义（适配 DeepCAD 原始数据）:
#   Line=0, Arc=1, Circle=2, EOS=3, SOL=4, Ext=5

# 参数维度说明（20 维参数，索引 0-19）：
#   [0:5] = Sketch 参数：x, y, alpha, f, r
#   [5:20] = Extrude 参数 (15 维)

# 命令 - 参数有效性掩码 (19 维参数)
# 每个命令只在其有效参数维度上计算 loss，避免无效梯度噪声
# 基于 DeepCAD 原始 6 种命令类型
CMD_PARAM_MASK = torch.tensor([
    # 0   1   2   3   4   | 5   6   7   8   9   10  11  12  13  14  15  16  17  18
    [1,  1,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # Line(0): x, y (终点坐标)
    [1,  1,  1,  1,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # Arc(1): x, y, alpha, f
    [1,  1,  0,  0,  1,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # Circle(2): x, y, r
    [0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # EOS(3): 无参数
    [0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # SOL(4): 无参数
    [0,  0,  0,  0,  0,   1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0],  # Ext(5): 11 维挤压参数
])


class CADLoss(nn.Module):
    """CAD 序列生成损失函数

    包含命令类型损失和参数预测损失
    改进：只对命令的有效参数维度计算损失
    """
    def __init__(self, cmd_weight=1.0, param_weight=0.5, use_cmd_mask=True):
        """
        Args:
            cmd_weight: 命令损失权重
            param_weight: 参数损失权重
            use_cmd_mask: 是否使用命令类型掩码（只对有效参数计算损失）
        """
        super().__init__()
        self.cmd_weight = cmd_weight
        self.param_weight = param_weight
        self.use_cmd_mask = use_cmd_mask

        self.cmd_criterion = nn.CrossEntropyLoss()
        self.param_criterion = nn.SmoothL1Loss()

        # 注册命令 - 参数掩码（buffer 不计算梯度）
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
        batch, seq_len, n_params = param_pred.shape

        # 命令损失
        cmd_loss = self.cmd_criterion(
            cmd_logits.view(-1, cmd_logits.shape[-1]),
            cmd_gt.view(-1)
        )

        # 参数损失 - 改进版本：只对有效参数维度计算损失
        if valid_mask.sum() > 0:
            if self.use_cmd_mask:
                # 确保 cmd_gt 在有效范围内，避免索引越界
                cmd_gt_clamped = cmd_gt.clamp(min=0, max=len(self.cmd_param_mask) - 1)
                # 将掩码移到与 cmd_gt 相同的设备
                cmd_param_mask = self.cmd_param_mask.to(cmd_gt.device)
                # 获取每个位置的命令类型掩码 [batch, seq_len, n_params]
                cmd_mask = cmd_param_mask[cmd_gt_clamped]  # [B, T, 19]

                # 结合有效掩码和命令掩码
                combined_mask = valid_mask.unsqueeze(-1) & (cmd_mask == 1)

                # 只计算有效位置的损失
                if combined_mask.sum() > 0:
                    param_loss = self.param_criterion(
                        param_pred[combined_mask],
                        param_gt[combined_mask]
                    )
                else:
                    param_loss = torch.tensor(0.0, device=param_pred.device)
            else:
                # 原始版本：对所有参数计算损失
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
