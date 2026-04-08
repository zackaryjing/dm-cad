"""
损失函数模块 - 实现 CAD 序列生成损失
基于设计文档 4.1 节

改进：
- 分层目标：结构、终止、参数三部分独立建模
- 参数损失按训练进度逐步增强，避免早期回归目标主导
- 参数损失按 token 归一化，避免高维命令主导
- 参数损失直接在归一化空间中计算 Huber，减少不必要的压缩失真
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
    """CAD 序列生成损失函数"""

    def __init__(
        self,
        cmd_weight=1.0,
        eos_weight=0.5,
        param_weight=0.5,
        use_cmd_mask=True,
        eos_token_id=3,
        label_smoothing=0.05,
        class_weights=None,
        param_scale=1.0,
        param_curriculum_start=0.1,
        param_curriculum_end=0.6,
        param_loss_cap=1.0,
        param_huber_delta=0.02,
    ):
        super().__init__()
        self.cmd_weight = cmd_weight
        self.eos_weight = eos_weight
        self.param_weight = param_weight
        self.use_cmd_mask = use_cmd_mask
        self.eos_token_id = eos_token_id
        self.label_smoothing = label_smoothing
        self.param_scale = max(float(param_scale), 1e-6)
        self.param_curriculum_start = float(param_curriculum_start)
        self.param_curriculum_end = float(param_curriculum_end)
        self.param_loss_cap = max(float(param_loss_cap), 1e-6)
        self.param_huber_delta = max(float(param_huber_delta), 1e-6)

        self.register_buffer('cmd_param_mask', CMD_PARAM_MASK)
        default_class_weights = torch.tensor([1.0, 1.1, 1.0, 1.5, 1.25, 1.15], dtype=torch.float32)
        if class_weights is None:
            class_weights = default_class_weights
        else:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer('class_weights', class_weights)
        self.eos_criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, cmd_logits, param_pred, cmd_gt, param_gt, valid_mask, progress=None):
        """
        Args:
            cmd_logits: [batch, seq_len, n_commands] - 命令预测 logits
            param_pred: [batch, seq_len, n_params] - 参数预测
            cmd_gt: [batch, seq_len] - 真实命令类型
            param_gt: [batch, seq_len, n_params] - 真实参数
            valid_mask: [batch, seq_len] - 有效位置掩码
            progress: float in [0, 1] - 当前训练进度，用于参数损失 curriculum
        Returns:
            total_loss: 总损失
            loss_dict: 各分项损失字典
        """
        valid_mask = valid_mask.bool()
        has_valid_tokens = bool(valid_mask.any())
        progress_tensor = self._resolve_progress(progress, cmd_logits.device)

        cmd_gt_clamped = cmd_gt.clamp(min=0, max=cmd_logits.shape[-1] - 1)
        safe_cmd_logits = torch.nan_to_num(cmd_logits, nan=0.0, posinf=20.0, neginf=-20.0)

        flat_cmd_loss = F.cross_entropy(
            safe_cmd_logits.reshape(-1, safe_cmd_logits.shape[-1]),
            cmd_gt_clamped.reshape(-1),
            weight=self.class_weights.to(cmd_gt_clamped.device),
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        flat_valid_mask = valid_mask.reshape(-1)
        if has_valid_tokens:
            cmd_loss = flat_cmd_loss[flat_valid_mask].mean()
        else:
            cmd_loss = flat_cmd_loss.new_zeros(())

        eos_loss = self._compute_eos_loss(safe_cmd_logits, cmd_gt_clamped, valid_mask)

        if has_valid_tokens:
            if self.use_cmd_mask:
                cmd_param_mask = self.cmd_param_mask.to(cmd_gt_clamped.device)
                cmd_mask = cmd_param_mask[cmd_gt_clamped]
                combined_mask = valid_mask.unsqueeze(-1) & cmd_mask
            else:
                combined_mask = valid_mask.unsqueeze(-1).expand_as(param_pred)

            param_loss = self._compute_param_loss(
                param_pred=param_pred,
                param_gt=param_gt,
                combined_mask=combined_mask,
                valid_mask=valid_mask
            )
        else:
            param_loss = param_pred.new_zeros(())

        param_weight = self.param_weight * self._compute_param_curriculum_weight(progress_tensor)
        total_loss = (
            self.cmd_weight * cmd_loss +
            self.eos_weight * eos_loss +
            param_weight * param_loss
        )
        total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=self.param_loss_cap, neginf=0.0)
        loss_dict = {
            'total_loss': total_loss.detach(),
            'cmd_loss': cmd_loss.detach(),
            'eos_loss': eos_loss.detach(),
            'param_loss': param_loss.detach(),
            'param_weight': param_weight.detach()
        }
        return total_loss, loss_dict

    def _resolve_progress(self, progress, device):
        if progress is None:
            return torch.tensor(1.0, device=device)
        return torch.tensor(float(progress), device=device).clamp(0.0, 1.0)

    def _compute_eos_loss(self, cmd_logits, cmd_gt, valid_mask):
        eos_logits = cmd_logits[..., self.eos_token_id]
        non_eos_logits = torch.logsumexp(
            torch.cat([
                cmd_logits[..., :self.eos_token_id],
                cmd_logits[..., self.eos_token_id + 1:]
            ], dim=-1),
            dim=-1
        )
        eos_binary_logits = eos_logits - non_eos_logits
        eos_target = (cmd_gt == self.eos_token_id).float()
        flat_loss = self.eos_criterion(eos_binary_logits, eos_target)
        if valid_mask.any():
            return flat_loss[valid_mask].mean()
        return flat_loss.new_zeros(())

    def _compute_param_loss(self, param_pred, param_gt, combined_mask, valid_mask):
        safe_param_pred = torch.nan_to_num(param_pred, nan=0.0, posinf=1e4, neginf=-1e4)
        safe_param_gt = torch.nan_to_num(param_gt, nan=0.0, posinf=1e4, neginf=-1e4)

        pred_norm = safe_param_pred / self.param_scale
        gt_norm = safe_param_gt / self.param_scale
        huber = F.huber_loss(
            pred_norm,
            gt_norm,
            reduction='none',
            delta=self.param_huber_delta
        )
        bounded_loss = torch.clamp(huber, max=self.param_loss_cap)
        masked_loss = bounded_loss * combined_mask.float()
        token_param_count = combined_mask.sum(dim=-1)
        token_loss = masked_loss.sum(dim=-1) / token_param_count.clamp_min(1).float()
        valid_tokens = valid_mask & (token_param_count > 0)

        if valid_tokens.any():
            return token_loss[valid_tokens].mean()
        return bounded_loss.new_zeros(())

    def _compute_param_curriculum_weight(self, progress):
        start = min(self.param_curriculum_start, self.param_curriculum_end)
        end = max(self.param_curriculum_start, self.param_curriculum_end)
        if end <= start:
            return torch.ones((), device=progress.device)
        ramp = (progress - start) / (end - start)
        return ramp.clamp(0.0, 1.0)


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
