"""
训练脚本 - 实现双模态 CAD 生成器训练
基于设计文档 4.2 节训练策略
"""

import os
import time
import math
from contextlib import nullcontext

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.dual_modal_cad import DualModalCADGenerator
from runtime_device import get_configured_visible_device_count
from train.loss import CADLoss


class Trainer:
    """双模态 CAD 生成器训练器"""

    def __init__(self, config, device='cuda'):
        self.config = config
        self.requested_device = device
        self.device_cfg = config.get('device', {})
        self.training_cfg = config.get('training', {})
        self.configured_visible_device_count = get_configured_visible_device_count(config)
        self.device = self._resolve_runtime_device(device)
        self.precision = self._resolve_precision_mode()
        self.autocast_enabled, self.autocast_dtype, self.use_grad_scaler = self._configure_precision()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_grad_scaler)
        self.profile_timing = bool(self.training_cfg.get('profile_timing', False))
        self.profile_steps = max(int(self.training_cfg.get('profile_steps', 20)), 1)
        self.profile_warmup_steps = max(int(self.training_cfg.get('profile_warmup_steps', 5)), 0)
        self.debug_monitor_enabled = bool(self.training_cfg.get('debug_monitor_enabled', False))
        self.debug_log_every_batches = max(int(self.training_cfg.get('debug_log_every_batches', 50)), 1)
        self.debug_fail_on_nonfinite = bool(self.training_cfg.get('debug_fail_on_nonfinite', False))
        self.global_train_step = 0

        print(
            f'Precision mode: {self.precision} '
            f'(autocast={"enabled" if self.autocast_enabled else "disabled"}, '
            f'grad_scaler={"enabled" if self.use_grad_scaler else "disabled"})'
        )

        base_model = DualModalCADGenerator(config.get('model', {}))
        self.model = self._wrap_model_for_parallel(base_model)
        self.model.to(self.device)

        loss_cfg = config.get('loss', {})
        self.criterion = CADLoss(
            cmd_weight=loss_cfg.get('cmd_weight', 1.0),
            eos_weight=loss_cfg.get('eos_weight', 0.5),
            param_weight=loss_cfg.get('param_weight', 0.5),
            use_cmd_mask=loss_cfg.get('use_cmd_mask', True),
            eos_token_id=loss_cfg.get('eos_token_id', 3),
            label_smoothing=loss_cfg.get('label_smoothing', 0.05),
            class_weights=loss_cfg.get('class_weights'),
            param_scale=loss_cfg.get('param_scale', 1.0),
            param_curriculum_start=loss_cfg.get('param_curriculum_start', 0.1),
            param_curriculum_end=loss_cfg.get('param_curriculum_end', 0.6),
            param_loss_cap=loss_cfg.get('param_loss_cap', 1.0),
        ).to(self.device)

        optimizer_cfg = config.get('optimizer', {})
        base_lr = optimizer_cfg.get('lr', 5e-5)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=base_lr,
            weight_decay=optimizer_cfg.get('weight_decay', 0.01)
        )

        scheduler_cfg = config.get('scheduler', {})
        self.lr_lambda = self._build_lr_lambda(scheduler_cfg)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)
        self._apply_lr_factor(0)

        self.epoch = 0
        self.best_val_loss = float('inf')

        log_cfg = config.get('log', {})
        self.log_dir = log_cfg.get('log_dir', config.get('log_dir', 'runs/dmcad'))
        self.writer = SummaryWriter(self.log_dir)

    def _resolve_precision_mode(self):
        precision = str(self.training_cfg.get('precision', 'fp32')).lower()
        if precision not in {'fp32', 'fp16', 'bf16'}:
            raise ValueError(f'Unsupported training.precision: {precision}')
        return precision

    def _configure_precision(self):
        if self.precision == 'fp32':
            return False, None, False

        if self.device.type != 'cuda':
            raise ValueError(f'training.precision={self.precision} requires CUDA, but runtime device is {self.device.type}')

        if self.precision == 'fp16':
            return True, torch.float16, True

        if not torch.cuda.is_bf16_supported():
            raise ValueError('training.precision=bf16 requires CUDA BF16 support, but the current device does not support it')
        return True, torch.bfloat16, False

    def _autocast_context(self):
        if not self.autocast_enabled:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)

    def _build_lr_lambda(self, scheduler_cfg):
        warmup_epochs = max(int(self.training_cfg.get('warmup_epochs', 0)), 0)
        total_epochs = max(int(self.training_cfg.get('num_epochs', scheduler_cfg.get('T_max', 1))), 1)
        eta_min = float(scheduler_cfg.get('eta_min', 1e-6))
        base_lr = float(self.optimizer.param_groups[0]['lr'])
        min_lr_ratio = min(max(eta_min / max(base_lr, 1e-12), 0.0), 1.0)

        def lr_lambda(epoch):
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return max((epoch + 1) / warmup_epochs, min_lr_ratio)

            cosine_span = max(total_epochs - warmup_epochs, 1)
            cosine_epoch = min(max(epoch - warmup_epochs, 0), cosine_span)
            cosine = 0.5 * (1.0 + math.cos(math.pi * cosine_epoch / cosine_span))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return lr_lambda

    def _training_progress(self, batch_idx, total_batches):
        total_epochs = max(int(self.training_cfg.get('progress_total_epochs', self.training_cfg.get('num_epochs', 1))), 1)
        epoch_progress = (batch_idx + 1) / max(total_batches, 1)
        return min((self.epoch + epoch_progress) / total_epochs, 1.0)

    def _apply_lr_factor(self, epoch):
        factor = self.lr_lambda(epoch)
        for group, base_lr in zip(self.optimizer.param_groups, self.scheduler.base_lrs):
            group['lr'] = base_lr * factor

    def _sync_profile_cuda(self):
        if self.device.type != 'cuda':
            return
        for device_idx in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device_idx)

    def _init_timing_stats(self):
        return {
            'count': 0,
            'data_time': 0.0,
            'h2d_time': 0.0,
            'forward_time': 0.0,
            'backward_time': 0.0,
            'optimizer_time': 0.0,
            'step_time': 0.0,
        }

    def _maybe_record_timing(self, timing_stats, batch_idx, data_time, h2d_time, forward_time, backward_time, optimizer_time, step_time):
        if not self.profile_timing:
            return
        if batch_idx < self.profile_warmup_steps:
            return
        if timing_stats['count'] >= self.profile_steps:
            return

        timing_stats['count'] += 1
        timing_stats['data_time'] += data_time
        timing_stats['h2d_time'] += h2d_time
        timing_stats['forward_time'] += forward_time
        timing_stats['backward_time'] += backward_time
        timing_stats['optimizer_time'] += optimizer_time
        timing_stats['step_time'] += step_time

    def _print_timing_summary(self, timing_stats, epoch):
        if not self.profile_timing or timing_stats['count'] == 0:
            return

        count = timing_stats['count']
        avg = {key: value / count for key, value in timing_stats.items() if key != 'count'}
        total = avg['step_time'] if avg['step_time'] > 0 else 1e-12
        print(
            f'[Timing][Epoch {epoch}] averaged over {count} steps '
            f'(after {self.profile_warmup_steps} warmup steps): '
            f'data={avg["data_time"] * 1000:.1f}ms ({avg["data_time"] / total * 100:.1f}%), '
            f'h2d={avg["h2d_time"] * 1000:.1f}ms ({avg["h2d_time"] / total * 100:.1f}%), '
            f'forward={avg["forward_time"] * 1000:.1f}ms ({avg["forward_time"] / total * 100:.1f}%), '
            f'backward={avg["backward_time"] * 1000:.1f}ms ({avg["backward_time"] / total * 100:.1f}%), '
            f'optimizer={avg["optimizer_time"] * 1000:.1f}ms ({avg["optimizer_time"] / total * 100:.1f}%), '
            f'step={avg["step_time"] * 1000:.1f}ms'
        )

    def _compute_grad_norm(self):
        total = 0.0
        for param in self.model.parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            total += grad.float().pow(2).sum().item()
        return math.sqrt(total) if total > 0 else 0.0

    def _tensor_finite_stats(self, tensor):
        detached = tensor.detach()
        finite_mask = torch.isfinite(detached)
        finite_ratio = float(finite_mask.float().mean().item()) if detached.numel() > 0 else 1.0
        nan_count = int(torch.isnan(detached).sum().item())
        posinf_count = int(torch.isposinf(detached).sum().item())
        neginf_count = int(torch.isneginf(detached).sum().item())
        if finite_mask.any():
            abs_max = float(detached[finite_mask].abs().max().item())
        else:
            abs_max = float('nan')
        return {
            'finite_ratio': finite_ratio,
            'nan_count': nan_count,
            'posinf_count': posinf_count,
            'neginf_count': neginf_count,
            'abs_max': abs_max,
            'all_finite': nan_count == 0 and posinf_count == 0 and neginf_count == 0,
        }

    def _scan_named_tensors_for_nonfinite(self, named_tensors):
        issues = []
        for name, tensor in named_tensors:
            if tensor is None:
                continue
            stats = self._tensor_finite_stats(tensor)
            if stats['all_finite']:
                continue
            issues.append((name, stats))
        return issues

    def _format_nonfinite_issues(self, issues, limit=3):
        if not issues:
            return 'none'
        parts = []
        for name, stats in issues[:limit]:
            parts.append(
                f'{name}(nan={stats["nan_count"]}, +inf={stats["posinf_count"]}, '
                f'-inf={stats["neginf_count"]}, finite_ratio={stats["finite_ratio"]:.6f})'
            )
        if len(issues) > limit:
            parts.append(f'... +{len(issues) - limit} more')
        return '; '.join(parts)

    def _debug_check_nonfinite_stage(self, stage, batch_idx, loss, lr):
        if not self.debug_monitor_enabled:
            return
        param_issues = self._scan_named_tensors_for_nonfinite(self._model_to_save().named_parameters())
        grad_issues = self._scan_named_tensors_for_nonfinite(
            (f'{name}.grad', param.grad) for name, param in self._model_to_save().named_parameters()
        )
        if not param_issues and not grad_issues:
            return

        print(
            f'[NonFiniteStage][Epoch {self.epoch} Batch {batch_idx + 1} Step {self.global_train_step}] '
            f'stage={stage} loss={loss.item():.4f} lr={lr:.8f} '
            f'params={self._format_nonfinite_issues(param_issues)} '
            f'grads={self._format_nonfinite_issues(grad_issues)}'
        )
        if self.debug_fail_on_nonfinite:
            raise RuntimeError(
                f'Non-finite tensors detected at stage={stage} epoch={self.epoch} batch={batch_idx + 1} step={self.global_train_step}'
            )

    def _maybe_fail_on_nonfinite(self, batch_idx, loss, lr, cmd_stats, param_stats, param_issues, grad_issues):
        if not self.debug_monitor_enabled:
            return
        has_issue = (
            not cmd_stats['all_finite'] or
            not param_stats['all_finite'] or
            bool(param_issues) or
            bool(grad_issues)
        )
        if not has_issue:
            return

        print(
            f'[NonFinite][Epoch {self.epoch} Batch {batch_idx + 1} Step {self.global_train_step}] '
            f'loss={loss.item():.4f} lr={lr:.8f} '
            f'cmd_logits(nan={cmd_stats["nan_count"]}, +inf={cmd_stats["posinf_count"]}, '
            f'-inf={cmd_stats["neginf_count"]}, finite_ratio={cmd_stats["finite_ratio"]:.6f}, '
            f'abs_max={cmd_stats["abs_max"]}) '
            f'param_pred(nan={param_stats["nan_count"]}, +inf={param_stats["posinf_count"]}, '
            f'-inf={param_stats["neginf_count"]}, finite_ratio={param_stats["finite_ratio"]:.6f}, '
            f'abs_max={param_stats["abs_max"]}) '
            f'params={self._format_nonfinite_issues(param_issues)} '
            f'grads={self._format_nonfinite_issues(grad_issues)}'
        )
        if self.debug_fail_on_nonfinite:
            raise RuntimeError(
                f'Non-finite tensors detected at epoch={self.epoch} batch={batch_idx + 1} step={self.global_train_step}'
            )

    def _compute_cmd_distribution(self, cmd_logits, cmd_gt, valid_mask):
        valid_mask = valid_mask.bool()
        if not valid_mask.any():
            zero = [0.0] * cmd_logits.shape[-1]
            return zero, zero

        pred = torch.argmax(cmd_logits.detach(), dim=-1)
        gt = cmd_gt.detach().clamp(min=0, max=cmd_logits.shape[-1] - 1)
        pred_valid = pred[valid_mask]
        gt_valid = gt[valid_mask]
        n_cmd = cmd_logits.shape[-1]

        pred_hist = torch.bincount(pred_valid, minlength=n_cmd).float()
        gt_hist = torch.bincount(gt_valid, minlength=n_cmd).float()
        pred_hist = (pred_hist / pred_hist.sum().clamp_min(1.0)).cpu().tolist()
        gt_hist = (gt_hist / gt_hist.sum().clamp_min(1.0)).cpu().tolist()
        return pred_hist, gt_hist

    def _log_debug_step(
        self,
        batch_idx,
        loss,
        loss_dict,
        grad_norm,
        grad_norm_clipped,
        scaler_scale,
        scaler_scale_next,
        overflow_detected,
        lr,
        cmd_stats,
        param_stats,
        param_issues,
        grad_issues,
        pred_hist,
        gt_hist
    ):
        if not self.debug_monitor_enabled:
            return
        if (batch_idx + 1) % self.debug_log_every_batches != 0:
            return

        step = self.global_train_step
        self.writer.add_scalar('debug/loss_step', loss.item(), step)
        self.writer.add_scalar('debug/cmd_loss_step', loss_dict['cmd_loss'].item(), step)
        self.writer.add_scalar('debug/eos_loss_step', loss_dict['eos_loss'].item(), step)
        self.writer.add_scalar('debug/param_loss_step', loss_dict['param_loss'].item(), step)
        self.writer.add_scalar('debug/grad_norm', grad_norm, step)
        self.writer.add_scalar('debug/grad_norm_clipped', grad_norm_clipped, step)
        self.writer.add_scalar('debug/grad_scaler_scale', scaler_scale, step)
        self.writer.add_scalar('debug/grad_scaler_scale_next', scaler_scale_next, step)
        self.writer.add_scalar('debug/amp_overflow', float(overflow_detected), step)
        self.writer.add_scalar('debug/lr_step', lr, step)
        self.writer.add_scalar('debug/cmd_logits_abs_max', cmd_stats['abs_max'], step)
        self.writer.add_scalar('debug/param_pred_abs_max', param_stats['abs_max'], step)
        self.writer.add_scalar('debug/cmd_logits_finite_ratio', cmd_stats['finite_ratio'], step)
        self.writer.add_scalar('debug/param_pred_finite_ratio', param_stats['finite_ratio'], step)
        self.writer.add_scalar('debug/cmd_logits_nan_count', cmd_stats['nan_count'], step)
        self.writer.add_scalar('debug/cmd_logits_posinf_count', cmd_stats['posinf_count'], step)
        self.writer.add_scalar('debug/cmd_logits_neginf_count', cmd_stats['neginf_count'], step)
        self.writer.add_scalar('debug/param_pred_nan_count', param_stats['nan_count'], step)
        self.writer.add_scalar('debug/param_pred_posinf_count', param_stats['posinf_count'], step)
        self.writer.add_scalar('debug/param_pred_neginf_count', param_stats['neginf_count'], step)
        self.writer.add_scalar('debug/nonfinite_param_tensor_count', len(param_issues), step)
        self.writer.add_scalar('debug/nonfinite_grad_tensor_count', len(grad_issues), step)

        for idx, value in enumerate(pred_hist):
            self.writer.add_scalar(f'debug/cmd_pred_frac_{idx}', value, step)
        for idx, value in enumerate(gt_hist):
            self.writer.add_scalar(f'debug/cmd_gt_frac_{idx}', value, step)

        pred_top = max(range(len(pred_hist)), key=lambda i: pred_hist[i]) if pred_hist else -1
        gt_top = max(range(len(gt_hist)), key=lambda i: gt_hist[i]) if gt_hist else -1
        print(
            f'[Debug][Epoch {self.epoch} Batch {batch_idx + 1} Step {step}] '
            f'loss={loss.item():.4f} cmd={loss_dict["cmd_loss"].item():.4f} '
            f'eos={loss_dict["eos_loss"].item():.4f} param={loss_dict["param_loss"].item():.4f} '
            f'grad_norm={grad_norm:.4f} grad_norm_clipped={grad_norm_clipped:.4f} '
            f'scaler={scaler_scale:.4f}->{scaler_scale_next:.4f} overflow={int(overflow_detected)} '
            f'lr={lr:.8f} cmd|max|={cmd_stats["abs_max"]:.4f} param|max|={param_stats["abs_max"]:.4f} '
            f'cmd_finite={cmd_stats["finite_ratio"]:.6f} param_finite={param_stats["finite_ratio"]:.6f} '
            f'bad_params={len(param_issues)} bad_grads={len(grad_issues)} '
            f'pred_top=cmd{pred_top}:{pred_hist[pred_top]:.3f} '
            f'gt_top=cmd{gt_top}:{gt_hist[gt_top]:.3f}'
        )

    def _resolve_runtime_device(self, requested_device):
        if requested_device != 'cuda' or not torch.cuda.is_available():
            return torch.device(requested_device)

        visible_gpu_count = torch.cuda.device_count()
        if visible_gpu_count == 0:
            return torch.device('cpu')

        output_device = int(self.device_cfg.get('output_device', 0))
        if output_device < 0 or output_device >= visible_gpu_count:
            print(
                f'Configured output_device={output_device} exceeds visible CUDA range [0, {visible_gpu_count - 1}]; '
                'falling back to cuda:0.'
            )
            output_device = 0

        return torch.device(f'cuda:{output_device}')

    def _wrap_model_for_parallel(self, model):
        if self.requested_device != 'cuda' or not torch.cuda.is_available():
            return model

        visible_gpu_count = torch.cuda.device_count()
        use_data_parallel = bool(self.device_cfg.get('use_data_parallel', False))
        if not use_data_parallel:
            return model

        if self.configured_visible_device_count is not None and self.configured_visible_device_count <= 1:
            print('Single visible device configured; disabling DataParallel and using one GPU.')
            return model

        if visible_gpu_count <= 1:
            print('DataParallel requested but fewer than 2 CUDA devices are visible; using single GPU.')
            return model

        output_device = self.device.index if self.device.index is not None else 0
        device_ids = list(range(visible_gpu_count))
        print(f'Enabling DataParallel on visible CUDA devices: {device_ids}, output_device={output_device}')
        return nn.DataParallel(model, device_ids=device_ids, output_device=output_device)

    def _model_to_save(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _load_state_dict_flexible(self, model, state_dict):
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            stripped = {
                key.replace('module.', '', 1) if key.startswith('module.') else key: value
                for key, value in state_dict.items()
            }
            model.load_state_dict(stripped)

    def train_one_epoch(self, dataloader):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        cmd_loss_total = 0.0
        eos_loss_total = 0.0
        param_loss_total = 0.0
        n_batches = len(dataloader)
        if n_batches == 0:
            raise ValueError('Training dataloader is empty')

        max_batches = self.training_cfg.get('max_train_batches')
        effective_batches = min(n_batches, max_batches) if max_batches else n_batches
        pbar = tqdm(dataloader, desc=f'Epoch {self.epoch}', total=effective_batches)

        processed_batches = 0
        skipped_batches = 0
        timing_stats = self._init_timing_stats()
        iteration_end_time = time.perf_counter()
        for batch_idx, batch in enumerate(pbar):
            step_start_time = time.perf_counter()
            data_time = step_start_time - iteration_end_time

            h2d_start_time = time.perf_counter()
            images = batch['images'].to(self.device, non_blocking=True)
            text_input_ids = batch['text_input_ids'].to(self.device, non_blocking=True)
            text_attention_mask = batch['text_attention_mask'].to(self.device, non_blocking=True)
            cad_seq = batch['cad_seq'].to(self.device, non_blocking=True)
            cad_valid_mask = batch['cad_valid_mask'].to(self.device, non_blocking=True)
            self._sync_profile_cuda()
            h2d_time = time.perf_counter() - h2d_start_time

            self.optimizer.zero_grad(set_to_none=True)
            forward_start_time = time.perf_counter()
            with self._autocast_context():
                cmd_logits, param_pred = self.model(
                    images, text_input_ids, text_attention_mask, cad_seq
                )

                cmd_gt = cad_seq[:, :, 0].long()
                param_gt = cad_seq[:, :, 1:]
                progress = self._training_progress(batch_idx, effective_batches)

                loss, loss_dict = self.criterion(
                    cmd_logits, param_pred, cmd_gt, param_gt, cad_valid_mask, progress=progress
                )
            self._sync_profile_cuda()
            forward_time = time.perf_counter() - forward_start_time
            self._debug_check_nonfinite_stage(
                stage='after_forward',
                batch_idx=batch_idx,
                loss=loss,
                lr=float(self.optimizer.param_groups[0]['lr'])
            )

            if not torch.isfinite(loss):
                skipped_batches += 1
                self.optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix({'loss': 'non-finite', 'skipped': skipped_batches})
                iteration_end_time = time.perf_counter()
                continue

            grad_clip = self.training_cfg.get('gradient_clip', 1.0)
            backward_start_time = time.perf_counter()
            scaler_scale = float(self.scaler.get_scale()) if self.use_grad_scaler else 1.0
            if self.use_grad_scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                self._debug_check_nonfinite_stage(
                    stage='after_backward_unscale',
                    batch_idx=batch_idx,
                    loss=loss,
                    lr=float(self.optimizer.param_groups[0]['lr'])
                )
                grad_norm = self._compute_grad_norm()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self._debug_check_nonfinite_stage(
                    stage='after_clip',
                    batch_idx=batch_idx,
                    loss=loss,
                    lr=float(self.optimizer.param_groups[0]['lr'])
                )
                grad_norm_clipped = self._compute_grad_norm()
            else:
                loss.backward()
                self._debug_check_nonfinite_stage(
                    stage='after_backward',
                    batch_idx=batch_idx,
                    loss=loss,
                    lr=float(self.optimizer.param_groups[0]['lr'])
                )
                grad_norm = self._compute_grad_norm()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self._debug_check_nonfinite_stage(
                    stage='after_clip',
                    batch_idx=batch_idx,
                    loss=loss,
                    lr=float(self.optimizer.param_groups[0]['lr'])
                )
                grad_norm_clipped = self._compute_grad_norm()
            self._sync_profile_cuda()
            backward_time = time.perf_counter() - backward_start_time

            optimizer_start_time = time.perf_counter()
            if self.use_grad_scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                scaler_scale_next = float(self.scaler.get_scale())
                overflow_detected = scaler_scale_next < scaler_scale
            else:
                self.optimizer.step()
                scaler_scale_next = 1.0
                overflow_detected = False
            self._debug_check_nonfinite_stage(
                stage='after_optimizer_step',
                batch_idx=batch_idx,
                loss=loss,
                lr=float(self.optimizer.param_groups[0]['lr'])
            )
            self._sync_profile_cuda()
            optimizer_time = time.perf_counter() - optimizer_start_time

            pred_hist, gt_hist = self._compute_cmd_distribution(cmd_logits, cmd_gt, cad_valid_mask)
            lr = float(self.optimizer.param_groups[0]['lr'])
            cmd_stats = self._tensor_finite_stats(cmd_logits)
            param_stats = self._tensor_finite_stats(param_pred)
            param_issues = self._scan_named_tensors_for_nonfinite(self._model_to_save().named_parameters())
            grad_issues = self._scan_named_tensors_for_nonfinite(
                (f'{name}.grad', param.grad) for name, param in self._model_to_save().named_parameters()
            )
            self._maybe_fail_on_nonfinite(
                batch_idx,
                loss,
                lr,
                cmd_stats,
                param_stats,
                param_issues,
                grad_issues
            )

            total_loss += loss.item()
            cmd_loss_total += loss_dict['cmd_loss'].item()
            eos_loss_total += loss_dict['eos_loss'].item()
            param_loss_total += loss_dict['param_loss'].item()
            processed_batches += 1
            iteration_end_time = time.perf_counter()
            step_time = iteration_end_time - step_start_time
            self.global_train_step += 1
            self._maybe_record_timing(
                timing_stats,
                batch_idx,
                data_time,
                h2d_time,
                forward_time,
                backward_time,
                optimizer_time,
                step_time
            )
            self._log_debug_step(
                batch_idx,
                loss,
                loss_dict,
                grad_norm,
                grad_norm_clipped,
                scaler_scale,
                scaler_scale_next,
                overflow_detected,
                lr,
                cmd_stats,
                param_stats,
                param_issues,
                grad_issues,
                pred_hist,
                gt_hist
            )

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cmd': f'{loss_dict["cmd_loss"].item():.4f}',
                'eos': f'{loss_dict["eos_loss"].item():.4f}',
                'param': f'{loss_dict["param_loss"].item():.4f}',
                'pw': f'{loss_dict["param_weight"].item():.3f}'
            })

            if max_batches and processed_batches >= max_batches:
                break

        if processed_batches == 0:
            raise RuntimeError(f'No valid training batches were processed; skipped_batches={skipped_batches}')

        self._print_timing_summary(timing_stats, self.epoch)

        return {
            'loss': total_loss / processed_batches,
            'cmd_loss': cmd_loss_total / processed_batches,
            'eos_loss': eos_loss_total / processed_batches,
            'param_loss': param_loss_total / processed_batches,
            'skipped_batches': skipped_batches,
            'lr': self.optimizer.param_groups[0]['lr']
        }

    @torch.no_grad()
    def evaluate(self, dataloader):
        """验证"""
        self.model.eval()
        total_loss = 0.0
        cmd_acc_total = 0.0
        param_acc_total = 0.0
        n_batches = len(dataloader)
        if n_batches == 0:
            raise ValueError('Validation dataloader is empty')

        max_batches = self.training_cfg.get('max_val_batches')
        effective_batches = min(n_batches, max_batches) if max_batches else n_batches
        processed_batches = 0

        for batch in tqdm(dataloader, desc='Validating', total=effective_batches):
            images = batch['images'].to(self.device, non_blocking=True)
            text_input_ids = batch['text_input_ids'].to(self.device, non_blocking=True)
            text_attention_mask = batch['text_attention_mask'].to(self.device, non_blocking=True)
            cad_seq = batch['cad_seq'].to(self.device, non_blocking=True)
            cad_valid_mask = batch['cad_valid_mask'].to(self.device, non_blocking=True)

            with self._autocast_context():
                cmd_logits, param_pred = self.model(
                    images, text_input_ids, text_attention_mask, cad_seq
                )

                cmd_gt = cad_seq[:, :, 0].long()
                param_gt = cad_seq[:, :, 1:]

                loss, _ = self.criterion(
                    cmd_logits, param_pred, cmd_gt, param_gt, cad_valid_mask, progress=1.0
                )
            total_loss += loss.item()
            processed_batches += 1

            if cad_valid_mask.any():
                cmd_pred = torch.argmax(cmd_logits, dim=-1)
                cmd_correct = (cmd_pred == cmd_gt.clamp(min=0))[cad_valid_mask].float().mean()
                cmd_acc_total += cmd_correct.item()

                param_error = torch.abs(param_pred - param_gt)
                cmd_mask = self.criterion.cmd_param_mask.to(param_error.device)[cmd_gt.clamp(min=0, max=self.criterion.cmd_param_mask.shape[0] - 1)]
                combined_mask = cad_valid_mask.unsqueeze(-1) & cmd_mask
                if combined_mask.any():
                    param_correct = (param_error < 0.1)[combined_mask].float().mean()
                    param_acc_total += param_correct.item()

            if max_batches and processed_batches >= max_batches:
                break

        return {
            'loss': total_loss / processed_batches,
            'cmd_acc': cmd_acc_total / processed_batches,
            'param_acc': param_acc_total / processed_batches
        }

    def train(self, train_loader, val_loader, num_epochs):
        """完整训练循环"""
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            self._apply_lr_factor(epoch)
            self.scheduler.last_epoch = epoch
            start_time = time.time()

            train_metrics = self.train_one_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            self._log_metrics(train_metrics, val_metrics, epoch)

            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint('best.pth')

            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'epoch_{epoch + 1}.pth')

            elapsed = time.time() - start_time
            print(f'Epoch {epoch}: train_loss={train_metrics["loss"]:.4f}, '
                  f'val_loss={val_metrics["loss"]:.4f}, time={elapsed:.1f}s')

        self.writer.close()

    def _log_metrics(self, train_metrics, val_metrics, epoch):
        """记录指标到 TensorBoard"""
        for name, value in train_metrics.items():
            self.writer.add_scalar(f'train/{name}', value, epoch)
        for name, value in val_metrics.items():
            self.writer.add_scalar(f'val/{name}', value, epoch)

    def save_checkpoint(self, filename):
        """保存检查点"""
        os.makedirs(os.path.join(self.log_dir, 'checkpoints'), exist_ok=True)
        checkpoint_path = os.path.join(self.log_dir, 'checkpoints', filename)
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self._model_to_save().state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }, checkpoint_path)
        print(f'Saved checkpoint to {checkpoint_path}')

    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self._load_state_dict_flexible(self._model_to_save(), checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        print(f'Loaded checkpoint from {checkpoint_path}')

    def load_model_weights(self, checkpoint_path):
        """仅从检查点加载模型参数，保留当前配置构建出的训练状态。"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self._load_state_dict_flexible(self._model_to_save(), checkpoint['model_state_dict'])
        print(f'Loaded model weights from {checkpoint_path}; optimizer, scheduler, epoch, and best_val_loss were reset for a new run')


def train_one_epoch(model, dataloader, criterion, optimizer, device, grad_clip=1.0):
    """训练一个 epoch 的函数接口"""
    model.train()
    total_loss = 0.0
    n_batches = len(dataloader)
    if n_batches == 0:
        raise ValueError('Training dataloader is empty')

    for batch in tqdm(dataloader):
        images = batch['images'].to(device)
        text_input_ids = batch['text_input_ids'].to(device)
        text_attention_mask = batch['text_attention_mask'].to(device)
        cad_seq = batch['cad_seq'].to(device)
        cad_valid_mask = batch['cad_valid_mask'].to(device)

        optimizer.zero_grad()
        cmd_logits, param_pred = model(
            images, text_input_ids, text_attention_mask, cad_seq
        )

        cmd_gt = cad_seq[:, :, 0].long()
        param_gt = cad_seq[:, :, 1:]

        loss, _ = criterion(cmd_logits, param_pred, cmd_gt, param_gt, cad_valid_mask)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """验证函数接口"""
    model.eval()
    total_loss = 0.0
    n_batches = len(dataloader)
    if n_batches == 0:
        raise ValueError('Validation dataloader is empty')

    for batch in dataloader:
        images = batch['images'].to(device)
        text_input_ids = batch['text_input_ids'].to(device)
        text_attention_mask = batch['text_attention_mask'].to(device)
        cad_seq = batch['cad_seq'].to(device)
        cad_valid_mask = batch['cad_valid_mask'].to(device)

        cmd_logits, param_pred = model(
            images, text_input_ids, text_attention_mask, cad_seq
        )

        cmd_gt = cad_seq[:, :, 0].long()
        param_gt = cad_seq[:, :, 1:]

        loss, _ = criterion(cmd_logits, param_pred, cmd_gt, param_gt, cad_valid_mask)
        total_loss += loss.item()

    return total_loss / n_batches
