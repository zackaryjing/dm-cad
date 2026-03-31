"""
训练脚本 - 实现双模态 CAD 生成器训练
基于设计文档 4.2 节训练策略
"""

import os
import time
import math

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
        self.use_amp = bool(self.training_cfg.get('use_amp', False)) and self.device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        if self.training_cfg.get('use_amp', False) and self.device.type != 'cuda':
            print('AMP requested but CUDA is not active; running without AMP.')
        elif self.use_amp:
            print('AMP mixed precision is enabled for training and evaluation.')

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
        base_lr = optimizer_cfg.get('lr', self.training_cfg.get('lr', 5e-5))
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
        total_epochs = max(int(self.training_cfg.get('num_epochs', 1)), 1)
        epoch_progress = (batch_idx + 1) / max(total_batches, 1)
        return min((self.epoch + epoch_progress) / total_epochs, 1.0)

    def _apply_lr_factor(self, epoch):
        factor = self.lr_lambda(epoch)
        for group, base_lr in zip(self.optimizer.param_groups, self.scheduler.base_lrs):
            group['lr'] = base_lr * factor

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
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(self.device, non_blocking=True)
            text_input_ids = batch['text_input_ids'].to(self.device, non_blocking=True)
            text_attention_mask = batch['text_attention_mask'].to(self.device, non_blocking=True)
            cad_seq = batch['cad_seq'].to(self.device, non_blocking=True)
            cad_valid_mask = batch['cad_valid_mask'].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                cmd_logits, param_pred = self.model(
                    images, text_input_ids, text_attention_mask, cad_seq
                )

                cmd_gt = cad_seq[:, :, 0].long()
                param_gt = cad_seq[:, :, 1:]
                progress = self._training_progress(batch_idx, effective_batches)

                loss, loss_dict = self.criterion(
                    cmd_logits, param_pred, cmd_gt, param_gt, cad_valid_mask, progress=progress
                )

            if not torch.isfinite(loss):
                skipped_batches += 1
                self.optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix({'loss': 'non-finite', 'skipped': skipped_batches})
                continue

            grad_clip = self.training_cfg.get('gradient_clip', 1.0)
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()

            total_loss += loss.item()
            cmd_loss_total += loss_dict['cmd_loss'].item()
            eos_loss_total += loss_dict['eos_loss'].item()
            param_loss_total += loss_dict['param_loss'].item()
            processed_batches += 1

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

            with torch.cuda.amp.autocast(enabled=self.use_amp):
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
