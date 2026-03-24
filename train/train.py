"""
训练脚本 - 实现双模态 CAD 生成器训练
基于设计文档 4.2 节训练策略
"""

import os
import time

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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
            param_weight=loss_cfg.get('param_weight', 0.5)
        ).to(self.device)

        optimizer_cfg = config.get('optimizer', {})
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=optimizer_cfg.get('lr', 5e-5),
            weight_decay=optimizer_cfg.get('weight_decay', 0.01)
        )

        scheduler_cfg = config.get('scheduler', {})
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=scheduler_cfg.get('T_max', 80),
            eta_min=scheduler_cfg.get('eta_min', 1e-6)
        )

        self.epoch = 0
        self.best_val_loss = float('inf')

        log_cfg = config.get('log', {})
        self.log_dir = log_cfg.get('log_dir', config.get('log_dir', 'runs/dmcad'))
        self.writer = SummaryWriter(self.log_dir)

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
        param_loss_total = 0.0
        n_batches = len(dataloader)
        if n_batches == 0:
            raise ValueError('Training dataloader is empty')

        max_batches = self.training_cfg.get('max_train_batches')
        effective_batches = min(n_batches, max_batches) if max_batches else n_batches
        pbar = tqdm(dataloader, desc=f'Epoch {self.epoch}', total=effective_batches)

        processed_batches = 0
        for batch in pbar:
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

                loss, loss_dict = self.criterion(
                    cmd_logits, param_pred, cmd_gt, param_gt, cad_valid_mask
                )

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
            param_loss_total += loss_dict['param_loss'].item()
            processed_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cmd': f'{loss_dict["cmd_loss"].item():.4f}',
                'param': f'{loss_dict["param_loss"].item():.4f}'
            })

            if max_batches and processed_batches >= max_batches:
                break

        return {
            'loss': total_loss / processed_batches,
            'cmd_loss': cmd_loss_total / processed_batches,
            'param_loss': param_loss_total / processed_batches
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
                    cmd_logits, param_pred, cmd_gt, param_gt, cad_valid_mask
                )
            total_loss += loss.item()
            processed_batches += 1

            if cad_valid_mask.any():
                cmd_pred = torch.argmax(cmd_logits, dim=-1)
                cmd_correct = (cmd_pred == cmd_gt.clamp(min=0))[cad_valid_mask].float().mean()
                cmd_acc_total += cmd_correct.item()

                param_error = torch.abs(param_pred - param_gt)
                param_correct = (param_error < 0.1)[cad_valid_mask].float().mean()
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
            start_time = time.time()

            train_metrics = self.train_one_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            self._log_metrics(train_metrics, val_metrics, epoch)

            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint('best.pth')

            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'epoch_{epoch + 1}.pth')

            self.scheduler.step()

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
