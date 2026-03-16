"""
训练脚本 - 实现双模态 CAD 生成器训练
基于设计文档 4.2 节训练策略
"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.dual_modal_cad import DualModalCADGenerator
from train.loss import CADLoss
from data.dataset import build_dataloader


class Trainer:
    """双模态 CAD 生成器训练器"""

    def __init__(self, config, device='cuda'):
        """
        Args:
            config: 训练配置 dict
            device: 训练设备
        """
        self.config = config
        self.device = device

        # 模型
        self.model = DualModalCADGenerator(config.get('model', {}))
        self.model.to(device)

        # 损失函数
        loss_cfg = config.get('loss', {})
        self.criterion = CADLoss(
            cmd_weight=loss_cfg.get('cmd_weight', 1.0),
            param_weight=loss_cfg.get('param_weight', 0.5)
        )

        # 优化器
        optimizer_cfg = config.get('optimizer', {})
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=optimizer_cfg.get('lr', 5e-5),
            weight_decay=optimizer_cfg.get('weight_decay', 0.01)
        )

        # 学习率调度器
        scheduler_cfg = config.get('scheduler', {})
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=scheduler_cfg.get('T_max', 80),
            eta_min=scheduler_cfg.get('eta_min', 1e-6)
        )

        # 训练状态
        self.epoch = 0
        self.best_val_loss = float('inf')

        # TensorBoard
        self.log_dir = config.get('log_dir', 'runs/dmcad')
        self.writer = SummaryWriter(self.log_dir)

    def train_one_epoch(self, dataloader):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        cmd_loss_total = 0
        param_loss_total = 0

        pbar = tqdm(dataloader, desc=f'Epoch {self.epoch}')
        for batch_idx, batch in enumerate(pbar):
            # 数据移到设备
            images = batch['images'].to(self.device)
            text_input_ids = batch['text_input_ids'].to(self.device)
            text_attention_mask = batch['text_attention_mask'].to(self.device)
            cad_seq = batch['cad_seq'].to(self.device)
            cad_valid_mask = batch['cad_valid_mask'].to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            cmd_logits, param_pred = self.model(
                images, text_input_ids, text_attention_mask, cad_seq
            )

            # 准备 ground truth
            cmd_gt = cad_seq[:, :, 0].long()
            param_gt = cad_seq[:, :, 1:]

            # 计算损失
            loss, loss_dict = self.criterion(
                cmd_logits, param_pred, cmd_gt, param_gt, cad_valid_mask
            )

            # 反向传播
            loss.backward()

            # 梯度裁剪
            grad_clip = self.config.get('training', {}).get('gradient_clip', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            self.optimizer.step()

            # 更新统计
            total_loss += loss.item()
            cmd_loss_total += loss_dict['cmd_loss'].item()
            param_loss_total += loss_dict['param_loss'].item()

            # 进度条更新
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cmd': f'{loss_dict["cmd_loss"].item():.4f}',
                'param': f'{loss_dict["param_loss"].item():.4f}'
            })

        # 计算平均损失
        n_batches = len(dataloader)
        return {
            'loss': total_loss / n_batches,
            'cmd_loss': cmd_loss_total / n_batches,
            'param_loss': param_loss_total / n_batches
        }

    @torch.no_grad()
    def evaluate(self, dataloader):
        """验证"""
        self.model.eval()
        total_loss = 0
        cmd_acc_total = 0
        param_acc_total = 0

        for batch in tqdm(dataloader, desc='Validating'):
            images = batch['images'].to(self.device)
            text_input_ids = batch['text_input_ids'].to(self.device)
            text_attention_mask = batch['text_attention_mask'].to(self.device)
            cad_seq = batch['cad_seq'].to(self.device)
            cad_valid_mask = batch['cad_valid_mask'].to(self.device)

            cmd_logits, param_pred = self.model(
                images, text_input_ids, text_attention_mask, cad_seq
            )

            cmd_gt = cad_seq[:, :, 0].long()
            param_gt = cad_seq[:, :, 1:]

            loss, _ = self.criterion(
                cmd_logits, param_pred, cmd_gt, param_gt, cad_valid_mask
            )
            total_loss += loss.item()

            # 计算命令准确率
            cmd_pred = torch.argmax(cmd_logits, dim=-1)
            cmd_correct = (cmd_pred == cmd_gt)[cad_valid_mask].sum()
            cmd_total = cad_valid_mask.sum()
            if cmd_total > 0:
                cmd_acc_total += cmd_correct.item() / cmd_total.item()

            # 计算参数准确率
            if cad_valid_mask.sum() > 0:
                param_error = torch.abs(param_pred - param_gt)[cad_valid_mask]
                param_correct = (param_error < 0.1).float().mean()
                param_acc_total += param_correct.item()

        n_batches = len(dataloader)
        return {
            'loss': total_loss / n_batches,
            'cmd_acc': cmd_acc_total / n_batches,
            'param_acc': param_acc_total / n_batches
        }

    def train(self, train_loader, val_loader, num_epochs):
        """完整训练循环"""
        for epoch in range(num_epochs):
            self.epoch = epoch
            start_time = time.time()

            # 训练
            train_metrics = self.train_one_epoch(train_loader)

            # 验证
            val_metrics = self.evaluate(val_loader)

            # 记录日志
            self._log_metrics(train_metrics, val_metrics, epoch)

            # 保存最佳模型
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint('best.pth')

            # 保存 epoch 检查点
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'epoch_{epoch}.pth')

            # 更新学习率
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
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }, checkpoint_path)
        print(f'Saved checkpoint to {checkpoint_path}')

    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        print(f'Loaded checkpoint from {checkpoint_path}')


def train_one_epoch(model, dataloader, criterion, optimizer, device, grad_clip=1.0):
    """训练一个 epoch 的函数接口"""
    model.train()
    total_loss = 0

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

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """验证函数接口"""
    model.eval()
    total_loss = 0

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

    return total_loss / len(dataloader)
