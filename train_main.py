#!/usr/bin/env python3
"""
训练入口脚本

用法:
    python train_main.py --config train/config.yaml --data_dir datasets/dataset_v0
"""

import argparse
import yaml
import torch
from train.train import Trainer
from data.dataset import build_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description='Train Dual-Modal CAD Generator')
    parser.add_argument('--config', type=str, default='train/config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='datasets/dataset_v0',
                        help='Path to data directory (default: datasets/dataset_v0)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training')
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 设置设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, using CPU')
        device = 'cpu'
    else:
        device = args.device

    # 创建训练器
    trainer = Trainer(config, device=device)

    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # 构建数据加载器
    print(f'Loading training data from {args.data_dir}...')
    train_loader = build_dataloader(
        data_root=args.data_dir,
        split='train',
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
    )
    print(f'  Loaded {len(train_loader.dataset)} samples')

    print(f'Loading validation data from {args.data_dir}...')
    val_loader = build_dataloader(
        data_root=args.data_dir,
        split='test',  # 使用 test 集作为验证集
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
    )
    print(f'  Loaded {len(val_loader.dataset)} samples')

    # 开始训练
    print(f'Starting training for {config["training"]["num_epochs"]} epochs...')
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=config['training']['num_epochs']
    )

    print('Training completed!')


if __name__ == '__main__':
    main()
