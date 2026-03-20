#!/usr/bin/env python3
"""
训练入口脚本

用法:
    python train_main.py --config train/config.yaml
"""

import argparse

import torch
import yaml

from data.dataset import build_dataloader
from train.train import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train Dual-Modal CAD Generator')
    parser.add_argument('--config', type=str, default='train/config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training')
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    data_cfg = config.get('data', {})
    training_cfg = config.get('training', {})
    data_root = data_cfg.get('data_root')
    if not data_root:
        raise ValueError('data_root not specified in config file')

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, using CPU')
        device = 'cpu'
    else:
        device = args.device

    trainer = Trainer(config, device=device)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    print(f'Loading training data from {data_root}...')
    train_ids_file = data_cfg.get('train_ids_file')
    train_loader = build_dataloader(
        data_root=data_root,
        split='train',
        batch_size=training_cfg['batch_size'],
        num_workers=training_cfg['num_workers'],
        ids_file=train_ids_file,
        img_size=data_cfg.get('img_size', 224),
        text_max_len=data_cfg.get('text_max_len', 64),
    )
    if train_ids_file:
        print(f'  Using ids file: {train_ids_file}')
    print(f'  Loaded {len(train_loader.dataset)} samples')

    print(f'Loading validation data from {data_root}...')
    test_ids_file = data_cfg.get('test_ids_file')
    val_loader = build_dataloader(
        data_root=data_root,
        split='test',
        batch_size=training_cfg['batch_size'],
        num_workers=training_cfg['num_workers'],
        ids_file=test_ids_file,
        img_size=data_cfg.get('img_size', 224),
        text_max_len=data_cfg.get('text_max_len', 64),
    )
    if test_ids_file:
        print(f'  Using ids file: {test_ids_file}')
    print(f'  Loaded {len(val_loader.dataset)} samples')

    print(f'Starting training for {training_cfg["num_epochs"]} epochs...')
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=training_cfg['num_epochs']
    )

    print('Training completed!')


if __name__ == '__main__':
    main()
