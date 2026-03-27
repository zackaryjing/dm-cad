#!/usr/bin/env python3
"""
训练入口脚本

用法:
    python train_main.py --config train/config.yaml
"""

import argparse
import copy
import os
import shutil
from datetime import datetime
from pathlib import Path

import yaml

from runtime_device import apply_visible_devices, resolve_device_type


def parse_args():
    parser = argparse.ArgumentParser(description='Train Dual-Modal CAD Generator')
    parser.add_argument('--config', type=str, default='train/config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                        help='Optional device override, e.g. cuda or cpu')
    return parser.parse_args()


def _ensure_unique_run_dir(base_log_dir, config_path):
    base_path = Path(base_log_dir)
    run_prefix = Path(config_path).stem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = base_path / f'{run_prefix}_{timestamp}'
    suffix = 1
    while run_dir.exists():
        run_dir = base_path / f'{run_prefix}_{timestamp}_{suffix:02d}'
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _resolve_log_dir(config, config_path, resume_path=None):
    log_cfg = config.setdefault('log', {})
    configured_log_dir = log_cfg.get('log_dir', config.get('log_dir', 'runs/dmcad'))

    if resume_path:
        checkpoint_path = Path(resume_path).resolve()
        try:
            run_dir = checkpoint_path.parent.parent
        except IndexError as exc:
            raise ValueError(f'Invalid resume checkpoint path: {resume_path}') from exc
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f'Resuming into existing run directory: {run_dir}')
    else:
        run_dir = _ensure_unique_run_dir(configured_log_dir, config_path)
        print(f'Created run directory: {run_dir}')

    log_cfg['log_dir'] = str(run_dir)
    config['log_dir'] = str(run_dir)
    return run_dir


def _save_resolved_config(config, source_config_path, run_dir):
    resolved_config_path = Path(run_dir) / 'config.resolved.yaml'
    with resolved_config_path.open('w') as f:
        yaml.safe_dump(config, f, sort_keys=False)

    source_config = Path(source_config_path)
    copied_config_path = Path(run_dir) / source_config.name
    if source_config.resolve() != copied_config_path.resolve():
        shutil.copy2(source_config, copied_config_path)


def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = copy.deepcopy(config)

    run_dir = _resolve_log_dir(config, args.config, args.resume)
    _save_resolved_config(config, args.config, run_dir)

    visible_devices = apply_visible_devices(config)

    import torch
    from data.dataset import build_dataloader
    from train.train import Trainer

    data_cfg = config.get('data', {})
    training_cfg = config.get('training', {})
    data_root = data_cfg.get('data_root')
    if not data_root:
        raise ValueError('data_root not specified in config file')

    requested_device = resolve_device_type(config, args.device)
    if requested_device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, using CPU')
        device = 'cpu'
    else:
        device = requested_device

    if visible_devices:
        print(f'Using CUDA_VISIBLE_DEVICES={visible_devices}')
    if device == 'cuda':
        print(f'Visible CUDA devices inside process: {torch.cuda.device_count()}')

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
