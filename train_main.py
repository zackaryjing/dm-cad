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

from runtime_device import apply_visible_devices, get_distributed_env, resolve_device_type


def parse_args():
    parser = argparse.ArgumentParser(description='Train Dual-Modal CAD Generator')
    parser.add_argument('--config', type=str, default='train/config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--no-resume-in-place', action='store_true',
                        help='When used with --resume, create a new run directory and only initialize model weights from the checkpoint')
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


def _resolve_log_dir(config, config_path, resume_path=None, resume_in_place=True):
    log_cfg = config.setdefault('log', {})
    configured_log_dir = log_cfg.get('log_dir', config.get('log_dir', 'runs/dmcad'))

    if resume_path and resume_in_place:
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


def _init_distributed_if_needed(config, requested_device):
    dist_env = get_distributed_env()
    if not dist_env['enabled']:
        return dist_env

    import torch
    import torch.distributed as dist

    if requested_device != 'cuda':
        raise ValueError('Distributed training currently requires CUDA')
    if not torch.cuda.is_available():
        raise ValueError('Distributed training requested but CUDA is not available')

    torch.cuda.set_device(dist_env['local_rank'])
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    return dist_env


def _destroy_distributed_if_needed(dist_env):
    if not dist_env.get('enabled'):
        return

    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def _sync_run_dir(config, config_path, resume_path, resume_in_place, dist_env):
    if not dist_env.get('enabled'):
        run_dir = _resolve_log_dir(config, config_path, resume_path, resume_in_place=resume_in_place)
        _save_resolved_config(config, config_path, run_dir)
        return run_dir

    import torch.distributed as dist

    shared = [None]
    if dist_env['is_main_process']:
        run_dir = _resolve_log_dir(config, config_path, resume_path, resume_in_place=resume_in_place)
        _save_resolved_config(config, config_path, run_dir)
        shared[0] = str(run_dir)
    dist.broadcast_object_list(shared, src=0)
    run_dir = Path(shared[0])
    config.setdefault('log', {})['log_dir'] = str(run_dir)
    config['log_dir'] = str(run_dir)
    dist.barrier()
    return run_dir


def _rank_print(dist_env, *args, **kwargs):
    if dist_env.get('is_main_process', True):
        print(*args, **kwargs)


def main():
    args = parse_args()
    dist_env = None

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        config = copy.deepcopy(config)

        resume_in_place = not args.no_resume_in_place
        if args.no_resume_in_place and not args.resume:
            print('--no-resume-in-place has no effect without --resume; proceeding with a new run directory.')

        visible_devices = apply_visible_devices(config)

        import torch
        from data.dataset import build_dataloader
        from train.train import Trainer

        requested_device = resolve_device_type(config, args.device)
        dist_env = _init_distributed_if_needed(config, requested_device)

        data_cfg = config.get('data', {})
        training_cfg = config.get('training', {})
        data_root = data_cfg.get('data_root')
        if not data_root:
            raise ValueError('data_root not specified in config file')
        data_backend = data_cfg.get('backend', 'files')
        lmdb_path = data_cfg.get('lmdb_path')

        if requested_device == 'cuda' and not torch.cuda.is_available():
            _rank_print(dist_env, 'CUDA not available, using CPU')
            device = 'cpu'
        else:
            device = requested_device

        run_dir = _sync_run_dir(config, args.config, args.resume, resume_in_place, dist_env)

        if visible_devices:
            _rank_print(dist_env, f'Using CUDA_VISIBLE_DEVICES={visible_devices}')
        if device == 'cuda':
            _rank_print(dist_env, f'Visible CUDA devices inside process: {torch.cuda.device_count()}')
        if dist_env.get('enabled'):
            _rank_print(
                dist_env,
                f'Distributed training enabled: world_size={dist_env["world_size"]}, '
                f'rank={dist_env["rank"]}, local_rank={dist_env["local_rank"]}'
            )

        trainer = Trainer(config, device=device)

        if args.resume:
            if resume_in_place:
                trainer.load_checkpoint(args.resume)
            else:
                trainer.load_model_weights(args.resume)

        default_num_workers = training_cfg['num_workers']
        train_num_workers = int(training_cfg.get('train_num_workers', default_num_workers))
        val_num_workers = int(training_cfg.get('val_num_workers', default_num_workers))

        _rank_print(dist_env, f'Loading training data from {data_root}...')
        _rank_print(dist_env, f'  Dataset backend: {data_backend}')
        if data_backend == 'lmdb':
            _rank_print(dist_env, f'  LMDB path: {lmdb_path or "cad_data.lmdb"}')
        _rank_print(dist_env, f'  Requested num_workers: {train_num_workers}')
        train_ids_file = data_cfg.get('train_ids_file')
        train_loader = build_dataloader(
            data_root=data_root,
            split='train',
            batch_size=training_cfg['batch_size'],
            num_workers=train_num_workers,
            ids_file=train_ids_file,
            img_size=data_cfg.get('img_size', 224),
            text_max_len=data_cfg.get('text_max_len', 64),
            backend=data_cfg.get('backend', 'files'),
            lmdb_path=data_cfg.get('lmdb_path'),
            pin_memory=data_cfg.get('pin_memory', True),
            persistent_workers=data_cfg.get('persistent_workers'),
            prefetch_factor=data_cfg.get('prefetch_factor', 1),
            max_prefetch_gb=data_cfg.get('max_prefetch_gb', 8.0),
            distributed=dist_env.get('enabled', False),
            rank=dist_env.get('rank', 0),
            world_size=dist_env.get('world_size', 1),
        )
        if train_ids_file:
            _rank_print(dist_env, f'  Using ids file: {train_ids_file}')
        _rank_print(dist_env, f'  Effective num_workers: {train_loader.num_workers}')
        if train_loader.num_workers > 0:
            _rank_print(dist_env, f'  Prefetch factor: {train_loader.prefetch_factor}')
        _rank_print(dist_env, f'  Estimated image memory per batch: {train_loader.estimated_batch_gb:.2f} GiB')
        if train_loader.num_workers > 0:
            _rank_print(dist_env, f'  Estimated prefetched batches: {train_loader.estimated_prefetched_batches}')
            _rank_print(dist_env, f'  Estimated prefetched image memory: {train_loader.estimated_prefetch_gb:.2f} GiB')
            _rank_print(dist_env, f'  Configured prefetch memory cap: {train_loader.max_prefetch_gb:.1f} GiB')
        _rank_print(dist_env, f'  Loaded {len(train_loader.dataset)} samples')

        _rank_print(dist_env, f'Loading validation data from {data_root}...')
        _rank_print(dist_env, f'  Dataset backend: {data_backend}')
        if data_backend == 'lmdb':
            _rank_print(dist_env, f'  LMDB path: {lmdb_path or "cad_data.lmdb"}')
        _rank_print(dist_env, f'  Requested num_workers: {val_num_workers}')
        test_ids_file = data_cfg.get('test_ids_file')
        val_loader = build_dataloader(
            data_root=data_root,
            split='test',
            batch_size=training_cfg['batch_size'],
            num_workers=val_num_workers,
            ids_file=test_ids_file,
            img_size=data_cfg.get('img_size', 224),
            text_max_len=data_cfg.get('text_max_len', 64),
            backend=data_cfg.get('backend', 'files'),
            lmdb_path=data_cfg.get('lmdb_path'),
            pin_memory=data_cfg.get('pin_memory', True),
            persistent_workers=data_cfg.get('persistent_workers'),
            prefetch_factor=data_cfg.get('prefetch_factor', 1),
            max_prefetch_gb=data_cfg.get('max_prefetch_gb', 8.0),
            distributed=dist_env.get('enabled', False),
            rank=dist_env.get('rank', 0),
            world_size=dist_env.get('world_size', 1),
        )
        if test_ids_file:
            _rank_print(dist_env, f'  Using ids file: {test_ids_file}')
        _rank_print(dist_env, f'  Effective num_workers: {val_loader.num_workers}')
        if val_loader.num_workers > 0:
            _rank_print(dist_env, f'  Prefetch factor: {val_loader.prefetch_factor}')
        _rank_print(dist_env, f'  Estimated image memory per batch: {val_loader.estimated_batch_gb:.2f} GiB')
        if val_loader.num_workers > 0:
            _rank_print(dist_env, f'  Estimated prefetched batches: {val_loader.estimated_prefetched_batches}')
            _rank_print(dist_env, f'  Estimated prefetched image memory: {val_loader.estimated_prefetch_gb:.2f} GiB')
            _rank_print(dist_env, f'  Configured prefetch memory cap: {val_loader.max_prefetch_gb:.1f} GiB')
        _rank_print(dist_env, f'  Loaded {len(val_loader.dataset)} samples')

        _rank_print(dist_env, f'Starting training for {training_cfg["num_epochs"]} epochs...')
        trainer.train(
            train_loader,
            val_loader,
            num_epochs=training_cfg['num_epochs']
        )

        _rank_print(dist_env, 'Training completed!')
    finally:
        if dist_env is not None:
            _destroy_distributed_if_needed(dist_env)


if __name__ == '__main__':
    main()
