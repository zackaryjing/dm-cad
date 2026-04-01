#!/usr/bin/env python3
"""
评估入口脚本

用法:
    python eval_main.py --checkpoint checkpoints/best.pth
    python eval_main.py --checkpoint checkpoints/best.pth --config train/config.yaml
"""

import argparse
from datetime import datetime
from pathlib import Path

import yaml

from runtime_device import apply_visible_devices, resolve_device_type


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Dual-Modal CAD Generator')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (optional, for dataloader settings)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to data directory (default: from config or datasets/dataset_v1)')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to evaluate (default: test)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for evaluation (default: from config or 32)')
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Maximum number of evaluation batches (default: from config or all)')
    parser.add_argument('--device', type=str, default=None,
                        help='Optional device override, e.g. cuda or cpu')
    parser.add_argument('--output', type=str, default=None,
                        help='Optional path to save generated sequences; relative paths are resolved under the run eval directory')
    parser.add_argument('--metrics_output', type=str, default=None,
                        help='Optional path to save evaluation metrics; relative paths are resolved under the run eval directory')
    return parser.parse_args()


def _load_state_dict_flexible(model, state_dict):
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        stripped = {
            key.replace('module.', '', 1) if key.startswith('module.') else key: value
            for key, value in state_dict.items()
        }
        model.load_state_dict(stripped)


def _resolve_run_eval_dir(checkpoint_path):
    checkpoint_path = Path(checkpoint_path).resolve()
    run_dir = checkpoint_path.parent.parent
    eval_dir = run_dir / 'eval'
    eval_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, eval_dir


def _resolve_output_path(path_value, eval_dir):
    if path_value is None:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = eval_dir / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _default_metrics_path(checkpoint_path, split, eval_dir):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_stem = Path(checkpoint_path).stem
    return eval_dir / f'{checkpoint_stem}_{split}_{timestamp}.metrics.yaml'


def _save_metrics(metrics, metrics_path, args, data_root, ids_file):
    payload = {
        'checkpoint': args.checkpoint,
        'config': args.config,
        'data_dir': data_root,
        'split': args.split,
        'ids_file': ids_file,
        'batch_size': args.batch_size,
        'max_batches': args.max_batches,
        'device': args.device,
        'metrics': metrics,
    }
    with metrics_path.open('w') as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def main():
    args = parse_args()

    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f'Loaded config from {args.config}')

    visible_devices = apply_visible_devices(config)

    import torch
    from data.dataset import build_dataloader
    from eval.evaluate import Evaluator
    from models.dual_modal_cad import DualModalCADGenerator

    data_cfg = config.get('data', {})
    training_cfg = config.get('training', {})

    requested_device = resolve_device_type(config, args.device)
    if requested_device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, using CPU')
        device = 'cpu'
    else:
        device = requested_device

    if visible_devices:
        print(f'Using CUDA_VISIBLE_DEVICES={visible_devices}')

    run_dir, eval_dir = _resolve_run_eval_dir(args.checkpoint)
    metrics_output_path = _resolve_output_path(args.metrics_output, eval_dir)
    if metrics_output_path is None:
        metrics_output_path = _default_metrics_path(args.checkpoint, args.split, eval_dir)
    generated_output_path = _resolve_output_path(args.output, eval_dir)

    print(f'Loading model from {args.checkpoint}...')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = checkpoint.get('config', {}).get('model', checkpoint.get('config', {}))
    model = DualModalCADGenerator(model_config)
    _load_state_dict_flexible(model, checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    data_root = args.data_dir or data_cfg.get('data_root', 'datasets/dataset_v1')
    batch_size = args.batch_size or training_cfg.get('batch_size', 32)
    max_batches = args.max_batches or training_cfg.get('max_val_batches')
    ids_file = data_cfg.get(f'{args.split}_ids_file')
    if ids_file is None and args.split == 'test':
        ids_file = data_cfg.get('test_ids_file')

    print(f'Run directory: {run_dir}')
    print(f'Evaluation outputs directory: {eval_dir}')
    print(f'Loading {args.split} data from {data_root}...')
    test_loader = build_dataloader(
        data_root=data_root,
        split=args.split,
        batch_size=batch_size,
        num_workers=training_cfg.get('num_workers', 4),
        ids_file=ids_file,
        img_size=data_cfg.get('img_size', 224),
        text_max_len=data_cfg.get('text_max_len', 64),
        backend=data_cfg.get('backend', 'files'),
        lmdb_path=data_cfg.get('lmdb_path'),
        pin_memory=data_cfg.get('pin_memory', True),
        persistent_workers=data_cfg.get('persistent_workers'),
        prefetch_factor=data_cfg.get('prefetch_factor', 1),
        max_prefetch_gb=data_cfg.get('max_prefetch_gb', 8.0),
    )
    if ids_file:
        print(f'  Using ids file: {ids_file}')
    print(f'  Loaded {len(test_loader.dataset)} samples')
    if max_batches:
        print(f'  Limiting evaluation to {max_batches} batches')

    evaluator = Evaluator(model, device=device)

    print('Evaluating...')
    metrics = evaluator.evaluate(test_loader, max_batches=max_batches)

    print('\n=== Evaluation Results ===')
    for name, value in metrics.items():
        print(f'{name}: {value:.4f}')

    _save_metrics(metrics, metrics_output_path, args, data_root, ids_file)
    print(f'Saved metrics to {metrics_output_path}')

    if generated_output_path:
        print(f'Saving generated sequences to {generated_output_path}...')
        evaluator.generate_and_save(test_loader, str(generated_output_path), max_batches=max_batches)

    print('Evaluation completed!')


if __name__ == '__main__':
    main()
