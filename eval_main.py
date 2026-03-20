#!/usr/bin/env python3
"""
评估入口脚本

用法:
    python eval_main.py --checkpoint checkpoints/best.pth
    python eval_main.py --checkpoint checkpoints/best.pth --config train/config.yaml
"""

import argparse

import torch
import yaml

from data.dataset import build_dataloader
from eval.evaluate import Evaluator
from models.dual_modal_cad import DualModalCADGenerator


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
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for evaluation')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save generated sequences')
    return parser.parse_args()


def main():
    args = parse_args()

    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f'Loaded config from {args.config}')

    data_cfg = config.get('data', {})
    training_cfg = config.get('training', {})

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, using CPU')
        device = 'cpu'
    else:
        device = args.device

    print(f'Loading model from {args.checkpoint}...')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = checkpoint.get('config', {}).get('model', checkpoint.get('config', {}))
    model = DualModalCADGenerator(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    data_root = args.data_dir or data_cfg.get('data_root', 'datasets/dataset_v1')
    batch_size = args.batch_size or training_cfg.get('batch_size', 32)
    max_batches = args.max_batches or training_cfg.get('max_val_batches')
    ids_file = data_cfg.get(f'{args.split}_ids_file')
    if ids_file is None and args.split == 'test':
        ids_file = data_cfg.get('test_ids_file')

    print(f'Loading {args.split} data from {data_root}...')
    test_loader = build_dataloader(
        data_root=data_root,
        split=args.split,
        batch_size=batch_size,
        num_workers=training_cfg.get('num_workers', 4),
        ids_file=ids_file,
        img_size=data_cfg.get('img_size', 224),
        text_max_len=data_cfg.get('text_max_len', 64),
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

    if args.output:
        print(f'Saving generated sequences to {args.output}...')
        evaluator.generate_and_save(test_loader, args.output, max_batches=max_batches)

    print('Evaluation completed!')


if __name__ == '__main__':
    main()
