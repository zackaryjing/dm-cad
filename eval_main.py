#!/usr/bin/env python3
"""
评估入口脚本

用法:
    python eval_main.py --checkpoint checkpoints/best.pth --data_dir datasets/dataset_v0
"""

import argparse
import torch
from models.dual_modal_cad import DualModalCADGenerator
from eval.evaluate import Evaluator
from data.dataset import build_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Dual-Modal CAD Generator')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='datasets/dataset_v0',
                        help='Path to data directory (default: datasets/dataset_v0)')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to evaluate (default: test)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for evaluation')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save generated sequences')
    return parser.parse_args()


def main():
    args = parse_args()

    # 设置设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, using CPU')
        device = 'cpu'
    else:
        device = args.device

    # 加载模型
    print(f'Loading model from {args.checkpoint}...')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = DualModalCADGenerator(checkpoint.get('config', {}))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # 构建数据加载器
    print(f'Loading {args.split} data from {args.data_dir}...')
    test_loader = build_dataloader(
        data_root=args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=4
    )
    print(f'  Loaded {len(test_loader.dataset)} samples')

    # 创建评估器
    evaluator = Evaluator(model, device=device)

    # 评估
    print('Evaluating...')
    metrics = evaluator.evaluate(test_loader)

    # 打印结果
    print('\n=== Evaluation Results ===')
    for name, value in metrics.items():
        print(f'{name}: {value:.4f}')

    # 保存生成结果
    if args.output:
        print(f'Saving generated sequences to {args.output}...')
        evaluator.generate_and_save(test_loader, args.output)

    print('Evaluation completed!')


if __name__ == '__main__':
    main()
