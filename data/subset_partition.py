#!/usr/bin/env python3
"""
从已有 train/test 划分中生成子集划分

用途：
- 生成小规模数据集用于快速验证（dataloader 测试、loss 下降测试等）

生成的划分：
- 5k:  train_ids_5k.txt (4k) + test_ids_5k.txt (1k)
- 20k: train_ids_20k.txt (16k) + test_ids_20k.txt (4k)

用法:
    python data/subset_partition.py --data_dir datasets/dataset_v0 --seed 42
"""

import os
import random
import argparse


def load_ids(filepath):
    """加载 IDs 文件"""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def save_ids(filepath, ids):
    """保存 IDs 到文件"""
    with open(filepath, 'w') as f:
        for id_ in ids:
            f.write(f"{id_}\n")
    print(f"Saved {len(ids)} IDs to {filepath}")


def stratified_sample(ids, n_samples, seed=42):
    """
    分层采样 - 按 group_id 保持原始分布
    """
    random.seed(seed)

    # 按 group_id 分组
    groups = {}
    for sample_id in ids:
        group_id = sample_id.split('/')[0]
        if group_id not in groups:
            groups[group_id] = []
        groups[group_id].append(sample_id)

    # 计算每组采样数量
    total = len(ids)
    sampled = []
    for group_id in sorted(groups.keys()):
        group_ids = groups[group_id]
        # 按比例采样
        n_group = max(1, int(len(group_ids) * n_samples / total))
        # 如果组大小小于采样数，则全取
        n_group = min(n_group, len(group_ids))
        random.shuffle(group_ids)
        sampled.extend(group_ids[:n_group])

    # 如果采样不足，从剩余中补充
    remaining = n_samples - len(sampled)
    if remaining > 0:
        all_sampled = set(sampled)
        pool = [s for s in ids if s not in all_sampled]
        random.shuffle(pool)
        sampled.extend(pool[:remaining])

    return sampled


def main():
    parser = argparse.ArgumentParser(description='Generate subset partitions from existing train/test split')
    parser.add_argument('--data_dir', type=str, default='datasets/dataset_v0',
                        help='Path to data directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--subset-sizes', type=int, nargs='+', default=[5000, 20000],
                        help='Target total sizes for subsets')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                        help='Test set ratio within each subset')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(base_dir, args.data_dir)

    # 加载原始划分
    train_file = os.path.join(data_dir, 'train_ids.txt')
    test_file = os.path.join(data_dir, 'test_ids.txt')

    print(f"Loading train IDs from {train_file}...")
    train_ids = load_ids(train_file)
    print(f"  Loaded {len(train_ids)} train samples")

    print(f"Loading test IDs from {test_file}...")
    test_ids = load_ids(test_file)
    print(f"  Loaded {len(test_ids)} test samples")

    all_ids = train_ids + test_ids
    print(f"Total available: {len(all_ids)} samples")

    # 生成各规模子集
    for target_size in args.subset_sizes:
        print(f"\n{'='*60}")
        print(f"Generating {target_size//1000}k subset (target: {target_size} total)...")
        print(f"{'='*60}")

        n_test = int(target_size * args.test_ratio)
        n_train = target_size - n_test

        # 从原始 train 中采样
        train_sampled = stratified_sample(train_ids, n_train, seed=args.seed)

        # 从原始 test 中采样
        test_sampled = stratified_sample(test_ids, n_test, seed=args.seed + 1)

        # 保存
        suffix = f"_{target_size//1000}k" if target_size >= 1000 else f"_{target_size}"

        train_out = os.path.join(data_dir, f'train_ids{suffix}.txt')
        test_out = os.path.join(data_dir, f'test_ids{suffix}.txt')

        save_ids(train_out, train_sampled)
        save_ids(test_out, test_sampled)

        print(f"\nSubset {target_size//1000}k summary:")
        print(f"  Train: {len(train_sampled)}")
        print(f"  Test:  {len(test_sampled)}")
        print(f"  Total: {len(train_sampled) + len(test_sampled)}")

    print(f"\n{'='*60}")
    print("Done! Usage in your code:")
    print(f"{'='*60}")
    print("""
# 使用 5k 子集测试 dataloader
from data.dataset import CADDataset
train_ds = CADDataset('datasets/dataset_v0', ids_file='datasets/dataset_v0/train_ids_5k.txt')
test_ds = CADDataset('datasets/dataset_v0', ids_file='datasets/dataset_v0/test_ids_5k.txt')

# 使用 20k 子集训练看 loss 下降
train_ds = CADDataset('datasets/dataset_v0', ids_file='datasets/dataset_v0/train_ids_20k.txt')
test_ds = CADDataset('datasets/dataset_v0', ids_file='datasets/dataset_v0/test_ids_20k.txt')
""")


if __name__ == "__main__":
    main()
