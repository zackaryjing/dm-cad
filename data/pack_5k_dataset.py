#!/usr/bin/env python3
"""
打包 5k 子集数据到 zip 文件
包含：
- train_ids_5k.txt, test_ids_5k.txt
- 对应的 cad_desc (按 group_id 去重)
- 对应的 cad_img (完整目录结构)
- 对应的 cad_vec

用法:
    python data/pack_5k_dataset.py --output datasets/dataset_v1.zip
"""

import os
import json
import shutil
import zipfile
import argparse
from pathlib import Path
from collections import defaultdict


def load_ids(filepath):
    """加载 IDs 文件"""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def pack_5k_dataset(source_dir, output_path, train_ids_file, test_ids_file):
    """
    打包 5k 子集数据集

    Args:
        source_dir: 源数据目录 (dataset_v0)
        output_path: 输出 zip 文件路径
        train_ids_file: 训练集 IDs 文件
        test_ids_file: 测试集 IDs 文件
    """
    source_dir = Path(source_dir)
    output_path = Path(output_path)

    # 加载 IDs
    print(f"Loading train IDs from {train_ids_file}...")
    train_ids = load_ids(train_ids_file)
    print(f"  {len(train_ids)} samples")

    print(f"Loading test IDs from {test_ids_file}...")
    test_ids = load_ids(test_ids_file)
    print(f"  {len(test_ids)} samples")

    all_ids = train_ids + test_ids
    print(f"Total: {len(all_ids)} samples")

    # 按 group_id 分组
    groups = defaultdict(list)
    for sample_id in all_ids:
        group_id = sample_id.split('/')[0]
        groups[group_id].append(sample_id)

    print(f"\n涉及 {len(groups)} 个组")

    # 收集需要的文本描述 (按 group_id)
    needed_desc_groups = set(groups.keys())
    print(f"需要 {len(needed_desc_groups)} 个 cad_desc 文件")

    # 收集需要的图像目录
    needed_img_dirs = set()
    for sample_id in all_ids:
        group_id, sample_name = sample_id.split('/')
        needed_img_dirs.add(f"{group_id}/{sample_name}")
    print(f"需要 {len(needed_img_dirs)} 个图像目录")

    # 收集需要的向量文件
    needed_vec_files = set()
    for sample_id in all_ids:
        group_id, sample_name = sample_id.split('/')
        needed_vec_files.add(f"{group_id}/{sample_name}.h5")
    print(f"需要 {len(needed_vec_files)} 个 cad_vec 文件")

    # 创建临时目录
    temp_dir = source_dir.parent / "dataset_v1_temp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    print(f"\n正在复制文件到临时目录 {temp_dir}...")

    # 复制 train_ids_5k.txt 和 test_ids_5k.txt
    shutil.copy(train_ids_file, temp_dir / "train_ids_5k.txt")
    shutil.copy(test_ids_file, temp_dir / "test_ids_5k.txt")
    print(f"  ✓ IDs 文件")

    # 复制 cad_desc
    desc_dir = temp_dir / "cad_desc"
    desc_dir.mkdir()
    for group_id in needed_desc_groups:
        src = source_dir / "cad_desc" / f"{group_id}.json"
        dst = desc_dir / f"{group_id}.json"
        if src.exists():
            shutil.copy(src, dst)
    print(f"  ✓ cad_desc: {len(list(desc_dir.glob('*.json')))} 文件")

    # 复制 cad_img (保持目录结构)
    img_dir = temp_dir / "cad_img"
    img_dir.mkdir()
    img_count = 0
    for item in needed_img_dirs:
        group_id, sample_name = item.split('/')
        src = source_dir / "cad_img" / group_id / sample_name
        dst = img_dir / group_id / sample_name
        if src.exists():
            dst.mkdir(parents=True, exist_ok=True)
            for img_file in src.glob("*.png"):
                shutil.copy(img_file, dst / img_file.name)
                img_count += 1
    print(f"  ✓ cad_img: {len(needed_img_dirs)} 目录，{img_count} 图像文件")

    # 复制 cad_vec (保持目录结构)
    vec_dir = temp_dir / "cad_vec"
    vec_dir.mkdir()
    vec_count = 0
    for item in needed_vec_files:
        group_id, filename = item.split('/')
        src = source_dir / "cad_vec" / group_id / filename
        dst = vec_dir / group_id / filename
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
            vec_count += 1
    print(f"  ✓ cad_vec: {vec_count} 文件")

    # 创建 zip
    print(f"\n正在创建 zip 文件 {output_path}...")
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(temp_dir.parent)
                zipf.write(file_path, arcname)

    # 获取 zip 大小
    zip_size = output_path.stat().st_size / (1024**3)  # GB
    print(f"  ✓ zip 文件大小：{zip_size:.2f} GB")

    # 清理临时目录
    print(f"\n清理临时目录...")
    shutil.rmtree(temp_dir)

    print(f"\n完成！数据集已打包到：{output_path}")
    print(f"\n使用 scp 下载到本地:")
    print(f"  scp root@server:{output_path} ./dataset_v1.zip")


def main():
    parser = argparse.ArgumentParser(description='Pack 5k subset dataset')
    parser.add_argument('--source', type=str, default='datasets/dataset_v0',
                        help='源数据目录')
    parser.add_argument('--output', type=str, default='datasets/dataset_v1.zip',
                        help='输出 zip 文件路径')
    parser.add_argument('--train-ids', type=str, default='datasets/dataset_v0/train_ids_5k.txt',
                        help='训练集 IDs 文件')
    parser.add_argument('--test-ids', type=str, default='datasets/dataset_v0/test_ids_5k.txt',
                        help='测试集 IDs 文件')
    args = parser.parse_args()

    pack_5k_dataset(args.source, args.output, args.train_ids, args.test_ids)


if __name__ == "__main__":
    main()
