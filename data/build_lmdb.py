#!/usr/bin/env python3
"""
将散文件 CAD 数据集打包为 LMDB。

默认行为：
- 读取 ids 文件收集样本
- 每个 sample 写成一个 LMDB record
- record 内容包含 8 视图 PNG bytes、文本描述、CAD 向量

推荐用法：
    python data/build_lmdb.py \
        --data-root datasets/dataset_v0 \
        --output datasets/dataset_v0/cad_data.lmdb \
        --ids-files train_ids.txt test_ids.txt
"""

import argparse
import os
import pickle
import shutil
from datetime import datetime, timezone

import numpy as np
from tqdm import tqdm

try:
    import lmdb
except ImportError as exc:
    raise SystemExit('lmdb is required to run this script. Please install lmdb first.') from exc

from data.dataset import (
    DEFAULT_LMDB_FILENAME,
    load_cad_array_from_files,
    load_ids_list,
    load_image_bytes_from_files,
    preload_text_descriptions,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Build LMDB dataset for DM-CAD')
    parser.add_argument('--data-root', type=str, required=True,
                        help='源数据目录，例如 datasets/dataset_v0')
    parser.add_argument('--output', type=str, default=None,
                        help=f'输出 LMDB 路径，默认 <data-root>/{DEFAULT_LMDB_FILENAME}')
    parser.add_argument('--ids-files', type=str, nargs='*', default=None,
                        help='需要打包的 ids 文件列表；相对路径基于 data-root')
    parser.add_argument('--map-size-gb', type=float, default=64.0,
                        help='LMDB map size in GB，默认 64')
    parser.add_argument('--commit-every', type=int, default=1024,
                        help='每写入多少个样本提交一次事务，默认 1024')
    parser.add_argument('--force', action='store_true',
                        help='若输出路径已存在则先删除')
    return parser.parse_args()


def discover_ids_files(data_root, explicit_ids_files):
    if explicit_ids_files:
        return explicit_ids_files

    preferred = ['train_ids.txt', 'test_ids.txt']
    resolved = [name for name in preferred if os.path.exists(os.path.join(data_root, name))]
    if resolved:
        return resolved

    fallback = sorted(
        filename for filename in os.listdir(data_root)
        if filename.endswith('.txt') and 'ids' in filename
    )
    if fallback:
        return fallback

    raise FileNotFoundError(f'No ids files found under {data_root}')


def collect_sample_ids(data_root, ids_files):
    all_ids = []
    seen = set()
    for ids_file in ids_files:
        sample_ids = load_ids_list(data_root, ids_file=ids_file)
        print(f'Loaded {len(sample_ids)} ids from {ids_file}')
        for sample_id in sample_ids:
            if sample_id not in seen:
                seen.add(sample_id)
                all_ids.append(sample_id)
    return all_ids


def build_record(sample_id, data_root, text_cache):
    group_id, sample_name = sample_id.split('/', 1)
    image_bytes = load_image_bytes_from_files(data_root, group_id, sample_name)
    cad_seq = load_cad_array_from_files(data_root, group_id, sample_name)
    text = text_cache.get(group_id, {}).get(sample_id, '')

    return {
        'sample_id': sample_id,
        'text': text,
        'cad_seq': np.asarray(cad_seq, dtype=np.float32),
        'image_bytes': image_bytes,
    }


def ensure_output_path(output_path, force):
    if not os.path.exists(output_path):
        return
    if not force:
        raise FileExistsError(f'Output path already exists: {output_path}. Use --force to overwrite.')
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    else:
        os.remove(output_path)


def main():
    args = parse_args()

    data_root = os.path.abspath(args.data_root)
    output_path = os.path.abspath(args.output or os.path.join(data_root, DEFAULT_LMDB_FILENAME))
    ids_files = discover_ids_files(data_root, args.ids_files)
    sample_ids = collect_sample_ids(data_root, ids_files)
    if not sample_ids:
        raise ValueError('No sample ids collected, aborting.')

    ensure_output_path(output_path, args.force)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f'Building text cache for {len(sample_ids)} samples...')
    text_cache = preload_text_descriptions(data_root, sample_ids)

    env = lmdb.open(
        output_path,
        map_size=int(args.map_size_gb * (1024 ** 3)),
        subdir=True,
        meminit=False,
        map_async=True,
        writemap=False,
        lock=True,
    )

    meta = {
        'version': 1,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'data_root': data_root,
        'sample_count': len(sample_ids),
        'ids_files': ids_files,
        'format': {
            'text': 'utf-8 string',
            'cad_seq': 'float32 ndarray [seq_len, 20]',
            'image_bytes': 'list[bytes|None], len=8',
        },
    }

    print(f'Writing {len(sample_ids)} samples to {output_path} ...')
    txn = env.begin(write=True)
    written = 0
    try:
        txn.put(b'__meta__', pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL))
        for index, sample_id in enumerate(tqdm(sample_ids, desc='LMDB build'), start=1):
            record = build_record(sample_id, data_root, text_cache)
            txn.put(
                sample_id.encode('utf-8'),
                pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)
            )
            written += 1

            if index % args.commit_every == 0:
                txn.commit()
                txn = env.begin(write=True)

        txn.commit()
        txn = None
        env.sync()
    finally:
        if txn is not None:
            txn.abort()
        env.close()

    print(f'LMDB build completed: {output_path}')
    print(f'Written samples: {written}')
    print('Use the following config fields to enable it during training:')
    print('  data.backend: lmdb')
    print(f'  data.lmdb_path: {os.path.relpath(output_path, data_root)}')


if __name__ == '__main__':
    main()
