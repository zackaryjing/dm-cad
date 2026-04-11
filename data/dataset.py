#!/usr/bin/env python3
"""
CAD 数据集类 - 实现双模态 CAD 数据加载
基于设计文档 5.3 节

支持两种后端：
- files: 原始散文件目录
- lmdb: 单库 KV 读取，减少大量随机小文件 I/O

目录结构适配:
- 图像：cad_img/{group_id}/{sample_name}/{sample_name}_{000-007}.png
- 文本：cad_desc/{group_id}.json (按 id 字段查找)
- CAD: cad_vec/{group_id}/{sample_name}.h5
"""

import io
import json
import os
import pickle

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from transformers import BertTokenizer

try:
    import lmdb
except ImportError:
    lmdb = None


DEFAULT_LMDB_FILENAME = 'cad_data.lmdb'
DEFAULT_MAX_PREFETCH_GB = 8.0


def resolve_ids_path(data_root, split='train', ids_file=None):
    """解析 ids 文件路径。"""
    if ids_file is not None:
        if os.path.isabs(ids_file):
            return ids_file
        return os.path.join(data_root, ids_file)
    return os.path.join(data_root, f'{split}_ids.txt')


def load_ids_list(data_root, split='train', ids_file=None):
    """加载样本 ID 列表。"""
    ids_path = resolve_ids_path(data_root, split=split, ids_file=ids_file)
    if os.path.exists(ids_path):
        with open(ids_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    print(f'Warning: ids file not found: {ids_path}')
    return []


def _normalize_cad_array(data):
    """将 CAD 向量规整为 [seq_len, 20] float32。"""
    if len(data.shape) == 1:
        data = data.reshape(-1, 17)

    if data.shape[1] == 17:
        padded = np.zeros((data.shape[0], 20), dtype=np.float32)
        padded[:, :17] = data.astype(np.float32)
        data = padded
    elif data.shape[1] != 20:
        raise ValueError(f'Unexpected CAD vector shape: {data.shape}')
    else:
        data = data.astype(np.float32, copy=False)

    return data


def load_cad_array_from_h5(vec_path):
    """从 h5 读取 CAD 向量。"""
    with h5py.File(vec_path, 'r') as f:
        for key in ['cad_seq', 'sequence', 'data', 'vec']:
            if key in f:
                data = f[key][:]
                break
        else:
            key = list(f.keys())[0]
            data = f[key][:]
    return _normalize_cad_array(data)


def load_cad_array_from_files(data_root, group_id, sample_name):
    """从散文件目录加载 CAD 向量。"""
    vec_path = os.path.join(data_root, 'cad_vec', group_id, f'{sample_name}.h5')
    if os.path.exists(vec_path):
        try:
            return load_cad_array_from_h5(vec_path)
        except Exception as e:
            print(f'Warning: Failed to load CAD vector {vec_path}: {e}')

    return np.zeros((0, 20), dtype=np.float32)


def load_image_bytes_from_files(data_root, group_id, sample_name, n_views=8):
    """从散文件目录读取多视图原始 PNG bytes。"""
    img_dir = os.path.join(data_root, 'cad_img', group_id, sample_name)
    image_bytes = []
    for i in range(n_views):
        img_path = os.path.join(img_dir, f'{sample_name}_{i:03d}.png')
        if os.path.exists(img_path):
            with open(img_path, 'rb') as f:
                image_bytes.append(f.read())
        else:
            image_bytes.append(None)
    return image_bytes


def preload_text_descriptions(data_root, sample_ids):
    """预加载所需 group 的文本描述。"""
    needed_groups = set()
    for sample_id in sample_ids:
        parts = sample_id.split('/')
        if len(parts) >= 2:
            needed_groups.add(parts[0])

    text_cache = {}
    for group_id in needed_groups:
        desc_file = os.path.join(data_root, 'cad_desc', f'{group_id}.json')
        if os.path.exists(desc_file):
            with open(desc_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            text_cache[group_id] = {
                entry['id']: entry.get('text caption', '')
                for entry in data if 'id' in entry
            }
        else:
            text_cache[group_id] = {}
    return text_cache


def resolve_lmdb_path(data_root, lmdb_path=None):
    """解析 LMDB 路径。"""
    if lmdb_path is None:
        return os.path.join(data_root, DEFAULT_LMDB_FILENAME)
    if os.path.isabs(lmdb_path):
        return lmdb_path
    return os.path.join(data_root, lmdb_path)


class CADDataset(Dataset):
    """双模态 CAD 数据集。"""

    def __init__(
        self,
        data_root,
        split='train',
        img_size=224,
        text_max_len=64,
        tokenizer_name='bert-base-uncased',
        ids_file=None,
        backend='files',
        lmdb_path=None,
        lmdb_readers=128,
    ):
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        self.text_max_len = text_max_len
        self.backend = backend
        self.lmdb_readers = lmdb_readers
        self.n_views = 8

        self.image_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.data_list = load_ids_list(data_root, split=split, ids_file=ids_file)
        self.text_cache = {}
        self._lmdb_env = None
        self._lmdb_txn = None
        self._lmdb_path = None

        self.backend = self._resolve_backend(backend, lmdb_path)
        if self.backend == 'files':
            self.text_cache = preload_text_descriptions(self.data_root, self.data_list)
        elif self.backend == 'lmdb':
            self._lmdb_path = resolve_lmdb_path(self.data_root, lmdb_path)
            self._validate_lmdb_available()
        else:
            raise ValueError(f'Unsupported dataset backend: {self.backend}')

    def _resolve_backend(self, backend, lmdb_path):
        if backend != 'auto':
            return backend

        resolved_lmdb_path = resolve_lmdb_path(self.data_root, lmdb_path)
        if os.path.exists(resolved_lmdb_path):
            return 'lmdb'
        return 'files'

    def _validate_lmdb_available(self):
        if lmdb is None:
            raise ImportError('lmdb is required for backend="lmdb". Please install lmdb first.')
        if not os.path.exists(self._lmdb_path):
            raise FileNotFoundError(f'LMDB path not found: {self._lmdb_path}')

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_lmdb_env'] = None
        state['_lmdb_txn'] = None
        return state

    def __del__(self):
        self._close_lmdb()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample_id = self.data_list[idx]
        parts = sample_id.split('/')
        if len(parts) < 2:
            raise ValueError(f'Invalid sample id format: {sample_id}')

        if self.backend == 'lmdb':
            images, text, cad_seq, cad_valid_mask = self._load_from_lmdb(sample_id)
        else:
            group_id, sample_name = parts[0], parts[1]
            images = self._load_images_from_files(group_id, sample_name)
            text = self._get_text(sample_id, group_id)
            cad_seq, cad_valid_mask = self._load_cad_vector_from_files(group_id, sample_name)

        text_encoding = self.tokenizer(
            text,
            max_length=self.text_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        text_input_ids = text_encoding['input_ids'].squeeze(0)
        text_attention_mask = text_encoding['attention_mask'].squeeze(0)

        return {
            'sample_id': sample_id,
            'images': images,
            'text': text,
            'text_input_ids': text_input_ids,
            'text_attention_mask': text_attention_mask,
            'cad_seq': cad_seq,
            'cad_valid_mask': cad_valid_mask
        }

    def _ensure_lmdb_open(self):
        if self._lmdb_env is not None and self._lmdb_txn is not None:
            return

        self._close_lmdb()

        self._validate_lmdb_available()
        self._lmdb_env = lmdb.open(
            self._lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=self.lmdb_readers,
            subdir=os.path.isdir(self._lmdb_path),
        )
        self._lmdb_txn = self._lmdb_env.begin(write=False)

    def _close_lmdb(self):
        txn = getattr(self, '_lmdb_txn', None)
        env = getattr(self, '_lmdb_env', None)

        if txn is not None:
            try:
                txn.abort()
            except Exception:
                pass
            self._lmdb_txn = None

        if env is not None:
            try:
                env.close()
            except Exception:
                pass
            self._lmdb_env = None

    def _load_from_lmdb(self, sample_id):
        self._ensure_lmdb_open()
        payload = self._lmdb_txn.get(sample_id.encode('utf-8'))
        if payload is None:
            raise KeyError(f'Sample id not found in LMDB: {sample_id}')

        record = pickle.loads(payload)
        image_bytes_list = record.get('image_bytes', [])
        images = self._load_images_from_bytes(image_bytes_list)

        text = record.get('text', '')
        cad_array = np.asarray(record.get('cad_seq', np.zeros((0, 20), dtype=np.float32)), dtype=np.float32)
        cad_seq = torch.from_numpy(cad_array).float()
        cad_valid_mask = (cad_seq[:, 0] >= 0).bool()
        return images, text, cad_seq, cad_valid_mask

    def _load_images_from_files(self, group_id, sample_name):
        image_bytes_list = load_image_bytes_from_files(
            self.data_root, group_id, sample_name, n_views=self.n_views
        )
        return self._load_images_from_bytes(image_bytes_list)

    def _load_images_from_bytes(self, image_bytes_list):
        images = []
        for image_bytes in image_bytes_list[:self.n_views]:
            if image_bytes:
                img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                img = self.image_transform(img)
            else:
                img = torch.ones(3, self.img_size, self.img_size)
            images.append(img)

        while len(images) < self.n_views:
            images.append(torch.ones(3, self.img_size, self.img_size))
        return torch.stack(images)

    def _get_text(self, sample_id, group_id):
        if group_id in self.text_cache and sample_id in self.text_cache[group_id]:
            return self.text_cache[group_id][sample_id]
        return ''

    def _load_cad_vector_from_files(self, group_id, sample_name):
        cad_array = load_cad_array_from_files(self.data_root, group_id, sample_name)
        cad_seq = torch.from_numpy(cad_array).float()
        cad_valid_mask = (cad_seq[:, 0] >= 0).bool()
        return cad_seq, cad_valid_mask


def collate_fn(batch):
    """Batch 数据合并。"""
    sample_ids = [item['sample_id'] for item in batch]
    images = torch.stack([item['images'] for item in batch])
    texts = [item['text'] for item in batch]
    text_input_ids = torch.stack([item['text_input_ids'] for item in batch])
    text_attention_mask = torch.stack([item['text_attention_mask'] for item in batch])

    max_seq_len = max((item['cad_seq'].shape[0] for item in batch), default=0)
    cad_seqs = []
    cad_valid_masks = []
    for item in batch:
        cad_seq = item['cad_seq']
        valid_mask = item['cad_valid_mask']

        if cad_seq.shape[0] < max_seq_len:
            pad_len = max_seq_len - cad_seq.shape[0]
            pad_seq = torch.zeros(pad_len, 20, dtype=cad_seq.dtype)
            pad_seq[:, 0] = -1
            pad_mask = torch.zeros(pad_len, dtype=torch.bool)
            cad_seq = torch.cat([cad_seq, pad_seq], dim=0)
            valid_mask = torch.cat([valid_mask, pad_mask], dim=0)

        cad_seqs.append(cad_seq)
        cad_valid_masks.append(valid_mask)

    if max_seq_len == 0:
        cad_seqs = torch.zeros(len(batch), 0, 20, dtype=torch.float32)
        cad_valid_masks = torch.zeros(len(batch), 0, dtype=torch.bool)
    else:
        cad_seqs = torch.stack(cad_seqs)
        cad_valid_masks = torch.stack(cad_valid_masks)

    return {
        'sample_ids': sample_ids,
        'images': images,
        'texts': texts,
        'text_input_ids': text_input_ids,
        'text_attention_mask': text_attention_mask,
        'cad_seq': cad_seqs,
        'cad_valid_mask': cad_valid_masks
    }


def build_dataloader(
    data_root,
    split='train',
    batch_size=32,
    num_workers=4,
    ids_file=None,
    img_size=224,
    text_max_len=64,
    tokenizer_name='bert-base-uncased',
    backend='files',
    lmdb_path=None,
    pin_memory=True,
    persistent_workers=None,
    prefetch_factor=1,
    max_prefetch_gb=DEFAULT_MAX_PREFETCH_GB,
    distributed=False,
    rank=0,
    world_size=1,
):
    """构建数据加载器。"""
    dataset = CADDataset(
        data_root,
        split=split,
        ids_file=ids_file,
        img_size=img_size,
        text_max_len=text_max_len,
        tokenizer_name=tokenizer_name,
        backend=backend,
        lmdb_path=lmdb_path,
    )

    effective_num_workers = num_workers
    effective_prefetch_factor = prefetch_factor
    per_sample_image_bytes = dataset.n_views * 3 * img_size * img_size * 4
    estimated_batch_bytes = per_sample_image_bytes * batch_size
    estimated_prefetched_batches = 0
    if num_workers > 0:
        if max_prefetch_gb is not None and max_prefetch_gb > 0:
            max_prefetch_bytes = int(max_prefetch_gb * (1024 ** 3))
            max_prefetched_batches = max(1, max_prefetch_bytes // max(estimated_batch_bytes, 1))
            max_workers_for_prefetch = max(1, max_prefetched_batches // max(prefetch_factor, 1))
            if effective_num_workers > max_workers_for_prefetch:
                print(
                    f'Reducing num_workers from {effective_num_workers} to {max_workers_for_prefetch} '
                    f'to keep estimated prefetched image memory under {max_prefetch_gb:.1f} GiB '
                    f'(batch_size={batch_size}, img_size={img_size}, n_views={dataset.n_views}).'
                )
                effective_num_workers = max_workers_for_prefetch
        estimated_prefetched_batches = effective_num_workers * max(effective_prefetch_factor, 1)

    sampler = None
    shuffle = (split == 'train')
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=False,
        )
        shuffle = False

    loader_kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': effective_num_workers,
        'collate_fn': collate_fn,
        'pin_memory': pin_memory,
    }
    if sampler is not None:
        loader_kwargs['sampler'] = sampler
    if effective_num_workers > 0:
        loader_kwargs['persistent_workers'] = False if persistent_workers is None else persistent_workers
        loader_kwargs['prefetch_factor'] = effective_prefetch_factor

    dataloader = DataLoader(**loader_kwargs)
    dataloader.distributed = distributed
    dataloader.rank = rank
    dataloader.world_size = world_size
    dataloader.sampler_for_epoch = sampler
    dataloader.estimated_batch_gb = estimated_batch_bytes / (1024 ** 3)
    dataloader.estimated_prefetched_batches = estimated_prefetched_batches
    dataloader.estimated_prefetch_gb = (
        dataloader.estimated_batch_gb * estimated_prefetched_batches if estimated_prefetched_batches > 0 else 0.0
    )
    dataloader.max_prefetch_gb = max_prefetch_gb
    return dataloader
