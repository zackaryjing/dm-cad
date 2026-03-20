#!/usr/bin/env python3
"""
CAD 数据集类 - 实现双模态 CAD 数据加载
基于设计文档 5.3 节

目录结构适配:
- 图像：cad_img/{group_id}/{sample_name}/{sample_name}_{000-007}.png
- 文本：cad_desc/{group_id}.json (按 id 字段查找)
- CAD: cad_vec/{group_id}/{sample_name}.h5
"""

import json
import os

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import BertTokenizer


class CADDataset(Dataset):
    """双模态 CAD 数据集

    加载图像、文本和 CAD 序列三元组数据
    """
    def __init__(self, data_root, split='train', img_size=224,
                 text_max_len=64, tokenizer_name='bert-base-uncased',
                 ids_file=None):
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        self.text_max_len = text_max_len

        self.image_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.text_cache = {}
        self.data_list = self._load_data_list(ids_file)
        self._preload_text_descriptions()

    def _load_data_list(self, ids_file):
        """加载数据索引"""
        if ids_file is not None:
            if os.path.isabs(ids_file):
                ids_path = ids_file
            else:
                ids_path = os.path.join(self.data_root, ids_file)
        else:
            ids_path = os.path.join(self.data_root, f'{self.split}_ids.txt')

        if os.path.exists(ids_path):
            with open(ids_path, 'r') as f:
                return [line.strip() for line in f if line.strip()]

        print(f'Warning: ids file not found: {ids_path}')
        return []

    def _preload_text_descriptions(self):
        """预加载所有需要的文本描述到缓存"""
        needed_groups = set()
        for sample_id in self.data_list:
            parts = sample_id.split('/')
            if len(parts) >= 2:
                needed_groups.add(parts[0])

        for group_id in needed_groups:
            desc_file = os.path.join(self.data_root, 'cad_desc', f'{group_id}.json')
            if os.path.exists(desc_file):
                with open(desc_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.text_cache[group_id] = {
                    entry['id']: entry.get('text caption', '')
                    for entry in data if 'id' in entry
                }
            else:
                self.text_cache[group_id] = {}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample_id = self.data_list[idx]
        parts = sample_id.split('/')
        if len(parts) < 2:
            raise ValueError(f'Invalid sample id format: {sample_id}')

        group_id, sample_name = parts[0], parts[1]
        images = self._load_images(group_id, sample_name)

        text = self._get_text(sample_id, group_id)
        text_encoding = self.tokenizer(
            text,
            max_length=self.text_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        text_input_ids = text_encoding['input_ids'].squeeze(0)
        text_attention_mask = text_encoding['attention_mask'].squeeze(0)

        cad_seq, cad_valid_mask = self._load_cad_vector(group_id, sample_name)

        return {
            'sample_id': sample_id,
            'images': images,
            'text': text,
            'text_input_ids': text_input_ids,
            'text_attention_mask': text_attention_mask,
            'cad_seq': cad_seq,
            'cad_valid_mask': cad_valid_mask
        }

    def _load_images(self, group_id, sample_name):
        """加载 8 视图图像"""
        images = []
        img_dir = os.path.join(self.data_root, 'cad_img', group_id, sample_name)

        for i in range(8):
            img_path = os.path.join(img_dir, f'{sample_name}_{i:03d}.png')
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                img = self.image_transform(img)
            else:
                img = torch.ones(3, self.img_size, self.img_size)
            images.append(img)
        return torch.stack(images)

    def _get_text(self, sample_id, group_id):
        """获取文本描述"""
        if group_id in self.text_cache and sample_id in self.text_cache[group_id]:
            return self.text_cache[group_id][sample_id]
        return ''

    def _load_cad_vector(self, group_id, sample_name):
        """加载 CAD 向量序列"""
        vec_path = os.path.join(self.data_root, 'cad_vec', group_id, f'{sample_name}.h5')
        if os.path.exists(vec_path):
            try:
                with h5py.File(vec_path, 'r') as f:
                    for key in ['cad_seq', 'sequence', 'data', 'vec']:
                        if key in f:
                            data = f[key][:]
                            break
                    else:
                        key = list(f.keys())[0]
                        data = f[key][:]

                if len(data.shape) == 1:
                    data = data.reshape(-1, 17)

                if data.shape[1] == 17:
                    padded = np.zeros((data.shape[0], 20), dtype=np.float32)
                    padded[:, :17] = data.astype(np.float32)
                    data = padded
                elif data.shape[1] != 20:
                    raise ValueError(f'Unexpected CAD vector shape: {data.shape}')

                cad_seq = torch.from_numpy(data).float()
                cad_valid_mask = cad_seq[:, 0] >= 0
                return cad_seq, cad_valid_mask.bool()
            except Exception as e:
                print(f'Warning: Failed to load CAD vector {vec_path}: {e}')

        empty_seq = torch.zeros(0, 20, dtype=torch.float32)
        empty_mask = torch.zeros(0, dtype=torch.bool)
        return empty_seq, empty_mask


def collate_fn(batch):
    """Batch 数据合并"""
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


def build_dataloader(data_root, split='train', batch_size=32, num_workers=4, ids_file=None,
                     img_size=224, text_max_len=64, tokenizer_name='bert-base-uncased'):
    """构建数据加载器"""
    dataset = CADDataset(
        data_root,
        split=split,
        ids_file=ids_file,
        img_size=img_size,
        text_max_len=text_max_len,
        tokenizer_name=tokenizer_name,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=collate_fn
    )
