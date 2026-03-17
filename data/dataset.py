#!/usr/bin/env python3
"""
CAD 数据集类 - 实现双模态 CAD 数据加载
基于设计文档 5.3 节

目录结构适配:
- 图像：cad_img/{group_id}/{sample_name}/{sample_name}_{000-007}.png
- 文本：cad_desc/{group_id}.json (按 id 字段查找)
- CAD: cad_vec/{group_id}/{sample_name}.h5
"""

import os
import json
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer


class CADDataset(Dataset):
    """双模态 CAD 数据集

    加载图像、文本和 CAD 序列三元组数据
    """
    def __init__(self, data_root, split='train', img_size=224,
                 text_max_len=64, tokenizer_name='bert-base-uncased',
                 ids_file=None):
        """
        Args:
            data_root: 数据根目录 (e.g., datasets/dataset_v0)
            split: 数据集划分 ('train', 'val', 'test')
            img_size: 图像大小
            text_max_len: 文本最大长度
            tokenizer_name: BERT tokenizer 名称
            ids_file: 可选的 ids 文件路径，如指定则从该文件加载样本列表
        """
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        self.text_max_len = text_max_len

        # 图像变换
        self.image_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # 文本 tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

        # 缓存文本描述 (group_id -> {sample_id -> text})
        self.text_cache = {}

        # 加载数据列表
        self.data_list = self._load_data_list(ids_file)

        # 预加载所有需要的文本描述
        self._preload_text_descriptions()

    def _load_data_list(self, ids_file):
        """加载数据索引"""
        # 优先使用 ids_file 参数
        if ids_file is not None:
            if os.path.exists(ids_file):
                with open(ids_file, 'r') as f:
                    return [line.strip() for line in f if line.strip()]
            else:
                print(f"Warning: ids file not found: {ids_file}")
                return []

        # 默认使用 {split}_ids.txt
        default_ids_file = os.path.join(self.data_root, f'{self.split}_ids.txt')
        if os.path.exists(default_ids_file):
            with open(default_ids_file, 'r') as f:
                return [line.strip() for line in f if line.strip()]

        return []

    def _preload_text_descriptions(self):
        """预加载所有需要的文本描述到缓存"""
        # 收集所有需要的 group_id
        needed_groups = set()
        for sample_id in self.data_list:
            group_id = sample_id.split('/')[0]
            needed_groups.add(group_id)

        # 加载每个 group 的 JSON 文件
        for group_id in needed_groups:
            desc_file = os.path.join(self.data_root, 'cad_desc', f'{group_id}.json')
            if os.path.exists(desc_file):
                with open(desc_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # 构建 {sample_id -> text} 映射
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
        group_id = sample_id.split('/')[0]
        sample_name = sample_id.split('/')[1]

        # 加载 8 视图图像 [8, 3, H, W]
        images = self._load_images(group_id, sample_name)

        # 加载文本描述
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

        # 加载 CAD 向量 [seq_len, 20]
        cad_seq = self._load_cad_vector(group_id, sample_name)
        cad_valid_mask = (cad_seq[:, 0] >= 0).bool()

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
                # 返回空白色图像作为占位符
                img = torch.ones(3, self.img_size, self.img_size)
            images.append(img)
        return torch.stack(images)

    def _get_text(self, sample_id, group_id):
        """获取文本描述"""
        if group_id in self.text_cache and sample_id in self.text_cache[group_id]:
            return self.text_cache[group_id][sample_id]
        return ""

    def _load_cad_vector(self, group_id, sample_name):
        """加载 CAD 向量序列"""
        vec_path = os.path.join(self.data_root, 'cad_vec', group_id, f'{sample_name}.h5')
        if os.path.exists(vec_path):
            try:
                with h5py.File(vec_path, 'r') as f:
                    # 尝试多个可能的键名
                    for key in ['cad_seq', 'sequence', 'data', 'vec']:
                        if key in f:
                            data = f[key][:]
                            break
                    else:
                        # 取第一个数据集
                        key = list(f.keys())[0]
                        data = f[key][:]
                    # 转换为 tensor
                    if len(data.shape) == 1:
                        data = data.reshape(-1, 17)
                    # 如果最后维度是 17，padding 到 20 (1 cmd + 19 params)
                    elif data.shape[1] == 17:
                        import numpy as np
                        padded = np.zeros((data.shape[0], 20), dtype=np.float32)
                        padded[:, :17] = data.astype(np.float32)
                        data = padded
                    return torch.from_numpy(data).float()
            except Exception as e:
                print(f"Warning: Failed to load CAD vector {vec_path}: {e}")
        # 返回空序列作为占位符 [120, 20]
        return torch.zeros(120, 20)


def collate_fn(batch):
    """Batch 数据合并"""
    sample_ids = [item['sample_id'] for item in batch]
    images = torch.stack([item['images'] for item in batch])
    texts = [item['text'] for item in batch]
    text_input_ids = torch.stack([item['text_input_ids'] for item in batch])
    text_attention_mask = torch.stack([item['text_attention_mask'] for item in batch])

    # 处理变长 CAD 序列
    max_seq_len = max(item['cad_seq'].shape[0] for item in batch)
    cad_seqs = []
    cad_valid_masks = []
    for item in batch:
        cad_seq = item['cad_seq']
        valid_mask = item['cad_valid_mask']

        # Padding
        if cad_seq.shape[0] < max_seq_len:
            pad_len = max_seq_len - cad_seq.shape[0]
            pad_seq = torch.zeros(pad_len, cad_seq.shape[1])
            pad_mask = torch.zeros(pad_len, dtype=torch.bool)
            cad_seq = torch.cat([cad_seq, pad_seq], dim=0)
            valid_mask = torch.cat([valid_mask, pad_mask], dim=0)

        cad_seqs.append(cad_seq)
        cad_valid_masks.append(valid_mask)

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


def build_dataloader(data_root, split='train', batch_size=32, num_workers=4, ids_file=None):
    """构建数据加载器"""
    dataset = CADDataset(data_root, split=split, ids_file=ids_file)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=collate_fn
    )
