"""
CAD 数据集类 - 实现双模态 CAD 数据加载
基于设计文档 5.3 节
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer


class CADDataset(Dataset):
    """双模态 CAD 数据集

    加载图像、文本和 CAD 序列三元组数据
    """
    def __init__(self, data_root, split='train', img_size=224,
                 text_max_len=64, tokenizer_name='bert-base-uncased'):
        """
        Args:
            data_root: 数据根目录
            split: 数据集划分 ('train', 'val', 'test')
            img_size: 图像大小
            text_max_len: 文本最大长度
            tokenizer_name: BERT tokenizer 名称
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

        # 加载数据列表
        self.data_list = self._load_data_list()

    def _load_data_list(self):
        """加载数据索引"""
        data_list_path = os.path.join(self.data_root, f'{self.split}.json')
        if os.path.exists(data_list_path):
            with open(data_list_path, 'r') as f:
                return json.load(f)
        return []

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        uid = sample['uid']

        # 加载 8 视图图像 [8, 3, H, W]
        images = self._load_images(uid)

        # 加载并编码文本
        text = sample.get('text', '')
        text_encoding = self.tokenizer(
            text,
            max_length=self.text_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        text_input_ids = text_encoding['input_ids'].squeeze(0)
        text_attention_mask = text_encoding['attention_mask'].squeeze(0)

        # 加载 CAD 序列 [seq_len, 20]
        cad_seq = self._load_cad_sequence(uid)
        cad_valid_mask = (cad_seq[:, 0] >= 0).bool()

        return {
            'uid': uid,
            'images': images,
            'text': text,
            'text_input_ids': text_input_ids,
            'text_attention_mask': text_attention_mask,
            'cad_seq': cad_seq,
            'cad_valid_mask': cad_valid_mask
        }

    def _load_images(self, uid):
        """加载 8 视图图像"""
        images = []
        for i in range(8):
            img_path = os.path.join(
                self.data_root, 'images', uid, f'view_{i:02d}.png'
            )
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                img = self.image_transform(img)
            else:
                # 返回空白色图像作为占位符
                img = torch.ones(3, self.img_size, self.img_size)
            images.append(img)
        return torch.stack(images)

    def _load_cad_sequence(self, uid):
        """加载 CAD 命令序列"""
        cad_path = os.path.join(self.data_root, 'cad_seq', f'{uid}.pt')
        if os.path.exists(cad_path):
            return torch.load(cad_path)
        # 返回空序列作为占位符
        return torch.zeros(20, 20)


def collate_fn(batch):
    """Batch 数据合并"""
    uids = [item['uid'] for item in batch]
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
        'uids': uids,
        'images': images,
        'texts': texts,
        'text_input_ids': text_input_ids,
        'text_attention_mask': text_attention_mask,
        'cad_seq': cad_seqs,
        'cad_valid_mask': cad_valid_masks
    }


def build_dataloader(data_root, split='train', batch_size=32, num_workers=4):
    """构建数据加载器"""
    dataset = CADDataset(data_root, split=split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=collate_fn
    )
