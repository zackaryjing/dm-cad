"""
数据增强模块 - 实现 CAD 训练数据增强
基于设计文档 4.3 节
"""

import random
import torch
from torchvision import transforms
from PIL import Image


class CADDataAugmentation:
    """CAD 训练数据增强

    包括图像增强和文本增强
    """
    def __init__(self, do_color_jitter=True, do_rotation=True, do_affine=True,
                 do_text_augment=False):
        """
        Args:
            do_color_jitter: 是否使用颜色抖动
            do_rotation: 是否使用旋转
            do_affine: 是否使用仿射变换
            do_text_augment: 是否使用文本增强
        """
        self.do_color_jitter = do_color_jitter
        self.do_rotation = do_rotation
        self.do_affine = do_affine
        self.do_text_augment = do_text_augment

        # 基础变换
        base_transforms = []

        if do_color_jitter:
            base_transforms.append(transforms.ColorJitter(
                brightness=0.2, contrast=0.2
            ))

        if do_rotation:
            base_transforms.append(transforms.RandomRotation(15))

        if do_affine:
            base_transforms.append(transforms.RandomAffine(
                degrees=0, translate=(0.1, 0.1)
            ))

        self.image_transforms = transforms.Compose(base_transforms)

    def augment_views(self, views):
        """增强多视图图像

        Args:
            views: [8, 3, 224, 224] - 8 个视图
        Returns:
            augmented: [8, 3, 224, 224] 增强后的视图
        """
        augmented = []
        for view in views:
            # Tensor -> PIL
            view_pil = transforms.ToPILImage()(view)
            # 增强
            view_aug = self.image_transforms(view_pil)
            # PIL -> Tensor
            view_aug = transforms.ToTensor()(view_aug)
            augmented.append(view_aug)
        return torch.stack(augmented)

    def augment_text(self, text, synonym_replace_prob=0.1, random_delete_prob=0.05):
        """增强文本描述

        Args:
            text: 原始文本
            synonym_replace_prob: 同义词替换概率
            random_delete_prob: 随机删除概率
        Returns:
            augmented_text: 增强后的文本
        """
        # 简单的文本增强实现
        words = text.split()
        augmented_words = []

        for word in words:
            # 随机删除
            if random.random() < random_delete_prob:
                continue

            # 同义词替换 (简化实现，实际可使用 nlpaug)
            if random.random() < synonym_replace_prob:
                # 这里可以集成 WordNet 等同义词库
                pass

            augmented_words.append(word)

        return ' '.join(augmented_words)

    def __call__(self, sample):
        """对样本进行增强

        Args:
            sample: 数据样本 dict
        Returns:
            augmented_sample: 增强后的样本
        """
        augmented_sample = sample.copy()

        # 图像增强
        if 'images' in sample:
            augmented_sample['images'] = self.augment_views(sample['images'])

        # 文本增强 (仅在训练时使用)
        if 'text' in sample and self.do_text_augment:
            augmented_sample['text'] = self.augment_text(sample['text'])

        return augmented_sample


class StrongAugmentation(CADDataAugmentation):
    """强数据增强"""

    def __init__(self):
        super().__init__(
            do_color_jitter=True,
            do_rotation=True,
            do_affine=True
        )
        # 添加更强的变换
        self.strong_transforms = transforms.Compose([
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3),
        ])

    def augment_views(self, views):
        views = super().augment_views(views)
        return self.strong_transforms(views)


class WeakAugmentation(CADDataAugmentation):
    """弱数据增强 (仅用于验证)"""

    def __init__(self):
        super().__init__(
            do_color_jitter=False,
            do_rotation=False,
            do_affine=False
        )
