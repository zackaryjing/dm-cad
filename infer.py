#!/usr/bin/env python3
"""
推理示例脚本

用法:
    python infer.py --checkpoint checkpoints/best.pth
"""

import argparse
import torch
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer
from models.dual_modal_cad import DualModalCADGenerator


def parse_args():
    parser = argparse.ArgumentParser(description='Inference with Dual-Modal CAD Generator')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--images', type=str, nargs='+', required=True,
                        help='Paths to 8 view images')
    parser.add_argument('--text', type=str, required=True,
                        help='Text description')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for inference')
    return parser.parse_args()


def load_image(img_path, img_size=224):
    """加载并预处理单张图像"""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    return transform(img)


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
    model.eval()

    # 加载图像
    print(f'Loading {len(args.images)} view images...')
    images = [load_image(img_path) for img_path in args.images]
    images = torch.stack(images).unsqueeze(0).to(device)  # [1, 8, 3, 224, 224]

    # 编码文本
    print(f'Encoding text: "{args.text}"')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text_encoding = tokenizer(
        args.text,
        max_length=64,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    text_input_ids = text_encoding['input_ids'].to(device)
    text_attention_mask = text_encoding['attention_mask'].to(device)

    # 推理
    print('Generating CAD sequence...')
    with torch.no_grad():
        generated = model.generate(
            images, text_input_ids, text_attention_mask
        )

    # 输出结果
    print('\n=== Generated CAD Sequence ===')
    cmd_names = ['START', 'SKETCH', 'EXTRUDE', 'END']
    for i, (cmd, params) in enumerate(generated):
        # cmd shape: [1, 1], get scalar value
        cmd_type = cmd[0, 0].item() if isinstance(cmd, torch.Tensor) else cmd
        cmd_name = cmd_names[cmd_type] if cmd_type < len(cmd_names) else f'CMD_{cmd_type}'
        params_np = params[0].cpu().numpy() if isinstance(params, torch.Tensor) else params
        print(f'Step {i}: {cmd_name}, params: {params_np[:5]}...')  # 显示前 5 个参数

    print('\nGeneration completed!')


if __name__ == '__main__':
    main()
