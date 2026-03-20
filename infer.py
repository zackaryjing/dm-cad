#!/usr/bin/env python3
"""
推理示例脚本

用法:
    python infer.py --checkpoint checkpoints/best.pth --images view_00.png ... view_07.png --text "..."
"""

import argparse

import torch
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer

from models.dual_modal_cad import DualModalCADGenerator


CMD_NAMES = ['Line', 'Arc', 'Circle', 'EOS', 'SOL', 'Ext']


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
    parser.add_argument('--max_steps', type=int, default=120,
                        help='Maximum number of CAD steps to generate')
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

    if len(args.images) != 8:
        raise ValueError(f'Expected 8 view images, got {len(args.images)}')

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, using CPU')
        device = 'cpu'
    else:
        device = args.device

    print(f'Loading model from {args.checkpoint}...')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = checkpoint.get('config', {}).get('model', checkpoint.get('config', {}))
    model = DualModalCADGenerator(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f'Loading {len(args.images)} view images...')
    images = [load_image(img_path) for img_path in args.images]
    images = torch.stack(images).unsqueeze(0).to(device)

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

    print('Generating CAD sequence...')
    with torch.no_grad():
        cmd_pred, param_pred = model.generate(
            images,
            text_input_ids,
            text_attention_mask,
            max_steps=args.max_steps
        )

    print('\n=== Generated CAD Sequence ===')
    cmd_seq = cmd_pred[0].cpu()
    param_seq = param_pred[0].cpu()
    for step in range(cmd_seq.shape[0]):
        cmd_type = int(cmd_seq[step].item())
        cmd_name = CMD_NAMES[cmd_type] if 0 <= cmd_type < len(CMD_NAMES) else f'CMD_{cmd_type}'
        params_np = param_seq[step].numpy()
        print(f'Step {step}: {cmd_name}, params: {params_np[:5]}...')
        if cmd_type == 3:
            break

    print('\nGeneration completed!')


if __name__ == '__main__':
    main()
