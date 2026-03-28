#!/usr/bin/env python3
"""
推理示例脚本

用法:
    python infer.py --checkpoint checkpoints/best.pth --images view_00.png ... view_07.png --text "..."
    python infer.py --checkpoint checkpoints/best.pth --config train/config_5k.yaml --split test --sample-index 0
    python infer.py --checkpoint checkpoints/best.pth --config train/config_5k.yaml --sample-ids-file sample_ids.txt
"""

import argparse
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer

from data.dataset import CADDataset
from models.dual_modal_cad import DualModalCADGenerator


CMD_NAMES = ['Line', 'Arc', 'Circle', 'EOS', 'SOL', 'Ext']


def parse_args():
    parser = argparse.ArgumentParser(description='Inference with Dual-Modal CAD Generator')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--images', type=str, nargs='+', default=None,
                        help='Paths to 8 view images')
    parser.add_argument('--text', type=str, default=None,
                        help='Text description')
    parser.add_argument('--config', type=str, default=None,
                        help='Optional config file for dataset-backed sample inference')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Optional data directory override for dataset-backed sample inference')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to sample from (default: test)')
    parser.add_argument('--sample-index', type=int, default=0,
                        help='Dataset sample index for dataset-backed sample inference (default: 0)')
    parser.add_argument('--sample-id', type=str, default=None,
                        help='Specific sample id for dataset-backed sample inference')
    parser.add_argument('--sample-ids-file', type=str, default=None,
                        help='Path to a text file containing one sample id per line for batch sample inference')
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


def load_config(config_path):
    if not config_path:
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def resolve_device(device_arg):
    if device_arg == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, using CPU')
        return 'cpu'
    return device_arg


def _load_state_dict_flexible(model, state_dict):
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        stripped = {
            key.replace('module.', '', 1) if key.startswith('module.') else key: value
            for key, value in state_dict.items()
        }
        model.load_state_dict(stripped)


def load_model(checkpoint_path, device):
    print(f'Loading model from {checkpoint_path}...')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get('config', {}).get('model', checkpoint.get('config', {}))
    model = DualModalCADGenerator(model_config)
    _load_state_dict_flexible(model, checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def load_manual_inputs(args):
    if args.images is None or args.text is None:
        return None

    if len(args.images) != 8:
        raise ValueError(f'Expected 8 view images, got {len(args.images)}')

    print(f'Loading {len(args.images)} view images...')
    images = [load_image(img_path) for img_path in args.images]
    images = torch.stack(images).unsqueeze(0)

    print(f'Encoding text: "{args.text}"')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text_encoding = tokenizer(
        args.text,
        max_length=64,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    return [{
        'sample_id': 'manual_input',
        'images': images,
        'text': args.text,
        'text_input_ids': text_encoding['input_ids'],
        'text_attention_mask': text_encoding['attention_mask'],
        'cad_seq': None,
    }]


def build_dataset(args, config):
    data_cfg = config.get('data', {})
    data_root = args.data_dir or data_cfg.get('data_root')
    if not data_root:
        raise ValueError('Dataset-backed inference requires --config or --data_dir')

    ids_file = data_cfg.get(f'{args.split}_ids_file')
    if ids_file is None and args.split == 'test':
        ids_file = data_cfg.get('test_ids_file')

    dataset = CADDataset(
        data_root=data_root,
        split=args.split,
        ids_file=ids_file,
        img_size=data_cfg.get('img_size', 224),
        text_max_len=data_cfg.get('text_max_len', 64),
    )
    if len(dataset) == 0:
        raise ValueError(f'No samples found for split={args.split}, ids_file={ids_file}')

    print(f'Loaded dataset split={args.split} with {len(dataset)} samples')
    if ids_file:
        print(f'Using ids file: {ids_file}')
    return dataset


def sample_from_dataset(dataset, sample_index):
    sample = dataset[sample_index]
    return {
        'sample_id': sample['sample_id'],
        'images': sample['images'].unsqueeze(0),
        'text': sample['text'],
        'text_input_ids': sample['text_input_ids'].unsqueeze(0),
        'text_attention_mask': sample['text_attention_mask'].unsqueeze(0),
        'cad_seq': sample['cad_seq'],
    }


def load_sample_ids(sample_ids_file):
    with open(sample_ids_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def load_dataset_samples(args, config):
    dataset = build_dataset(args, config)

    if args.sample_ids_file:
        requested_ids = load_sample_ids(args.sample_ids_file)
        if not requested_ids:
            raise ValueError(f'No sample ids found in {args.sample_ids_file}')
        dataset_index = {sample_id: idx for idx, sample_id in enumerate(dataset.data_list)}
        samples = []
        for sample_id in requested_ids:
            if sample_id not in dataset_index:
                raise ValueError(f'Sample id not found in dataset split: {sample_id}')
            sample_index = dataset_index[sample_id]
            samples.append(sample_from_dataset(dataset, sample_index))
        print(f'Loaded {len(samples)} samples from {args.sample_ids_file}')
        return samples

    if args.sample_id is not None:
        try:
            sample_index = dataset.data_list.index(args.sample_id)
        except ValueError as exc:
            raise ValueError(f'Sample id not found in dataset split: {args.sample_id}') from exc
    else:
        sample_index = args.sample_index

    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f'sample-index {sample_index} out of range [0, {len(dataset) - 1}]')

    print(f'Loaded dataset sample {dataset.data_list[sample_index]} from split={args.split} (index={sample_index})')
    return [sample_from_dataset(dataset, sample_index)]


def format_step(cmd_type, params):
    cmd_name = CMD_NAMES[cmd_type] if 0 <= cmd_type < len(CMD_NAMES) else f'CMD_{cmd_type}'
    params_np = params.numpy()
    return f'{cmd_name}, params: {params_np[:5]}...'


def print_sequence(title, cmd_tensor, param_tensor):
    print(f'\n=== {title} ===')
    for step in range(cmd_tensor.shape[0]):
        cmd_type = int(cmd_tensor[step].item())
        print(f'Step {step}: {format_step(cmd_type, param_tensor[step].cpu())}')
        if cmd_type == 3:
            break


def print_ground_truth(cad_seq):
    if cad_seq is None or cad_seq.shape[0] == 0:
        return

    gt_cmd = cad_seq[:, 0].long().clamp(min=0).cpu()
    gt_param = cad_seq[:, 1:].cpu()
    print_sequence('Ground Truth CAD Sequence', gt_cmd, gt_param)


def run_inference(model, sample, device, max_steps):
    print(f'\nSample ID: {sample["sample_id"]}')
    print(f'Text: "{sample["text"]}"')

    images = sample['images'].to(device)
    text_input_ids = sample['text_input_ids'].to(device)
    text_attention_mask = sample['text_attention_mask'].to(device)

    print('Generating CAD sequence...')
    with torch.no_grad():
        cmd_pred, param_pred = model.generate(
            images,
            text_input_ids,
            text_attention_mask,
            max_steps=max_steps
        )

    print_sequence('Generated CAD Sequence', cmd_pred[0].cpu(), param_pred[0].cpu())
    print_ground_truth(sample['cad_seq'])


def main():
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device)

    samples = load_manual_inputs(args)
    if samples is None:
        samples = load_dataset_samples(args, config)

    total_samples = len(samples)
    for idx, sample in enumerate(samples, start=1):
        if total_samples > 1:
            print('\n' + '=' * 24 + f' Sample {idx}/{total_samples} ' + '=' * 24)
        run_inference(model, sample, device, args.max_steps)

    print('\nGeneration completed!')


if __name__ == '__main__':
    main()
