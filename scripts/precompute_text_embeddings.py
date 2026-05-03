#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_ids(ids_file: Path) -> list[str]:
    with ids_file.open("r") as f:
        return [line.strip() for line in f if line.strip()]


class TextDataset(Dataset):
    def __init__(self, ids_file: Path, data_root: Path):
        self.sample_ids = load_ids(ids_file)
        self.data_root = data_root
        self.group_cache: dict[str, dict[str, str]] = {}

    def _load_group(self, group_id: str):
        if group_id in self.group_cache:
            return
        desc_path = self.data_root / "cad_desc" / f"{group_id}.json"
        items = json.loads(desc_path.read_text())
        self.group_cache[group_id] = {
            item["id"]: item.get("text caption", "")
            for item in items
            if "id" in item
        }

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index: int):
        sample_id = self.sample_ids[index]
        group_id = sample_id.split("/")[0]
        self._load_group(group_id)
        return sample_id, self.group_cache[group_id].get(sample_id, "")


def collate_text(batch):
    sample_ids = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    return sample_ids, texts


def masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute frozen BERT text embeddings")
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/dataset_v0"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="bert-base-uncased")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--shard-size", type=int, default=50000)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.model_name, local_files_only=True)
    model = BertModel.from_pretrained(args.model_name, local_files_only=True).to(args.device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    dataset = TextDataset(args.ids, args.data_root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_text,
    )

    shard_sample_ids: list[str] = []
    shard_embs: list[torch.Tensor] = []
    shard_index = 0

    def flush():
        nonlocal shard_sample_ids, shard_embs, shard_index
        if not shard_sample_ids:
            return
        payload = {
            "sample_ids": shard_sample_ids,
            "text_emb": torch.cat(shard_embs, dim=0).cpu().to(torch.float16),
            "source_ids_file": str(args.ids),
            "model_name": args.model_name,
            "pooling": "masked_mean",
        }
        path = args.output_dir / f"shard_{shard_index:05d}.pt"
        torch.save(payload, path)
        print(f"saved {path} ({len(shard_sample_ids)} samples)")
        shard_sample_ids = []
        shard_embs = []
        shard_index += 1

    with torch.no_grad():
        for sample_ids, texts in loader:
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(args.device) for k, v in encoded.items()}
            outputs = model(**encoded)
            text_emb = masked_mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            shard_sample_ids.extend(sample_ids)
            shard_embs.append(text_emb.detach())
            if len(shard_sample_ids) >= args.shard_size:
                flush()

    flush()


if __name__ == "__main__":
    main()
