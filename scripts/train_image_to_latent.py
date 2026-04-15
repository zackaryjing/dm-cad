#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepcad_latent.data import ImageLatentDataset, RunningAverage, collate_image_latent
from deepcad_latent.model import MultiViewLatentRegressor


def parse_args():
    parser = argparse.ArgumentParser(description="Train image-to-DeepCAD-latent regressor")
    parser.add_argument("--train-ids", type=Path, required=True)
    parser.add_argument("--test-ids", type=Path, required=True)
    parser.add_argument("--train-latent-root", type=Path, required=True)
    parser.add_argument("--test-latent-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/dataset_v0"))
    parser.add_argument("--lmdb-path", type=str, default="cad_data.lmdb")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--n-views", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def setup_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = 0
    local_rank = 0
    if distributed:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return distributed, rank, local_rank, world_size


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def build_loader(dataset, batch_size: int, num_workers: int, distributed: bool, shuffle: bool):
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=False)
        shuffle = False
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_image_latent,
        persistent_workers=num_workers > 0,
    )
    return loader, sampler


def evaluate(model, loader, device):
    model.eval()
    mse_meter = RunningAverage()
    cos_meter = RunningAverage()
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            target = batch["z"].to(device, non_blocking=True)
            pred = model(images)
            mse = F.mse_loss(pred, target)
            cos = F.cosine_similarity(pred, target, dim=-1).mean()
            mse_meter.update(float(mse.item()), images.shape[0])
            cos_meter.update(float(cos.item()), images.shape[0])
    return {"mse": mse_meter.avg, "cosine": cos_meter.avg}


def main():
    args = parse_args()
    setup_seed(args.seed)
    distributed, rank, local_rank, world_size = setup_distributed()

    if args.device == "cuda":
        device = torch.device(f"cuda:{local_rank}" if distributed else "cuda")
    else:
        device = torch.device(args.device)

    train_dataset = ImageLatentDataset(
        ids_file=args.train_ids,
        latent_root=args.train_latent_root,
        data_root=args.data_root,
        lmdb_path=args.lmdb_path,
        img_size=args.img_size,
        n_views=args.n_views,
    )
    test_dataset = ImageLatentDataset(
        ids_file=args.test_ids,
        latent_root=args.test_latent_root,
        data_root=args.data_root,
        lmdb_path=args.lmdb_path,
        img_size=args.img_size,
        n_views=args.n_views,
    )

    train_loader, train_sampler = build_loader(
        train_dataset, args.batch_size, args.num_workers, distributed, shuffle=True
    )
    test_loader, _ = build_loader(
        test_dataset, args.batch_size, args.num_workers, distributed, shuffle=False
    )

    model = MultiViewLatentRegressor(
        backbone_name=args.backbone,
        n_views=args.n_views,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_test_mse = float("inf")

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        train_mse_meter = RunningAverage()
        train_cos_meter = RunningAverage()
        for step, batch in enumerate(train_loader, start=1):
            images = batch["images"].to(device, non_blocking=True)
            target = batch["z"].to(device, non_blocking=True)

            pred = model(images)
            mse = F.mse_loss(pred, target)
            cos = F.cosine_similarity(pred, target, dim=-1).mean()

            optimizer.zero_grad(set_to_none=True)
            mse.backward()
            optimizer.step()

            train_mse_meter.update(float(mse.item()), images.shape[0])
            train_cos_meter.update(float(cos.item()), images.shape[0])

            if is_main_process(rank) and step % args.log_interval == 0:
                print(
                    f"epoch {epoch:03d} step {step:05d}/{len(train_loader):05d} "
                    f"train_mse={train_mse_meter.avg:.6f} train_cos={train_cos_meter.avg:.6f}"
                )

        test_metrics = evaluate(model, test_loader, device)
        train_metrics = {"mse": train_mse_meter.avg, "cosine": train_cos_meter.avg}
        epoch_metrics = {
            "epoch": epoch,
            "train_mse": train_metrics["mse"],
            "train_cosine": train_metrics["cosine"],
            "test_mse": test_metrics["mse"],
            "test_cosine": test_metrics["cosine"],
        }
        history.append(epoch_metrics)

        if is_main_process(rank):
            print(json.dumps(epoch_metrics, ensure_ascii=True))
            latest_path = args.output_dir / "latest.pt"
            model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            torch.save(
                {
                    "epoch": epoch,
                    "model": model_state,
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                    "metrics": epoch_metrics,
                },
                latest_path,
            )
            if test_metrics["mse"] < best_test_mse:
                best_test_mse = test_metrics["mse"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model_state,
                        "optimizer": optimizer.state_dict(),
                        "args": vars(args),
                        "metrics": epoch_metrics,
                    },
                    args.output_dir / "best.pt",
                )
            with (args.output_dir / "history.json").open("w") as f:
                json.dump(history, f, indent=2)

    cleanup_distributed()


if __name__ == "__main__":
    main()
