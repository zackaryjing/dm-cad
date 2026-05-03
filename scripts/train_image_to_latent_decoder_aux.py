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

from deepcad_latent.adapter import DeepCADAdapter
from deepcad_latent.data import ImageLatentCadDataset, RunningAverage, collate_image_latent_cad
from deepcad_latent.model import MultiViewLatentRegressor


def parse_args():
    parser = argparse.ArgumentParser(description="Train image-to-latent regressor with frozen DeepCAD decoder loss")
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
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--latent-loss-weight", type=float, default=1.0)
    parser.add_argument("--cmd-loss-weight", type=float, default=0.1)
    parser.add_argument("--param-loss-weight", type=float, default=0.1)
    parser.add_argument("--deepcad-checkpoint", type=Path)
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
        collate_fn=collate_image_latent_cad,
        persistent_workers=num_workers > 0,
    )
    return loader, sampler


def _get_visibility_mask(commands: torch.Tensor, eos_idx: int) -> torch.Tensor:
    seq_len = commands.size(-1)
    return (commands == eos_idx).sum(dim=-1) < seq_len - 1


def _get_padding_mask(commands: torch.Tensor, eos_idx: int, extended: bool = False) -> torch.Tensor:
    padding_mask = ((commands == eos_idx).cumsum(dim=-1) == 0).float()
    if extended and commands.size(-1) > 3:
        seq_len = commands.size(-1)
        padding_mask[..., 3:seq_len] = (padding_mask[..., 3:seq_len] + padding_mask[..., : seq_len - 3]).clamp(max=1.0)
    return padding_mask


def compute_decoder_aux_losses(
    command_logits: torch.Tensor,
    args_logits: torch.Tensor,
    tgt_commands: torch.Tensor,
    tgt_args: torch.Tensor,
    cmd_args_mask: torch.Tensor,
    eos_idx: int,
):
    visibility_mask = _get_visibility_mask(tgt_commands, eos_idx=eos_idx)
    padding_mask = _get_padding_mask(tgt_commands, eos_idx=eos_idx, extended=True) * visibility_mask.unsqueeze(-1)
    valid_token_mask = padding_mask.squeeze(-1).bool()
    valid_arg_mask = cmd_args_mask[tgt_commands.long()]

    loss_cmd = F.cross_entropy(
        command_logits[valid_token_mask].reshape(-1, command_logits.shape[-1]),
        tgt_commands[valid_token_mask].reshape(-1).long(),
    )
    loss_args = F.cross_entropy(
        args_logits[valid_arg_mask].reshape(-1, args_logits.shape[-1]),
        tgt_args[valid_arg_mask].reshape(-1).long() + 1,
    )
    return loss_cmd, loss_args


def evaluate(model, decoder_adapter, loader, device, args):
    model.eval()
    total_meter = RunningAverage()
    latent_meter = RunningAverage()
    cmd_meter = RunningAverage()
    param_meter = RunningAverage()
    cos_meter = RunningAverage()

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            target_z = batch["z"].to(device, non_blocking=True)
            cad_vec = batch["cad_vec"].to(device, non_blocking=True)
            tgt_commands = cad_vec[:, :, 0]
            tgt_args = cad_vec[:, :, 1:]

            pred_z = model(images)
            latent_loss = F.mse_loss(pred_z, target_z)
            cos = F.cosine_similarity(pred_z, target_z, dim=-1).mean()
            decoder_outputs = decoder_adapter.decode_logits(pred_z)
            cmd_loss, param_loss = compute_decoder_aux_losses(
                command_logits=decoder_outputs["command_logits"],
                args_logits=decoder_outputs["args_logits"],
                tgt_commands=tgt_commands,
                tgt_args=tgt_args,
                cmd_args_mask=decoder_adapter.cmd_args_mask,
                eos_idx=decoder_adapter.eos_idx,
            )
            total_loss = (
                args.latent_loss_weight * latent_loss
                + args.cmd_loss_weight * cmd_loss
                + args.param_loss_weight * param_loss
            )

            batch_size = images.shape[0]
            total_meter.update(float(total_loss.item()), batch_size)
            latent_meter.update(float(latent_loss.item()), batch_size)
            cmd_meter.update(float(cmd_loss.item()), batch_size)
            param_meter.update(float(param_loss.item()), batch_size)
            cos_meter.update(float(cos.item()), batch_size)

    return {
        "total_loss": total_meter.avg,
        "latent_mse": latent_meter.avg,
        "cmd_loss": cmd_meter.avg,
        "param_loss": param_meter.avg,
        "cosine": cos_meter.avg,
    }


def main():
    args = parse_args()
    setup_seed(args.seed)
    distributed, rank, local_rank, world_size = setup_distributed()

    if args.device == "cuda":
        device = torch.device(f"cuda:{local_rank}" if distributed else "cuda")
    else:
        device = torch.device(args.device)

    decoder_adapter = DeepCADAdapter(checkpoint_path=args.deepcad_checkpoint, device=device)
    train_dataset = ImageLatentCadDataset(
        ids_file=args.train_ids,
        latent_root=args.train_latent_root,
        data_root=args.data_root,
        max_total_len=decoder_adapter.max_total_len,
        eos_vec=decoder_adapter.eos_vec,
        lmdb_path=args.lmdb_path,
        img_size=args.img_size,
        n_views=args.n_views,
    )
    test_dataset = ImageLatentCadDataset(
        ids_file=args.test_ids,
        latent_root=args.test_latent_root,
        data_root=args.data_root,
        max_total_len=decoder_adapter.max_total_len,
        eos_vec=decoder_adapter.eos_vec,
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

    if args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
        missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
        if is_main_process(rank):
            print(json.dumps({"image_checkpoint_init": {"missing_keys": missing, "unexpected_keys": unexpected}}))

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_test_total = float("inf")

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        total_meter = RunningAverage()
        latent_meter = RunningAverage()
        cmd_meter = RunningAverage()
        param_meter = RunningAverage()
        cos_meter = RunningAverage()

        for step, batch in enumerate(train_loader, start=1):
            images = batch["images"].to(device, non_blocking=True)
            target_z = batch["z"].to(device, non_blocking=True)
            cad_vec = batch["cad_vec"].to(device, non_blocking=True)
            tgt_commands = cad_vec[:, :, 0]
            tgt_args = cad_vec[:, :, 1:]

            pred_z = model(images)
            latent_loss = F.mse_loss(pred_z, target_z)
            cos = F.cosine_similarity(pred_z, target_z, dim=-1).mean()
            decoder_outputs = decoder_adapter.decode_logits_with_grad(pred_z)
            cmd_loss, param_loss = compute_decoder_aux_losses(
                command_logits=decoder_outputs["command_logits"],
                args_logits=decoder_outputs["args_logits"],
                tgt_commands=tgt_commands,
                tgt_args=tgt_args,
                cmd_args_mask=decoder_adapter.cmd_args_mask,
                eos_idx=decoder_adapter.eos_idx,
            )
            total_loss = (
                args.latent_loss_weight * latent_loss
                + args.cmd_loss_weight * cmd_loss
                + args.param_loss_weight * param_loss
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            batch_size = images.shape[0]
            total_meter.update(float(total_loss.item()), batch_size)
            latent_meter.update(float(latent_loss.item()), batch_size)
            cmd_meter.update(float(cmd_loss.item()), batch_size)
            param_meter.update(float(param_loss.item()), batch_size)
            cos_meter.update(float(cos.item()), batch_size)

            if is_main_process(rank) and step % args.log_interval == 0:
                print(
                    f"epoch {epoch:03d} step {step:05d}/{len(train_loader):05d} "
                    f"train_total={total_meter.avg:.6f} train_mse={latent_meter.avg:.6f} "
                    f"train_cmd={cmd_meter.avg:.6f} train_param={param_meter.avg:.6f} "
                    f"train_cos={cos_meter.avg:.6f}"
                )

        test_metrics = evaluate(model, decoder_adapter, test_loader, device, args)
        epoch_metrics = {
            "epoch": epoch,
            "train_total_loss": total_meter.avg,
            "train_latent_mse": latent_meter.avg,
            "train_cmd_loss": cmd_meter.avg,
            "train_param_loss": param_meter.avg,
            "train_cosine": cos_meter.avg,
            "test_total_loss": test_metrics["total_loss"],
            "test_latent_mse": test_metrics["latent_mse"],
            "test_cmd_loss": test_metrics["cmd_loss"],
            "test_param_loss": test_metrics["param_loss"],
            "test_cosine": test_metrics["cosine"],
        }
        history.append(epoch_metrics)

        if is_main_process(rank):
            print(json.dumps(epoch_metrics, ensure_ascii=True))
            latest_path = args.output_dir / "latest.pt"
            model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            payload = {
                "epoch": epoch,
                "model": model_state,
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "metrics": epoch_metrics,
            }
            torch.save(payload, latest_path)
            if test_metrics["total_loss"] < best_test_total:
                best_test_total = test_metrics["total_loss"]
                torch.save(payload, args.output_dir / "best.pt")
            with (args.output_dir / "history.json").open("w") as f:
                json.dump(history, f, indent=2)

    cleanup_distributed()


if __name__ == "__main__":
    main()
