"""DeepCAD autoencoder adapter for encode/decode workflows."""

from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import torch


DEEP_CAD_ROOT = Path("/root/projects/CAD-MLLM/3rd_party/DeepCAD")
PRETRAINED_TAR = DEEP_CAD_ROOT / "proj_log" / "pretrained.tar"
AE_CHECKPOINT_MEMBER = "pretrained/model/ckpt_epoch1000.pth"


def _ensure_deepcad_importable() -> None:
    root_str = str(DEEP_CAD_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _build_cfg():
    _ensure_deepcad_importable()
    from cadlib.macro import ARGS_DIM, N_ARGS, ALL_COMMANDS, MAX_N_EXT, MAX_N_LOOPS, MAX_N_CURVES, MAX_TOTAL_LEN

    return SimpleNamespace(
        args_dim=ARGS_DIM,
        n_args=N_ARGS,
        n_commands=len(ALL_COMMANDS),
        n_layers=4,
        n_layers_decode=4,
        n_heads=8,
        dim_feedforward=512,
        d_model=256,
        dropout=0.1,
        dim_z=256,
        use_group_emb=True,
        max_n_ext=MAX_N_EXT,
        max_n_loops=MAX_N_LOOPS,
        max_n_curves=MAX_N_CURVES,
        max_num_groups=30,
        max_total_len=MAX_TOTAL_LEN,
    )


def _load_checkpoint_bytes(checkpoint_path: str | Path | None = None):
    if checkpoint_path is not None:
        return torch.load(str(checkpoint_path), map_location="cpu")

    with tarfile.open(PRETRAINED_TAR, "r") as tf:
        member = tf.getmember(AE_CHECKPOINT_MEMBER)
        with tf.extractfile(member) as f:
            buffer = io.BytesIO(f.read())
    return torch.load(buffer, map_location="cpu")


class DeepCADAdapter:
    """Thin wrapper around the DeepCAD autoencoder."""

    def __init__(self, checkpoint_path: str | Path | None = None, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        checkpoint = _load_checkpoint_bytes(checkpoint_path)
        self.cfg = self._build_cfg_from_checkpoint(checkpoint)
        self.model = self._build_model()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.max_total_len = int(self.cfg.max_total_len)

        _ensure_deepcad_importable()
        from cadlib.macro import EOS_IDX, EOS_VEC, CMD_ARGS_MASK

        self.eos_idx = int(EOS_IDX)
        self.eos_vec = np.asarray(EOS_VEC, dtype=np.int64)
        self.cmd_args_mask = torch.as_tensor(CMD_ARGS_MASK, dtype=torch.bool, device=self.device)

    def _build_model(self):
        _ensure_deepcad_importable()
        from model import CADTransformer

        return CADTransformer(self.cfg)

    def _build_cfg_from_checkpoint(self, checkpoint):
        cfg = _build_cfg()
        state_dict = checkpoint["model_state_dict"]
        encoder_pos_len = int(state_dict["encoder.embedding.pos_encoding.position"].shape[0])
        decoder_pos_len = int(state_dict["decoder.embedding.PE.position"].shape[0])
        # In DeepCAD, encoder positional table is max_total_len + 2, decoder table is max_total_len.
        inferred_max_total_len = decoder_pos_len
        if encoder_pos_len >= 2:
            inferred_max_total_len = min(inferred_max_total_len, encoder_pos_len - 2)
        cfg.max_total_len = inferred_max_total_len
        return cfg

    def pad_cad_vec(self, cad_vec: np.ndarray) -> np.ndarray:
        array = np.asarray(cad_vec, dtype=np.int64)
        if array.ndim == 1:
            array = array.reshape(-1, 17)
        if array.shape[0] > self.max_total_len:
            raise ValueError(f"cad_vec length {array.shape[0]} exceeds max_total_len={self.max_total_len}")
        if array.shape[0] == self.max_total_len:
            return array
        pad_len = self.max_total_len - array.shape[0]
        padding = np.repeat(self.eos_vec[None, :], pad_len, axis=0)
        return np.concatenate([array, padding], axis=0)

    def cad_vec_to_tensors(self, cad_vec_batch: np.ndarray | Iterable[np.ndarray]):
        if isinstance(cad_vec_batch, np.ndarray) and cad_vec_batch.ndim == 3:
            padded = np.asarray(cad_vec_batch, dtype=np.int64)
        else:
            padded = np.stack([self.pad_cad_vec(item) for item in cad_vec_batch], axis=0)
        commands = torch.as_tensor(padded[:, :, 0], dtype=torch.long, device=self.device)
        args = torch.as_tensor(padded[:, :, 1:], dtype=torch.long, device=self.device)
        return commands, args

    @torch.no_grad()
    def encode(self, cad_vec_batch: np.ndarray | Iterable[np.ndarray]) -> torch.Tensor:
        commands, args = self.cad_vec_to_tensors(cad_vec_batch)
        z = self.model(commands, args, encode_mode=True)
        return z[:, 0, :]

    def _prepare_z_batch(self, z_batch: torch.Tensor | np.ndarray) -> torch.Tensor:
        if not torch.is_tensor(z_batch):
            z_batch = torch.as_tensor(z_batch, dtype=torch.float32, device=self.device)
        else:
            z_batch = z_batch.to(device=self.device, dtype=torch.float32)
        if z_batch.ndim == 2:
            z_batch = z_batch.unsqueeze(1)
        elif z_batch.ndim != 3:
            raise ValueError(f"Expected z batch with ndim 2 or 3, got shape {tuple(z_batch.shape)}")
        return z_batch

    def decode_logits_with_grad(self, z_batch: torch.Tensor | np.ndarray) -> dict[str, torch.Tensor]:
        z_batch = self._prepare_z_batch(z_batch)
        return self.model(None, None, z=z_batch, return_tgt=False)

    @torch.no_grad()
    def decode_logits(self, z_batch: torch.Tensor | np.ndarray) -> dict[str, torch.Tensor]:
        z_batch = self._prepare_z_batch(z_batch)
        return self.model(None, None, z=z_batch, return_tgt=False)

    @torch.no_grad()
    def decode(self, z_batch: torch.Tensor | np.ndarray, trim_eos: bool = True) -> list[np.ndarray]:
        outputs = self.decode_logits(z_batch)
        command_logits = outputs["command_logits"]
        args_logits = outputs["args_logits"]

        out_command = torch.argmax(torch.softmax(command_logits, dim=-1), dim=-1)
        out_args = torch.argmax(torch.softmax(args_logits, dim=-1), dim=-1) - 1
        mask = ~self.cmd_args_mask[out_command.long()]
        out_args[mask] = -1
        cad_vec = torch.cat([out_command.unsqueeze(-1), out_args], dim=-1).detach().cpu().numpy()

        if not trim_eos:
            return [item for item in cad_vec]

        trimmed = []
        for item in cad_vec:
            eos_positions = np.where(item[:, 0] == self.eos_idx)[0]
            seq_len = int(eos_positions[0]) if len(eos_positions) > 0 else item.shape[0]
            trimmed.append(item[:seq_len])
        return trimmed
